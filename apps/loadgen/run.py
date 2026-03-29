#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import yaml


@dataclass
class Phase:
    name: str
    duration_s: float
    rps: float
    max_tokens: int
    temperature: float


@dataclass
class Tenant:
    name: str
    endpoint: str
    model: str
    prompt: str
    system_prompt: str
    concurrency: int
    timeout_s: float
    phases: list[Phase]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MIG QoS benchmark scenario.")
    parser.add_argument(
        "--scenario",
        required=True,
        help="Path to the YAML scenario file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where raw requests and summaries will be written.",
    )
    return parser.parse_args()


def resolve_path(base: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base / candidate).resolve()


def load_text(path: Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(encoding="utf-8").strip()


def expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}
    return value


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * pct
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def load_scenario(path: Path) -> tuple[str, list[Tenant], dict[str, Any]]:
    raw = expand_env(yaml.safe_load(path.read_text(encoding="utf-8")))
    scenario_dir = path.parent
    api_key_env = raw.get("api_key_env", "VLLM_API_KEY")
    timeout_s = float(raw.get("timeout_s", 180))
    system_prompt = load_text(resolve_path(scenario_dir, raw.get("system_prompt_path")))

    tenants: list[Tenant] = []
    for name, tenant_raw in raw["tenants"].items():
        if "${" in tenant_raw["model"]:
            raise SystemExit(
                f"Scenario model for tenant {name} is unresolved: {tenant_raw['model']}. "
                "Export MODEL_ID before running the benchmark."
            )
        prompt = load_text(resolve_path(scenario_dir, tenant_raw["prompt_path"]))
        phases = [
            Phase(
                name=phase["name"],
                duration_s=float(phase["duration_s"]),
                rps=float(phase["rps"]),
                max_tokens=int(phase["max_tokens"]),
                temperature=float(phase.get("temperature", 0.0)),
            )
            for phase in tenant_raw["phases"]
        ]
        tenants.append(
            Tenant(
                name=name,
                endpoint=tenant_raw["endpoint"].rstrip("/"),
                model=tenant_raw["model"],
                prompt=prompt,
                system_prompt=system_prompt,
                concurrency=int(tenant_raw.get("concurrency", 1)),
                timeout_s=float(tenant_raw.get("timeout_s", timeout_s)),
                phases=phases,
            )
        )

    return api_key_env, tenants, raw


def build_payload(tenant: Tenant, phase: Phase) -> dict[str, Any]:
    return {
        "model": tenant.model,
        "messages": [
            {"role": "system", "content": tenant.system_prompt},
            {"role": "user", "content": tenant.prompt},
        ],
        "temperature": phase.temperature,
        "max_tokens": phase.max_tokens,
        "stream": False,
    }


async def issue_request(
    client: httpx.AsyncClient,
    api_key: str,
    tenant: Tenant,
    phase: Phase,
    started_at: float,
) -> dict[str, Any]:
    request_id = str(uuid.uuid4())
    payload = build_payload(tenant, phase)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Request-Id": request_id,
    }

    t0 = time.perf_counter()
    try:
        response = await client.post(
            f"{tenant.endpoint}/chat/completions",
            headers=headers,
            json=payload,
            timeout=tenant.timeout_s,
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        data = response.json()
        usage = data.get("usage", {}) if isinstance(data, dict) else {}
        error = ""
        if response.status_code >= 400:
            error = json.dumps(data)
        return {
            "relative_start_s": started_at,
            "tenant": tenant.name,
            "phase": phase.name,
            "request_id": request_id,
            "status_code": response.status_code,
            "ok": response.status_code < 400,
            "latency_ms": round(latency_ms, 3),
            "endpoint": tenant.endpoint,
            "model": tenant.model,
            "prompt_chars": len(tenant.prompt),
            "max_tokens": phase.max_tokens,
            "temperature": phase.temperature,
            "prompt_tokens": usage.get("prompt_tokens", ""),
            "completion_tokens": usage.get("completion_tokens", ""),
            "total_tokens": usage.get("total_tokens", ""),
            "error": error,
        }
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "relative_start_s": started_at,
            "tenant": tenant.name,
            "phase": phase.name,
            "request_id": request_id,
            "status_code": 0,
            "ok": False,
            "latency_ms": round(latency_ms, 3),
            "endpoint": tenant.endpoint,
            "model": tenant.model,
            "prompt_chars": len(tenant.prompt),
            "max_tokens": phase.max_tokens,
            "temperature": phase.temperature,
            "prompt_tokens": "",
            "completion_tokens": "",
            "total_tokens": "",
            "error": str(exc),
        }


async def validate_scenario(
    api_key: str,
    tenants: list[Tenant],
) -> None:
    seen: set[tuple[str, int, float]] = set()
    for tenant in tenants:
        limits = httpx.Limits(
            max_keepalive_connections=tenant.concurrency,
            max_connections=tenant.concurrency * 2,
        )
        async with httpx.AsyncClient(limits=limits) as client:
            for phase in tenant.phases:
                key = (tenant.name, phase.max_tokens, phase.temperature)
                if key in seen:
                    continue
                seen.add(key)

                response = await client.post(
                    f"{tenant.endpoint}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "X-Request-Id": f"preflight-{uuid.uuid4()}",
                    },
                    json=build_payload(tenant, phase),
                    timeout=tenant.timeout_s,
                )
                if response.status_code >= 400:
                    try:
                        body = json.dumps(response.json())
                    except Exception:  # noqa: BLE001
                        body = response.text
                    raise SystemExit(
                        "Scenario validation failed before the timed run "
                        f"for tenant={tenant.name} phase={phase.name} "
                        f"max_tokens={phase.max_tokens}: {body}"
                    )


async def run_phase(
    client: httpx.AsyncClient,
    api_key: str,
    tenant: Tenant,
    phase: Phase,
    start_time: float,
    rows: list[dict[str, Any]],
) -> None:
    if phase.rps <= 0:
        await asyncio.sleep(phase.duration_s)
        return

    semaphore = asyncio.Semaphore(tenant.concurrency)
    phase_tasks: list[asyncio.Task[None]] = []
    interval = 1.0 / phase.rps
    end_time = time.perf_counter() + phase.duration_s

    async def worker(scheduled_at: float) -> None:
        async with semaphore:
            row = await issue_request(
                client=client,
                api_key=api_key,
                tenant=tenant,
                phase=phase,
                started_at=scheduled_at - start_time,
            )
            rows.append(row)

    next_fire = time.perf_counter()
    while next_fire < end_time:
        now = time.perf_counter()
        if now < next_fire:
            await asyncio.sleep(next_fire - now)
        scheduled_at = time.perf_counter()
        phase_tasks.append(asyncio.create_task(worker(scheduled_at)))
        next_fire += interval

    if phase_tasks:
        await asyncio.gather(*phase_tasks)


async def run_tenant(
    api_key: str,
    tenant: Tenant,
    start_time: float,
    rows: list[dict[str, Any]],
) -> None:
    limits = httpx.Limits(max_keepalive_connections=tenant.concurrency, max_connections=tenant.concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        for phase in tenant.phases:
            await run_phase(client, api_key, tenant, phase, start_time, rows)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["tenant"], row["phase"]), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (tenant, phase), group in sorted(grouped.items()):
        latencies = [float(item["latency_ms"]) for item in group if item["ok"]]
        total = len(group)
        ok = sum(1 for item in group if item["ok"])
        errors = total - ok
        avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else ""
        summary_rows.append(
            {
                "tenant": tenant,
                "phase": phase,
                "requests": total,
                "successes": ok,
                "errors": errors,
                "success_rate": round(ok / total, 4) if total else 0,
                "avg_latency_ms": avg_latency,
                "p50_latency_ms": round(percentile(latencies, 0.50), 3) if latencies else "",
                "p95_latency_ms": round(percentile(latencies, 0.95), 3) if latencies else "",
                "p99_latency_ms": round(percentile(latencies, 0.99), 3) if latencies else "",
            }
        )
    return summary_rows


async def async_main() -> None:
    args = parse_args()
    scenario_path = Path(args.scenario).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key_env, tenants, raw_scenario = load_scenario(scenario_path)
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise SystemExit(f"Environment variable {api_key_env} is required.")

    shutil.copy2(scenario_path, output_dir / "scenario.yaml")
    (output_dir / "scenario.resolved.json").write_text(
        json.dumps(raw_scenario, indent=2),
        encoding="utf-8",
    )

    start_time = time.perf_counter()
    rows: list[dict[str, Any]] = []
    await validate_scenario(api_key, tenants)
    await asyncio.gather(*(run_tenant(api_key, tenant, start_time, rows) for tenant in tenants))
    rows.sort(key=lambda item: float(item["relative_start_s"]))

    raw_fields = [
        "relative_start_s",
        "tenant",
        "phase",
        "request_id",
        "status_code",
        "ok",
        "latency_ms",
        "endpoint",
        "model",
        "prompt_chars",
        "max_tokens",
        "temperature",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "error",
    ]
    write_csv(output_dir / "requests.csv", rows, raw_fields)

    summary_rows = build_summary(rows)
    summary_fields = [
        "tenant",
        "phase",
        "requests",
        "successes",
        "errors",
        "success_rate",
        "avg_latency_ms",
        "p50_latency_ms",
        "p95_latency_ms",
        "p99_latency_ms",
    ]
    write_csv(output_dir / "summary.csv", summary_rows, summary_fields)

    manifest = {
        "scenario": str(scenario_path),
        "output_dir": str(output_dir),
        "request_count": len(rows),
        "tenants": [tenant.name for tenant in tenants],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
