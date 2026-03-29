#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

_CACHE_ROOT = (Path.cwd() / ".cache").resolve()
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "fontconfig").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str((_CACHE_ROOT / "matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PHASE_ORDER = ["warmup", "quiet", "burst", "burst_2"]
TENANT_ORDER = ["tenant_a", "tenant_b"]
TENANT_COLORS = {
    "tenant_a": "#1f77b4",
    "tenant_b": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate charts for a single experiment run.")
    parser.add_argument("--requests-csv", required=True, help="Path to request-level CSV.")
    parser.add_argument("--summary-csv", required=True, help="Path to summary CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for output charts.")
    parser.add_argument("--label", required=True, help="Label for chart titles.")
    return parser.parse_args()


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["phase"] = pd.Categorical(df["phase"], categories=PHASE_ORDER, ordered=True)
    df["tenant"] = pd.Categorical(df["tenant"], categories=TENANT_ORDER, ordered=True)
    return df.sort_values(["tenant", "phase"])


def plot_latency_timeline(requests: pd.DataFrame, output_dir: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    for tenant in TENANT_ORDER:
        subset = requests[requests["tenant"] == tenant]
        if subset.empty:
            continue
        ax.plot(
            subset["relative_start_s"],
            subset["latency_ms"],
            marker="o",
            linestyle="-",
            linewidth=1.0,
            markersize=2.5,
            alpha=0.55,
            color=TENANT_COLORS[tenant],
            label=tenant,
        )

    ax.set_title(f"{label}: request latency over time")
    ax.set_xlabel("Seconds since scenario start")
    ax.set_ylabel("Latency (ms)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "latency_timeline.png", dpi=180)
    plt.close(fig)


def plot_phase_metric(
    summary: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    phases = PHASE_ORDER
    x = range(len(phases))
    width = 0.35

    for idx, tenant in enumerate(TENANT_ORDER):
        tenant_rows = (
            summary[summary["tenant"] == tenant]
            .set_index("phase")
            .reindex(phases)
        )
        values = tenant_rows[value_column].fillna(0).astype(float).tolist()
        offsets = [item + (idx - 0.5) * width for item in x]
        ax.bar(offsets, values, width=width, label=tenant, color=TENANT_COLORS[tenant], alpha=0.85)

    ax.set_xticks(list(x))
    ax.set_xticklabels(phases)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requests = pd.read_csv(args.requests_csv)
    summary = pd.read_csv(args.summary_csv)
    requests = ordered(requests)
    summary = ordered(summary)

    plot_latency_timeline(requests, output_dir, args.label)
    plot_phase_metric(
        summary,
        value_column="p95_latency_ms",
        ylabel="p95 latency (ms)",
        title=f"{args.label}: phase p95 latency",
        output_path=output_dir / "phase_p95.png",
    )
    plot_phase_metric(
        summary,
        value_column="success_rate",
        ylabel="Success rate",
        title=f"{args.label}: phase success rate",
        output_path=output_dir / "phase_success_rate.png",
    )


if __name__ == "__main__":
    main()
