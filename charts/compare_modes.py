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
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

PHASE_ORDER = ["quiet", "burst", "burst_2"]
PHASE_LABELS = {"quiet": "Quiet", "burst": "Burst", "burst_2": "Burst+"}
MODE_COLORS = {
    "shared": "#d62728",
    "mig": "#1f77b4",
}
MODE_LABELS = {
    "shared": "Shared GPU (7 tenants)",
    "mig": "MIG (7 × 1g.5gb slices)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare shared and MIG summary CSVs.")
    parser.add_argument("--shared-summary", required=True, help="Path to shared summary CSV.")
    parser.add_argument("--mig-summary", required=True, help="Path to MIG summary CSV.")
    parser.add_argument("--output-dir", required=True, help="Directory for output charts.")
    return parser.parse_args()


def load_summary(path: str, mode: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["mode"] = mode
    return df


def plot_tenant_a_hero(df: pd.DataFrame, output_path: Path) -> None:
    """Hero chart: Tenant A p95 bar chart per phase, shared vs MIG, with isolation delta annotation."""
    subset = df[df["tenant"] == "tenant_a"]
    phases = PHASE_ORDER
    x = np.arange(len(phases))
    width = 0.32

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, mode in enumerate(["shared", "mig"]):
        rows = subset[subset["mode"] == mode].set_index("phase")
        vals = [rows.loc[p, "p95_latency_ms"] if p in rows.index else 0 for p in phases]
        bars = ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
            alpha=0.88,
            zorder=3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 30,
                f"{v:.0f}ms",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=MODE_COLORS[mode],
                fontweight="bold",
            )

    shared_rows = subset[subset["mode"] == "shared"].set_index("phase")
    mig_rows = subset[subset["mode"] == "mig"].set_index("phase")
    for j, p in enumerate(phases):
        if p in shared_rows.index and p in mig_rows.index:
            s_val = shared_rows.loc[p, "p95_latency_ms"]
            m_val = mig_rows.loc[p, "p95_latency_ms"]
            delta = s_val - m_val
            pct = delta / s_val * 100
            ax.annotate(
                f"−{delta:.0f}ms\n(−{pct:.0f}%)",
                xy=(j, (s_val + m_val) / 2),
                xytext=(j + 0.42, (s_val + m_val) / 2),
                fontsize=7.5,
                color="#444",
                va="center",
                arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.8),
            )

    ax.set_xticks(x)
    ax.set_xticklabels([PHASE_LABELS[p] for p in phases], fontsize=11)
    ax.set_ylabel("p95 Latency (ms)", fontsize=11)
    ax.set_title("Tenant A — Protected Tenant p95 Latency\nShared GPU vs MIG Isolation", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(subset["p95_latency_ms"]) * 1.22)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_all_tenants_p95(df: pd.DataFrame, output_path: Path) -> None:
    """Show all tenants' quiet-phase p95 in shared vs MIG — proves saturation and isolation."""
    quiet = df[df["phase"] == "quiet"].copy()
    tenants = sorted(quiet["tenant"].unique())

    x = np.arange(len(tenants))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))

    for i, mode in enumerate(["shared", "mig"]):
        rows = quiet[quiet["mode"] == mode].set_index("tenant")
        vals = [rows.loc[t, "p95_latency_ms"] if t in rows.index else 0 for t in tenants]
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
            alpha=0.88,
            zorder=3,
        )

    xlabels = ["Tenant A\n(protected)" if t == "tenant_a" else t.replace("tenant_", "Tenant ") for t in tenants]
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("p95 Latency (ms) — Quiet Phase", fontsize=11)
    ax.set_title("All Tenants — Quiet-Phase p95: Shared GPU vs MIG\n(MIG hardware isolation eliminates cross-tenant bandwidth contention)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    a_shared = quiet[(quiet["tenant"] == "tenant_a") & (quiet["mode"] == "shared")]["p95_latency_ms"].values
    a_mig = quiet[(quiet["tenant"] == "tenant_a") & (quiet["mode"] == "mig")]["p95_latency_ms"].values
    if len(a_shared) and len(a_mig):
        ax.axhline(a_mig[0], color=MODE_COLORS["mig"], linestyle="--", linewidth=1, alpha=0.5)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_burst_comparison(df: pd.DataFrame, output_path: Path) -> None:
    """Tenant A p95 across all phases as a line, showing both modes. Classic latency timeline."""
    subset = df[df["tenant"] == "tenant_a"]
    phases = ["warmup"] + PHASE_ORDER

    fig, ax = plt.subplots(figsize=(9, 4))
    for mode in ["shared", "mig"]:
        rows = subset[subset["mode"] == mode].set_index("phase")
        vals = [rows.loc[p, "p95_latency_ms"] if p in rows.index else None for p in phases]
        ax.plot(
            [PHASE_LABELS.get(p, p.capitalize()) for p in phases],
            vals,
            marker="o",
            linewidth=2.5,
            markersize=7,
            label=MODE_LABELS[mode],
            color=MODE_COLORS[mode],
        )

    shared_vals = subset[subset["mode"] == "shared"].set_index("phase").reindex(phases)["p95_latency_ms"].ffill().tolist()
    mig_vals = subset[subset["mode"] == "mig"].set_index("phase").reindex(phases)["p95_latency_ms"].ffill().tolist()
    ax.fill_between(
        range(len(phases)),
        shared_vals,
        mig_vals,
        alpha=0.08,
        color="#888",
    )

    ax.set_ylabel("p95 Latency (ms)", fontsize=11)
    ax.set_xlabel("Experiment Phase", fontsize=11)
    ax.set_title("Tenant A p95 Latency Timeline — Shared vs MIG", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    shared = load_summary(args.shared_summary, "shared")
    mig = load_summary(args.mig_summary, "mig")
    combined = pd.concat([shared, mig], ignore_index=True)

    plot_tenant_a_hero(combined, output_dir / "tenant_a_isolation_hero.png")
    print("  ✓ tenant_a_isolation_hero.png")

    plot_all_tenants_p95(combined, output_dir / "all_tenants_quiet_p95.png")
    print("  ✓ all_tenants_quiet_p95.png")

    plot_burst_comparison(combined, output_dir / "tenant_a_p95_timeline.png")
    print("  ✓ tenant_a_p95_timeline.png")


if __name__ == "__main__":
    main()
