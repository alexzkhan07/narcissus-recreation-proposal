"""
Recreate Figure 3 from NARCISSUS (arXiv:2204.05255).

Two side-by-side panels:
  - Left:  Tar-ACC (%) vs Target-class poison ratio (%)
  - Right: ASR     (%) vs Target-class poison ratio (%)

Four trigger methods plotted: BadNets, Blend, Label-Consistent, Ours.

Input CSV schema (long format, one row per (method, poison_ratio, seed)):
    method,poison_ratio,seed,tar_acc,asr

Mean curves are drawn as lines+markers. Shaded bands show ±1 std across seeds
when more than one seed is present; with a single seed the band collapses
to the line.

Usage:
    python plot_figure3.py \
        --csv ../results/figure3_results.csv \
        --out ../figures/figure3.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHOD_ORDER = ["BadNets", "Blend", "Label-Consistent", "Ours"]

STYLE = {
    "BadNets":          {"color": "#E89A7A", "marker": "o", "mfc": "white"},
    "Blend":            {"color": "#5FA8C4", "marker": "D", "mfc": "white"},
    "Label-Consistent": {"color": "#7FA674", "marker": "^", "mfc": "white"},
    "Ours":             {"color": "#7A2828", "marker": "s", "mfc": "white"},
}


def aggregate(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = df.groupby(["method", "poison_ratio"])[value_col]
    return g.agg(["mean", "std"]).reset_index().fillna({"std": 0.0})


def plot_panel(ax, df: pd.DataFrame, value_col: str, ylabel: str, ylim, yticks):
    agg = aggregate(df, value_col)
    for method in METHOD_ORDER:
        sub = agg[agg["method"] == method].sort_values("poison_ratio")
        if sub.empty:
            continue
        x = sub["poison_ratio"].to_numpy()
        m = sub["mean"].to_numpy()
        s = sub["std"].to_numpy()
        st = STYLE[method]
        ax.fill_between(x, m - s, m + s, color=st["color"], alpha=0.18, linewidth=0)
        ax.plot(
            x, m,
            color=st["color"], marker=st["marker"], mfc=st["mfc"],
            markersize=6, linewidth=1.5, label=method,
        )
    ax.set_xlabel("Target-class poison ratio (%)")
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70])
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="path to results CSV")
    parser.add_argument("--out", type=Path, required=True, help="output image path")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    expected = {"method", "poison_ratio", "seed", "tar_acc", "asr"}
    missing = expected - set(df.columns)
    if missing:
        raise SystemExit(f"CSV missing columns: {sorted(missing)}")

    unknown = sorted(set(df["method"]) - set(METHOD_ORDER))
    if unknown:
        raise SystemExit(
            f"Unknown method(s) in CSV: {unknown}. Expected one of {METHOD_ORDER}."
        )

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(9.0, 3.4))

    plot_panel(
        ax_left, df, value_col="tar_acc",
        ylabel="Tar-ACC (%)",
        ylim=(82, 95), yticks=[82, 84, 86, 88, 90, 92, 94],
    )
    plot_panel(
        ax_right, df, value_col="asr",
        ylabel="ASR (%)",
        ylim=(0, 105), yticks=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    )

    handles, labels = ax_left.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center", ncol=len(METHOD_ORDER),
        bbox_to_anchor=(0.5, 1.02),
        frameon=True, framealpha=1.0, edgecolor="0.7",
        handletextpad=0.5, columnspacing=1.5,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
