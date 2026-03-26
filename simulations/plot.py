#!/usr/bin/env python3

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

KERNEL_BENCHMARKS = ["Godunov", "TimeStep",  "ConsToPrim", "PrimToCons"]
ALL_BENCHMARKS = KERNEL_BENCHMARKS
ALL_BENCHMARKS.append("EulerSimulation")

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

FILES={
    "a100":   "./results/ruche/a100/[412780]_ALL-a100_bm_a100.json",
     "v100":   "./results/ruche/v100/[412885]_ALL-v100_bm_v100.json",
     "skx":   "./results/ruche/skx/[419180]_ALL-skx_bm_skx.json",

}

# each tuple is (slower, faster) → speedup = time[slower] / time[faster]
SPEEDUPS = [
    ("v100", "a100"),
]

OUT_DIR = "results/plots"

# ---------------------------------------------------------
# Load
# ---------------------------------------------------------
def load(path, benchmarks):
    with open(path) as f:
        data = json.load(f)["benchmarks"]
    rows = []
    for b in data:
        name = b["name"]
        match = next((bm for bm in benchmarks if bm in name), None)
        if match is None:
            continue
        rows.append({
            "benchmark": match,
            "size": int(name.split("/")[-1]),
            "time": b["cpu_time"],
            "cells_per_second": b.get("cells_per_second"),
            "bytes_per_second": b.get("bytes_per_second"),
        })
    return pd.DataFrame(rows)

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------
def plot_time(files, speedups, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {label: load(path, ALL_BENCHMARKS) for label, path in files.items()}

    for bm in ALL_BENCHMARKS:
        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        # --- times ---
        for label, df in datasets.items():
            sub = df[df["benchmark"] == bm].sort_values("size")
            if sub.empty:
                continue
            ax1.plot(sub["size"], sub["time"], marker="o", label=label)

        # --- speedups ---
        for slow, fast in speedups:
            if slow not in datasets or fast not in datasets:
                continue
            df_slow = datasets[slow][datasets[slow]["benchmark"] == bm].sort_values("size")
            df_fast = datasets[fast][datasets[fast]["benchmark"] == bm].sort_values("size")
            merged = df_slow.merge(df_fast, on="size", suffixes=(f"_{slow}", f"_{fast}"))
            merged["speedup"] = merged[f"time_{slow}"] / merged[f"time_{fast}"]
            ax2.plot(merged["size"], merged["speedup"], linestyle="--",
                     marker="s", label=f"{slow}→{fast} speedup", color="red")

        ax1.set_title(bm)
        ax1.set_xlabel("nx")
        ax1.set_ylabel("Time (ns)")
        ax2.set_ylabel("Speedup v100/a100")
        ax2.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8)

        ax1.grid(True)
        ax1.set_yscale("log", base=2)
        plt.tight_layout()
        path = out_dir / f"{bm}_times.png"
        plt.savefig(path, dpi=200)
        plt.close()
        print("saved:", path)


def plot_items(files,out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {label: load(path, KERNEL_BENCHMARKS) for label, path in files.items()}

    for bm in KERNEL_BENCHMARKS:

        fig, ax1 = plt.subplots(figsize=(9, 5))

        # --- cells_per_second ---
        for label, df in datasets.items():
            sub = df[df["benchmark"] == bm].sort_values("size")
            if sub.empty:
                continue
            ax1.plot(sub["size"], sub["cells_per_second"], marker="o", label=label)


        
        ax1.set_title(bm)
        ax1.set_xlabel("nx")
        ax1.set_ylabel("cells/s")

        lines1, labels1 = ax1.get_legend_handles_labels()
        ax1.legend(lines1, labels1, fontsize=8)

        ax1.grid(True)
        plt.tight_layout()
        path = out_dir / f"{bm}_items.png"
        plt.savefig(path, dpi=200)
        plt.close()
        print("saved:", path)

plot_time(FILES, SPEEDUPS, OUT_DIR)
plot_items(FILES, OUT_DIR)


