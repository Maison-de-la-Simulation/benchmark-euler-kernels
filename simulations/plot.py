"""
Plotting script for plotting/comparing Google Benchmark JSON files.

This script has two main functionalities:
    - plotting scalar vs vectorized benchmark outputs
    - creating CSV files comparing benchmark variants
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

HW_COLORS = {
    "skx": "C0",
    "genoa": "C1",
    "mi300": "C2",
    "gh200": "C3",
    "a100": "C4",
    "unknown": "gray",
}

SIMD_WIDTH = {
    "skx": 8,
    "genoa": 8,
    "gh200": 2,
}

HW_LABELS = {
    "skx": "Intel Xeon Gold",
    "genoa": "AMD Genoa",
    "mi300": "MI300A (APU)",
    "gh200": "GH200 ARM (CPU)",
    "a100": "NVIDIA A100 (GPU)",
    "unknown": "Unknown",
}

# pylint: disable=too-many-locals
# pylint: disable=too-many-statements


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------


def load_one(path):
    """Load and parse a Google Benchmark JSON file."""

    with open(path, encoding="utf-8") as file:
        raw = json.load(file)

    caches = {
        cache["level"]: cache["size"]
        for cache in raw["context"]["caches"]
        if cache["type"] == "Unified"
    }

    rows = []

    for benchmark in raw["benchmarks"]:
        name = benchmark["name"]

        rows.append(
            {
                "benchmark": name.split("/")[0],
                "size": int(name.split("/")[-2]),
                "cells_per_second": benchmark.get("cells_per_second"),
                "bytes_per_second": benchmark.get("bytes_per_second"),
                "real_time_ns": benchmark.get("real_time"),
            }
        )

    return pd.DataFrame(rows), caches


# -----------------------------------------------------------------------------
# Plot helpers
# -----------------------------------------------------------------------------


def _plot_series(ax, df_series, y_key, style):
    """Plot a benchmark series."""

    df_series = df_series[df_series[y_key].notna()]

    if df_series.empty:
        return

    ax.plot(
        df_series["size"],
        df_series[y_key],
        linestyle=style["linestyle"],
        color=style["color"],
        label=style["label"],
        alpha=0.5,
    )

    ax.scatter(
        df_series["size"],
        df_series[y_key],
        marker="o",
        color=style["color"],
        zorder=5,
        alpha=0.5,
        s=8,
    )


def get_hw(path):
    """Infer hardware name from filename."""

    name = Path(path).name.lower()

    for tag in ["skx", "genoa", "mi300", "gh200", "a100"]:
        if tag in name:
            return tag

    return "unknown"


def hw_label_speedup(hw):
    """Generate speedup plot legend label."""

    base = HW_LABELS.get(hw, hw)
    width = SIMD_WIDTH.get(hw, "N/A")

    return f"{base} | (W={width})"


# -----------------------------------------------------------------------------
# Benchmark comparison
# -----------------------------------------------------------------------------


def compare_benchmarks(path_a, path_b, out_csv, cols=None):
    """Compare two benchmark JSON files."""

    df_a, _ = load_one(path_a)
    df_b, _ = load_one(path_b)

    merged = pd.merge(
        df_a,
        df_b,
        on=["benchmark", "size"],
        suffixes=("_a", "_b"),
        how="inner",
    )

    merged["real_time_speedup"] = (
        merged["real_time_ns_a"] / merged["real_time_ns_b"]
    )

    for col in ("cells_per_second", "bytes_per_second"):
        a_col = f"{col}_a"
        b_col = f"{col}_b"

        if a_col in merged and b_col in merged:
            merged[f"{col}_speedup"] = merged[b_col] / merged[a_col]

    if cols:
        merged = merged[cols]

    mean_row = merged.mean(numeric_only=True).to_frame().T
    mean_row["benchmark"] = "MEAN"
    mean_row["size"] = pd.NA

    merged = pd.concat([merged, mean_row], ignore_index=True)

    rounding = {c: 5 for c in merged.columns if "speedup" in c}
    rounding |= {c: 5 for c in merged.columns if "time" in c}
    rounding |= {
        c: 5
        for c in merged.columns
        if "cells_per_second" in c or "bytes_per_second" in c
    }

    merged = merged.round(rounding)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out_csv, index=False)

    return merged


def compare_godunov_benchmarks(
    path,
    out_csv,
    base_name="Godunov",
    opti_name="GodunovOpti",
    cols=None,
):
    """Compare Godunov baseline and optimized variants."""

    df, _ = load_one(path)

    df_base = df[df["benchmark"] == base_name].copy()
    df_opti = df[df["benchmark"] == opti_name].copy()

    merged = pd.merge(
        df_base,
        df_opti,
        on=["size"],
        suffixes=("_base", "_opti"),
        how="inner",
    )

    merged["real_time_speedup"] = (
        merged["real_time_ns_base"] / merged["real_time_ns_opti"]
    )

    for col in ("cells_per_second", "bytes_per_second"):
        base_col = f"{col}_base"
        opti_col = f"{col}_opti"

        if base_col in merged.columns and opti_col in merged.columns:
            merged[f"{col}_speedup"] = (
                merged[opti_col] / merged[base_col]
            )

    if cols:
        merged = merged[cols]

    mean_row = merged.mean(numeric_only=True).to_frame().T
    mean_row["benchmark"] = "MEAN"
    mean_row["size"] = pd.NA

    merged = pd.concat([merged, mean_row], ignore_index=True)

    rounding = {c: 5 for c in merged.columns if "speedup" in c}
    rounding |= {c: 5 for c in merged.columns if "time" in c}
    rounding |= {
        c: 5
        for c in merged.columns
        if "cells_per_second" in c or "bytes_per_second" in c
    }

    merged = merged.round(rounding)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out_csv, index=False)

    return merged


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_hw_speedup(res_dir, out_dir):
    """Plot vector/scalar speedup."""

    res_dir = Path(res_dir)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(res_dir.glob("*.json"))

    if not files:
        raise FileNotFoundError(f"No JSON files in {res_dir}")

    data = []

    for file in files:
        df, _ = load_one(str(file))
        df["hardware"] = get_hw(file)
        data.append(df)

    df = pd.concat(data, ignore_index=True)

    bases = {
        benchmark.replace("Vectorized", "")
        for benchmark in df["benchmark"].unique()
    }

    for base in bases:

        _, ax = plt.subplots(figsize=(8, 5))

        metric = (
            "bytes_per_second"
            if base != "EulerSimulation"
            else "real_time_ns"
        )

        for hw in df["hardware"].unique():

            subset = df[df["hardware"] == hw]

            scalar = subset[
                subset["benchmark"] == base
            ].sort_values("size")

            vector = subset[
                subset["benchmark"] == f"{base}Vectorized"
            ].sort_values("size")

            if scalar.empty or vector.empty:
                continue

            merged = pd.merge(
                scalar[["size", metric]],
                vector[["size", metric]],
                on="size",
                suffixes=("_scalar", "_vector"),
            )

            if merged.empty:
                continue

            if metric == "bytes_per_second":
                speedup = (
                    merged[f"{metric}_vector"]
                    / merged[f"{metric}_scalar"]
                )
            else:
                speedup = (
                    merged[f"{metric}_scalar"]
                    / merged[f"{metric}_vector"]
                )

            ax.plot(
                merged["size"],
                speedup,
                color=HW_COLORS.get(hw, "black"),
                label=hw_label_speedup(hw),
                marker="o",
                markersize=3,
                alpha=0.7,
            )

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)

        ax.set_title(f"{base} Speedup (Vectorized vs Scalar)")
        ax.set_xlabel("n (cube width in cells)")
        ax.set_ylabel("Speedup")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()

        unique_handles = []
        unique_labels = []

        for handle, label in zip(handles, labels):
            if label not in unique_labels:
                unique_handles.append(handle)
                unique_labels.append(label)

        ax.legend(unique_handles, unique_labels, fontsize=8)

        plt.tight_layout()

        plt.savefig(
            out_dir / f"{base}_speedup.png",
            dpi=200,
        )

        plt.close()


def plot_hw_scalar_vector(res_dir, out_dir, title=""):
    """Plot scalar vs vectorized benchmark throughput."""

    res_dir = Path(res_dir)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(res_dir.glob("*.json"))

    if not files:
        raise FileNotFoundError(f"No JSON files in {res_dir}")

    data = []

    for file in files:
        df, _ = load_one(str(file))
        df["hardware"] = get_hw(file)
        data.append(df)

    df = pd.concat(data, ignore_index=True)

    bases = {
        benchmark.replace("Vectorized", "")
        for benchmark in df["benchmark"].unique()
    }

    for base in bases:

        _, ax_cells = plt.subplots(figsize=(8, 5))

        is_euler = base == "EulerSimulation"

        ax_bytes = ax_cells.twinx() if not is_euler else None

        hardware_handles = {}

        for hw in df["hardware"].unique():

            subset = df[df["hardware"] == hw]

            scalar = subset[
                subset["benchmark"] == base
            ].sort_values("size")

            vector = subset[
                subset["benchmark"] == f"{base}Vectorized"
            ].sort_values("size")

            if scalar.empty or vector.empty:
                continue

            color = HW_COLORS.get(hw, "black")

            scalar_style = {
                "color": color,
                "label": "_nolegend_",
                "linestyle": "-",
            }

            vector_style = {
                "color": color,
                "label": "_nolegend_",
                "linestyle": "--",
            }

            _plot_series(
                ax_cells,
                scalar,
                "cells_per_second",
                scalar_style,
            )

            _plot_series(
                ax_cells,
                vector,
                "cells_per_second",
                vector_style,
            )

            if hw not in hardware_handles:
                hardware_handles[hw] = ax_cells.lines[-2]

            if not is_euler and ax_bytes is not None:

                _plot_series(
                    ax_bytes,
                    scalar,
                    "bytes_per_second",
                    scalar_style,
                )

                _plot_series(
                    ax_bytes,
                    vector,
                    "bytes_per_second",
                    vector_style,
                )

        legend1 = ax_cells.legend(
            hardware_handles.values(),
            [hw_label_speedup(hw) for hw in hardware_handles],
            bbox_to_anchor=(0.7, 0.3),
            loc="best",
            fontsize=8,
            title="Hardware",
        )

        ax_cells.add_artist(legend1)

        (scalar_proxy,) = ax_cells.plot([], [], "-", color="black")
        (vector_proxy,) = ax_cells.plot([], [], "--", color="black")

        ax_cells.legend(
            [scalar_proxy, vector_proxy],
            ["scalar", "vector"],
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            fontsize=8,
            title="Mode",
        )

        ax_cells.set_yscale("log")
        ax_cells.set_ylabel("cells per second")

        if not is_euler and ax_bytes is not None:
            ax_bytes.set_yscale("log")
            ax_bytes.set_ylabel("bytes per second")

        ax_cells.set_xlabel("n (cube width in cells)")
        ax_cells.set_title(f"{title} {base}".strip())
        ax_cells.grid(True)

        plt.tight_layout()

        plt.savefig(
            out_dir / f"{base}_hw_compare.png",
            dpi=200,
        )

        plt.close()


# -----------------------------------------------------------------------------
# Scaling plots
# -----------------------------------------------------------------------------


def plot_scaling_dir(path):
    """Plot strong and weak scaling CSV files."""

    files = sorted(
        file for file in os.listdir(path) if file.endswith(".csv")
    )

    strong_scalar = None
    strong_vector = None
    weak_scalar = None
    weak_vector = None

    for file in files:

        full = os.path.join(path, file)

        if "strong" in file and "scalar" in file:
            strong_scalar = pd.read_csv(full)

        elif "strong" in file and "vector" in file:
            strong_vector = pd.read_csv(full)

        elif "weak" in file and "scalar" in file:
            weak_scalar = pd.read_csv(full)

        elif "weak" in file and "vector" in file:
            weak_vector = pd.read_csv(full)

    def plot_strong(df, ax, title_text):
        """Plot strong scaling."""

        threads = df["threads"].to_numpy()
        time = df["time_s"].to_numpy()

        speedup = time[0] / time
        efficiency = 100 * speedup / threads

        ax.plot(threads, speedup, marker="o", label="speedup")
        ax.plot(threads, threads, "--", label="ideal")

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Threads")
        ax.set_ylabel("Speedup")
        ax.set_title(title_text)
        ax.grid(True)

        ax2 = ax.twinx()

        ax2.plot(
            threads,
            efficiency,
            marker="s",
            linestyle=":",
            label="efficiency",
        )

        ax2.set_ylabel("Efficiency (%)")

        l1, lab1 = ax.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()

        ax.legend(l1 + l2, lab1 + lab2, loc="upper left")

    def plot_weak(df, ax, title_text):
        """Plot weak scaling."""

        threads = df["threads"].to_numpy()
        time = df["time_s"].to_numpy()

        normalized = time / time[0]

        ax.plot(
            threads,
            normalized,
            marker="o",
            label="runtime normalized",
        )

        ax.plot(
            threads,
            threads,
            "--",
            label="ideal weak scaling",
        )

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Threads")
        ax.set_ylabel("T / T1")
        ax.set_title(title_text)
        ax.grid(True)
        ax.legend()

    _, axes = plt.subplots(2, 2, figsize=(12, 9))

    if strong_scalar is not None:
        plot_strong(strong_scalar, axes[0, 0], "Strong Scalar")

    if strong_vector is not None:
        plot_strong(strong_vector, axes[0, 1], "Strong Vector")

    if weak_scalar is not None:
        plot_weak(weak_scalar, axes[1, 0], "Weak Scalar")

    if weak_vector is not None:
        plot_weak(weak_vector, axes[1, 1], "Weak Vector")

    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

COLS = [
    "benchmark_base",
    "size",
    "cells_per_second_speedup",
    "cells_per_second_base",
    "cells_per_second_opti",
]

# compare_godunov_benchmarks(
#     "./results/adastra/mi250/4971156_mi250x__Godunov_GodunovOpti_GodunovVectorized__.json",
#     "opti-base.csv",
#     base_name="Godunov",
#     opti_name="GodunovOpti",
#     cols=COLS,
# )

plot_hw_scalar_vector("./results", "./results/test_polts/")
# plot_hw_speedup("./results", "./results/test_polts/")
# plot_scaling_dir("./results/scaling/genoa/")
