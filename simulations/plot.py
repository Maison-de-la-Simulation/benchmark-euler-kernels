"""
Plotting script for plotting/comparing Google Benchmark JSON files.

This script has two main functionalities:
    - plotting scalar vs vectorized benchmark outputs
    - creating a csv file comparing two different benchmarks
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# %%


def load_one(path):
    """Load and parse a Google Benchmark JSON file."""

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    caches = {c["level"]: c["size"] for c in raw["context"]["caches"] if c["type"] == "Unified"}

    rows = []

    for b in raw["benchmarks"]:
        name = b["name"]

        row = {
            "benchmark": name.split("/")[0],
            "size": int(name.split("/")[-2]),
            "cells_per_second": b.get("cells_per_second"),
            "bytes_per_second": b.get("bytes_per_second"),
            "real_time_ns": b.get("real_time"),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    return df, caches


def _plot_series(ax, df_series, color, label, y_key, linestyle="-"):
    """Plot a benchmark series with aligned and unaligned data points."""

    df_series = df_series[df_series[y_key].notna()]

    if df_series.empty:
        return

    ax.plot(
        df_series["size"],
        df_series[y_key],
        linestyle,
        color=color,
        label=label,
        alpha=0.5,
    )

    ax.scatter(
        df_series["size"],
        df_series[y_key],
        marker="o",
        color=color,
        zorder=5,
        alpha=0.5,
        s=8,  # 👈 fix
    )


def compare_benchmarks(path_a, path_b, out_csv, cols=None):
    """Compare two benchmark results and generate a CSV with speedup metrics.

    Args:
        path_a: Path to the first benchmark result JSON file.
        path_b: Path to the second benchmark result JSON file.
        out_csv: Path to the output CSV file.
        cols: Optional list of columns to include in the output CSV.

    Returns:
        DataFrame containing the merged and computed comparison results.
    """
    df_a, _ = load_one(path_a)
    df_b, _ = load_one(path_b)

    merged = pd.merge(
        df_a,
        df_b,
        on=["benchmark", "size"],
        suffixes=("_a", "_b"),
        how="inner",
    )

    merged["real_time_speedup"] = merged["real_time_ns_a"] / merged["real_time_ns_b"]

    for col in ("cells_per_second", "bytes_per_second"):
        a_col = f"{col}_a"
        b_col = f"{col}_b"
        if a_col in merged and b_col in merged:
            merged[f"{col}_speedup"] = merged[b_col] / merged[a_col]

    merged = merged[merged.benchmark == "PrimToConsVectorized"]
    if cols:
        merged = merged[cols]

    mean_row = merged.mean(numeric_only=True).to_frame().T
    mean_row["benchmark"] = "MEAN"
    mean_row["size"] = pd.NA
    merged = pd.concat([merged, mean_row], ignore_index=True)

    rounding = {c: 5 for c in merged.columns if "speedup" in c}
    rounding |= {c: 5 for c in merged.columns if "time" in c}
    rounding |= {c: 5 for c in merged.columns if "cells_per_second" in c or "bytes_per_second" in c}
    merged = merged.round(rounding)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


# %%


from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# shared helpers
# ----------------------------


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
    # GPUs intentionally omitted
}
HW_LABELS = {
    "skx": "Intel Xeon Gold ",
    "genoa": "AMD Genoa",
    "mi300": "MI300A (APU)",
    "gh200": "GH200 ARM (CPU)",
    "a100": "NVIDIA A100 (GPU)",
    "unknown": "Unknown",
}

# CPU SIMD "upper bounds"


def get_hw(path):
    name = Path(path).name.lower()
    for tag in ["skx", "genoa", "mi300", "gh200", "a100"]:
        if tag in name:
            return tag
    return "unknown"


def hw_label(hw, mode):
    base = HW_LABELS.get(hw, hw)
    if mode is None:
        return ""

    if mode == "scalar":
        return f"{base} | scalar"

    w = f"(W={SIMD_WIDTH.get(hw) or 'N/A'})"
    return f"{base} | vector {w}"


def hw_label_speedup(hw):
    base = HW_LABELS.get(hw, hw)

    w = f"(W={SIMD_WIDTH.get(hw) or 'N/A'})"
    return f"{base} | {w}"


def plot_hw_speedup(res_dir, out_dir):

    res_dir = Path(res_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(res_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files in {res_dir}")

    # load
    data = []
    for f in files:
        df, _ = load_one(str(f))
        df["hardware"] = get_hw(f)
        data.append(df)

    df = pd.concat(data, ignore_index=True)

    bases = {b.replace("Vectorized", "") for b in df["benchmark"].unique()}

    for base in bases:

        fig, ax = plt.subplots(figsize=(8, 5))
        metric = "bytes_per_second" if base != "EulerSimulation" else "real_time_ns"

        for hw in df["hardware"].unique():

            d = df[df["hardware"] == hw]

            scalar = d[d["benchmark"] == base].sort_values("size")
            vector = d[d["benchmark"] == base + "Vectorized"].sort_values("size")

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
                speedup = merged[f"{metric}_vector"] / merged[f"{metric}_scalar"]
            else:
                speedup = merged[f"{metric}_scalar"] / merged[f"{metric}_vector"]

            label = hw_label_speedup(hw)

            ax.plot(
                merged["size"],
                speedup,
                color=HW_COLORS.get(hw, "black"),
                label=label,
                marker="o",
                markersize=3,  # ← controls point size
                alpha=0.7,
            )

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)

        ax.set_title(f"{base} Speedup (Vectorized vs Scalar)")
        ax.set_xlabel("n (cube width in cells)")
        ax.set_ylabel("Speedup")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        handles, labels = ax.get_legend_handles_labels()

        new_h, new_l = [], []

        for h, l in zip(handles, labels):

            if l not in new_l:
                new_h.append(h)
                new_l.append(l)

        ax.legend(new_h, new_l, fontsize=8)

        plt.tight_layout()
        plt.savefig(out_dir / f"{base}_speedup.png", dpi=200)
        plt.close()


def plot_hw_scalar_vector(res_dir, out_dir, title=""):

    res_dir = Path(res_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(res_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files in {res_dir}")

    # ----------------------------
    # load
    # ----------------------------
    data = []
    for f in files:
        df, _ = load_one(str(f))
        df["hardware"] = get_hw(f)
        data.append(df)

    df = pd.concat(data, ignore_index=True)

    bases = {b.replace("Vectorized", "") for b in df["benchmark"].unique()}

    for base in bases:

        fig, ax_cells = plt.subplots(figsize=(8, 5))

        # EulerSimulation: no bytes axis
        is_euler = base == "EulerSimulation"

        ax_bytes = ax_cells.twinx() if not is_euler else None

        hardware_handles = {}

        for hw in df["hardware"].unique():

            d = df[df["hardware"] == hw]

            scalar = d[d["benchmark"] == base].sort_values("size")
            vector = d[d["benchmark"] == base + "Vectorized"].sort_values("size")

            if scalar.empty or vector.empty:
                continue

            color = HW_COLORS.get(hw, "black")

            # ----------------------------
            # LEFT AXIS (cells/s only)
            # ----------------------------
            _plot_series(
                ax_cells,
                scalar,
                color,
                "_nolegend_",
                "cells_per_second",
            )

            _plot_series(
                ax_cells,
                vector,
                color,
                "_nolegend_",
                "cells_per_second",
            )

            ax_cells.lines[-2].set_linestyle("-")
            ax_cells.lines[-1].set_linestyle("--")

            # store one representative handle per HW
            if hw not in hardware_handles:
                hardware_handles[hw] = ax_cells.lines[-2]

            # ----------------------------
            # RIGHT AXIS (bytes/s only for non-Euler)
            # ----------------------------
            if not is_euler and ax_bytes is not None:

                _plot_series(
                    ax_bytes,
                    scalar,
                    color,
                    "_nolegend_",
                    "bytes_per_second",
                )

                _plot_series(
                    ax_bytes,
                    vector,
                    color,
                    "_nolegend_",
                    "bytes_per_second",
                )

                ax_bytes.lines[-2].set_linestyle("-")
                ax_bytes.lines[-1].set_linestyle("--")

        # ----------------------------
        # LEGEND 1: hardware
        # ----------------------------
        legend1 = ax_cells.legend(
            hardware_handles.values(),
            [hw_label_speedup(hw) for hw in hardware_handles.keys()],
            bbox_to_anchor=(0.7, 0.3),
            loc="best",
            fontsize=8,
            title="Hardware",
        )

        ax_cells.add_artist(legend1)

        # ----------------------------
        # LEGEND 2: scalar/vector mode
        # ----------------------------
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

        # ----------------------------
        # formatting
        # ----------------------------
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

import os 
def plot_scaling_dir(path):
    files = sorted([f for f in os.listdir(path) if f.endswith(".csv")])

    strong_scalar = None
    strong_vector = None
    weak_scalar = None
    weak_vector = None

    for f in files:
        full = os.path.join(path, f)

        if "strong" in f and "scalar" in f:
            strong_scalar = pd.read_csv(full)
        elif "strong" in f and "vector" in f:
            strong_vector = pd.read_csv(full)
        elif "weak" in f and "scalar" in f:
            weak_scalar = pd.read_csv(full)
        elif "weak" in f and "vector" in f:
            weak_vector = pd.read_csv(full)

    def plot_strong(df, ax, title):
        t = df["threads"].to_numpy()
        time = df["time_s"].to_numpy()

        speedup = time[0] / time
        eff = 100 * speedup / t

        ax.plot(t, speedup, marker="o", label="speedup")
        ax.plot(t, t, "--", label="ideal")

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Threads")
        ax.set_ylabel("Speedup")
        ax.set_title(title)
        ax.grid(True)

        ax2 = ax.twinx()
        ax2.plot(t, eff, marker="s", linestyle=":", label="efficiency")
        ax2.set_ylabel("Efficiency (%)")

        l1, lab1 = ax.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lab1 + lab2, loc="upper left")

    def plot_weak(df, ax, title):
        t = df["threads"].to_numpy()
        time = df["time_s"].to_numpy()

        norm = time / time[0]

        ax.plot(t, norm, marker="o", label="runtime normalized")
        ax.plot(t, t, "--", label="ideal weak scaling")

        ax.set_xscale("log", base=2)
        ax.set_xlabel("Threads")
        ax.set_ylabel("T / T1")
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

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

from pathlib import Path
import pandas as pd


def compare_godunov_benchmarks(path, out_csv, base_name="Godunov", opti_name="GodunovOpti", cols=None):
    """
    Compare Godunov baseline vs optimized versions within the same benchmark JSON.
    """

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

    # -------------------------
    # Core speedup
    # -------------------------
    merged["real_time_speedup"] = (
        merged["real_time_ns_base"] / merged["real_time_ns_opti"]
    )

    # -------------------------
    # Throughput speedups
    # -------------------------
    for col in ("cells_per_second", "bytes_per_second"):
        base_col = f"{col}_base"
        opti_col = f"{col}_opti"

        if base_col in merged.columns and opti_col in merged.columns:
            merged[f"{col}_speedup"] = merged[opti_col] / merged[base_col]

    # -------------------------
    # Optional column filtering
    # -------------------------
    if cols:
        merged = merged[cols]

    # -------------------------
    # Mean row
    # -------------------------
    mean_row = merged.mean(numeric_only=True).to_frame().T
    mean_row["benchmark"] = "MEAN"
    mean_row["size"] = pd.NA
    merged = pd.concat([merged, mean_row], ignore_index=True)

    # -------------------------
    # Rounding
    # -------------------------
    rounding = {c: 5 for c in merged.columns if "speedup" in c}
    rounding |= {c: 5 for c in merged.columns if "time" in c}
    rounding |= {c: 5 for c in merged.columns if "cells_per_second" in c or "bytes_per_second" in c}

    merged = merged.round(rounding)

    # -------------------------
    # Output
    # -------------------------
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    return merged
COLS=["benchmark_base", "size", "cells_per_second_speedup", "cells_per_second_base", "cells_per_second_opti"]
# compare_godunov_benchmarks("./results/adastra/mi250x/4971156_mi250x__Godunov_GodunovOpti_GodunovVectorized__.json", "opti-base.csv",base_name="Godunov", opti_name='GodunovOpti' ,cols=COLS)
# compare_godunov_benchmarks("./results/adastra/mi250x/4971156_mi250x__Godunov_GodunovOpti_GodunovVectorized__.json", "vec-opti.csv",base_name="GodunovVectorized", opti_name='GodunovOpti' ,cols=COLS)

# plot_scaling_dir("./results/scaling/genoa/")
plot_hw_scalar_vector("./results", "./results/test_polts/")
# plot_hw_speedup("./results", "./results/test_polts/")
