"""
Plotting script for plotting/comparing Google Benchmark JSON files.

This script has two main functionalities:
    - plotting scalar vs vectorized benchmark outputs
    - creating a csv file comparing two different benchmarks
"""

import glob
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

KERNEL_BENCHMARKS = [
    "Godunov",
    "TimeStep",
    "ConsToPrim",
    "PrimToConsVectorized",
    "PrimToCons",
]
ALL_BENCHMARKS = KERNEL_BENCHMARKS
ALL_BENCHMARKS.append("EulerSimulation")


OUT_DIR = "results/plots"
RES_DIR = "results/ruche/skx/"

import numpy as np
import pandas as pd


def validate_bytes_cells(df, out_csv=None):
    """
    Checks whether bytes_per_second and cells_per_second
    are proportional with a stable bytes-per-cell factor.

    Also saves per-benchmark/hardware ratios if out_csv is provided.
    """

    if "bytes_per_second" not in df.columns or "cells_per_second" not in df.columns:
        raise ValueError("Missing required columns")

    df = df.copy()
    df["benchmark"] = (
        df["benchmark"]
        .str.replace("Vectorized", "", regex=False)
        .str.replace("WorstRem", "", regex=False)
    )
    df["bytes_per_cell_estimate"] = df["bytes_per_second"] / df["cells_per_second"]
    summary = df.groupby(["benchmark"])["bytes_per_cell_estimate"].agg(["min", "max"]).reset_index()
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        summary.round(3).to_csv(out_csv, index=False)

    return summary


def latest_result(res_dir=RES_DIR, pattern="*.json"):
    """Find and return the most recently modified benchmark JSON file.

    Args:
        res_dir: Directory to search for benchmark files (default: RES_DIR).
        pattern: Glob pattern to match files (default: "*.json").

    Returns:
        Path to the most recently modified file matching the pattern.

    Raises:
        FileNotFoundError: If no files matching the pattern are found.
    """
    files = glob.glob(os.path.join(res_dir, pattern))
    print(files)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {res_dir}")
    return max(files, key=os.path.getmtime)


def result_by_job_id(job_id, res_dir=RES_DIR):
    """Retrieve a benchmark result file by job ID.

    Args:
        job_id: The job ID to search for (used as filename prefix).
        res_dir: Directory to search for benchmark files (default: RES_DIR).

    Returns:
        Path to the result file for the given job ID.

    Raises:
        FileNotFoundError: If no result file is found for the given job ID.
    """
    prefix = f"[{job_id}]"
    files = os.listdir(res_dir)
    for f in files:
        if f.startswith(prefix):
            return os.path.join(res_dir, f)
    raise FileNotFoundError(f"No result found for job {job_id} in {res_dir}")


def extract_label(path):
    """Extract a timestamp label from a benchmark result file path.

    Args:
        path: Path to the benchmark result file.

    Returns:
        A timestamp string extracted from the filename, with a trailing underscore.
    """
    name = Path(path).name
    timestamp = name.split("[")[1].split("]")[0]
    return timestamp + "_"


# %%

BYTES_PER_CELL = 10 * 8
CACHE_COLORS = {1: "green", 2: "orange", 3: "red"}


def load_one(path):
    """Load and parse a Google Benchmark JSON file.

    Args:
        path: Path to the JSON benchmark file.

    Returns:
        A tuple of (DataFrame, caches_dict) where:
        - DataFrame contains benchmark data with columns: benchmark, size,
          cells_per_second, bytes_per_second, real_time_ns
        - caches_dict is a mapping of cache level to size in bytes
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    caches = {c["level"]: c["size"] for c in raw["context"]["caches"] if c["type"] == "Unified"}
    rows = []
    for b in raw["benchmarks"]:
        name = b["name"]
        rows.append(
            {
                "benchmark": name.split("/")[0],
                "size": int(name.split("/")[-2]),
                "cells_per_second": b.get("cells_per_second"),
                "bytes_per_second": b.get("bytes_per_second"),
                "real_time_ns": b.get("real_time"),
            }
        )
    return pd.DataFrame(rows), caches


def _draw_cache_lines(ax, caches):
    """Draw vertical lines on a plot indicating cache level boundaries.
       (Read directly from Google Benchmark => only relevant for cpu.

    Args:
        ax: Matplotlib axis object to draw on.
        caches: Dictionary mapping cache level to size in bytes.
    """
    for level, size_bytes in sorted(caches.items()):
        n_cache = (size_bytes / BYTES_PER_CELL) ** (1 / 3)
        color = CACHE_COLORS.get(level, "gray")
        ax.axvline(
            n_cache,
            linestyle="--",
            color=color,
            alpha=0.7,
            label=f"L{level} ({size_bytes // 1024} KB) → n≈{n_cache:.0f}",
        )


def _plot_series(ax, df_series, color, label, y_key, linestyle="-"):
    """Plot a benchmark series with aligned and unaligned data points."""

    # ----------------------------
    # FILTER invalid values (keep your fix)
    # ----------------------------
    df_series = df_series[df_series[y_key].notna()]
    # df_series = df_series[df_series[y_key] > 0]

    if df_series.empty:
        return

    aligned = df_series[df_series["size"] % 8 == 0]
    unaligned = df_series[df_series["size"] % 8 != 0]
    ax.plot(
        df_series["size"],
        df_series[y_key],
        linestyle,
        color=color,
        label=label,
        alpha=0.5,
    )

    ax.scatter(
        aligned["size"],
        aligned[y_key],
        marker="o",
        color=color,
        zorder=5,
        alpha=0.5,
        s=8,  # 👈 fix
    )


def plot_time_and_speedup(ax_right, ax_speedup, s, v, caches):
    """Plot wall time and speedup comparison between scalar and vectorized implementations.

    Args:
        ax_right: Matplotlib axis for wall time plot.
        ax_speedup: Secondary axis for speedup overlay.
        s: DataFrame of scalar benchmark results.
        v: DataFrame of vectorized benchmark results.
        caches: Dictionary mapping cache level to size in bytes.
    """
    _plot_series(ax_right, s, "C0", "scalar ns", "real_time_ns")
    _plot_series(ax_right, v, "C1", "vectorized ns", "real_time_ns")

    merged = pd.merge(
        s[["size", "real_time_ns"]],
        v[["size", "real_time_ns"]],
        on="size",
        suffixes=("_s", "_v"),
    ).dropna()

    merged["speedup"] = merged["real_time_ns_s"] / merged["real_time_ns_v"]

    ax_speedup.plot(
        merged["size"],
        merged["speedup"],
        "-",
        color="C2",
        label="speedup (×)",
        linewidth=1.5,
    )
    ax_speedup.scatter(
        merged["size"],
        merged["speedup"],
        marker="D",
        color="C2",
        zorder=5,
        s=25,
    )
    ax_speedup.axhline(1.0, linestyle=":", color="C2", alpha=0.5)

    _draw_cache_lines(ax_right, caches)


def plot_throughput(ax_left, ax_bytes, s, v, caches):
    """Plot throughput comparison (cells/s and bytes/s) between scalar and vectorized.

    Args:
        ax_left: Matplotlib axis for cells per second plot.
        ax_bytes: Secondary axis for bytes per second overlay.
        s: DataFrame of scalar benchmark results.
        v: DataFrame of vectorized benchmark results.
        caches: Dictionary mapping cache level to size in bytes.
    """
    for df_series, color, label in [
        (s, "C0", "scalar"),
        (v, "C1", "vectorized"),
    ]:
        _plot_series(ax_left, df_series, color, f"{label} cells/s", "cells_per_second")
        _plot_series(ax_bytes, df_series, color, f"{label} bytes/s", "bytes_per_second")

    _draw_cache_lines(ax_left, caches)


def plot_pair(benchmarks, caches, base_name, bm_label, out_dir):
    """Create a two-panel figure comparing scalar vs vectorized performance metrics.

    Args:
        benchmarks: Pair of DataFrames of scalar and vectorized benchmark results.
        caches: Dictionary mapping cache level to size in bytes.
        base_name: Base name of the benchmark (without "Vectorized" suffix).
        bm_label: Label for the benchmark (used in filename and title).
        out_dir: Output directory path for saving the figure.
    """
    s, v = benchmarks
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 5))

    ax_bytes = ax_left.twinx()
    ax_speedup = ax_right.twinx()

    fig.suptitle(f"{base_name} — {bm_label}", fontsize=12)

    plot_throughput(ax_left, ax_bytes, s, v, caches)
    plot_time_and_speedup(ax_right, ax_speedup, s, v, caches)

    ax_left.set_xlabel("n (cube width in cells)")
    ax_right.set_xlabel("n (cube width in cells)")

    ax_left.set_title("Throughput")
    ax_right.set_title("Wall Time & Speedup")

    plt.tight_layout()
    plt.savefig(out_dir / f"{bm_label}_{base_name}.png", dpi=200)
    plt.close()


def get_scalar_vector(df, base_name):
    """Extract scalar and vectorized benchmark data for a given base benchmark name.

    Args:
        df: DataFrame containing benchmark results.
        base_name: Base name of the benchmark (without "Vectorized" suffix).

    Returns:
        A tuple of (scalar_df, vectorized_df) sorted by size.
    """
    s = df[df["benchmark"] == base_name].sort_values("size")
    v = df[df["benchmark"] == base_name + "Vectorized"].sort_values("size")
    return s, v


def collect_all_benchmarks(files):
    """Collect all unique benchmark names from a set of result files.

    Args:
        files: Dictionary mapping environment names to file paths.

    Returns:
        A set of unique benchmark names found across all files.
    """
    all_names = set()
    for path in files.values():
        df, _ = load_one(path)
        all_names.update(df["benchmark"].unique())
    return all_names


def process_base_name(files, out_dir, base_name):
    """Process and plot scalar vs vectorized comparisons for a single benchmark.

    Args:
        files: Dictionary mapping environment names to file paths.
        out_dir: Output directory for saving plots.
        base_name: Base name of the benchmark to process.
    """
    for environment, path in files.items():
        df, caches = load_one(path)
        bm_label = extract_label(path)

        s, v = get_scalar_vector(df, base_name)
        if s.empty or v.empty:
            print(f"skipping {base_name} for {environment}")
            continue

        plot_pair((s, v), caches, base_name, bm_label, out_dir)


def plot_scalar_vs_vector(files, out_dir):
    """Generate scalar vs vectorized comparison plots for all benchmarks.

    Args:
        files: Dictionary mapping environment names to benchmark result file paths.
        out_dir: Output directory for saving generated plots.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_names = collect_all_benchmarks(files)
    base_names = [b for b in all_names if b + "Vectorized" in all_names]

    for base_name in base_names:
        process_base_name(files, out_dir, base_name)


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
    validate_bytes_cells(df, "throughput_metric.csv")

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
            # LEFT AXIS (always cells/s now)
            # ----------------------------
            _plot_series(ax_cells, scalar, color, "_nolegend_", "cells_per_second")
            _plot_series(ax_cells, vector, color, "_nolegend_", "cells_per_second")

            ax_cells.lines[-2].set_linestyle("-")
            ax_cells.lines[-1].set_linestyle("--")

            # store one representative handle per HW
            if hw not in hardware_handles:
                hardware_handles[hw] = ax_cells.lines[-2]

            # ----------------------------
            # RIGHT AXIS (skip for Euler)
            # ----------------------------
            if not is_euler and ax_bytes:
                _plot_series(ax_bytes, scalar, color, "_nolegend_", "bytes_per_second")
                _plot_series(ax_bytes, vector, color, "_nolegend_", "bytes_per_second")

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

        if not is_euler and ax_bytes:
            ax_bytes.set_yscale("log")
            ax_bytes.set_ylabel("bytes per second")

        ax_cells.set_xlabel("n (cube width in cells)")
        ax_cells.set_title(f"{title} {base}".strip())
        ax_cells.grid(True)

        plt.tight_layout()
        plt.savefig(out_dir / f"{base}_hw_compare.png", dpi=200)
        plt.close()


plot_hw_scalar_vector("./results", "./results/new/")
plot_hw_speedup("./results", "./results/new/")
# FILES = {
#     "skx_new": latest_result("."),
# }
# plot_scalar_vs_vector(FILES, OUT_DIR)
# COLS = ["benchmark", "size", "real_time_speedup"]
# compare_benchmarks(
#     FILES["skx_new"],
#     FILES["skx_new"],
#     "store.csv",
#     cols=COLS,
# )
#
