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


OUT_DIR = "results/adastra/genoa/plots"
RES_DIR = "results/adastra/genoa/bm_json/mt/"
# OUT_DIR = "results/adastra/mi300/plots"
# RES_DIR = "results/adastra/mi300/bm_json/"



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

    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {res_dir}")

    # sort by modification time (newest first)
    files = sorted(files, key=os.path.getmtime, reverse=True)

    return files[0]


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
                "cpu_time_ns": b.get("cpu_time"),
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


def _plot_series(ax, df_series, color, label, y_key):
    """Plot a benchmark series with aligned and unaligned data points.

    Args:
        ax: Matplotlib axis object to plot on.
        df_series: DataFrame containing the series data.
        color: Color for the plot line and markers.
        label: Label for the series.
        y_key: Column name to plot on the y-axis.
    """
    aligned = df_series[df_series["size"] % 8 == 0]
    unaligned = df_series[df_series["size"] % 8 != 0]
    ax.plot(df_series["size"], df_series[y_key], "-", color=color, label=label, alpha=0.5)
    ax.scatter(aligned["size"], aligned[y_key], marker="o", color=color, zorder=5, alpha=0.5)
    ax.scatter(
        unaligned["size"],
        unaligned[y_key],
        marker="x",
        color=color,
        zorder=5,
        alpha=0.5,
    )

    ax.plot(
        df_series["size"],
        df_series[y_key],
        color=color,
        label=label,
        alpha=0.5,
    )



def plot_throughput(ax_left, ax_bytes, bm_dfs, caches):
    """Plot throughput comparison (cells/s and bytes/s) between scalar and vectorized.

    Args:
        ax_left: Matplotlib axis for cells per second plot.
        ax_bytes: Secondary axis for bytes per second overlay.
        bm_dfs: Tuple of (scalar_df, vectorized_df).
        caches: Dictionary mapping cache level to size in bytes.
    """
    s, v = bm_dfs

    for df_series, color, label in [
        (s, "C0", "scalar"),
        (v, "C1", "vectorized"),
    ]:
        _plot_series(ax_left, df_series, color, f"{label} cells/s", "cells_per_second")
        _plot_series(ax_bytes, df_series, color, f"{label} bytes/s", "bytes_per_second")

    _draw_cache_lines(ax_left, caches)

    # --- Axis labels ---
    ax_left.set_ylabel("Cells per second")
    ax_bytes.set_ylabel("Bytes per second")

    # --- Combined legend (both axes) ---
    handles_left, labels_left = ax_left.get_legend_handles_labels()

    ax_left.legend(
        handles_left,
        labels_left,
        loc="best",
        fontsize=9,
    )


def plot_time_and_speedup(ax_right, ax_speedup, bm_dfs, caches, plot_cpu_time=True):
    """Plot wall time and (optionally) CPU time + speedup comparison."""

    s, v = bm_dfs

    # --- Wall time (solid lines) ---
    _plot_series(ax_right, s, "C0", "scalar (wall ns)", "real_time_ns")
    _plot_series(ax_right, v, "C1", "vectorized (wall ns)", "real_time_ns")

    # --- CPU time (dashed lines, optional) ---
    if plot_cpu_time:
        ax_right.plot(
            s["size"], s["cpu_time_ns"],
            linestyle="--", color="C0",
            label="scalar (cpu ns)", linewidth=1.2
        )
        ax_right.plot(
            v["size"], v["cpu_time_ns"],
            linestyle="--", color="C1",
            label="vectorized (cpu ns)", linewidth=1.2
        )

    # --- Speedup ---
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

    # --- Cache markers ---
    _draw_cache_lines(ax_right, caches)

    # --- Axis labels ---
    ax_right.set_ylabel("Time (ns)")
    ax_speedup.set_ylabel("Speedup (×)")

    # --- Combined legend (both axes) ---
    handles_r, labels_r = ax_right.get_legend_handles_labels()
    handles_s, labels_s = ax_speedup.get_legend_handles_labels()

    ax_right.legend(
        handles_r + handles_s,
        labels_r + labels_s,
        loc="best",
        fontsize=9,
    )














def plot_pair(benchmarks, caches, base_name, plot_title, out_dir):
    """Create a two-panel figure comparing scalar vs vectorized performance metrics.

    Args:
        benchmarks: Pair of DataFrames of scalar and vectorized benchmark results.
        caches: Dictionary mapping cache level to size in bytes.
        base_name: Base benchmark name (used in filename).
        plot_title: Title shown on the figure.
        out_dir: Output directory path for saving the figure.
    """
    s, v = benchmarks

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 5))

    ax_bytes = ax_left.twinx()
    ax_speedup = ax_right.twinx()

    # Changed: use filename-derived benchmark title only
    fig.suptitle(plot_title, fontsize=12)

    bm_dfs = (s, v)

    plot_throughput(ax_left, ax_bytes, bm_dfs, caches)
    plot_time_and_speedup(ax_right, ax_speedup, bm_dfs, caches)

    ax_left.set_xlabel("n (cube width in cells)")
    ax_right.set_xlabel("n (cube width in cells)")

    ax_left.set_title("Throughput")
    ax_right.set_title("Wall Time & Speedup")

    plt.tight_layout()

    plt.savefig(out_dir / f"{plot_title}_{base_name}.png", dpi=200)
    plt.close()


def get_scalar_vector(df, base_name):
    """Extract scalar and vectorized benchmark data for a given benchmark name."""
    s = df[df["benchmark"] == base_name].sort_values("size")
    v = df[df["benchmark"] == base_name + "Vectorized"].sort_values("size")
    return s, v


def collect_all_benchmarks(files):
    """Collect all unique benchmark names from all result files."""
    all_names = set()

    for path in files.values():
        df, _ = load_one(path)
        all_names.update(df["benchmark"].unique())

    return all_names


def extract_plot_title(path):
    """Use filename without .json as plot title."""
    return Path(path).stem


def process_base_name(files, out_dir, base_name):
    """Generate plots for one benchmark across all environments."""
    for environment, path in files.items():
        df, caches = load_one(path)

        # Changed: title now comes from JSON filename
        plot_title = extract_plot_title(path)

        s, v = get_scalar_vector(df, base_name)

        if s.empty or v.empty:
            print(f"skipping {base_name} for {environment}")
            continue

        print(plot_title)
        plot_pair((s, v), caches, base_name, plot_title, out_dir)


def plot_scalar_vs_vector(files, out_dir):
    """Generate scalar vs vectorized comparison plots for all benchmarks."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_names = collect_all_benchmarks(files)
    base_names = [b for b in all_names if b + "Vectorized" in all_names]

    for base_name in base_names:
        process_base_name(files, out_dir, base_name)
# %%


def compare_benchmarks(path_a, path_b, out_dir, cols=None):
    """Compare two benchmark result files and save a deterministic CSV.

    Output filename format:
        <file_a_stem>__to__<file_b_stem>.csv

    Args:
        path_a: Path to first benchmark JSON file.
        path_b: Path to second benchmark JSON file.
        out_dir: Directory where CSV will be written.
        cols: Optional subset of columns to keep.

    Returns:
        DataFrame containing merged comparison results.
    """
    path_a = Path(path_a)
    path_b = Path(path_b)
    out_dir = Path(out_dir)

    df_a, _ = load_one(path_a)
    df_b, _ = load_one(path_b)

    merged = pd.merge(
        df_a,
        df_b,
        on=["benchmark", "size"],
        suffixes=("_a", "_b"),
        how="inner",
    )

    # Lower time is better
    merged["real_time_ns_speedup"] = (
        merged["real_time_ns_a"] / merged["real_time_ns_b"]
    )
    merged["cpu_time_ns_speedup"] = (
        merged["cpu_time_ns_a"] / merged["cpu_time_ns_b"]
    )

    # Higher throughput is better
    for col in ("cells_per_second", "bytes_per_second"):
        a_col = f"{col}_a"
        b_col = f"{col}_b"

        if a_col in merged.columns and b_col in merged.columns:
            merged[f"{col}_speedup"] = merged[b_col] / merged[a_col]

    if cols:
        merged = merged[cols]

    # Mean summary row
    mean_row = merged.mean(numeric_only=True).to_frame().T
    mean_row["benchmark"] = "MEAN"
    mean_row["size"] = pd.NA

    merged = pd.concat([merged, mean_row], ignore_index=True)

    # Round numeric outputs
    rounding = {c: 5 for c in merged.columns if "speedup" in c}
    rounding |= {c: 5 for c in merged.columns if "time" in c}
    rounding |= {
        c: 5
        for c in merged.columns
        if "cells_per_second" in c or "bytes_per_second" in c
    }

    merged = merged.round(rounding)
    path_a = Path(path_a)
    path_b = Path(path_b)
    out_dir = Path(out_dir)

    ts_a = path_a.stem.split("[")[1].split("]")[0]
    ts_b = path_b.stem.split("[")[1].split("]")[0]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{ts_a}__to__{ts_b}.csv"

    merged.to_csv(out_csv, index=False)

    return merged

COLS = ["benchmark", "size", "real_time_ns_speedup", "cpu_time_ns_speedup"]

FILES = {
    # "genoa": latest_result(RES_DIR)
    "genoa_base": result_by_job_id(4870888, ),
    "genoa_t10": latest_result(RES_DIR)
}
# plot_scalar_vs_vector(FILES, OUT_DIR)
compare_benchmarks(FILES["genoa_base"], FILES["genoa_t10"], OUT_DIR, cols=COLS)
