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




def compare_hllc_opti_pairs(path_json, out_csv, cols=None):
    """Compare benchmarks against their HllcOpti variants.

    Example:
        Godunov                vs GodunovHllcOpti
        GodunovOpti            vs GodunovOptiHllcOpti
        GodunovVectorized      vs GodunovVectorizedHllcOpti

    Args:
        path_json: Path to benchmark JSON file.
        out_csv: Output CSV path.
        cols: Optional list of columns to keep.

    Returns:
        DataFrame with speedup metrics.
    """
    df, _ = load_one(path_json)

    base = df[~df["benchmark"].str.endswith("HllcOpti")].copy()
    opti = df[df["benchmark"].str.endswith("HllcOpti")].copy()

    opti["benchmark"] = opti["benchmark"].str.replace(
        "HllcOpti", "", regex=False
    )

    merged = pd.merge(
        base,
        opti,
        on=["benchmark", "size"],
        suffixes=("_base", "_opti"),
        how="inner",
    )

    merged["real_time_speedup"] = (
        merged["real_time_ns_base"] / merged["real_time_ns_opti"]
    )

    for col in ("cells_per_second", "bytes_per_second"):
        base_col = f"{col}_base"
        opti_col = f"{col}_opti"

        if base_col in merged and opti_col in merged:
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
# %%
COLS = ["benchmark", "size", "real_time_speedup", "cells_per_second_base", "cells_per_second_opti"]

# compare_hllc_opti_pairs("./results/adastra/genoa/[4976036]_Godunov_hllc-opti1x.json", "opti1x-genoa.csv", COLS)
compare_hllc_opti_pairs("./results/adastra/genoa/[4976121]_Godunov_hllc-opti2.json", "opti2-genoa.csv", COLS)
compare_hllc_opti_pairs("./results/adastra/mi300/[4976127]_mi300_Godunov_hllc-opti2.json", "opti2-mi300.csv", COLS)
# compare_hllc_opti_pairs("./results/adastra/mi300/[4976050]_mi300_Godunov_hllc-opti1x.json", "opti1x-mi300.csv", COLS)
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
