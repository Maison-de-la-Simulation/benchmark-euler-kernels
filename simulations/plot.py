#!/usr/bin/env python3

import json
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

KERNEL_BENCHMARKS = ["Godunov", "TimeStep",  "ConsToPrim", "PrimToConsVectorized","PrimToCons" ]
ALL_BENCHMARKS = KERNEL_BENCHMARKS
ALL_BENCHMARKS.append("EulerSimulation")

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------


OUT_DIR = "results/plots/skx/mt/"

import os
import glob

RES_DIR = "results/ruche/skx/mt/"

# def latest_result(res_dir=RES_DIR, pattern="*.json", rank=1):
#     files = glob.glob(os.path.join(res_dir, pattern))
#     if not files:
#         raise FileNotFoundError(f"No files matching {pattern} in {res_dir}")
#     # return max(files, key=os.path.getmtime)
#     return max(files, key=os.path.getmtime)

def latest_result(res_dir, pattern="*.json", rank=1):
    files = glob.glob(os.path.join(res_dir, pattern))

    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {res_dir}")

    # sort by modification time (newest first)
    files = sorted(files, key=os.path.getmtime, reverse=True)

    if rank < 1 or rank > len(files):
        raise IndexError(f"rank={rank} out of range (1..{len(files)})")

    return files[rank - 1]

def result_by_job_id(job_id, res_dir=RES_DIR):
    prefix = f"[{job_id}]"
    files = os.listdir(res_dir)
    for f in files:
        if f.startswith(prefix):
            return os.path.join(res_dir, f)
    raise FileNotFoundError(f"No result found for job {job_id} in {res_dir}")

def extract_label(path):
    name = Path(path).name
    label = name.split("_")[1]
    timestamp = name.split("[")[1].split("]")[0]
    return timestamp + "_" 



# %%
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BYTES_PER_CELL = 10 * 8
CACHE_COLORS = {1: "green", 2: "orange", 3: "red"}


def load_one(path):
    with open(path) as f:
        raw = json.load(f)
    caches = {
        c["level"]: c["size"]
        for c in raw["context"]["caches"]
        if c["type"] == "Unified"
    }
    rows = []
    for b in raw["benchmarks"]:
        name = b["name"]
        rows.append({
            "benchmark":       name.split("/")[0],
            "size":            int(name.split("/")[-2 if "real_time" in name else -1]),
            "cells_per_second": b.get("cells_per_second"),
            "bytes_per_second": b.get("bytes_per_second"),
            "real_time_ns":     b.get("real_time"),
            "cpu_time_ns":     b.get("cpu_time"),
        })
    return pd.DataFrame(rows), caches


def _draw_cache_lines(ax, caches):
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


def _plot_series(ax, df_series, color, label, y_key, alpha=1.0, linestyle="-"):
    aligned   = df_series[df_series["size"] % 8 == 0]
    unaligned = df_series[df_series["size"] % 8 != 0]

    ax.plot(
        df_series["size"],
        df_series[y_key],
        linestyle,
        color=color,
        label=label,
        alpha=alpha,
    )

    ax.scatter(aligned["size"], aligned[y_key],
               marker="o", color=color, zorder=5, alpha=alpha)
    ax.scatter(unaligned["size"], unaligned[y_key],
               marker="x", color=color, zorder=5, alpha=alpha)

def plot_scalar_vs_vector(files, out_dir, draw_caches=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect benchmark names present in every file
    all_names = set()
    for path in files.values():
        df, _ = load_one(path)
        all_names.update(df["benchmark"].unique())

    base_names = [b for b in all_names if b + "Vectorized" in all_names]

    for base_name in base_names:

        # if "Godunov" not in base_name:
        #     continue
        vec_name = base_name + "Vectorized"

        for environment, path in files.items():
            df, caches = load_one(path)
            bm_label = extract_label(path)

            s = df[df["benchmark"] == base_name].sort_values("size")
            v = df[df["benchmark"] == vec_name].sort_values("size")

            if s.empty or v.empty:
                print(f"skipping {base_name} for {environment}")
                continue

            fig, (ax_left, ax_right) = plt.subplots(
                1, 2, figsize=(16, 5), sharey=False
            )
            fig.suptitle(f"{base_name} — {bm_label}", fontsize=12)

            # ── left plot: throughput ─────────────────────────────────────
            ax_bytes = ax_left.twinx()

            for df_series, color, label in [
                (s, "C0", "scalar"),
                (v, "C1", "vectorized"),
            ]:
                _plot_series(ax_left,  df_series, color, f"{label} cells/s",
                             "cells_per_second")
                _plot_series(ax_bytes, df_series, color, f"{label} bytes/s",
                             "bytes_per_second", alpha=0.4)

            if draw_caches:
                _draw_cache_lines(ax_left, caches)

            ax_left.set_xlabel("n (cube width in cells)")
            ax_left.set_ylabel("cells / s")
            ax_bytes.set_ylabel("bytes / s")
            ax_left.set_title("Throughput")
            ax_right.set_xscale("log")
            ax_right.set_yscale("log")
            ax_left.legend(fontsize=7)
            ax_left.grid(True, alpha=0.3)

            # ── right plot: wall time + speedup ──────────────────────────
            ax_speedup = ax_right.twinx()


            _plot_series(ax_right, s, "C0", "scalar real time", "real_time_ns", linestyle="-")
            _plot_series(ax_right, v, "C1", "vectorized real time", "real_time_ns", linestyle="-")

            _plot_series(ax_right, s, "C0", "scalar cpu time", "cpu_time_ns", linestyle="--")
            _plot_series(ax_right, v, "C1", "vectorized cpu time", "cpu_time_ns", linestyle="--")

            # speedup: scalar / vectorized on shared sizes
            merged = pd.merge(
                s[["size", "real_time_ns"]],
                v[["size", "real_time_ns"]],
                on="size",
                suffixes=("_s", "_v"),
            ).dropna()
            merged["speedup"] = merged["real_time_ns_s"] / merged["real_time_ns_v"]

            ax_speedup.plot(
                merged["size"], merged["speedup"],
                "-", color="C2", label="speedup (×)", linewidth=1.5,
            )
            ax_speedup.scatter(
                merged["size"], merged["speedup"],
                marker="D", color="C2", zorder=5, s=25,
            )
            ax_speedup.axhline(1.0, linestyle=":", color="C2", alpha=0.5)

            if draw_caches:
                _draw_cache_lines(ax_right, caches)

            ax_right.set_xlabel("n (cube width in cells)")
            ax_right.set_ylabel("real time (ns)")
            ax_speedup.set_ylabel("speedup (×)")
            ax_right.set_title("Wall Time & Speedup")

            # merge legends from both right-plot axes
            lines_r, labels_r = ax_right.get_legend_handles_labels()
            lines_s, labels_s = ax_speedup.get_legend_handles_labels()
            ax_right.legend(lines_r + lines_s, labels_r + labels_s, fontsize=7)
            ax_right.grid(True, alpha=0.3)

            plt.tight_layout()
            print("bm_label = " , bm_label)
            save_name = out_dir / f"{bm_label}_{base_name}.png"
            print("saving : ", save_name)
            plt.savefig(save_name, dpi=200)
            plt.close()

def compare_benchmarks(path_a, path_b, out_csv,label_a="a", label_b="b",cols=None):
    df_a, _ = load_one(path_a)
    df_b, _ = load_one(path_b)
    merged = pd.merge(
        df_a,
        df_b,
        on=["benchmark", "size"],
        suffixes=(f"_{label_a}", f"_{label_b}"),
        how="inner",
    )
    # merged = merged[
    #     merged.benchmark.isin(["GodunovVectorized", "Godunov"])
    # ]

    # time speedup: lower is better, so a/b (b is faster if > 1)
    time_cols = ["real_time_ns", "cpu_time_ns"]
    for time_col in time_cols:
        a_t, b_t = f"{time_col}_{label_a}", f"{time_col}_{label_b}"
        if a_t in merged and b_t in merged:
            merged[f"{time_col}_speedup"] = merged[a_t] / merged[b_t]

    # bytes/s speedup: higher is better, so b/a (b is faster if > 1)
    bw_col = "bytes_per_second"
    a_bw, b_bw = f"{bw_col}_{label_a}", f"{bw_col}_{label_b}"
    if a_bw in merged and b_bw in merged:
        merged[f"{bw_col}_speedup"] = merged[b_bw] / merged[a_bw]

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

COLS = ["benchmark", "size", "real_time_ns_speedup" , "cpu_time_ns_speedup"]

plot_scalar_vs_vector({"skx" : latest_result(RES_DIR)}, OUT_DIR + "numa/")
# plot_scalar_vs_vector({"skx" : latest_result(RES_DIR , rank=2)}, OUT_DIR + "numa/")

# compare_benchmarks( result_by_job_id(644750, RES_DIR), result_by_job_id(646848, RES_DIR), OUT_DIR + "mt1-mt40.csv", cols=COLS)
# compare_benchmarks( result_by_job_id(644750, RES_DIR), latest_result(RES_DIR), OUT_DIR + "spread_mt1-mt20.csv", cols=COLS)
#
#

def extract_threads(filename: str) -> int:
    match = re.search(r"mt1-mt(\d+)", filename)
    if not match:
        raise ValueError(f"Cannot parse thread count from {filename}")
    return int(match.group(1))


def plot_speedup_from_dir(directory: str):
    csv_files = sorted(glob.glob(os.path.join(directory, "*.csv")))

    if not csv_files:
        raise ValueError(f"No CSV files found in {directory}")

    # Map thread count -> color
    thread_counts = sorted(
        extract_threads(os.path.basename(f)) for f in csv_files
    )

    cmap = plt.get_cmap("tab10")
    color_map = {
        t: cmap(i % 10) for i, t in enumerate(thread_counts)
    }

    plt.figure()

    for filepath in csv_files:
        df = pd.read_csv(filepath)

        df = df[df["benchmark"] != "MEAN"]
        df = df.dropna(subset=["size", "real_time_ns_speedup"])

        filename = os.path.basename(filepath).replace(".csv", "")
        threads = extract_threads(filename)
        if threads < 16:
            continue
        color = color_map[threads]

        for bench_name, group in df.groupby("benchmark"):
            group = group.sort_values("size")

            # Style based on kernel
            if bench_name == "Godunov":
                linestyle = "-"
                marker = "o"
            elif bench_name == "GodunovVectorized":
                linestyle = "--"
                marker = "s"
            else:
                linestyle = ":"
                marker = "x"

            label = f"{threads} threads - {bench_name}"

            plt.plot(
                group["size"],
                group["real_time_ns_speedup"],
                color=color,
                linestyle=linestyle,
                marker=marker,
                label=label
            )

    plt.xlabel("Problem Size")
    plt.ylabel("Real Time Speedup")
    plt.title("Speedup vs Size (Color = Threads, Style = Kernel)")

    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)
    plt.tight_layout()
    # plt.show()
    plt.savefig("godunov_mt_speedup_threshold", dpi=200)


# plot_speedup_from_dir(OUT_DIR)
