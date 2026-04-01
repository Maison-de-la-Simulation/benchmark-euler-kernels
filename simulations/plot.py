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
FILES = {
    "skx_store":  "results/ruche/skx/[460979]_skx-PrimToCons_bm_ruche.json",
    "skx_ref2":  "results/ruche/skx/[460449]_skx-PrimToCons_bm_ruche.json",
    "skx_ptr":  "results/ruche/skx/[462651]_skx-PrimToCons_bm_ruche.json",


    }

OUT_DIR = "results/plots"


def extract_label(path):
    name = Path(path).name
    label = name.split("_")[1]
    timestamp = name.split("[")[1].split("]")[0]
    return timestamp + "_" + label



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
            "size":            int(name.split("/")[-1]),
            "cells_per_second": b.get("cells_per_second"),
            "bytes_per_second": b.get("bytes_per_second"),
            "real_time_ns":     b.get("real_time"),
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


def _plot_series(ax, df_series, color, label, y_key, alpha=1.0):
    aligned   = df_series[df_series["size"] % 8 == 0]
    unaligned = df_series[df_series["size"] % 8 != 0]
    ax.plot(df_series["size"], df_series[y_key], "-", color=color,
            label=label, alpha=alpha)
    ax.scatter(aligned["size"],   aligned[y_key],   marker="o",
               color=color, zorder=5, alpha=alpha)
    ax.scatter(unaligned["size"], unaligned[y_key], marker="x",
               color=color, zorder=5, alpha=alpha)


def plot_scalar_vs_vector(files, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect benchmark names present in every file
    all_names = set()
    for path in files.values():
        df, _ = load_one(path)
        all_names.update(df["benchmark"].unique())

    base_names = [b for b in all_names if b + "Vectorized" in all_names]

    for base_name in base_names:
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

            _plot_series(ax_right, s, "C0", "scalar ns",    "real_time_ns")
            _plot_series(ax_right, v, "C1", "vectorized ns", "real_time_ns")

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
            plt.savefig(out_dir / f"{bm_label}_{base_name}.png", dpi=200)
            plt.close()

def compare_benchmarks(path_a, path_b, out_csv, label_a="a", label_b="b"):
    df_a, _ = load_one(path_a)
    df_b, _ = load_one(path_b)

    merged = pd.merge(
        df_a,
        df_b,
        on=["benchmark", "size"],
        suffixes=(f"_{label_a}", f"_{label_b}"),
        how="inner",
    )

    merged["real_time_speedup"] = (
        merged[f"real_time_ns_{label_a}"] / merged[f"real_time_ns_{label_b}"]
    )

    for col in ("cells_per_second", "bytes_per_second"):
        a_col = f"{col}_{label_a}"
        b_col = f"{col}_{label_b}"
        if a_col in merged and b_col in merged:
            merged[f"{col}_speedup"] = merged[b_col] / merged[a_col]

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
plot_scalar_vs_vector(FILES, OUT_DIR)
# compare_benchmarks(FILES["skx_ref2"], FILES["skx_store"], "store.csv", "ref2", "store")

