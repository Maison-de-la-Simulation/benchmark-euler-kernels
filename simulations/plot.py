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
    "skx_kokkos5.0.0":  "results/ruche/skx/[457139]_skx-PrimToCons_bm_ruche.json",
    # "skx_rem":  "results/ruche/skx/[457078]_skx-PrimToCons_bm_ruche.json",
    # "skx_rem":"results/ruche/skx/[457041]_skx-PrimToCons_bm_ruche.json",

    # "skx_10": "results/ruche/skx/[453127]_cpus-10-ref_bm_ruche.json",
    # "skx_20": "results/ruche/skx/[453128]_cpus-20-ref_bm_ruche.json",
    # "skx_30": "results/ruche/skx/[453129]_cpus-30-ref_bm_ruche.json",
    # "skx_40": "results/ruche/skx/[453130]_cpus-40-ref_bm_ruche.json",
}
OUT_DIR = "results/plots"


def extract_label(path):
    name = Path(path).name
    label = name.split("_")[1]
    timestamp = name.split("[")[1].split("]")[0]
    return timestamp + "_" + label




def plot_scalar_vs_vector(files, out_dir):
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    BYTES_PER_CELL = 10 * 8
    cache_colors = {1: "green", 2: "orange", 3: "red"}

    def load_one(path):
        with open(path) as f:
            raw = json.load(f)
        caches = {c["level"]: c["size"] for c in raw["context"]["caches"] if c["type"] == "Unified"}
        rows = []
        for b in raw["benchmarks"]:
            name = b["name"]
            rows.append({
                "benchmark": name.split("/")[0],
                "size": int(name.split("/")[-1]),
                "cells_per_second": b.get("cells_per_second"),
                "bytes_per_second": b.get("bytes_per_second"),
            })
        return pd.DataFrame(rows), caches

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # collect all benchmark names across every file
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

            fig, ax1 = plt.subplots(figsize=(9, 5))
            ax2 = ax1.twinx()

            for df_series, color, label in [(s, "C0", "scalar"), (v, "C1", "vectorized")]:
                aligned   = df_series[df_series["size"] % 8 == 0]
                unaligned = df_series[df_series["size"] % 8 != 0]
                ax1.plot(df_series["size"], df_series["cells_per_second"], "-",  color=color, label=label + " cells/s")
                ax2.plot(df_series["size"], df_series["bytes_per_second"], "--", color=color, label=label + " bytes/s", alpha=0.4)
                ax1.scatter(aligned["size"],   aligned["cells_per_second"],   marker="o", color=color, zorder=5)
                ax1.scatter(unaligned["size"], unaligned["cells_per_second"], marker="x", color=color, zorder=5)
                ax2.scatter(aligned["size"],   aligned["bytes_per_second"],   marker="o", color=color, alpha=0.4)
                ax2.scatter(unaligned["size"], unaligned["bytes_per_second"], marker="x", color=color, alpha=0.4)

            for level, size_bytes in sorted(caches.items()):
                n_cache = (size_bytes / BYTES_PER_CELL) ** (1/3)
                color = cache_colors.get(level, "gray")
                ax1.axvline(n_cache, linestyle="--", color=color, alpha=0.7,
                            label=f"L{level} ({size_bytes // 1024} KB) → n≈{n_cache:.0f}")

            ax1.set_title(f"{base_name} — {bm_label}")
            ax1.set_xlabel("n (cube width in cells)")
            ax1.set_ylabel("cells/s")
            ax2.set_ylabel("bytes/s")
            ax1.legend(fontsize=8)
            ax1.grid(True)
            plt.tight_layout()
            plt.savefig(out_dir / f"{bm_label}_{base_name}.png", dpi=200)
            plt.close()
plot_scalar_vs_vector(FILES, OUT_DIR)
