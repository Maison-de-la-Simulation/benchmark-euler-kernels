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
    # "skx_1":  "results/ruche/skx/[455228]_cpus-1_bm_ruche.json",
    "skx":  "results/ruche/skx/[455363]_skx-PrimToCons_bm_ruche.json",
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

def plot_scalar_vs_vector(files, out_dir, base_name):
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    def load_one(path):
        with open(path) as f:
            data = json.load(f)["benchmarks"]
        rows = []
        for b in data:
            name = b["name"]
            if base_name not in name and (base_name + "Vectorized") not in name:
                continue
            rows.append({
                "benchmark": name.split("/")[0],
                "size": int(name.split("/")[-1]),
                "cells_per_second": b.get("cells_per_second"),
                "bytes_per_second": b.get("bytes_per_second"),
            })
        return pd.DataFrame(rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    vec_name = base_name + "Vectorized"
    print(files)

    for environment, path in files.items():
        df = load_one(path)
        bm_label = extract_label(path)

        s = df[df["benchmark"] == base_name].sort_values("size")
        v = df[df["benchmark"] == vec_name].sort_values("size")

        if s.empty or v.empty:
            print(f"skipping {base_name}")
            continue  

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax2 = ax1.twinx()

        # cells/s
        ax1.plot(s["size"], s["cells_per_second"], "o-", label="scalar")
        ax1.plot(v["size"], v["cells_per_second"], "s-", label="vectorized")

        # bytes/s
        ax2.plot(s["size"], s["bytes_per_second"], "--")
        ax2.plot(v["size"], v["bytes_per_second"], ":")

        ax1.set_title(base_name +" " + bm_label)
        ax1.set_xlabel("nx")
        ax1.set_ylabel("cells/s")
        ax2.set_ylabel("bytes/s")

        ax1.legend(fontsize=8)
        ax1.grid(True)
        plt.tight_layout()

        print("saving ...")
        plt.savefig(out_dir / f"{environment}_{bm_label}_{base_name}.png", dpi=200)
        plt.close()

plot_scalar_vs_vector(FILES, OUT_DIR, "PrimToCons")
