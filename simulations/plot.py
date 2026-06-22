#

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.lines import Line2D


def load_one(path):
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


def compare_benchmarks(path_a, path_b, out_csv, cols=None):
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

    rounding = {
        c: 5
        for c in merged.columns
        if "speedup" in c or "time" in c or "cells_per_second" in c or "bytes_per_second" in c
    }
    merged = merged.round(rounding)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


SAVE_PLOTS = True
FONT_SIZE = 6
LOG_BASE_Y = 10
LOG_BASE_X = 2

HW_COLORS = {
    "skx": "C0",
    "genoa": "C1",
    "mi300a": "C2",
    "gh200": "C3",
    "a100": "C4",
    "mi250": "C5",
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
    "mi250": "MI250X (GPU)",
    "unknown": "Unknown",
}

# %%
HW_MARKERS_DB = {
    "genoa": [
        # (8, "CCX / Shared L3"),
        # (24, "CCD"),
        (96, "1st Socket sat"),
    ],
    "gh200": [
        (72, "1st Socket sat"),
        (142, "2nd Socket sat"),
        (184, "3rd Socket sat"),
    ],
}
# %%

DPI_SIZE = 120


def get_hw(path):
    name = Path(path).name.lower()
    for tag in ["skx", "genoa", "mi300", "gh200", "a100", "mi250"]:
        if tag in name:
            return tag
    return "unknown"


def hw_label_speedup(hw):
    base = HW_LABELS.get(hw, hw)
    w = f"(W={SIMD_WIDTH.get(hw) or 'N/A'})"
    return f"{base} | {w}"


def setup_log_scales(ax, x_log=True, y_log=False):
    if x_log:
        ax.set_xscale("log", base=LOG_BASE_X)
    if y_log:
        ax.set_yscale("log", base=LOG_BASE_Y)


def setup_power_of_two_ticks(ax, df):
    tick_df = df[df["size"].apply(lambda x: x > 0 and (x & (x - 1)) == 0)]
    ax.set_xticks(sorted(tick_df["size"].unique()))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(plt.NullFormatter())


def add_legend_pair(ax, hardware_handles, linestyles_dict):
    legend1 = ax.legend(
        hardware_handles.values(),
        [hw_label_speedup(hw) for hw in hardware_handles.keys()],
        loc="center right",
        fontsize=FONT_SIZE,
    )
    ax.add_artist(legend1)

    proxies = []
    labels = []
    for style, label in linestyles_dict.items():
        (proxy,) = ax.plot([], [], style, color="black")
        proxies.append(proxy)
        labels.append(label)

    ax.legend(
        proxies,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.0),
        fontsize=FONT_SIZE,
        title="Kernel",
    )


def plot_series(ax, data, color, metric, linestyle="-"):
    ax.plot(
        data["size"],
        data[metric],
        color=color,
        marker="o",
        markersize=3,
        alpha=0.7,
        linestyle=linestyle,
    )


def add_hw_markers(ax, hw, ymax):
    if hw not in HW_MARKERS_DB:
        return

    for x, label in HW_MARKERS_DB[hw]:
        ax.axvline(
            x,
            linestyle="--",
            linewidth=1.0,
            color=HW_COLORS.get(hw, "black"),
            alpha=0.5,
        )

        ax.annotate(
            label,
            xy=(x, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
            clip_on=False,
            color=HW_COLORS.get(hw, "black"),
        )


def save_or_show(out_dir, filename):
    if SAVE_PLOTS:
        plt.savefig(out_dir / filename, dpi=DPI_SIZE, orientation="portrait")
    else:
        plt.show()


def load_data_from_files(res_dir, data_loader):
    res_dir = Path(res_dir)
    files = list(res_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files in {res_dir}")

    data = []
    for f in files:
        df, _ = load_one(str(f))
        df["hardware"] = get_hw(f)
        data.append(df)

    return pd.concat(data, ignore_index=True)


def plot_hw_speedup(res_dir, out_dir):
    res_dir = Path(res_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data_from_files(res_dir, load_one)

    bases = ["Godunov", "EulerSimulation"]

    for base in bases:
        fig, ax = plt.subplots(figsize=(8, 5))

        metric = "bytes_per_second" if base != "EulerSimulation" else "real_time_ns"
        vector_name = f"{base}Vectorized"
        opti_name = f"{base}Opti"

        hardware_handles = {}

        for hw in df["hardware"].unique():
            d = df[df["hardware"] == hw]
            opti = d[d["benchmark"] == opti_name].sort_values("size")
            vector = d[d["benchmark"] == vector_name].sort_values("size")

            if opti.empty or vector.empty:
                continue

            merged = pd.merge(
                opti[["size", metric]],
                vector[["size", metric]],
                on="size",
                suffixes=("_opti", "_vector"),
            )

            if merged.empty:
                continue

            if metric == "bytes_per_second":
                speedup = merged[f"{metric}_vector"] / merged[f"{metric}_opti"]
            else:
                speedup = merged[f"{metric}_opti"] / merged[f"{metric}_vector"]

            color = HW_COLORS.get(hw, "black")
            ax.plot(
                merged["size"],
                speedup,
                color=color,
                marker="o",
                markersize=3,
                alpha=0.7,
            )

            if hw not in hardware_handles:
                hardware_handles[hw] = ax.lines[-1]

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        setup_log_scales(ax, x_log=True, y_log=False)
        setup_power_of_two_ticks(ax, df)

        ax.set_title(f"{base} Speedup (Vectorized vs Scalar)")
        ax.set_xlabel("n (cube width in cells)")
        ax.set_ylabel("Speedup")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        legend = ax.legend(
            hardware_handles.values(),
            [hw_label_speedup(hw) for hw in hardware_handles.keys()],
            fontsize=FONT_SIZE,
        )
        ax.add_artist(legend)

        plt.tight_layout()
        save_or_show(out_dir, f"{base}_speedup.png")
        plt.close()


def plot_hw_scalar_vector(res_dir, out_dir, title=""):
    res_dir = Path(res_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data_from_files(res_dir, load_one)
    bases = ["Godunov", "EulerSimulation"]

    for base in bases:
        vector_name = f"{base}Vectorized"
        opti_name = f"{base}Opti"

        fig, ax_cells = plt.subplots(figsize=(8, 5))

        is_euler = base == "EulerSimulation"
        ax_bytes = ax_cells.twinx() if not is_euler else None

        hardware_handles = {}

        for hw in df["hardware"].unique():
            d = df[df["hardware"] == hw]
            opti = d[d["benchmark"] == opti_name].sort_values("size")
            vector = d[d["benchmark"] == vector_name].sort_values("size")

            if opti.empty or vector.empty:
                continue

            color = HW_COLORS.get(hw, "black")

            plot_series(ax_cells, opti, color, "cells_per_second", linestyle="-")
            plot_series(ax_cells, vector, color, "cells_per_second", linestyle="--")

            if hw not in hardware_handles:
                hardware_handles[hw] = ax_cells.lines[-2]

            if not is_euler and ax_bytes is not None:
                plot_series(ax_bytes, opti, color, "bytes_per_second", linestyle="-")
                plot_series(ax_bytes, vector, color, "bytes_per_second", linestyle="--")

        add_legend_pair(ax_cells, hardware_handles, {"-": "scalar", "--": "vectorized"})

        setup_log_scales(ax_cells, x_log=True, y_log=True)
        ax_cells.grid(True, which="both", linestyle="--", alpha=0.5)

        # setup_power_of_two_ticks(ax_cells, df)

        ax_cells.set_ylabel("cells per second")

        if not is_euler and ax_bytes is not None:
            setup_log_scales(ax_bytes, x_log=True, y_log=True)
            ax_bytes.set_ylabel("bytes per second")

        ax_cells.set_xlabel("n (cube width in cells)")
        ax_cells.set_title(f"{title} {base}".strip())
        ax_cells.grid(True)
        ax_cells.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax_cells.get_xaxis().set_minor_formatter(plt.NullFormatter())

        plt.tight_layout()
        save_or_show(out_dir, f"{base}_hw_compare.png")
        plt.close()


GPU_BASELINES = {
    "a100": 2681.6,
    "mi250": 2427.0,
    "mi300": 5639.1,
}


def plot_scaling_dir(res_dir, out_dir, title=""):
    res_dir = Path(res_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(res_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {res_dir}")

    data = []
    for f in files:
        df = pd.read_csv(f)
        hw = get_hw(f)
        mode = "vector" if "vector" in f.stem.lower() else "scalar"
        df["hardware"] = hw
        df["kernel_mode"] = mode
        data.append(df)

    df = pd.concat(data, ignore_index=True)

    for scaling in sorted(df["mode"].unique()):
        dscale = df[df["mode"] == scaling]
        fig, ax_thr = plt.subplots(1, 1, figsize=(8, 5))

        hardware_handles = {}
        baseline_handles = {}

        for hw in sorted(dscale["hardware"].unique()):
            dhw = dscale[dscale["hardware"] == hw]
            color = HW_COLORS.get(hw, "black")

            for kernel_mode, linestyle in [("scalar", "-"), ("vector", "--")]:
                dk = dhw[dhw["kernel_mode"] == kernel_mode]
                if dk.empty:
                    continue

                dk = dk.sort_values("threads")
                (line,) = ax_thr.plot(
                    dk["threads"],
                    dk["mcells_s"],
                    marker="o",
                    markersize=4,
                    linestyle=linestyle,
                    color=color,
                    alpha=0.8,
                )

                if hw not in hardware_handles and kernel_mode == "scalar":
                    hardware_handles[hw] = line

        for hw, baseline in GPU_BASELINES.items():
            (line,) = ax_thr.plot([], [], ":", color=HW_COLORS.get(hw), linewidth=1.5)
            baseline_handles[hw] = line
            ax_thr.axhline(
                baseline,
                color=HW_COLORS.get(hw),
                linestyle=":",
                linewidth=1.5,
                alpha=0.6,
            )

        setup_log_scales(ax_thr, x_log=True, y_log=True)
        ax_thr.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax_thr.get_xaxis().set_minor_formatter(plt.NullFormatter())

        ax_thr.set_xlabel("OpenMP threads")
        ax_thr.set_ylabel("Million cells / second")
        ax_thr.grid(True, which="both", linestyle="--", alpha=0.5)

        ymax = ax_thr.get_ylim()[0] + 70
        for hw in sorted(dscale["hardware"].unique()):
            add_hw_markers(ax_thr, hw, ymax)

        legend1 = ax_thr.legend(
            hardware_handles.values(),
            [hw_label_speedup(hw) for hw in hardware_handles.keys()],
            loc="center right",
            fontsize=FONT_SIZE,
        )
        ax_thr.add_artist(legend1)

        (scalar_proxy,) = ax_thr.plot([], [], "-", color="black")
        (vector_proxy,) = ax_thr.plot([], [], "--", color="black")

        kernel_legend = ax_thr.legend(
            [scalar_proxy, vector_proxy],
            ["scalar", "vectorized"],
            loc="upper center",
            fontsize=FONT_SIZE,
            title="Kernel",
        )
        ax_thr.add_artist(kernel_legend)

        ax_thr.set_title(f" EulerSimulation Throughput Scaling".strip())
        if baseline_handles:
            baseline_legend = ax_thr.legend(
                baseline_handles.values(),
                [f"{hw} {GPU_BASELINES[hw]} Mcells/s" for hw in baseline_handles.keys()],
                loc="upper left",
                fontsize=FONT_SIZE,
                title="GPU Peak",
            )
            ax_thr.add_artist(baseline_legend)

        plt.tight_layout()
        save_or_show(out_dir, f"{scaling}_scaling.png")
        plt.close()


# %%


def plot_hw_euler_vectorized_256(res_dir, out_dir):
    res_dir = Path(res_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data_from_files(res_dir, load_one)

    vec = df[(df["benchmark"] == "EulerSimulationVectorized") & (df["size"] == 256)][
        ["hardware", "cells_per_second"]
    ]

    opti = df[(df["benchmark"] == "EulerSimulationOpti") & (df["size"] == 256)][
        ["hardware", "cells_per_second"]
    ]

    data = vec.merge(opti, on="hardware", suffixes=("_vec", "_opti"))
    if data.empty:
        print("no data")
        return

    data = data.sort_values("cells_per_second_vec")

    x = np.arange(len(data))
    width = 0.4

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(
        x + width / 2,
        data["cells_per_second_vec"] * 1e-6,
        width,
        label="Kokkos SIMD",
        color="tab:blue",
        alpha=0.8,
    )
    bars2 = ax.bar(
        x - width / 2,
        data["cells_per_second_opti"] * 1e-6,
        width,
        label="Kokkos Scalar",
        color="tab:orange",
        alpha=0.8,
    )

    # ax.bar_label(bars1, fmt="%.1f", fontsize=FONT_SIZE + 2, padding=3)
    # ax.bar_label(bars2, fmt="%.1f", fontsize=FONT_SIZE + 2, padding=3)
    ax.bar_label(bars2, fmt="%.0f", color="tab:orange", fontsize=8)
    ax.bar_label(bars1, fmt="%.0f", color="tab:blue", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [hw_label_speedup(hw).split("|")[0] for hw in data["hardware"]],
        # [hw.upper() for hw in data["hardware"]],
        # data["hardware"],
        # rotation=90,
        # ha="right",
        fontsize=8,
    )

    ax.set_ylabel("Million cells/s")
    ax.set_xlabel("Hardware")
    # ax.set_yscale("log", base=LOG_BASE_Y)
    ax.set_title("EulerSimulation (n = 256)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()
    save_or_show(out_dir, "EulerSimulation_256_hw_comparison.png")
    plt.close()


# %%
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_best_hw_vectorized_from_csvs(csv_dir, out_dir):
    """
    For each hardware:
      1. Find matching scalar/vector CSVs.
      2. Find thread counts present in BOTH runs.
      3. Choose the thread count with the highest vector performance.
      4. Plot scalar vs vector performance at that same thread count.

    Expected filenames:
        1032031_strong-skx_vector.csv
        1032105_strong-skx_scalar.csv
        4979270_strong-genoa_vector_FINAL.csv
        4979653_strong-genoa_scalar_FINAL.csv
        ...
    """

    csv_dir = Path(csv_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def get_hw(fname):
        """
        Extract:
            skx
            genoa
            gh200-amd
        from filenames.
        """
        m = re.search(
            r"strong-([A-Za-z0-9\-]+)_(scalar|vector)",
            Path(fname).stem,
        )
        return m.group(1) if m else None

    vector_files = {}
    scalar_files = {}

    for f in csv_dir.glob("*.csv"):
        hw = get_hw(f.name)

        if hw is None:
            continue

        if "_vector" in f.stem:
            vector_files[hw] = f
        elif "_scalar" in f.stem:
            scalar_files[hw] = f

    rows = []

    for hw in sorted(vector_files.keys()):

        if hw not in scalar_files:
            print(f"missing scalar file for {hw}")
            continue

        vec_file = vector_files[hw]
        scalar_file = scalar_files[hw]

        vec = pd.read_csv(vec_file)
        scalar = pd.read_csv(scalar_file)

        common_threads = sorted(set(vec["threads"]) & set(scalar["threads"]))

        if not common_threads:
            print(f"no common thread counts for {hw}")
            continue

        vec_common = vec[vec["threads"].isin(common_threads)]

        # Best vectorized configuration
        best_vec = vec_common.loc[vec_common["mcells_s"].idxmax()]

        threads = int(best_vec["threads"])

        scalar_row = scalar[scalar["threads"] == threads].iloc[0]

        rows.append(
            {
                "hardware": hw,
                "threads": threads,
                "mcells_vec": float(best_vec["mcells_s"]),
                "mcells_scalar": float(scalar_row["mcells_s"]),
                "speedup": float(best_vec["mcells_s"]) / float(scalar_row["mcells_s"]),
            }
        )

    data = pd.DataFrame(rows)

    if data.empty:
        print("no data")
        return

    data = data.sort_values("mcells_vec")

    x = np.arange(len(data))
    width = 0.4

    fig, ax = plt.subplots(figsize=(8, 5))

    bars_vec = ax.bar(
        x + width / 2,
        data["mcells_vec"],
        width,
        label="Kokkos SIMD",
        color="tab:blue",
        alpha=0.8,
    )

    bars_scalar = ax.bar(
        x - width / 2,
        data["mcells_scalar"],
        width,
        label="Kokkos Scalar",
        color="tab:orange",
        alpha=0.8,
    )

    ax.bar_label(
        bars_scalar,
        fmt="%.0f",
        color="tab:orange",
        fontsize=8,
    )

    ax.bar_label(
        bars_vec,
        fmt="%.0f",
        color="tab:blue",
        fontsize=8,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            f"{hw_label_speedup(hw).split('|')[0]}\n({threads} cores)"
            for hw, threads in zip(
                data["hardware"],
                data["threads"],
            )
        ],
        fontsize=8,
    )

    ax.set_ylabel("Million cells/s")
    ax.set_xlabel("Hardware")
    ax.set_title("Best SIMD Configuration per Hardware")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend()

    plt.tight_layout()

    save_or_show(
        out_dir,
        "best_vectorized_hw_comparison.png",
    )

    print(
        data[
            [
                "hardware",
                "threads",
                "mcells_scalar",
                "mcells_vec",
                "speedup",
            ]
        ]
    )

    plt.close()


plot_best_hw_vectorized_from_csvs(
    "./results/opti-godunov/scaling/", "./results/opti-godunov/scaling/"
)


# %%


def plot_hw_euler_vectorized_256_all(res_dir, csv_dir, out_dir):
    import re

    res_dir = Path(res_dir)
    csv_dir = Path(csv_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #
    # Existing benchmark results
    #
    df = load_data_from_files(res_dir, load_one)

    vec = df[(df["benchmark"] == "EulerSimulationVectorized") & (df["size"] == 256)][
        ["hardware", "cells_per_second"]
    ].copy()

    opti = df[(df["benchmark"] == "EulerSimulationOpti") & (df["size"] == 256)][
        ["hardware", "cells_per_second"]
    ].copy()

    #
    # CSV hardware -> benchmark hardware names
    #
    CSV_HW_TO_HW = {
        "skx": "skx",
        "genoa": "genoa",
        "gh200-arm": "gh200",
    }

    #
    # Parse CSV filenames
    #
    def get_hw(fname):
        m = re.search(
            r"strong-([A-Za-z0-9\-]+)_(scalar|vector)",
            Path(fname).stem,
        )
        return m.group(1) if m else None

    vector_files = {}
    scalar_files = {}

    for f in csv_dir.glob("*.csv"):
        hw = get_hw(f.name)

        if hw is None:
            continue

        if "_vector" in f.stem:
            vector_files[hw] = f
        elif "_scalar" in f.stem:
            scalar_files[hw] = f

    extra_vec = []
    extra_scalar = []

    #
    # For each hardware:
    # choose the best vectorized configuration
    # using a thread count available in BOTH scalar/vector runs
    #
    for hw in sorted(vector_files):

        if hw not in scalar_files:
            continue

        vec_df = pd.read_csv(vector_files[hw])
        scalar_df = pd.read_csv(scalar_files[hw])

        common_threads = sorted(set(vec_df["threads"]) & set(scalar_df["threads"]))

        if not common_threads:
            continue

        vec_common = vec_df[vec_df["threads"].isin(common_threads)]

        best_vec = vec_common.loc[vec_common["mcells_s"].idxmax()]

        threads = int(best_vec["threads"])

        scalar_row = scalar_df[scalar_df["threads"] == threads].iloc[0]

        canonical_hw = CSV_HW_TO_HW.get(hw, hw)

        hardware_key = f"{canonical_hw}_multicore"

        extra_vec.append(
            {
                "hardware": hardware_key,
                "cells_per_second": best_vec["mcells_s"] * 1e6,
                "threads": threads,
                "base_hw": canonical_hw,
            }
        )

        extra_scalar.append(
            {
                "hardware": hardware_key,
                "cells_per_second": scalar_row["mcells_s"] * 1e6,
                "threads": threads,
                "base_hw": canonical_hw,
            }
        )

    extra_vec_df = pd.DataFrame(extra_vec)
    extra_scalar_df = pd.DataFrame(extra_scalar)

    if not extra_vec_df.empty:
        vec = pd.concat(
            [vec, extra_vec_df[["hardware", "cells_per_second"]]],
            ignore_index=True,
        )

    if not extra_scalar_df.empty:
        opti = pd.concat(
            [opti, extra_scalar_df[["hardware", "cells_per_second"]]],
            ignore_index=True,
        )

    #
    # Merge scalar/vector results
    #
    data = vec.merge(
        opti,
        on="hardware",
        suffixes=("_vec", "_opti"),
    )

    if data.empty:
        print("no data")
        return

    #
    # Metadata for multicore labels
    #
    thread_map = {}
    base_hw_map = {}

    if not extra_vec_df.empty:
        thread_map = extra_vec_df.set_index("hardware")["threads"].to_dict()

        base_hw_map = extra_vec_df.set_index("hardware")["base_hw"].to_dict()

    data = data.sort_values("cells_per_second_vec")

    x = np.arange(len(data))
    width = 0.4

    fig, ax = plt.subplots(figsize=(12, 5))

    bars1 = ax.bar(
        x + width / 2,
        data["cells_per_second_vec"] * 1e-6,
        width,
        label="Kokkos SIMD",
        color="tab:blue",
        alpha=0.8,
    )

    bars2 = ax.bar(
        x - width / 2,
        data["cells_per_second_opti"] * 1e-6,
        width,
        label="Kokkos Scalar",
        color="tab:orange",
        alpha=0.8,
    )

    ax.bar_label(
        bars2,
        fmt="%.0f",
        color="tab:orange",
        fontsize=8,
    )

    ax.bar_label(
        bars1,
        fmt="%.0f",
        color="tab:blue",
        fontsize=8,
    )

    #
    # Labels:
    # Intel Xeon Gold
    # (1 core)
    #
    # Intel Xeon Gold
    # (40 cores)
    #
    labels = []

    for hw in data["hardware"]:

        if hw in thread_map:
            base_hw = base_hw_map[hw]

            labels.append(f"{HW_LABELS.get(base_hw, base_hw)}\n({thread_map[hw]} cores)")
        else:
            labels.append(f"{HW_LABELS.get(hw, hw)}\n(1 core)")

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        fontsize=8,
    )

    ax.set_ylabel("Million cells/s")
    ax.set_xlabel("Hardware")
    ax.set_title("EulerSimulation (n = 256)")

    ax.set_yscale("log", base=LOG_BASE_Y)

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.5,
    )

    ax.legend()

    plt.tight_layout()

    save_or_show(
        out_dir,
        "EulerSimulation_256_hw_comparison_all.png",
    )

    plt.close()


plot_hw_euler_vectorized_256_all(
    "./results/opti-godunov/",
    "./results/opti-godunov/scaling/",
    "./results/opti-godunov/scaling/",
)
# %%

SAVE_PLOTS = True
plot_hw_euler_vectorized_256("./results/opti-godunov/", "./results/opti-godunov/new_plots/")

# %%
plot_scaling_dir("./results/opti-godunov/scaling", "./results/opti-godunov/new_plots/")
plot_hw_scalar_vector("./results/opti-godunov", "./results/opti-godunov/new_plots/")
plot_hw_speedup("./results/opti-godunov", "./results/opti-godunov/new_plots/")
