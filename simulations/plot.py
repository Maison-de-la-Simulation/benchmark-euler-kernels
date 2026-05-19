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


def plot_strong_scaling(path):
    df = pd.read_csv(path)

    threads = df["threads"].to_numpy()
    time_s = df["time_s"].to_numpy()
    mcells_s = df["mcells_s"].to_numpy()

    speedup = time_s[0] / time_s
    efficiency = 100 * speedup / threads

    fig, ax1 = plt.subplots(figsize=(7, 5))

    ax1.plot(threads, speedup, marker="o", label="Measured speedup")
    ax1.plot(threads, threads, linestyle="--", label="Ideal speedup")

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Threads")
    ax1.set_ylabel("Speedup")
    ax1.grid(True)

    ax2 = ax1.twinx()

    ax2.plot(threads, efficiency, marker="s", linestyle=":", label="Efficiency")
    ax2.set_ylabel("Efficiency (%)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    p
    plt.tight_layout()
    plt.show()

plot_strong_scaling("./results/scaling/base-non_multi/strong-non_multi_scaling.csv")
# plot_hw_scalar_vector("./results", "./results/test_polts/")
# plot_hw_speedup("./results", "./results/test_polts/")
