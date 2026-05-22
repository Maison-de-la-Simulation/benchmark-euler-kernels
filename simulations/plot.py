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
    # GPUs intentionally omitted
}
HW_LABELS = {
    "skx": "Intel Xeon Gold ",
    "genoa": "AMD Genoa",
    "mi300": "MI300A (APU)",
    "gh200": "GH200 ARM (CPU)",
    "a100": "NVIDIA A100 (GPU)",
    "mi250": "MI250X (GPU)",
    "unknown": "Unknown",
}


def get_hw(path):
    name = Path(path).name.lower()
    for tag in ["skx", "genoa", "mi300", "gh200", "a100", "mi250"]:
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


def _plot_series(ax, data, color, label, metric):
    ax.plot(
        data["size"],
        data[metric],
        color=color,
        label=label,
        marker="o",
        markersize=3,
        alpha=0.7,
    )


def plot_hw_speedup(res_dir, out_dir):
    res_dir = Path(res_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = list(res_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON files in {res_dir}")

    data = []
    for f in files:
        df, _ = load_one(str(f))
        df["hardware"] = get_hw(f)
        data.append(df)

    df = pd.concat(data, ignore_index=True)

    BASES = ["Godunov", "EulerSimulation"]

    for base in BASES:
        fig, ax = plt.subplots(figsize=(8, 5))

        metric = "bytes_per_second" if base != "EulerSimulation" else "real_time_ns"

        vector_name = f"{base}Vectorized"
        opti_name = f"{base}Opti"

        hardware_handles = {}

        for hw in df["hardware"].unique():
            print("hw = ", hw)

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

        ax.set_title(f"{base} Speedup (Vectorized vs Scalar)")
        ax.set_xlabel("n (cube width in cells)")
        ax.set_ylabel("Speedup")
        ax.grid(True, which="both", linestyle="--", alpha=0.5)

        legend = ax.legend(
            hardware_handles.values(),
            [hw_label_speedup(hw) for hw in hardware_handles.keys()],
            fontsize=8,
        )
        ax.add_artist(legend)

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

    data = []
    for f in files:
        df, _ = load_one(str(f))
        df["hardware"] = get_hw(f)
        data.append(df)

    df = pd.concat(data, ignore_index=True)

    BASES = ["Godunov", "EulerSimulation"]

    for base in BASES:
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

            _plot_series(ax_cells, opti, color, "_nolegend_", "cells_per_second")
            _plot_series(ax_cells, vector, color, "_nolegend_", "cells_per_second")

            ax_cells.lines[-2].set_linestyle("-")
            ax_cells.lines[-1].set_linestyle("--")

            if hw not in hardware_handles:
                hardware_handles[hw] = ax_cells.lines[-2]

            if not is_euler and ax_bytes is not None:
                _plot_series(ax_bytes, opti, color, "_nolegend_", "bytes_per_second")
                _plot_series(ax_bytes, vector, color, "_nolegend_", "bytes_per_second")

                ax_bytes.lines[-2].set_linestyle("-")
                ax_bytes.lines[-1].set_linestyle("--")

        legend1 = ax_cells.legend(
            hardware_handles.values(),
            [hw_label_speedup(hw) for hw in hardware_handles.keys()],
            bbox_to_anchor=(0.7, 0.3),
            loc="best",
            fontsize=8,
            title="Hardware",
        )

        ax_cells.add_artist(legend1)

        (opti_proxy,) = ax_cells.plot([], [], "-", color="black")
        (vector_proxy,) = ax_cells.plot([], [], "--", color="black")

        ax_cells.legend(
            [opti_proxy, vector_proxy],
            ["scalar", "vectorized"],
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            fontsize=8,
            title="Kernel",
        )

        ax_cells.set_yscale("log")
        ax_cells.set_ylabel("cells per second")

        if not is_euler and ax_bytes is not None:
            ax_bytes.set_yscale("log")
            ax_bytes.set_ylabel("bytes per second")

        ax_cells.set_xlabel("n (cube width in cells)")
        ax_cells.set_title(f"{title} {base}".strip())
        ax_cells.grid(True)

        plt.tight_layout()
        # plt.show()
        plt.savefig(out_dir / f"{base}_hw_compare.png", dpi=200)
        plt.close()


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

        fig, (ax_thr, ax_spd) = plt.subplots(1, 2, figsize=(14, 5))

        hardware_handles_thr = {}
        hardware_handles_spd = {}

        for hw in sorted(dscale["hardware"].unique()):
            dhw = dscale[dscale["hardware"] == hw]
            color = HW_COLORS.get(hw, "black")

            for kernel_mode, linestyle in [("scalar", "-"), ("vector", "--")]:
                dk = dhw[dhw["kernel_mode"] == kernel_mode]

                if dk.empty:
                    continue

                dk = dk.sort_values("threads")

                ax_thr.plot(
                    dk["threads"],
                    dk["mcells_s"],
                    marker="o",
                    markersize=4,
                    linestyle=linestyle,
                    color=color,
                    alpha=0.8,
                )

                baseline = dk.iloc[0]["time_s"]
                speedup = baseline / dk["time_s"]

                ax_spd.plot(
                    dk["threads"],
                    speedup,
                    marker="o",
                    markersize=4,
                    linestyle=linestyle,
                    color=color,
                    alpha=0.8,
                )

                if kernel_mode == "scalar" and hw not in hardware_handles_thr:
                    hardware_handles_thr[hw] = ax_thr.lines[-1]
                    hardware_handles_spd[hw] = ax_spd.lines[-1]

        ideal_threads = sorted(dscale["threads"].unique())

        ax_spd.plot(
            ideal_threads,
            ideal_threads,
            linestyle="--",
            color="black",
            linewidth=1,
        )

        legend1_thr = ax_thr.legend(
            hardware_handles_thr.values(),
            [hw_label_speedup(hw) for hw in hardware_handles_thr.keys()],
            bbox_to_anchor=(0.7, 0.3),
            loc="best",
            fontsize=8,
            title="Hardware",
        )

        ax_thr.add_artist(legend1_thr)

        (scalar_proxy_thr,) = ax_thr.plot([], [], "-", color="black")
        (vector_proxy_thr,) = ax_thr.plot([], [], "--", color="black")

        ax_thr.legend(
            [scalar_proxy_thr, vector_proxy_thr],
            ["scalar", "vectorized"],
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            fontsize=8,
            title="Kernel",
        )

        legend1_spd = ax_spd.legend(
            hardware_handles_spd.values(),
            [hw_label_speedup(hw) for hw in hardware_handles_spd.keys()],
            bbox_to_anchor=(0.7, 0.3),
            loc="best",
            fontsize=8,
            title="Hardware",
        )

        ax_spd.add_artist(legend1_spd)

        (scalar_proxy_spd,) = ax_spd.plot([], [], "-", color="black")
        (vector_proxy_spd,) = ax_spd.plot([], [], "--", color="black")

        ax_spd.legend(
            [scalar_proxy_spd, vector_proxy_spd],
            ["scalar", "vectorized"],
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            fontsize=8,
            title="Kernel",
        )

        ax_thr.set_xscale("log")
        ax_thr.set_xlabel("OpenMP threads")
        ax_thr.set_ylabel("Million cells / second")
        ax_thr.set_title(f"{title} {scaling.capitalize()} Scaling Throughput".strip())
        ax_thr.grid(True, which="both", linestyle="--", alpha=0.5)

        ax_spd.set_xscale("log")
        ax_spd.set_yscale("log")
        ax_spd.set_xlabel("OpenMP threads")
        ax_spd.set_ylabel("Speedup")
        ax_spd.set_title(f"{title} {scaling.capitalize()} Scaling Speedup".strip())
        ax_spd.grid(True, which="both", linestyle="--", alpha=0.5)

        plt.tight_layout()
        # plt.show()
        plt.savefig(out_dir / f"{scaling}_scaling.png", dpi=200)
        plt.close()


plot_scaling_dir("./results/opti-godunov/scaling", "./results/opti-godunov/plots/")
plot_hw_scalar_vector("./results/opti-godunov", "./results/opti-godunov/plots/")
plot_hw_speedup("./results/opti-godunov", "./results/opti-godunov/plots/")
