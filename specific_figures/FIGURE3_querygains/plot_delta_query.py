import argparse
import csv
import json
import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
THEORY_BASE_DIR = SCRIPT_DIR.parents[1] / "theory_base"
if THEORY_BASE_DIR.exists() and str(THEORY_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(THEORY_BASE_DIR))

CORR0_PTR_TO_CORR_LEN = {
    128: 0,
    98: 1,
    60: 2,
    42: 3,
    32: 4,
    26: 5,
    22: 6,
    19: 7,
    17: 8,
    15: 9,
    14: 10,
}
CORR_LEN_TO_PTR = {corr_len: p_tr for p_tr, corr_len in CORR0_PTR_TO_CORR_LEN.items()}

MODEL_TITLES = {
    "4": "Full-Parameter linear attention",
    "5": "Softmax attention",
    "9": "Softmax attention with MLP",
}

SPINE_COLOR = "#A1AAB5"
AXES_FACE_COLOR = "#F7F7F7"
GRID_COLOR = "white"
EMPIRICAL_FILL_ALPHA = 0.32


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot delta_query = best independent-query fresh-test error "
            "- best dependent-query fresh-test error."
        )
    )
    parser.add_argument(
        "--corr_csv",
        type=str,
        default="saved_data/p128_corrlen_sweep.csv",
        help="Path to the correlated-token sweep CSV.",
    )
    parser.add_argument(
        "--reduced_csv",
        type=str,
        default="saved_data/reduced_effective_compare.csv",
        help="Path to the reduced-model effective-compare cache CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="plots/delta_query.pdf",
        help="Output image path.",
    )
    return parser.parse_args()


def load_rows(csv_path):
    csv.field_size_limit(sys.maxsize)
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def configure_plot_style(matplotlib):
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "mathtext.fontset": "dejavuserif",
            "axes.unicode_minus": False,
            "axes.formatter.use_mathtext": True,
            "legend.fontsize": 12,
        }
    )


def best_fresh_test_loss_from_row(row):
    fresh_test_losses = json.loads(row.get("fresh_test_losses_json", "[]"))
    if fresh_test_losses:
        return min(float(loss) for loss in fresh_test_losses)

    best_loss = row.get("best_fresh_test_loss")
    if best_loss in (None, ""):
        raise ValueError("Row is missing both fresh_test_losses_json and best_fresh_test_loss")
    return float(best_loss)


def mean_and_std(values):
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(variance)


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def corr_len_from_row(row):
    return int(round(float(row["corr_len"])))


def compute_theory_curves(x_values, theory_params):
    try:
        from theory import icl_correlated_REARRANGED, icl_uncorrelated, kernel_exp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Theory overlays require the dependencies needed by theory.py. "
            "Install them in the active Python environment and rerun plot_delta_query.py."
        ) from exc

    B = theory_params["B"]
    d = theory_params["d"]
    correlated_p_tr = theory_params["P_tr"]
    K = theory_params["K"]
    rho = theory_params["rho"]

    tau = B / (d ** 2)
    kappa = K / d
    correlated_alpha = correlated_p_tr / d

    no_query_values = []
    effective_values = []
    full_values = []
    for corr in x_values:
        if corr not in CORR_LEN_TO_PTR:
            raise ValueError(
                f"Missing effective-sample-size P_tr mapping for corr_len={corr}. "
                f"Known corr_len values: {sorted(CORR_LEN_TO_PTR)}"
            )
        effective_p_tr = CORR_LEN_TO_PTR[corr]
        effective_alpha = effective_p_tr / d

        corrmatrix = kernel_exp(int(correlated_p_tr), corr)
        no_query, tail = icl_correlated_REARRANGED(tau, correlated_alpha, kappa, rho, corrmatrix)
        effective = icl_uncorrelated(tau, effective_alpha, effective_alpha, kappa, rho, 1, 1)
        no_query_values.append(no_query)
        effective_values.append(effective)
        full_values.append(no_query + tail)

    return {
        "no_query": (x_values, no_query_values),
        "effective": (x_values, effective_values),
        "full": (x_values, full_values),
    }


def hex_to_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def interpolate_color(left, right, weight):
    left_rgb = hex_to_rgb(left)
    right_rgb = hex_to_rgb(right)
    return rgb_to_hex(
        tuple(
            round((1.0 - weight) * left_channel + weight * right_channel)
            for left_channel, right_channel in zip(left_rgb, right_rgb)
        )
    )


LOCAL_CURVE_STYLES = {
    "linear_reduced": "#7A3FE3",  # purple, close to current but cleaner/brighter
    "linear_full": "#5E6DE6",     # blue-violet, more cohesive with purple
}
SOFTMAX_COLORS = [
    "#36B6D0",  # Softmax: cyan from your cohesive palette
    "#0DA190",  # Softmax+MLP: darker teal for contrast
    "#3FBE73",  # Softmax+MLP x 4: green from your cohesive palette
    "#62B064",
    "#1E8A7F",
    "#3F8F46",
]
SOFTMAX_COLOR_BY_ARCHITECTURE = {
    ("5", 1): SOFTMAX_COLORS[0],
    ("9", 1): SOFTMAX_COLORS[1],
    ("9", 4): SOFTMAX_COLORS[2],
}

BASE_ARCHITECTURE_LABELS = {
    "reduced": "Reduced LA",
    "4": "Full LA",
    "5": "Softmax",
    "9": "Softmax+MLP",
}

BASE_DELTA_STYLES = {
    "reduced": {
        "color": LOCAL_CURVE_STYLES["linear_reduced"],
        "marker": "o",
        "linestyle": "-",
    },
    "4": {
        "color": LOCAL_CURVE_STYLES["linear_full"],
        "marker": "o",
        "linestyle": "-",
    },
}
LAYER_MARKERS = ["s", "s", "s", "s", "s", "s"]
SOFTMAX_MARKERS = ["s", "s", "s", "s", "s", "s"]


def parse_layer(row):
    return int(round(float(row.get("L", 1))))


def architecture_label(series_key):
    if series_key == "reduced":
        return BASE_ARCHITECTURE_LABELS["reduced"]

    model_type, layer = series_key
    label = BASE_ARCHITECTURE_LABELS.get(model_type, MODEL_TITLES.get(model_type, f"model_type={model_type}"))
    if layer == 1:
        return label
    return f"{label} x{layer}"


def architecture_sort_key(series_key):
    if series_key == "reduced":
        return (-1, 0)

    model_type, layer = series_key
    try:
        model_sort = int(model_type)
    except ValueError:
        model_sort = float("inf")
    return (model_sort, layer)


def softmax_color(series_key):
    if series_key in SOFTMAX_COLOR_BY_ARCHITECTURE:
        return SOFTMAX_COLOR_BY_ARCHITECTURE[series_key]

    model_type, layer = series_key
    try:
        model_index = int(model_type)
    except ValueError:
        model_index = sum(ord(character) for character in model_type)
    return SOFTMAX_COLORS[(model_index + layer) % len(SOFTMAX_COLORS)]


def softmax_marker(series_key):
    model_type, layer = series_key
    try:
        model_index = int(model_type)
    except ValueError:
        model_index = sum(ord(character) for character in model_type)
    return SOFTMAX_MARKERS[(model_index + layer) % len(SOFTMAX_MARKERS)]


def architecture_style(series_key):
    if series_key == "reduced":
        style = BASE_DELTA_STYLES["reduced"].copy()
        style["label"] = architecture_label(series_key)
        return style

    model_type, layer = series_key
    if model_type in BASE_DELTA_STYLES:
        style = BASE_DELTA_STYLES[model_type].copy()
    else:
        style = {
            "color": softmax_color(series_key),
            "marker": softmax_marker(series_key),
            "linestyle": "-",
        }
    if layer != 1:
        if model_type in BASE_DELTA_STYLES:
            style["color"] = interpolate_color(style["color"], LOCAL_CURVE_STYLES["linear_reduced"], 0.35)
            style["marker"] = LAYER_MARKERS[(layer - 1) % len(LAYER_MARKERS)]
        style["linestyle"] = "-"
    style["label"] = architecture_label(series_key)
    return style


def loss_std_from_row(row):
    value = row.get("best_fresh_test_loss_std")
    if value in (None, ""):
        return None
    return float(value)


def aggregate_query_stats_by_x(rows, x_getter):
    values_by_query_and_x = {
        False: defaultdict(list),
        True: defaultdict(list),
    }
    stds_by_query_and_x = {
        False: defaultdict(list),
        True: defaultdict(list),
    }

    for row in rows:
        correlated_query = parse_bool(row.get("correlated_query", "False"))
        x_value = x_getter(row)
        values_by_query_and_x[correlated_query][x_value].append(best_fresh_test_loss_from_row(row))

        row_std = loss_std_from_row(row)
        if row_std is not None:
            stds_by_query_and_x[correlated_query][x_value].append(row_std)

    stats_by_query = {}
    for correlated_query in [False, True]:
        stats_by_x = {}
        for x_value, values in values_by_query_and_x[correlated_query].items():
            mean, std = mean_and_std(values)
            row_stds = stds_by_query_and_x[correlated_query].get(x_value, [])
            if row_stds:
                row_std_rms = math.sqrt(sum(std_value**2 for std_value in row_stds) / len(row_stds))
                std = math.sqrt(std**2 + row_std_rms**2)
            stats_by_x[x_value] = {
                "mean": mean,
                "std": std,
                "n": len(values),
            }
        stats_by_query[correlated_query] = stats_by_x

    return stats_by_query


def delta_query_series_from_rows(rows, x_getter, series_label=None):
    stats_by_query = aggregate_query_stats_by_x(rows, x_getter)
    independent_stats = stats_by_query[False]
    dependent_stats = stats_by_query[True]
    x_values = sorted(set(independent_stats) & set(dependent_stats))
    if not x_values:
        raise ValueError("No correlation lengths have both independent and dependent query rows.")

    missing_independent = sorted(set(dependent_stats) - set(independent_stats))
    missing_dependent = sorted(set(independent_stats) - set(dependent_stats))
    if missing_independent or missing_dependent:
        label_suffix = f" for {series_label}" if series_label else ""
        warnings.warn(
            f"Ignoring unmatched query rows{label_suffix}: "
            f"missing_independent={missing_independent}, missing_dependent={missing_dependent}",
            stacklevel=2,
        )

    delta_values = []
    delta_stds = []
    counts = []
    for x_value in x_values:
        independent = independent_stats[x_value]
        dependent = dependent_stats[x_value]
        delta_values.append(independent["mean"] - dependent["mean"])
        delta_stds.append(math.sqrt(independent["std"] ** 2 + dependent["std"] ** 2))
        counts.append((independent["n"], dependent["n"]))

    return x_values, delta_values, delta_stds, counts


def reduced_corr_len(row):
    return int(round(float(row.get("plot_corr_len", row["corr_len"]))))


def infer_theory_params(corr_csv):
    rows = load_rows(Path(corr_csv))
    if not rows:
        raise ValueError(f"No rows found in {corr_csv}")

    preferred_rows = [
        row
        for row in rows
        if row.get("model_type") == "4" and parse_layer(row) == 1 and not parse_bool(row.get("correlated_query", "False"))
    ]
    fallback_rows = [row for row in rows if not parse_bool(row.get("correlated_query", "False"))]
    row = (preferred_rows or fallback_rows or rows)[0]

    return {
        "B": float(row["B"]),
        "d": float(row["d"]),
        "P_tr": float(row["P_tr"]),
        "K": float(row["K"]),
        "rho": float(row["rho"]),
    }


def compute_theory_delta_query(x_values, theory_params):
    theory_curves = compute_theory_curves(x_values, theory_params)
    theory_x, independent_query = theory_curves["no_query"]
    _, dependent_query = theory_curves["full"]
    return theory_x, [
        independent_loss - dependent_loss
        for independent_loss, dependent_loss in zip(independent_query, dependent_query)
    ]


def load_delta_query_data(corr_csv, reduced_csv):
    corr_rows = load_rows(Path(corr_csv))
    reduced_rows = load_rows(Path(reduced_csv))

    corr_rows_by_architecture = defaultdict(list)
    for row in corr_rows:
        model_type = row["model_type"]
        layer = parse_layer(row)
        if model_type == "4" and layer > 1:
            continue
        architecture_key = (model_type, layer)
        corr_rows_by_architecture[architecture_key].append(row)

    if not corr_rows_by_architecture:
        raise ValueError(f"No rows found in {corr_csv}")

    series_by_name = {}
    reduced_fixed_ptr_rows = [
        row
        for row in reduced_rows
        if row.get("model_type") == "reduced"
        and (
            (row.get("curve_name") == "p128" and not parse_bool(row.get("correlated_query", "False")))
            or (row.get("curve_name") == "corr_query" and parse_bool(row.get("correlated_query", "False")))
        )
    ]
    if reduced_fixed_ptr_rows:
        series_by_name["reduced"] = delta_query_series_from_rows(
            reduced_fixed_ptr_rows,
            reduced_corr_len,
            series_label=architecture_label("reduced"),
        )

    for architecture_key, rows in sorted(corr_rows_by_architecture.items(), key=lambda item: architecture_sort_key(item[0])):
        series_by_name[architecture_key] = delta_query_series_from_rows(
            rows,
            corr_len_from_row,
            series_label=architecture_label(architecture_key),
        )

    return series_by_name


def plot_delta_query(
    corr_csv="saved_data/p128_corrlen_sweep.csv",
    reduced_csv="saved_data/reduced_effective_compare.csv",
    output="plots/delta_query.png",
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        configure_plot_style(matplotlib)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to render the plot. Install it in the active Python environment "
            "and rerun plot_delta_query.py."
        ) from exc

    series_by_name = load_delta_query_data(
        corr_csv,
        reduced_csv,
    )

    fig, ax = plt.subplots(figsize=(5, 3.5))
    all_x_values = set()
    ordered_names = sorted(series_by_name, key=architecture_sort_key)

    for series_name in ordered_names:
        x_values, delta_values, delta_stds, _counts = series_by_name[series_name]
        all_x_values.update(x_values)
        style = architecture_style(series_name)
        ax.fill_between(
            x_values,
            [delta - std for delta, std in zip(delta_values, delta_stds)],
            [delta + std for delta, std in zip(delta_values, delta_stds)],
            color=style["color"],
            alpha=EMPIRICAL_FILL_ALPHA * 0.7,
            linewidth=0,
            zorder=1,
        )
        ax.plot(
            x_values,
            delta_values,
            marker=style["marker"],
            ms=5,
            linewidth=1.25,
            color=style["color"],
            linestyle=style["linestyle"],
            label=style["label"],
            zorder=3,
        )

    all_x_values = sorted(all_x_values)
    theory_x, theory_delta = compute_theory_delta_query(all_x_values, infer_theory_params(corr_csv))
    ax.plot(
        theory_x,
        theory_delta,
        marker=None,
        linewidth=1.6,
        color=LOCAL_CURVE_STYLES["linear_reduced"],
        linestyle="--",
        label="Theory",
        zorder=4,
    )

    ax.axhline(0.0, color=SPINE_COLOR, linewidth=1.0, alpha=0.8, zorder=0)
    ax.set_xlabel(r"Correlation length", fontsize=10)
    ax.set_ylabel(r"$\Delta_{\mathrm{query}}$", fontsize=12)
    ax.set_xticks(all_x_values)
    ax.tick_params(axis="both", labelsize=8, color=SPINE_COLOR, labelcolor=SPINE_COLOR)
    ax.set_facecolor(AXES_FACE_COLOR)
    ax.grid(
        True,
        which="major",
        color=GRID_COLOR,
        linewidth=1.0,
        alpha=0.9,
    )
    ax.grid(
        True,
        which="minor",
        color=GRID_COLOR,
        linewidth=0.5,
        alpha=0.45,
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(1.0)

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    handle_by_label = {label: handle for handle, label in zip(legend_handles, legend_labels)}
    legend_labels_ordered = [
        "Theory",
        "Softmax",
        "Reduced LA",
        "Softmax+MLP",
        "Full LA",
        "Softmax+MLP x4",
    ]
    fig.legend(
        [handle_by_label[label] for label in legend_labels_ordered],
        legend_labels_ordered,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=10,
        columnspacing=1.4,
        handlelength=2.0,
        handletextpad=0.5,
        bbox_to_anchor=(0.5, 1.05),
    )
    # fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.78))

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def main():
    args = parse_args()
    output_path = plot_delta_query(
        corr_csv=args.corr_csv,
        reduced_csv=args.reduced_csv,
        output=args.output,
    )
    print(output_path)


if __name__ == "__main__":
    main()
