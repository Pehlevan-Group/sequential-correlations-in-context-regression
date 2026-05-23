import argparse
import csv
import json
import math
import sys
from pathlib import Path

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
    "reduced": "Reduced-Parameter linear attention",
    "4": "Full-Parameter linear attention",
    "9": "Softmax attention with MLP",
}

CURVE_STYLES = {
    "p128": {
        "color": "#5E60CE",
        "label": r"Correlated tokens + independent query",
    },
    "corr0": {
        "color": "#6FA8DC",
        "label": r"Effective $\ell$ surrogate",
    },
    "corr_query": {
        "color": "#2A9D8F",
        "label": r"Correlated tokens + correlated query",
    },
}
LEGEND_ORDER = [
    CURVE_STYLES["p128"]["label"],
    CURVE_STYLES["corr_query"]["label"],
    CURVE_STYLES["corr0"]["label"],
    "theory",
]

SPINE_COLOR = "#A1AAB5"
THEORY_COLOR = "#7A7A7A"
THEORY_ALPHA = 0.45
EMPIRICAL_FILL_ALPHA = 0.32
REDUCED_NUM_RUNS = 20
REDUCED_CACHE_FIELDNAMES = [
    "model_type",
    "curve_name",
    "plot_corr_len",
    "corr_len",
    "P_tr",
    "correlated_query",
    "B",
    "d",
    "K",
    "rho",
    "num_runs",
    "best_fresh_test_loss",
    "best_fresh_test_loss_std",
    "fresh_test_losses_json",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare best fresh-test loss against corr_len across two sweep CSVs."
    )
    parser.add_argument(
        "--p128_csv",
        type=str,
        default="saved_data/p128_corrlen_sweep.csv",
        help="Path to the corr_len sweep CSV with fixed P_tr=128.",
    )
    parser.add_argument(
        "--corr0_csv",
        type=str,
        default="saved_data/corr0_ptr_sweep.csv",
        help="Path to the effective-sample-size sweep CSV at corr_len=0.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="queryfree/effective_compare.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--model_type",
        type=int,
        nargs="*",
        default=None,
        help="Optional model_type values to plot. Default plots all model types present.",
    )
    parser.add_argument(
        "--layer",
        "--L",
        dest="layer",
        type=int,
        default=1,
        help="Layer count L to plot. Default plots only L=1 rows.",
    )
    parser.add_argument(
        "--theory",
        action="store_true",
        help="Overlay theory curves on the linear-attention subplot.",
    )
    parser.add_argument(
        "--corrquery",
        action="store_true",
        help="Include correlated-query (red) curves.",
    )
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="Include a reduced-model subplot on the left.",
    )
    parser.add_argument(
        "--reduced_csv",
        type=str,
        default="saved_data/reduced_effective_compare.csv",
        help="Cache CSV for reduced-model curves.",
    )
    return parser.parse_args()


def load_rows(csv_path):
    csv.field_size_limit(sys.maxsize)
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def load_rows_if_exists(csv_path):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return []
    return load_rows(csv_path)


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


def aggregate_stats_by_x(rows, x_getter):
    values_by_x = {}
    for row in rows:
        x_value = x_getter(row)
        y_value = best_fresh_test_loss_from_row(row)
        values_by_x.setdefault(x_value, []).append(y_value)

    if not values_by_x:
        raise ValueError("No valid rows found for plotting")

    xs = sorted(values_by_x)
    ys = []
    stds = []
    for x in xs:
        mean, std = mean_and_std(values_by_x[x])
        ys.append(mean)
        stds.append(std)
    return xs, ys, stds


def parse_bool(value):
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def filter_rows_by_correlated_query(rows, correlated_query):
    return [row for row in rows if parse_bool(row.get("correlated_query", "False")) == correlated_query]


def filter_rows_by_layer(rows, layer):
    return [row for row in rows if int(round(float(row["L"]))) == layer]


def aggregate_stats_from_sources(source_rows):
    values_by_x = {}
    for rows, x_getter in source_rows:
        for row in rows:
            x_value = x_getter(row)
            y_value = best_fresh_test_loss_from_row(row)
            values_by_x.setdefault(x_value, []).append(y_value)

    if not values_by_x:
        return None

    xs = sorted(values_by_x)
    ys = []
    stds = []
    for x in xs:
        mean, std = mean_and_std(values_by_x[x])
        ys.append(mean)
        stds.append(std)
    return xs, ys, stds


def group_rows_by_model_type(rows, selected_model_types=None):
    grouped_rows = {}
    allowed = None if selected_model_types is None else {str(model_type) for model_type in selected_model_types}

    for row in rows:
        model_type = row["model_type"]
        if allowed is not None and model_type not in allowed:
            continue
        grouped_rows.setdefault(model_type, []).append(row)

    if not grouped_rows:
        if selected_model_types is None:
            raise ValueError("No rows found for plotting")
        raise ValueError(f"No rows found for model_type values {sorted(selected_model_types)}")

    return dict(sorted(grouped_rows.items(), key=lambda item: int(item[0])))


def p128_corr_len(row):
    return int(round(float(row["corr_len"])))


def corr0_effective_corr_len(row):
    p_tr = int(row["P_tr"])
    if p_tr not in CORR0_PTR_TO_CORR_LEN:
        raise ValueError(
            f"P_tr={p_tr} is not in the specified effective corr_len mapping: "
            f"{sorted(CORR0_PTR_TO_CORR_LEN)}"
        )
    return CORR0_PTR_TO_CORR_LEN[p_tr]


def load_effective_compare_data(
    p128_csv,
    corr0_csv,
    selected_model_types=None,
    selected_layer=1,
    include_corrquery=False,
):
    p128_rows = filter_rows_by_layer(load_rows(Path(p128_csv)), selected_layer)
    corr0_rows = filter_rows_by_layer(load_rows(Path(corr0_csv)), selected_layer)
    p128_false_by_model = group_rows_by_model_type(
        filter_rows_by_correlated_query(p128_rows, False),
        selected_model_types,
    )
    p128_true_by_model = {}
    if include_corrquery:
        p128_true_rows = filter_rows_by_correlated_query(p128_rows, True)
        p128_true_by_model = group_rows_by_model_type(p128_true_rows, selected_model_types) if p128_true_rows else {}
    corr0_false_by_model = group_rows_by_model_type(
        filter_rows_by_correlated_query(corr0_rows, False),
        selected_model_types,
    )

    model_types = sorted(set(p128_false_by_model) | set(corr0_false_by_model), key=int)
    series_by_model = {}
    for model_type in model_types:
        if model_type not in p128_false_by_model or model_type not in corr0_false_by_model:
            raise ValueError(
                f"model_type={model_type} is missing from one of the input CSVs: "
                f"p128_present={model_type in p128_false_by_model}, corr0_present={model_type in corr0_false_by_model}"
            )
        p128_x, p128_y, p128_std = aggregate_stats_by_x(p128_false_by_model[model_type], p128_corr_len)
        corr0_x, corr0_y, corr0_std = aggregate_stats_by_x(corr0_false_by_model[model_type], corr0_effective_corr_len)
        corr_query = None
        if include_corrquery:
            corr_query = aggregate_stats_from_sources(
                [
                    (p128_true_by_model.get(model_type, []), p128_corr_len),
                ]
            )
        theory_row = p128_false_by_model[model_type][0]
        series_by_model[model_type] = {
            "p128": (p128_x, p128_y, p128_std),
            "corr0": (corr0_x, corr0_y, corr0_std),
            "corr_query": corr_query,
            "theory_params": {
                "B": float(theory_row["B"]),
                "d": float(theory_row["d"]),
                "P_tr": float(theory_row["P_tr"]),
                "K": float(theory_row["K"]),
                "rho": float(theory_row["rho"]),
            },
        }

    return series_by_model


def compute_theory_curves(x_values, theory_params):
    try:
        from theory import icl_correlated_REARRANGED, icl_uncorrelated, kernel_exp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Theory overlays require the dependencies needed by theory.py. "
            "Install them in the active Python environment and rerun effective_compare.py."
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


THEORY_CURVE_STYLE_MAP = {
    "no_query": "p128",
    "effective": "corr0",
    "full": "corr_query",
}


def required_reduced_specs(x_values, theory_params, include_corrquery=False):
    fixed_p_tr = int(round(theory_params["P_tr"]))
    specs = []
    for corr in x_values:
        adaptive_p_tr = CORR_LEN_TO_PTR[corr]
        specs.append(
            {
                "curve_name": "p128",
                "plot_corr_len": corr,
                "corr_len": corr,
                "P_tr": fixed_p_tr,
                "correlated_query": False,
            }
        )
        if include_corrquery:
            specs.append(
                {
                    "curve_name": "corr_query",
                    "plot_corr_len": corr,
                    "corr_len": corr,
                    "P_tr": fixed_p_tr,
                    "correlated_query": True,
                }
            )
        specs.append(
            {
                "curve_name": "corr0",
                "plot_corr_len": corr,
                "corr_len": 0,
                "P_tr": adaptive_p_tr,
                "correlated_query": False,
            }
        )
    return specs


def reduced_cache_key(row):
    return (
        row["curve_name"],
        int(round(float(row["plot_corr_len"]))),
        int(round(float(row["corr_len"]))),
        int(round(float(row["P_tr"]))),
        parse_bool(row["correlated_query"]),
        int(round(float(row["B"]))),
        int(round(float(row["d"]))),
        int(round(float(row["K"]))),
        float(row["rho"]),
    )


def is_valid_reduced_cache_row(row):
    try:
        return (
            row.get("model_type") == "reduced"
            and int(row.get("num_runs", "0")) == REDUCED_NUM_RUNS
            and row.get("best_fresh_test_loss") not in (None, "")
            and row.get("best_fresh_test_loss_std") not in (None, "")
        )
    except ValueError:
        return False


def reduced_spec_key(spec, theory_params):
    return (
        spec["curve_name"],
        int(spec["plot_corr_len"]),
        int(spec["corr_len"]),
        int(spec["P_tr"]),
        bool(spec["correlated_query"]),
        int(round(theory_params["B"])),
        int(round(theory_params["d"])),
        int(round(theory_params["K"])),
        float(theory_params["rho"]),
    )


def append_reduced_cache_rows(reduced_csv, rows):
    reduced_csv = Path(reduced_csv)
    reduced_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not reduced_csv.exists()
    with open(reduced_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REDUCED_CACHE_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_reduced_cache_rows(reduced_csv, rows):
    reduced_csv = Path(reduced_csv)
    reduced_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(reduced_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REDUCED_CACHE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def reduced_cache_needs_upgrade(rows):
    if not rows:
        return False
    row_keys = set(rows[0].keys())
    return (
        "num_runs" not in row_keys
        or "best_fresh_test_loss_std" not in row_keys
    )


def load_reduced_panel_data(reduced_csv, x_values, theory_params, include_corrquery=False):
    cached_rows = load_rows_if_exists(reduced_csv)
    if reduced_cache_needs_upgrade(cached_rows):
        cached_rows = []
        write_reduced_cache_rows(reduced_csv, [])
    cached_by_key = {}
    for row in cached_rows:
        if not is_valid_reduced_cache_row(row):
            continue
        cached_by_key[reduced_cache_key(row)] = row

    specs = required_reduced_specs(x_values, theory_params, include_corrquery=include_corrquery)
    missing_specs = [spec for spec in specs if reduced_spec_key(spec, theory_params) not in cached_by_key]
    if missing_specs:
        try:
            from reduced import average_over_replicate
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Reduced-model curves require the dependencies needed by reduced.py. "
                "Install them in the active Python environment and rerun effective_compare.py."
            ) from exc

        new_rows = []
        for spec in missing_specs:
            mean, std = average_over_replicate(
                    int(round(theory_params["B"])),
                    int(round(theory_params["d"])),
                    int(spec["P_tr"]),
                    int(round(theory_params["K"])),
                    float(theory_params["rho"]),
                    int(spec["corr_len"]),
                    num_average=REDUCED_NUM_RUNS,
                    query=bool(spec["correlated_query"]),
                )
            row = {
                "model_type": "reduced",
                "curve_name": spec["curve_name"],
                "plot_corr_len": int(spec["plot_corr_len"]),
                "corr_len": int(spec["corr_len"]),
                "P_tr": int(spec["P_tr"]),
                "correlated_query": str(bool(spec["correlated_query"])),
                "B": int(round(theory_params["B"])),
                "d": int(round(theory_params["d"])),
                "K": int(round(theory_params["K"])),
                "rho": float(theory_params["rho"]),
                "num_runs": REDUCED_NUM_RUNS,
                "best_fresh_test_loss": float(mean),
                "best_fresh_test_loss_std": float(std),
                "fresh_test_losses_json": json.dumps([float(mean)]),
            }
            new_rows.append(row)
            cached_by_key[reduced_spec_key(spec, theory_params)] = row
        append_reduced_cache_rows(reduced_csv, new_rows)

    series = {}
    for curve_name in ["p128", "corr0"]:
        curve_specs = [spec for spec in specs if spec["curve_name"] == curve_name]
        curve_specs.sort(key=lambda spec: spec["plot_corr_len"])
        xs = [spec["plot_corr_len"] for spec in curve_specs]
        ys = [
            float(cached_by_key[reduced_spec_key(spec, theory_params)]["best_fresh_test_loss"])
            for spec in curve_specs
        ]
        stds = [
            float(cached_by_key[reduced_spec_key(spec, theory_params)].get("best_fresh_test_loss_std", 0.0))
            for spec in curve_specs
        ]
        series[curve_name] = (xs, ys, stds)
    if include_corrquery:
        curve_specs = [spec for spec in specs if spec["curve_name"] == "corr_query"]
        curve_specs.sort(key=lambda spec: spec["plot_corr_len"])
        xs = [spec["plot_corr_len"] for spec in curve_specs]
        ys = [
            float(cached_by_key[reduced_spec_key(spec, theory_params)]["best_fresh_test_loss"])
            for spec in curve_specs
        ]
        stds = [
            float(cached_by_key[reduced_spec_key(spec, theory_params)].get("best_fresh_test_loss_std", 0.0))
            for spec in curve_specs
        ]
        series["corr_query"] = (xs, ys, stds)
    else:
        series["corr_query"] = None

    if cached_rows == [] and cached_by_key:
        write_reduced_cache_rows(reduced_csv, cached_by_key.values())
    return series


def plot_effective_compare(
    p128_csv,
    corr0_csv,
    output,
    selected_model_types=None,
    selected_layer=1,
    show_theory=False,
    include_corrquery=False,
    show_reduced=False,
    reduced_csv="queryfree/reduced_effective_compare.csv",
):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to render the plot. Install it in the active Python environment "
            "and rerun effective_compare.py."
        ) from exc

    series_by_model = load_effective_compare_data(
        p128_csv,
        corr0_csv,
        selected_model_types,
        selected_layer,
        include_corrquery=include_corrquery,
    )
    output_path = Path(output)

    model_types = list(series_by_model.keys())
    panel_specs = []
    if show_reduced:
        if "4" not in series_by_model:
            raise ValueError("--reduced requires model_type=4 data to infer the reduced-model hyperparameters.")
        reduced_x_values = sorted(CORR_LEN_TO_PTR)
        reduced_series = load_reduced_panel_data(
            reduced_csv,
            reduced_x_values,
            series_by_model["4"]["theory_params"],
            include_corrquery=include_corrquery,
        )
        panel_specs.append(
            {
                "panel_key": "reduced",
                "title": MODEL_TITLES["reduced"],
                "series": reduced_series,
                "theory_params": series_by_model["4"]["theory_params"],
            }
        )
    for model_type in model_types:
        panel_specs.append(
            {
                "panel_key": model_type,
                "title": MODEL_TITLES.get(model_type, f"model_type={model_type}"),
                "series": {
                    "p128": series_by_model[model_type]["p128"],
                    "corr0": series_by_model[model_type]["corr0"],
                    "corr_query": series_by_model[model_type]["corr_query"],
                },
                "theory_params": series_by_model[model_type]["theory_params"],
            }
        )

    fig, axes = plt.subplots(
        1,
        len(panel_specs),
        figsize=(12, 3.5) if show_reduced else (8, 3.5),
        sharey=False,
    )
    if len(panel_specs) == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []
    for ax, panel in zip(axes, panel_specs):
        panel_key = panel["panel_key"]
        p128_x, p128_y, p128_std = panel["series"]["p128"]
        corr0_x, corr0_y, corr0_std = panel["series"]["corr0"]
        corr_query = panel["series"]["corr_query"]
        all_x_values = set(p128_x) | set(corr0_x)
        if corr_query is not None:
            all_x_values.update(corr_query[0])
        all_x_values = sorted(all_x_values)

        ax.fill_between(
            p128_x,
            [y - s for y, s in zip(p128_y, p128_std)],
            [y + s for y, s in zip(p128_y, p128_std)],
            color=CURVE_STYLES["p128"]["color"],
            alpha=EMPIRICAL_FILL_ALPHA,
            linewidth=0,
            zorder=1,
        )
        p128_line, = ax.plot(
            p128_x,
            p128_y,
            marker='o',
            ms=5,
            linewidth=1.00,
            color=CURVE_STYLES["p128"]["color"],
            linestyle="-",
            label=CURVE_STYLES["p128"]["label"],
            zorder=3,
        )
        ax.fill_between(
            corr0_x,
            [y - s for y, s in zip(corr0_y, corr0_std)],
            [y + s for y, s in zip(corr0_y, corr0_std)],
            color=CURVE_STYLES["corr0"]["color"],
            alpha=EMPIRICAL_FILL_ALPHA,
            linewidth=0,
            zorder=1,
        )
        corr0_line, = ax.plot(
            corr0_x,
            corr0_y,
            marker='d',
            ms=5,
            linewidth=1.00,
            color=CURVE_STYLES["corr0"]["color"],
            linestyle="-",
            label=CURVE_STYLES["corr0"]["label"],
            zorder=3,
        )
        corr_query_line = None
        if corr_query is not None:
            corr_query_x, corr_query_y, corr_query_std = corr_query
            ax.fill_between(
                corr_query_x,
                [y - s for y, s in zip(corr_query_y, corr_query_std)],
                [y + s for y, s in zip(corr_query_y, corr_query_std)],
                color=CURVE_STYLES["corr_query"]["color"],
                alpha=EMPIRICAL_FILL_ALPHA,
                linewidth=0,
                zorder=1,
            )
            corr_query_line, = ax.plot(
                corr_query_x,
                corr_query_y,
                marker='o',
                ms=5,
                linewidth=1.00,
                color=CURVE_STYLES["corr_query"]["color"],
                linestyle="-",
                label=CURVE_STYLES["corr_query"]["label"],
                zorder=3,
            )
        if not legend_handles:
            legend_handles = [p128_line, corr0_line]
            legend_labels = [CURVE_STYLES["p128"]["label"], CURVE_STYLES["corr0"]["label"]]
            if corr_query_line is not None:
                legend_handles.append(corr_query_line)
                legend_labels.append(CURVE_STYLES["corr_query"]["label"])

        theory_on_this_panel = show_theory and (
            (show_reduced and panel_key == "reduced") #or (panel_key == "4")
        )
        if theory_on_this_panel:
            theory_curves = compute_theory_curves(all_x_values, panel["theory_params"])
            for theory_name in ["no_query", "effective", "full"]:
                theory_x, theory_y = theory_curves[theory_name]
                style_name = THEORY_CURVE_STYLE_MAP[theory_name]
                ax.plot(
                    theory_x,
                    theory_y,
                    linewidth=1.5,
                    color=CURVE_STYLES[style_name]["color"],
                    alpha=THEORY_ALPHA,
                    linestyle="--",
                    label="theory",
                    zorder=1,
                )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=THEORY_COLOR,
                    linestyle="--",
                    linewidth=1.5,
                )
            )
            legend_labels.append("theory")

        ax.set_title(panel["title"], fontsize=10, fontweight="bold")
        ax.set_xlabel(r"Correlation length", fontsize=10)
        ax.set_xticks(all_x_values)
        ax.tick_params(axis="both", labelsize=8, color=SPINE_COLOR, labelcolor=SPINE_COLOR)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_color(SPINE_COLOR)
            spine.set_linewidth(1.0)

    axes[0].set_ylabel("ICL Test Loss", fontsize=10)
    handle_by_label = {label: handle for handle, label in zip(legend_handles, legend_labels)}
    ordered_labels = [label for label in LEGEND_ORDER if label in handle_by_label]
    ordered_handles = [handle_by_label[label] for label in ordered_labels]
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        ncol=len(ordered_labels),
        frameon=False,
        fontsize=12,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def main():
    args = parse_args()
    output_path = plot_effective_compare(
        args.p128_csv,
        args.corr0_csv,
        args.output,
        selected_model_types=args.model_type,
        selected_layer=args.layer,
        show_theory=args.theory,
        include_corrquery=args.corrquery,
        show_reduced=args.reduced,
        reduced_csv=args.reduced_csv,
    )
    print(output_path)


if __name__ == "__main__":
    main()
