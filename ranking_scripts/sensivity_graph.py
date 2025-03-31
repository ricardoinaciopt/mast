import os
import sys
import json
import itertools
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def extract_q_number(q):
    return int(q.split("_")[0][1:]) if "_" in q else None


def plot_ranks(rankings, filename, plot_trend=False):
    CATEGORIES = {
        "Oversampling Features": ["ADASYN", "SMOTE", "BorderlineSMOTE", "SVMSMOTE"],
        "Undersampling Features": ["RandomUnderSampler"],
        "Generating Time Series": [
            "DBA",
            "Jittering",
            "Scaling",
            "MagnitudeWarping",
            "TimeWarping",
            "TSMixup",
        ],
        "No Resampling": ["No Resampling"],
    }

    COLOR_MAP = {
        "Oversampling Features": "#4daf4a",
        "Undersampling Features": "#e41a1c",
        "Generating Time Series": "#ff7f00",
        "No Resampling": "#999999",
    }

    MARKERS = ["o", "s", "D", "^", "H", "<", ">", "p", "*", "v", "X", "h", "x", "d"]

    method_data = defaultdict(list)
    category_data = defaultdict(lambda: defaultdict(list))
    x_values = []

    sorted_q = sorted(
        ((q, extract_q_number(q)) for q in rankings.keys()), key=lambda x: x[1]
    )

    for q_dir, q_num in sorted_q:
        if q_num is None:
            continue
        x_values.append(q_num)
        for method, rank in rankings[q_dir].items():
            method_data[method].append((q_num, rank))
            category = next(
                (cat for cat, methods in CATEGORIES.items() if method in methods),
                None,
            )
            if category in ["Oversampling Features", "Generating Time Series"]:
                category_data[category][q_num].append(rank)

    plt.figure(figsize=(16, 8))
    ax = plt.gca()

    all_methods = [method for methods in CATEGORIES.values() for method in methods]
    method_markers = {method: marker for method, marker in zip(all_methods, MARKERS)}

    marker_cycle = itertools.cycle(MARKERS)
    method_markers = defaultdict(lambda: next(marker_cycle), method_markers)

    for method in sorted(method_data.keys(), key=lambda x: x.lower()):
        points = method_data[method]
        x = [p[0] for p in points]
        y = [p[1] for p in points]

        category = next(
            (cat for cat, methods in CATEGORIES.items() if method in methods), "Other"
        )
        color = COLOR_MAP.get(category, "#999999")

        ax.plot(
            x,
            y,
            marker=method_markers[method],
            color=color,
            markersize=10,
            linewidth=2,
            alpha=0.9,
            label=method,
        )

    trend_legend_elements = []
    if plot_trend:
        linestyle_cycle = itertools.cycle(["--", ":"])
        for category, color in [
            ("Oversampling Features", "#4daf4a"),
            ("Generating Time Series", "#ff7f00"),
        ]:
            if category in category_data:
                x_trend = sorted(category_data[category].keys())
                y_trend = [np.mean(category_data[category][x]) for x in x_trend]

                if len(x_trend) > 1:
                    coef = np.polyfit(x_trend, y_trend, 1)
                    poly_eq = np.poly1d(coef)
                    y_fit = poly_eq(x_trend)

                    trend_linestyle = next(linestyle_cycle)
                    ax.plot(
                        x_trend,
                        y_fit,
                        color=color,
                        linestyle=trend_linestyle,
                        linewidth=3,
                        label=f"{category} Trend",
                    )

                    trend_legend_elements.append(
                        plt.Line2D(
                            [0],
                            [0],
                            linestyle=trend_linestyle,
                            color=color,
                            lw=3,
                            label=f"{category} Trend",
                        )
                    )

    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=4, label=f"{cat}")
        for cat, color in COLOR_MAP.items()
    ]

    for method in sorted(method_data.keys(), key=lambda x: x.lower()):
        category = next(
            (cat for cat, methods in CATEGORIES.items() if method in methods), "Other"
        )
        color = COLOR_MAP.get(category, "#999999")
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker=method_markers[method],
                color="w",
                label=method,
                markerfacecolor=color,
                markersize=12,
                linestyle="None",
            )
        )

    plt.xlabel("Percentile Threshold", fontsize=16)
    plt.ylabel("Average Rank", fontsize=16)
    plt.xticks(sorted(x_values), fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, alpha=0.2)

    legend1 = ax.legend(
        handles=legend_elements[: len(COLOR_MAP)],
        loc="upper left",
        bbox_to_anchor=(0, 1.3),
        title="Categories",
        title_fontsize=14,
        fontsize=14,
    )
    legend2 = ax.legend(
        handles=legend_elements[len(COLOR_MAP) :] + trend_legend_elements,
        loc="upper right",
        bbox_to_anchor=(1, 1.3),
        title="Methods & Trends",
        title_fontsize=14,
        fontsize=12,
        ncol=3,
    )
    ax.add_artist(legend1)

    folder_suffix = filename[-7:-5]
    save_dir = os.path.join("figs_ranks", folder_suffix)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"sensitivity_plot_{folder_suffix}.pdf")
    plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <json_file> [--trend]")
        sys.exit(1)

    json_file = sys.argv[1]
    plot_trend = "--trend" in sys.argv

    try:
        with open(json_file, "r") as f:
            rankings = json.load(f)
        plot_ranks(rankings, json_file, plot_trend)
        print(f"Plot saved successfully in figs_ranks/{json_file[-7:-5]}/")
    except Exception as e:
        print(f"Error: {e}")
