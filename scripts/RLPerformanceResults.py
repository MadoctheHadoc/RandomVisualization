import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"

# Data organized by grid -> algorithm -> parameter category -> {param_value: score}
data = {
    "A1": {
        "Value Iteration": {
            "Default": {"-": 1.0000},
            "Sigma": {"0.0": 1.0000, "0.2": 1.0000, "0.3": 1.0000},
            "Gamma": {
                "0.7": 1.0000,
                "0.8": 1.0000,
                "0.95": 1.0000,
                "0.99": 1.0000,
                "0.999": 1.0000,
                "0.99999": 1.0000,
            },
            "Epsilon": {"0.0": 1.0000, "0.2": 1.0000, "0.5": 1.0000},
            "LR": {"0.01": 1.0000, "0.1": 1.0000, "0.3": 1.0000, "0.5": 1.0000, "0.7": 1.0000},
            "Step Ratio": {"0.5": 1.0000, "1": 1.0000, "4": 1.0000},
        },
        "SARSA": {
            "Default": {"-": 1.1714},
            "Sigma": {"0.0": 1.0000, "0.2": 1.3571, "0.3": 1.2143},
            "Gamma": {
                "0.7": 1.1429,
                "0.8": 1.0714,
                "0.95": 1.0714,
                "0.99": 1.0000,
                "0.999": None,
                "0.99999": None,
            },
            "Epsilon": {"0.0": 1.2143, "0.2": 1.0000, "0.5": None},
            "LR": {"0.01": None, "0.1": None, "0.3": 1.0000, "0.5": None, "0.7": None},
            "Step Ratio": {"0.5": None, "1": 1.2857, "4": 1.1429},
        },
        "Monte Carlo On-Policy": {
            "Default": {"-": 2.3714},
            "Sigma": {"0.0": 1.3643, "0.2": 6.5982, "0.3": 6.7964},
            "Gamma": {
                "0.7": 1.6643,
                "0.8": 1.6607,
                "0.95": 1.8357,
                "0.99": None,
                "0.999": None,
                "0.99999": None,
            },
            "Epsilon": {"0.0": None, "0.2": 1.4750, "0.5": 2.3321},
            "LR": {"0.01": None, "0.1": None, "0.3": None, "0.5": None, "0.7": None},
            "Step Ratio": {"0.5": None, "1": 3.7268, "4": 2.4464},
        },
    },
    "Large Grid": {
        "Value Iteration": {
            "Default": {"-": 1.0000},
            "Sigma": {"0.0": 1.0000, "0.2": 1.0000, "0.3": 1.0000},
            "Gamma": {
                "0.7": 1.0000,
                "0.8": 1.0000,
                "0.95": 1.0000,
                "0.99": 1.0000,
                "0.999": 1.0000,
                "0.99999": 1.0000,
            },
            "Epsilon": {"0.0": 1.0000, "0.2": 1.0000, "0.5": 1.0000},
            "LR": {"0.01": 1.0000, "0.1": 1.0000, "0.3": 1.0000, "0.5": 1.0000, "0.7": 1.0000},
            "Step Ratio": {"0.5": 1.0000, "1": 1.0000, "4": 1.0000},
        },
        "SARSA": {
            "Default": {"-": 1.0952},
            "Sigma": {"0.0": 1.1905, "0.2": 1.0000, "0.3": 1.1905},
            "Gamma": {
                "0.7": 1.0000,
                "0.8": 1.0000,
                "0.95": 1.0000,
                "0.99": 1.0000,
                "0.999": 1.0000,
                "0.99999": 1.1905,
            },
            "Epsilon": {"0.0": 1.1905, "0.2": 1.0000, "0.5": 1.0000},
            "LR": {"0.01": None, "0.1": 1.0000, "0.3": 1.0000, "0.5": 1.0000, "0.7": None},
            "Step Ratio": {"0.5": 1.1905, "1": 1.1905, "4": 1.0000},
        },
        "Monte Carlo On-Policy": {
            "Default": {"-": None},
            "Sigma": {"0.0": None, "0.2": None, "0.3": 5.6405},
            "Gamma": {
                "0.7": None,
                "0.8": 3.4143,
                "0.95": 2.2643,
                "0.99": 7.0381,
                "0.999": None,
                "0.99999": None,
            },
            "Epsilon": {"0.0": None, "0.2": 3.0595, "0.5": 2.6524},
            "LR": {"0.01": None, "0.1": None, "0.3": None, "0.5": None, "0.7": None},
            "Step Ratio": {"0.5": None, "1": None, "4": None},
        },
    },
    "Super Hard": {
        "Value Iteration": {
            "Default": {"-": 1.0000},
            "Sigma": {"0.0": 1.0000, "0.2": 1.0000, "0.3": 1.0000},
            "Gamma": {
                "0.7": None,
                "0.8": None,
                "0.95": 1.0000,
                "0.99": 1.0000,
                "0.999": 1.0000,
                "0.99999": 1.0000,
            },
            "Epsilon": {"0.0": 1.0000, "0.2": 1.0000, "0.5": 1.0000},
            "LR": {"0.01": 1.0000, "0.1": 1.0000, "0.3": 1.0000, "0.5": 1.0000, "0.7": 1.0000},
            "Step Ratio": {"0.5": 1.0000, "1": 1.0000, "4": 1.0000},
        },
        "SARSA": {
            "Default": {"-": 1.3345},
            "Sigma": {"0.0": 1.3103, "0.2": 1.2759, "0.3": 1.2069},
            "Gamma": {
                "0.7": 1.4828,
                "0.8": 1.4828,
                "0.95": 1.0000,
                "0.99": None,
                "0.999": None,
                "0.99999": None,
            },
            "Epsilon": {"0.0": 1.7586, "0.2": 1.3103, "0.5": None},
            "LR": {"0.01": None, "0.1": 1.1034, "0.3": 1.2069, "0.5": None, "0.7": None},
            "Step Ratio": {"0.5": None, "1": 1.3448, "4": 1.2069},
        },
        "Monte Carlo On-Policy": {
            "Default": {"-": None},
            "Sigma": {"0.0": None, "0.2": None, "0.3": None},
            "Gamma": {
                "0.7": None,
                "0.8": None,
                "0.95": 2.9802,
                "0.99": None,
                "0.999": None,
                "0.99999": None,
            },
            "Epsilon": {"0.0": None, "0.2": 2.2034, "0.5": 3.7241},
            "LR": {"0.01": None, "0.1": None, "0.3": None, "0.5": None, "0.7": None},
            "Step Ratio": {"0.5": None, "1": None, "4": None},
        },
    },
}
# Default values for vertical reference lines
default_params = {
    "Sigma": "0.1",
    "Gamma": "0.9",
    "Epsilon": "0.1",
    "LR": "0.2",
    "Step Ratio": "2",
}


# Add default parameter values to the data if they are missing
for grid in data:
    for algo in data[grid]:
        for category in default_params:
            default_param_value = default_params[category]
            if default_param_value not in data[grid][algo][category]:
                # Copy the default performance score
                default_score = data[grid][algo]["Default"]["-"]
                data[grid][algo][category][default_param_value] = default_score

# Now, sort the parameter values for consistent plotting
for grid in data:
    for algo in data[grid]:
        for category in data[grid][algo]:
            if category != "Default":
                # Sort the keys numerically (or lexicographically if needed)
                sorted_keys = sorted(
                    data[grid][algo][category].keys(),
                    key=lambda x: float(x) if x.replace(".", "").isdigit() else x,
                )
                # Reorder the dictionary
                data[grid][algo][category] = {
                    k: data[grid][algo][category][k] for k in sorted_keys
                }

categories = ["Sigma", "Gamma", "Epsilon", "LR", "Step Ratio"]
grids = ["A1", "Large Grid", "Super Hard"]
algos = ["Value Iteration", "SARSA", "Monte Carlo On-Policy"]

colors = {
    "Value Iteration": "#AA2222",
    "SARSA": "#56A520",
    "Monte Carlo On-Policy": "#2B46CE",
}

markers = {
    "Value Iteration": "s",
    "SARSA": "^",
    "Monte Carlo On-Policy": "o",
}


# Actual Rendering code:

n_rows = len(grids)
n_cols = len(categories)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 4 * n_rows))

# Calculate y-axis limits for each row (grid)
y_limits = {}
for row, grid in enumerate(grids):
    all_y_values = []
    for col, category in enumerate(categories):
        for algo in algos:
            param_dict = data[grid][algo][category]
            for val in param_dict.values():
                if val is not None:
                    all_y_values.append(float(val))
    if all_y_values:
        _min = min(all_y_values)
        _max = max(all_y_values)
        _range = (_max - _min) * 0.1  # 10% padding
        y_min = _min - _range
        y_max = _max + _range
        y_limits[grid] = (y_min, y_max)
    else:
        y_limits[grid] = (0, 1)  # Fallback

# Plot the data and set y-axis limits for each row
for row, grid in enumerate(grids):
    for col, category in enumerate(categories):
        ax = axes[row, col]

        for algo in algos:
            param_dict = data[grid][algo][category]
            x_labels = list(param_dict.keys())
            y_values = [np.nan if val is None else float(val) for val in param_dict.values()]
            x_positions = range(len(x_labels))

            ax.plot(
                x_positions,
                y_values,
                marker=markers[algo],
                linestyle="-",
                color=colors[algo],
                linewidth=2,
                markersize=7,
                label=algo
            )

        # Set x-axis ticks to show all categorical labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=40, ha="right")

        # Add vertical line for the default value
        if category in default_params:
            default_x = default_params[category]
            if default_x in x_labels:
                default_idx = x_labels.index(default_x)
                ax.axvline(
                    default_idx,
                    color="gray",
                    linestyle="--",
                    linewidth=1.5,
                )

        # Keep grid lines for all subplots
        ax.grid(True, linestyle="--", alpha=1, which="both")
        
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)

        # Set y-axis limits for this row
        ax.set_ylim(y_limits[grid])

        # Enable grid lines for both x and y axes
        ax.grid(True, which="both", linestyle="--", alpha=1)

        # Disable y-axis for all but the leftmost column
        if col != 0:
            ax.yaxis.set_tick_params(labelleft=False, left=False)
        else:
            ax.tick_params(axis="y", labelsize=15)

        # Disable x-axis for all but the bottom row
        if row != n_rows - 1:
            ax.xaxis.set_tick_params(labelbottom=False, bottom=False)
        else:
            ax.tick_params(axis="x", labelsize=15)

        # Column titles on top row only
        if row == 0:
            ax.set_title(category, fontsize=20, pad=10)

        # X-axis labels on bottom row only
        if row == n_rows - 1:
            ax.set_xlabel("Value", fontsize=18)

        # Row labels on leftmost column
        if col == 0:
            ax.set_ylabel(f"{grid}\nPerformance", fontsize=18)

# Add legend once
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    fontsize=18,
    bbox_to_anchor=(0.5, 0.96),
    frameon=False,
)

fig.suptitle(
    "Algorithm Performance by Grid and Parameter",
    fontsize=24,
    y=1.00,
)

plt.subplots_adjust(
    wspace=-1,  # Very small horizontal spacing
    hspace=0,   # Very small vertical spacings
)


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("visualizations/RLPerformanceResults.png", dpi=300, bbox_inches="tight")
