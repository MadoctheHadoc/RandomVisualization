import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"

# Data organized by grid -> algorithm -> parameter category -> {param_value: score}
data = {
    "A1": {
        "Value Iteration": {
            "Default": {"-": 1.0921},
            "Sigma": {"0.0": 1.0000, "0.2": 1.1929, "0.3": 1.3857},
            "Gamma": {
                "0.7": 1.1107,
                "0.8": 1.1107,
                "0.9": 1.1107,
                "0.99": 1.1107,
                "0.999": 1.1107,
                "0.99999": 1.1107,
            },
            "Epsilon": {"0.0": None, "0.2": None, "0.5": None},
            "LR": {"0.01": None, "0.1": None, "0.3": None, "0.5": None, "0.7": None},
            "Step Ratio": {"0.5": None, "1": None, "4": None},
        },
        "SARSA": {
            "Default": {"-": 1.0986},
            "Sigma": {"0.0": 1.0000, "0.2": 1.2214, "0.3": 1.4500},
            "Gamma": {
                "0.7": 1.1179,
                "0.8": 1.1179,
                "0.9": 1.1143,
                "0.99": 1.2143,
                "0.999": 1.1357,
                "0.99999": 1.1143,
            },
            "Epsilon": {"0.0": 1.2464, "0.2": 1.1071, "0.5": 1.1143},
            "LR": {"0.01": None, "0.1": 1.2071, "0.3": 1.1321, "0.5": 1.1071, "0.7": 1.3607},
            "Step Ratio": {"0.5": None, "1": 1.1107, "4": 1.1143},
        },
        "Monte Carlo On-Policy": {
            "Default": {"-": 1.6032},
            "Sigma": {"0.0": None, "0.2": 2.2179, "0.3": 2.0607},
            "Gamma": {
                "0.7": None,
                "0.8": 1.5786,
                "0.9": 1.3000,
                "0.99": 2.0393,
                "0.999": 2.3036,
                "0.99999": 3.5304,
            },
            "Epsilon": {"0.0": None, "0.2": 2.2179, "0.5": 2.9821},
            "LR": {"0.01": None, "0.1": None, "0.3": None, "0.5": None, "0.7": None},
            "Step Ratio": {"0.5": None, "1": 1.9250, "4": 1.3929},
        },
    },
    "Large Grid": {
        "Value Iteration": {
            "Default": {"-": 1.0590},
            "Sigma": {"0.0": 1.0000, "0.2": 1.1190, "0.3": 1.2048},
            "Gamma": {
                "0.7": 1.0619,
                "0.8": 1.0619,
                "0.9": 1.0619,
                "0.99": 1.0619,
                "0.999": 1.0619,
                "0.99999": 1.0619,
            },
            "Epsilon": {"0.0": None, "0.2": None, "0.5": None},
            "LR": {"0.01": None, "0.1": None, "0.3": None, "0.5": None, "0.7": None},
            "Step Ratio": {"0.5": None, "1": None, "4": None},
        },
        "SARSA": {
            "Default": {"-": 1.0738},
            "Sigma": {"0.0": 1.0000, "0.2": 1.1238, "0.3": 1.2000},
            "Gamma": {
                "0.7": 1.0619,
                "0.8": 1.0619,
                "0.9": 1.0619,
                "0.99": 1.1381,
                "0.999": 1.0619,
                "0.99999": 1.0619,
            },
            "Epsilon": {"0.0": 4.8286, "0.2": 1.0619, "0.5": 1.0619},
            "LR": {"0.01": 8.4262, "0.1": 1.0619, "0.3": 1.0619, "0.5": 1.2476, "0.7": 1.2429},
            "Step Ratio": {"0.5": 1.0762, "1": 1.1952, "4": 1.0619},
        },
        "Monte Carlo On-Policy": {
            "Default": {"-": 1.3776},
            "Sigma": {"0.0": 1.0524, "0.2": 1.5476, "0.3": 1.5000},
            "Gamma": {
                "0.7": 1.1238,
                "0.8": 1.1333,
                "0.9": 1.2619,
                "0.99": 1.8857,
                "0.999": 1.4095,
                "0.99999": 1.7238,
            },
            "Epsilon": {"0.0": None, "0.2": 1.2714, "0.5": 1.7143},
            "LR": {"0.01": None, "0.1": None, "0.3": None, "0.5": None, "0.7": None},
            "Step Ratio": {"0.5": 1.1238, "1": 1.1619, "4": 1.3238},
        },
    },
    "Super Hard": {
        "Value Iteration": {
            "Default": {"-": 1.1179},
            "Sigma": {"0.0": 1.0000, "0.2": 1.2414, "0.3": 1.4397},
            "Gamma": {
                "0.7": None,
                "0.8": None,
                "0.9": 1.1293,
                "0.99": 1.1293,
                "0.999": 1.1293,
                "0.99999": 1.1293,
            },
            "Epsilon": {"0.0": None, "0.2": None, "0.5": None},
            "LR": {"0.01": None, "0.1": None, "0.3": None, "0.5": None, "0.7": None},
            "Step Ratio": {"0.5": None, "1": None, "4": None},
        },
        "SARSA": {
            "Default": {"-": 1.3458},
            "Sigma": {"0.0": 1.0345, "0.2": 1.5448, "0.3": 2.3776},
            "Gamma": {
                "0.7": 1.4983,
                "0.8": 1.1345,
                "0.9": 1.1741,
                "0.99": 1.1172,
                "0.999": 1.1828,
                "0.99999": 1.1690,
            },
            "Epsilon": {"0.0": 1.1724, "0.2": 1.1483, "0.5": 1.1276},
            "LR": {"0.01": 13.9069, "0.1": 1.6793, "0.3": 1.1586, "0.5": 1.1672, "0.7": 1.5534},
            "Step Ratio": {"0.5": 1.2638, "1": 1.1603, "4": 1.1293},
        },
        "Monte Carlo On-Policy": {
            "Default": {"-": None},
            "Sigma": {"0.0": None, "0.2": None, "0.3": 2.5569},
            "Gamma": {
                "0.7": None,
                "0.8": None,
                "0.9": None,
                "0.99": None,
                "0.999": None,
                "0.99999": None,
            },
            "Epsilon": {"0.0": None, "0.2": None, "0.5": 2.6224},
            "LR": {"0.01": None, "0.1": None, "0.3": None, "0.5": None, "0.7": None},
            "Step Ratio": {"0.5": None, "1": None, "4": None},
        },
    },
}

# Default values for vertical reference lines
default_params = {
    "Sigma": "0.1",
    "Gamma": "0.95",
    "Epsilon": "0.1",
    "LR": "0.2",
    "Step Ratio": "2",
}


# Add default parameter values to the data if they are missing
for grid in data:
    for algo in data[grid]:
        for category in default_params:
            default_param_value = default_params[category]
            param_dict = data[grid][algo][category]

            # Check if all values (excluding default) are None
            all_none = all(
                val is None
                for key, val in param_dict.items()
                if key != default_param_value
            )

            # Only add default if not all other values are None
            if default_param_value not in param_dict and not all_none:
                default_score = data[grid][algo]["Default"]["-"]
                param_dict[default_param_value] = default_score

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
        _range = (_max - _min) * 0.1
        y_min = _min - _range
        y_max = _max + _range
        y_limits[grid] = (y_min, y_max)
    else:
        y_limits[grid] = (0, 1)

# Find the number of x-ticks for each category
category_x_limits = {}
for category in categories:
    max_ticks = 0
    for grid in grids:
        for algo in algos:
            param_dict = data[grid][algo][category]
            num_ticks = len(param_dict)
            if num_ticks > max_ticks:
                max_ticks = num_ticks
    category_x_limits[category] = (-0.5, max_ticks - 0.5)

# For each category, collect all unique x-axis labels
category_x_labels = {}
for category in categories:
    all_labels = set()
    for grid in grids:
        for algo in algos:
            all_labels.update(data[grid][algo][category].keys())
    category_x_labels[category] = sorted(
        all_labels,
        key=lambda x: float(x) if x.replace(".", "").isdigit() else x,
    )

# Plot the data and set y-axis limits for each row
for row, grid in enumerate(grids):
    for col, category in enumerate(categories):
        ax = axes[row, col]
        x_labels = category_x_labels[category]  # Use the global labels for this category
        x_positions = range(len(x_labels))

        for algo in algos:
            param_dict = data[grid][algo][category]
            y_values = [param_dict.get(label, np.nan) for label in x_labels]
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

        # Set x-axis ticks and labels for this category
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
        
        ax.set_xlim(category_x_limits[category])

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
