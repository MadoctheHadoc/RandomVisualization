import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"

# Data organized by grid -> algorithm -> parameter category -> {param_value: score}
data = {
    "A1": {
        "Value Iteration": {
            "Default": {"-": 18.0},
            "Sigma": {"0.0": 18.0, "0.2": 18.0, "0.3": 18.0},
            "Gamma": {
                "0.7": 18.0,
                "0.8": 19.0,
                "0.9": 18.0,
                "0.99": 17.0,
                "0.999": 17.0,
                "0.99999": 16.0,
            },
            "Epsilon": {"0.0": None, "0.2": None, "0.5": None},
            "LR": {"0.01": None, "0.1": None, "0.3": None, "0.5": None, "0.7": None},
            "Step Ratio": {"0.5": None, "1": None, "4": None},
        },
        "SARSA": {
            "Default": {"-": 178.6},
            "Sigma": {"0.0": 156.0, "0.2": 221.0, "0.3": 175.0},
            "Gamma": {
                "0.7": 218.0,
                "0.8": 184.0,
                "0.9": 176.0,
                "0.99": 168.0,
                "0.999": 173.0,
                "0.99999": 183.0,
            },
            "Epsilon": {"0.0": 757.0, "0.2": 188.0, "0.5": 213.0},
            "LR": {"0.01": None, "0.1": 205.0, "0.3": 166.0, "0.5": 143.0, "0.7": 355.0},
            "Step Ratio": {"0.5": None, "1": 186.0, "4": 157.0},
        },
        "Monte Carlo On-Policy": {
            "Default": {"-": 2190.7},
            "Sigma": {"0.0": None, "0.2": 2354.0, "0.3": 1848.0},
            "Gamma": {
                "0.7": None,
                "0.8": 4254.0,
                "0.9": 601.0,
                "0.99": 693.0,
                "0.999": 4962.0,
                "0.99999": 10000.0,
            },
            "Epsilon": {"0.0": None, "0.2": 1358.0, "0.5": 219.0},
            "LR": {"0.01": None, "0.1": None, "0.3": None, "0.5": None, "0.7": None},
            "Step Ratio": {"0.5": None, "1": 429.0, "4": 213.0},
        },
    },
    "Large Grid": {
        "Value Iteration": {
            "Default": {"-": 18.0},
            "Sigma": {"0.0": 18.0, "0.2": 18.0, "0.3": 18.0},
            "Gamma": {
                "0.7": 18.0,
                "0.8": 18.0,
                "0.9": 18.0,
                "0.99": 18.0,
                "0.999": 18.0,
                "0.99999": 18.0,
            },
            "Epsilon": {"0.0": None, "0.2": None, "0.5": None},
            "LR": {"0.01": None, "0.1": None, "0.3": None, "0.5": None, "0.7": None},
            "Step Ratio": {"0.5": None, "1": None, "4": None},
        },
        "SARSA": {
            "Default": {"-": 145.4},
            "Sigma": {"0.0": 132.0, "0.2": 136.0, "0.3": 136.0},
            "Gamma": {
                "0.7": 150.0,
                "0.8": 148.0,
                "0.9": 256.0,
                "0.99": 137.0,
                "0.999": 144.0,
                "0.99999": 137.0,
            },
            "Epsilon": {"0.0": 10000.0, "0.2": 150.0, "0.5": 217.0},
            "LR": {"0.01": 692.0, "0.1": 177.0, "0.3": 136.0, "0.5": 222.0, "0.7": 221.0},
            "Step Ratio": {"0.5": 149.0, "1": 148.0, "4": 140.0},
        },
        "Monte Carlo On-Policy": {
            "Default": {"-": 507.1},
            "Sigma": {"0.0": 203.0, "0.2": 246.0, "0.3": 274.0},
            "Gamma": {
                "0.7": 860.0,
                "0.8": 880.0,
                "0.9": 189.0,
                "0.99": 133.0,
                "0.999": 239.0,
                "0.99999": 10000.0,
            },
            "Epsilon": {"0.0": None, "0.2": 162.0, "0.5": 175.0},
            "LR": {"0.01": 207.0, "0.1": 406.0, "0.3": 271.0, "0.5": 125.0, "0.7": 175.0},
            "Step Ratio": {"0.5": 341.0, "1": 121.0, "4": 314.0},
        },
    },
    "Super Hard": {
        "Value Iteration": {
            "Default": {"-": 49.0},
            "Sigma": {"0.0": 49.0, "0.2": 49.0, "0.3": 49.0},
            "Gamma": {
                "0.7": None,
                "0.8": None,
                "0.9": 49.0,
                "0.99": 47.0,
                "0.999": 47.0,
                "0.99999": 46.0,
            },
            "Epsilon": {"0.0": None, "0.2": None, "0.5": None},
            "LR": {"0.01": None, "0.1": None, "0.3": None, "0.5": None, "0.7": None},
            "Step Ratio": {"0.5": None, "1": None, "4": None},
        },
        "SARSA": {
            "Default": {"-": 444.4},
            "Sigma": {"0.0": 409.0, "0.2": 446.0, "0.3": 513.0},
            "Gamma": {
                "0.7": 608.0,
                "0.8": 542.0,
                "0.9": 459.0,
                "0.99": 385.0,
                "0.999": 458.0,
                "0.99999": 442.0,
            },
            "Epsilon": {"0.0": 443.0, "0.2": 496.0, "0.5": 912.0},
            "LR": {"0.01": 5385.0, "0.1": 711.0, "0.3": 441.0, "0.5": 359.0, "0.7": 852.0},
            "Step Ratio": {"0.5": 597.0, "1": 471.0, "4": 398.0},
        },
        "Monte Carlo On-Policy": {
            "Default": {"-": None},
            "Sigma": {"0.0": None, "0.2": None, "0.3": 2203.0},
            "Gamma": {
                "0.7": None,
                "0.8": None,
                "0.9": None,
                "0.99": None,
                "0.999": None,
                "0.99999": None,
            },
            "Epsilon": {"0.0": None, "0.2": None, "0.5": 1237.0},
            "LR": {"0.01": None, "0.1": None, "0.3": None, "0.5": None, "0.7": 4324.0},
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
            ax.set_ylabel(f"{grid}\nEpisodes", fontsize=18)

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
    "Algorithm Episodes until Convergence by Grid and Parameter",
    fontsize=24,
    y=1.00,
)

plt.subplots_adjust(
    wspace=-1,  # Very small horizontal spacing
    hspace=0,   # Very small vertical spacings
)


plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("visualizations/RLConvergenceResults.png", dpi=300, bbox_inches="tight")
