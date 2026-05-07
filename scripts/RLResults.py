import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "serif"

# Data organized by grid -> algorithm -> parameter category -> {param_value: score}
data = {
    "A1": {
        "ValIt": {
            "Default": {"-": 0.950},
            "Sigma": {"1": 0.920, "2": 0.880, "3": 0.810},
            "Gamma": {"0.5": 0.900, "0.7": 0.850, "0.9": 0.920, "0.99": 0.950, "0.9999": 0.940},
            "Epsilon": {"1": 0.930, "2": 0.870},
            "LR": {"0.1": 0.900, "0.01": 0.940},
        },
        "SARSA": {
            "Default": {"-": 0.910},
            "Sigma": {"1": 0.880, "2": 0.840, "3": 0.770},
            "Gamma": {"0.5": 0.600, "0.7": 0.800, "0.9": 0.650, "0.99": 0.910, "0.9999": 0.900},
            "Epsilon": {"1": 0.890, "2": 0.830},
            "LR": {"0.1": 0.860, "0.01": 0.900},
        },
        "MC": {
            "Default": {"-": 0.870},
            "Sigma": {"1": 0.850, "2": 0.800, "3": 0.730},
            "Gamma": {"0.5": 0.650, "0.7": 0.760, "0.9": 0.840, "0.99": 0.870, "0.9999": 0.860},
            "Epsilon": {"1": 0.850, "2": 0.790},
            "LR": {"0.1": 0.820, "0.01": 0.860},
        },
    },
    "Large Grid": {
        "ValIt": {
            "Default": {"-": 0.890},
            "Sigma": {"1": 0.860, "2": 0.810, "3": 0.740},
            "Gamma": {"0.5": 0.900, "0.7": 0.850, "0.9": 0.920, "0.99": 0.950, "0.9999": 0.940},            "Epsilon": {"1": 0.870, "2": 0.800},
            "LR": {"0.1": 0.840, "0.01": 0.880},
        },
        "SARSA": {
            "Default": {"-": 0.840},
            "Sigma": {"1": 0.810, "2": 0.760, "3": 0.690},
            "Gamma": {"0.5": 0.600, "0.7": 0.800, "0.9": 0.650, "0.99": 0.910, "0.9999": 0.900},
            "Epsilon": {"1": 0.820, "2": 0.750},
            "LR": {"0.1": 0.790, "0.01": 0.830},
        },
        "MC": {
            "Default": {"-": 0.790},
            "Sigma": {"1": 0.760, "2": 0.710, "3": 0.640},
            "Gamma": {"0.5": 0.650, "0.7": 0.760, "0.9": 0.840, "0.99": 0.870, "0.9999": 0.860},            "Epsilon": {"1": 0.770, "2": 0.700},
            "LR": {"0.1": 0.740, "0.01": 0.780},
        },
    },
    "Super Hard": {
        "ValIt": {
            "Default": {"-": 0.780},
            "Sigma": {"1": 0.750, "2": 0.700, "3": 0.630},
            "Gamma": {"0.5": 0.900, "0.7": 0.850, "0.9": 0.920, "0.99": 0.950, "0.9999": 0.940},            "Epsilon": {"1": 0.760, "2": 0.690},
            "LR": {"0.1": 0.730, "0.01": 0.770},
        },
        "SARSA": {
            "Default": {"-": 0.720},
            "Sigma": {"1": 0.690, "2": 0.640, "3": 0.570},
            "Gamma": {"0.5": 0.600, "0.7": 0.800, "0.9": 0.650, "0.99": 0.910, "0.9999": 0.900},
            "Epsilon": {"1": 0.700, "2": 0.630},
            "LR": {"0.1": 0.670, "0.01": 0.710},
        },
        "MC": {
            "Default": {"-": 0.660},
            "Sigma": {"1": 0.630, "2": 0.580, "3": 0.510},
            "Gamma": {"0.5": 0.650, "0.7": 0.760, "0.9": 0.840, "0.99": 0.870, "0.9999": 0.860},            "Epsilon": {"1": 0.640, "2": 0.570},
            "LR": {"0.1": 0.610, "0.01": 0.650},
        },
    },
}

# Default values for vertical reference lines
default_values = {
    "Sigma": 2,
    "Gamma": 0.9,
    "Epsilon": 1,
    "LR": 0.1,
}

offsets = {
    "Sigma": 0.12,
    "Gamma": 0.18,
    "Epsilon": 0.06,
    "LR": 0.006,
}

colors = {
    "ValIt": "#30D1CE",
    "SARSA": "#6030D1",
    "MC": "#9EAE26",
}

markers = {
    "ValIt": "s",
    "SARSA": "^",
    "MC": "o",
}

log_scale_categories = {"Gamma"} # list of variables to plot with logs

categories = ["Sigma", "Gamma", "Epsilon", "LR"]
grids = ["A1", "Large Grid", "Super Hard"]
 
n_rows = len(grids)
n_cols = len(categories)
 
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows), sharey=True)
 
for row, grid in enumerate(grids):
    for col, category in enumerate(categories):
        ax = axes[row, col]
 
        for algo in ["ValIt", "SARSA", "MC"]:
            param_dict = data[grid][algo][category]
            x = [float(k) for k in param_dict.keys()]
            y = list(param_dict.values())
 
            ax.plot(x, y, marker=markers[algo], linestyle='-',
                    color=colors[algo], linewidth=2, markersize=7, label=algo)
 
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=14)
 
        # Apply log scale with readable tick labels
        if category in log_scale_categories:
            ax.set_xscale('log')
            # Collect all unique x values for this category across algorithms
            tick_vals = sorted(set(
                float(k)
                for algo in ["ValIt", "SARSA", "MC"]
                for k in data[grid][algo][category].keys()
            ))
            ax.set_xticks(tick_vals)
            # Format labels: show integers where possible, otherwise decimals
            ax.set_xticklabels([
                f'{v:g}' for v in tick_vals
            ])
            from matplotlib.ticker import NullFormatter
            ax.xaxis.set_minor_formatter(NullFormatter())
 
        # Add vertical line for the default value
        if category in default_values:
            ax.axvline(default_values[category], color='gray', linestyle='--', linewidth=1.5)
            # For log-scale axes, use multiplicative offset; for linear, additive
            if category in log_scale_categories:
                text_x = default_values[category] * (1 - offsets[category])
            else:
                text_x = default_values[category] - offsets[category]
            ax.text(text_x, ax.get_ylim()[0] + 0.02,
                    'default value', rotation=90, color='gray', fontsize=12,
                    ha='right', va='bottom')
 
        # Column titles on top row only
        if row == 0:
            ax.set_title(category, fontsize=18, pad=6)
 
        # X-axis labels on bottom row only
        if row == n_rows - 1:
            ax.set_xlabel('Value', fontsize=16)
 
        # Row labels on leftmost column
        if col == 0:
            ax.set_ylabel(f'{grid}\nPerformance', fontsize=16)

# Add legend once
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=15,
           bbox_to_anchor=(0.5, 0.98), frameon=False)

fig.suptitle("Algorithm Performance by Grid and Parameter",
             fontsize=20, y=1.02)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("visualizations/RLPerformanceResults.png", dpi=300, bbox_inches='tight')
