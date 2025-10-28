import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"

# Data
data = {
    "Top-P": {
        "0.6": 0.9925,
        "0.9": 0.9900,
        "0.95": 0.9725,
        "1.0": 0.9425
    },
    "Top-K": {
        "5": 0.9275,
        "10": 0.9200,
        "40": 0.9725,
        "100": 0.9725
    },
    "Temperature": {
        "0.0": 0.9550,
        "0.2": 1.0000,
        "0.5": 0.9875,
        "0.8": 0.9725,
        "1.0": 0.94001,
        "1.4": 0.8300,
        "2.0": 0.8525
    },
    "Repeat Penalty": {
        "1.0": 0.9475,
        "1.1": 0.9725,
        "1.2": 0.8125,
        "1.3": 0.6625,
        "1.4": 0.6775,
        "1.5": 0.5850,
        "1.6": 0.4925
    }
}

# Normal (baseline) values for vertical reference lines
normal_values = {
    "Top-P": 0.95,
    "Temperature": 0.8,
    "Repeat Penalty": 1.1,
    "Top-K": 40
}

colors = [
    "#30D1CE",
    "#6030D1",
    "#9EAE26",
    "#AE2626"
]
offsets = [0.015, 3.5, 0.075, 0.018]

# Create horizontal sequence of subplots
n = len(data)
fig, axes = plt.subplots(1, n, figsize=(3 * n, 4), sharey=True)

# Ensure axes is iterable
if n == 1:
    axes = [axes]

for ax, (category, values), c, offset in zip(axes, data.items(), colors, offsets):
    x = [float(k) for k in values.keys()]
    y = list(values.values())
    if(category == 'Top-K'):
        ax.set_xlim(0,105)

    ax.plot(x, y, marker='o', linestyle='-', color=c, linewidth=2)
    ax.set_title(category, fontsize=18, pad=1)
    ax.set_xlabel('Value', fontsize=18)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Set tick label font sizes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)

    # Add vertical line for the normal value
    if category in normal_values:
        ax.axvline(normal_values[category], color='gray', linestyle='--', linewidth=1.5)
        ax.text(normal_values[category] - offset, 0.78,
                'normal value', rotation=90, color='gray', fontsize=14,
                ha='center', va='top')

axes[0].set_ylabel('AUROC (%)', fontsize=18)
fig.suptitle("LLM Meta-Parameters vs Binoculars AUROC score",
             fontsize=20, y=0.93)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("visualizations/PromptingResults.png", dpi=300, bbox_inches='tight')
