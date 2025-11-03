import matplotlib.pyplot as plt
import numpy as np

# Style
plt.rcParams["font.family"] = "Times New Roman"

# Custom colors
colors = {
    "SlateGrey": "#2E2E2E",
    "LightGrey": "#7D7D7D",
    "PastelRed": "#8F0D0D",
    "DarkPastelRed": "#450808",
}

benchmarks = {
    "HuffmanEncoding": [
        [6043699],
        [5696235],
        [2051524]
    ],
    "Permutation": [
        [619662],
        [578459],
        [638401]
    ],
    "RecursiveMatrix\nExponentiation": [
        [7394715],
        [5546214],
        [4807624]
    ],
    "schemeInterpreter": [
        [5356445],
        [5984708],
        [3577313]
    ],
    "SieveOfAtkin": [
        [57043610],
        [44070798],
        [1546697]
    ]
}

benchmarks = {
    "HuffmanEncoding": [
        [4955957],
        [2051524]],
    "Permutation": [
        [619662],
        [638401]],
    "RecursiveMatrix\nExponentiation": [
        [6894824],
        [4807624]],
    "schemeInterpreter": [
        [5356445],
        [3577313]],
    "SieveOfAtkin": [
        [42894080],
        [1546697]]
    }

# Categories for the legend
categories = ["Our Impl.", "Reference\nImpl."]

# Benchmark names and number of benchmarks
benchmark_names = list(benchmarks.keys())
num_benchmarks = len(benchmark_names)

# Figure setup with shared y-axis
fig, axes = plt.subplots(nrows=1, ncols=num_benchmarks, figsize=(12, 3), facecolor='white', sharey=True)

# Add a big title for the entire figure
fig.text(0.5, 0.97, "Time Comparison Across Benchmarks",
         fontsize=24, color=colors["DarkPastelRed"], ha='center', fontweight='bold')

# Iterate over each benchmark
for i, (benchmark_name, data) in enumerate(benchmarks.items()):
    ax = axes[i]
    y_pos = np.arange(len(categories))

    # Convert data from [[x], [y], [z]] to [x, y, z]
    values = [row[0] for row in data]

    # Determine the min between Our Impl. and JIT Impl. (indices 0 and 1)
    min_index = 0 if values[0] < values[1] else 1

    # Set default colors
    bar_colors = [colors["LightGrey"], colors["SlateGrey"]]

    # Plot all three bars
    bars = ax.barh(
        y_pos,
        values,
        height=0.6,
        color=bar_colors,
        edgecolor="white",
        linewidth=0.0
    )

    # Set y-ticks and labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=12, color=colors["DarkPastelRed"])

    # Add benchmark name as title
    ax.set_title(benchmark_name, fontsize=14, color=colors["DarkPastelRed"], pad=10, fontweight='bold')

    # X labels and ticks
    ax.tick_params(axis='x', colors=colors["SlateGrey"], labelsize=12, width=2)
    ax.tick_params(axis='y', colors=colors["SlateGrey"], labelsize=12, width=2)

    # Axis styling
    for spine_name, spine in ax.spines.items():
        if spine_name == 'left':
            spine.set_linewidth(1.5)
            spine.set_color(colors["SlateGrey"])
        else:
            spine.set_visible(False)

    # Gridlines
    ax.xaxis.grid(True, linestyle='--', alpha=1.0, color=colors["SlateGrey"], linewidth=1.0)
    ax.set_axisbelow(True)

# Remove y-axis labels for all but the first subplot
for ax in axes[1:]:
    ax.tick_params(labelleft=False)

# Overall x-axis label
fig.text(0.5, 0.02, "Time (ns)", fontsize=18, color=colors["SlateGrey"], ha='center')

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("visualizations/TimeComparisonGrouped.png", dpi=300, bbox_inches="tight", facecolor='white')
