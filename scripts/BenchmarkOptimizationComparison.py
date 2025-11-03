import matplotlib.pyplot as plt
import numpy as np

# Style
plt.rcParams["font.family"] = "Times New Roman"

# Custom colors
colors = {
    "SlateGrey": "#2E2E2E",
    "LightGrey": "#7D7D7D",
    "PastelRed": "#8F0D0D",
    "DarkPastelRed": "#450808"
}

# Data
combinations = [
    "no_opts",
    "string-ropes",
    "array-strategies",
    "inline-caches",
    "string-ropes\n-array-strategies",
    "string-ropes\n-inline-caches",
    "array-strategies\n-inline-caches",
    "string-ropes\n-array-strategies\n-inline-caches"
]

runs = [
    [5529102, 4882868, 4951407, 5168266, 4854537, 5003172, 5181951, 5002084],
    [963064, 996749, 963188, 939270, 1014873, 967014, 864045, 915798],
    [6405310, 6280396, 5455430, 7178010, 8274308, 10340201, 8372130, 7874961],
    [8837606, 8665692, 8299986, 8985361, 8631137, 8857658, 9513663, 8768728],
    [57389010, 58636158, 57258367, 63787329, 56779282, 48806648, 40148699, 56598082]
]

# Made-up group labels (e.g., different test environments)
group_labels = [
    "HuffmanEncoding",
    "Permutation",
    "RecursiveMatrix\nExponentiation",
    "schemeInterpreter",
    "SieveOfAtkin"
]

# Figure setup with shared y-axis
fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(12, 6), facecolor='white', sharey=True)

# Add a big title for the entire figure
fig.text(0.5, 0.92, "Performance Comparison Across Runs and Combinations",
         fontsize=24, color=colors["DarkPastelRed"], ha='center', fontweight='bold')

# Iterate over each run
for i, ax in enumerate(axes):
    run_data = runs[i]
    min_value = min(run_data)
    min_index = run_data.index(min_value)

    # Create horizontal bars
    bars = ax.barh(
        combinations,
        run_data,
        color=[colors["PastelRed"] if j == min_index else colors["LightGrey"] for j in range(len(combinations))],
        edgecolor="white",
        linewidth=0.5
    )

    # X labels and ticks
    ax.set_xlabel(f"Time (ns)", fontsize=12, color=colors["SlateGrey"])
    ax.tick_params(axis='x', colors=colors["SlateGrey"], labelsize=10, width=1)
    ax.tick_params(axis='y', colors=colors["SlateGrey"], labelsize=10, width=1)

    # Axis styling
    for spine_name, spine in ax.spines.items():
        if spine_name == 'left':
            spine.set_linewidth(1.5)
            spine.set_color(colors["SlateGrey"])
        else:
            spine.set_visible(False)

    # Gridlines
    ax.xaxis.grid(True, linestyle='--', alpha=1.0, color=colors["SlateGrey"], linewidth=0.7)
    ax.set_axisbelow(True)

    # Title for each subplot
    ax.set_title(f"{group_labels[i]}", fontsize=14, color=colors["DarkPastelRed"], pad=2, fontweight='bold')

# Remove y-axis labels for all but the first subplot
for ax in axes[1:]:
    ax.tick_params(labelleft=False)

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.9])  # Adjust the top to make space for the big title
plt.savefig("visualizations/BenchmarkOptComp.png", dpi=300, bbox_inches="tight", facecolor='white')
