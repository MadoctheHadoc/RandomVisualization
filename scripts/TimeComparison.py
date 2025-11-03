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
data = [
    [20800, 3360300, 2096800],
    [17300, 2639200, 100960200, 268900]
]

categories = ["Unoptimized", "JIT-compiled"]

# Labels for each data segment
segment_labels = [
    ["AST\nto\nOp", "Builtins", "Execution"],
    ["AST\nto\nOp", "Builtins", "JIT-Compilation", "Execution"],
]

# Normalize lengths
max_len = max(len(row) for row in data)
data_padded = np.array([row + [0]*(max_len - len(row)) for row in data])
y_pos = np.arange(len(categories))

# Figure setup
fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')

# Plot stacked bars with labels
left = np.zeros(len(categories))
for i in range(data_padded.shape[1]):
    c = colors["LightGrey"]
    if i==1:
        c = colors["PastelRed"]
    bars = ax.barh(
        y_pos,
        data_padded[:, i],
        left=left,
        color=c,
        edgecolor="white",
        linewidth=2,
        height=0.5
    )

    # Add labels on each segment (if the segment is not zero)
    for j, bar in enumerate(bars):
        width = data_padded[j, i]
        if width > 0:
            ax.text(
                left[j] + width / 2,
                bar.get_y() + bar.get_height() / 2,
                segment_labels[j][i],
                ha='center',
                va='center',
                color='white',
                fontsize=12,
                fontweight='bold'
            )

    left += data_padded[:, i]

# Y labels and ticks
ax.set_yticks(y_pos)
ax.set_yticklabels(categories, fontsize=18, color=colors["DarkPastelRed"])

# X labels and ticks
ax.tick_params(axis='x', colors=colors["SlateGrey"], labelsize=14, width=2)
ax.tick_params(axis='y', colors=colors["SlateGrey"], labelsize=16, width=2)

# Axis styling (only keep y-axis)
for spine_name, spine in ax.spines.items():
    if spine_name == 'left':
        spine.set_linewidth(2.5)
        spine.set_color(colors["SlateGrey"])
    else:
        spine.set_visible(False)

# Gridlines aligned with y-ticks
ax.xaxis.grid(True, linestyle='--', alpha=1.0, color=colors["SlateGrey"], linewidth=1.5)
ax.set_axisbelow(True)

# Labels and title
ax.set_xlabel("Time (ms)", fontsize=18, color=colors["SlateGrey"], labelpad=10)
ax.set_title("Time Comparison", fontsize=20, color=colors["DarkPastelRed"], pad=15, fontweight='bold')

plt.tight_layout()
plt.savefig("visualizations/TimeComparison.png", dpi=300, bbox_inches="tight", facecolor='white')

