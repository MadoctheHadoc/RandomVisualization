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

data = {
    "categories": [
        "CALL", "DIV", "LOAD_CONST_BIG", "ADD", "LOAD_CONST_LONG",
        "PUSH_STRING", "PUSH_LONG", "MUL", "SUB", "NOT",
        "LOAD_CONST_STRING", "LOAD_CONST_BOOL", "LOAD_CONST_FUNCT",
        "RETURN", "JUMP", "READ_ARGUMENT", "HALT", "STORE_LOCAL",
        "READ_LOCAL", "EQUAL", "LE", "LT", "JUMP_IF_FALSE", "MOVE",
        "WRITE_PROPERTY", "READ_PROPERTY", "NEG", "CHECK_BOOL",
        "LOAD_RAW_BOOL"
    ],
    "values": [
        12316200.00, 5731200.00, 819100.00, 2400.00, 300.00,
        200.00, 100.00, 1200.00, 100.00, 100.00,
        100.00, 100.00, 0.00, 48700.00, 0.00, 100.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00
    ]
}

categories = data['categories']
values = data['values']

# Sort categories and values by value (descending)
sorted_indices = np.argsort(values)[::-1]
categories = [categories[i] for i in sorted_indices]
offset = 85000
values = [values[i] + offset for i in sorted_indices]

# Figure setup
fig, ax = plt.subplots(figsize=(10, 5), facecolor='white')

# Plot bars
bar_colors = [colors["PastelRed"] if cat == "CALL" else colors["LightGrey"] for cat in categories]
bars = ax.bar(
    range(len(categories)),
    values,
    color=bar_colors,
    edgecolor="white",
    linewidth=0.0,
    width=0.85,
)

# X labels and ticks
ax.set_xticks(range(len(categories)))
ax.set_xticklabels(categories, fontsize=12, color=colors["DarkPastelRed"], rotation=40, ha='right')

# Y labels and ticks
ax.tick_params(axis='y', colors=colors["SlateGrey"], labelsize=14, width=2)
ax.tick_params(axis='x', colors=colors["SlateGrey"], labelsize=12, width=2)

# Axis styling
for spine_name, spine in ax.spines.items():
    if spine_name == 'bottom':
        spine.set_linewidth(2.5)
        spine.set_color(colors["SlateGrey"])
    else:
        spine.set_visible(False)

# Gridlines
ax.yaxis.grid(True, linestyle='--', alpha=1.0, color=colors["SlateGrey"], linewidth=1.0)
ax.set_axisbelow(True)

# Labels and title
ax.set_ylabel("Time (ns)", fontsize=18, color=colors["SlateGrey"], labelpad=10)
ax.set_title("Execution Durations of Opcodes", fontsize=20, color=colors["DarkPastelRed"], pad=15, fontweight='bold')

ax.text(15, 3000000, "most opcodes have average durations very close to zero", ha='center', va='center', color=colors["LightGrey"], fontsize=12)

plt.tight_layout()
plt.savefig("visualizations/TimeComparison.png", dpi=300, bbox_inches="tight", facecolor='white')
