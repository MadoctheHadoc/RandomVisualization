import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"


def plot_grouped_score_bars_vertical(data, colors):
    # Extract labels and values
    categories = [row[0] for row in data]
    values = [row[1:-2] for row in data]  # Exclude the overall score and the kept_techniques flag
    kept_techniques = [row[-1] for row in data]  # Extract the kept_techniques flag

    # Group data into Memory, Eval Time (synth + real), and Other
    grouped_values = []
    for row in values:
        syn_eval_time = row[1]  # Synth
        real_eval_time = row[2]  # Real
        other = row[0] / 500 + row[3] / 1000 + 20 * row[4]  # Prep/500 + Load/1000 + 20*Peak Mem
        grouped_values.append([real_eval_time, syn_eval_time, other])

    # Labels for each group
    group_labels = ["Real\nEvaluation\nTime", "Syn Eval", "Other"]

    # Figure setup
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')

    # Plot stacked bars (vertically)
    x_pos = np.arange(len(categories))
    bottom = np.zeros(len(categories))
    for i in range(len(grouped_values[0])):
        bars = ax.bar(
            x_pos,
            [row[i] for row in grouped_values],
            bottom=bottom,
            color= [(colors["DarkPastelRed"] if i==2 else colors["SlateGrey"])  if not kept_techniques[j] else (colors["PastelRed"] if i==2 else colors["LightGrey"]) for j in range(len(categories))],
            edgecolor='white',
            linewidth=0.5,
            width=0.8
        )

        # Add labels on the first bar only
        for j, bar in enumerate(bars):
            if j == 0:
                height = [row[i] for row in grouped_values][j]
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bottom[j] + height / 2,
                        group_labels[i],
                        ha='center',
                        va='center',
                        color='white',
                        fontsize=12,
                        fontweight='bold'
                    )

        bottom += np.array([row[i] for row in grouped_values])


    # X labels and ticks (categories)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=10, color=colors["DarkPastelRed"])

    # Y labels and ticks (score)
    ax.tick_params(axis='y', colors=colors["SlateGrey"], labelsize=17, width=2)
    ax.tick_params(axis='x', colors=colors["SlateGrey"], labelsize=17, width=2)

    # Axis styling
    for spine_name, spine in ax.spines.items():
        if spine_name == 'bottom':
            spine.set_linewidth(2)
            spine.set_color(colors["SlateGrey"])
        else:
            spine.set_visible(False)

    # Gridlines
    ax.yaxis.grid(True, linestyle='--', alpha=1.0, color=colors["SlateGrey"], linewidth=1.5)
    ax.set_axisbelow(True)

    # Labels and title
    ax.set_ylabel("Score Contribution", fontsize=24, color=colors["DarkPastelRed"], labelpad=0)
    ax.set_title("Score History by Technique", fontsize=30, color=colors["DarkPastelRed"], pad=0, fontweight='bold')

    plt.tight_layout()
    plt.savefig("visualizations/PerformanceComparison.png", dpi=300, bbox_inches="tight", facecolor='white')



# Custom colors (as provided)
colors = {
    "SlateGrey": "#2E2E2E",
    "LightGrey": "#7D7D7D",
    "PastelRed": "#8F0D0D",
    "DarkPastelRed": "#450808"
}

data = [ # Last variable indicates whether the technique made it to the final implementation
    # Basic implementation
    # ["Basic\nImplementation", 697.19, 89.41, 17643.63, 3877.55, 7.76, 17893.51, True],
    ["Basic\nImplementation  ", 0, 546.77, 13984.72, 3842.32, 24.07, 15016.73, True],
    
    # Bushy
    ["Bushy\nEval", 730.08, 87.79, 16785.89, 4025.89, 9.33, 17065.77, False],
        
    # CSR/CSC
    ["CSR/\nCSC", 695.83, 520.24, 12266.92, 3920.49, 25.08, 13294.07, True],

    # Waveguide Original
    ["Waveguide\nOriginal", 706.71, 251.29, 11958.52, 4134.58, 25.08, 12716.96, False],

    # ZigZag + In-Place Join + Semi-Transitive Closure
    ["ZigZag\n+ Others", 739.38, 423.17, 9050.19, 3902.90, 25.67, 9992.14, False],

    # Union Cardinality Ordering
    # ["Union Card\nOrdering", 785.73, 423.88, 9369.03, 3993.75, 25.67, 10311.87, False],

    # PSO and POS
    ["PSO/\nPOS", 709.13, 395.06, 8494.53, 4733.06, 28.53, 9466.34, True],

    # Parallelism
    ["Multi-\nThreading", 730.89, 375.19, 8465.62, 4787.30, 27.79, 9402.80, False],

    # Waveguide
    ["Waveguide", 707.53, 52.87, 1531.17, 4750.70, 5.21, 1694.41, True],

    # Smart Memory Allocation
    ["Smart\nAllocation", 715.43, 31.99463, 1071.97, 4779.72, 5.21, 1214.38, True],

    # Final Touches
    ["Final\nTouches", 711.29, 31.08, 1026.21, 4771.87, 5.21, 1167.69, True]
]


# Call the function
plot_grouped_score_bars_vertical(data, colors)
