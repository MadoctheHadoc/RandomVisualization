import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# ======== CONSTANTS ======
# =========================
SCRIPT_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(SCRIPT_DIR, '../data/quicksort_results.csv')

BACKGROUND = '#2a2a2a'
GRID_COLOR = "#747474"
POINT_COLOR = "#1f73b4"


def load_data():
    # Load CSV results from Java benchmark
    df = pd.read_csv(DATA_PATH)
    # Compute time per element
    df['TimePerElement'] = df['TimeMillis'] * 10000/ df['ArraySize']
    return df


def plot_results(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    # Compute average time per element per pivot for each size
    size_groups = df.groupby(['ArraySize', 'Pivots'], as_index=False)['TimePerElement'].mean()

    # Unique array sizes for coloring
    sizes = sorted(df['ArraySize'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(sizes)))

    # Plot each size group as a separate line
    for size, color in zip(sizes, colors):
        subset = size_groups[size_groups['ArraySize'] == size]
        ax.plot(
            subset['Pivots'], subset['TimePerElement'],
            marker='o', markersize=4, linewidth=1.2,
            color=color, alpha=0.8, label=f'Size {size}'
        )

    # Use logarithmic scales
    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.set_xlabel('Number of Pivots (log scale)', color='white', fontsize=12)
    ax.set_ylabel('Time per Element (ms / size, log scale)', color='white', fontsize=12)
    ax.set_title('QuickSort Performance vs Number of Pivots by Array Size (Log-Log Scale)', color='white', fontsize=14, pad=20)

    ax.tick_params(colors='white')
    ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.7, which='both')
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(facecolor=BACKGROUND, edgecolor='white', labelcolor='white', fontsize=9)
    plt.tight_layout()

    plt.savefig(os.path.join(SCRIPT_DIR, '../visualizations/QuickSortPivotsScatter.png'),
                dpi=300, bbox_inches='tight')


def main():
    df = load_data()
    plot_results(df)


if __name__ == "__main__":
    main()
