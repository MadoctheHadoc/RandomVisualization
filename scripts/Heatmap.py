import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # For better heatmap styling

DETECTORS = [
    'Binoculars', 'Fast-Detect', 'GPTZero',
    'Turnitin', 'Originality.ai', 'Copyleaks'
]

PROMPT_SUCCESSES = {
    "Reply like a human": [45.5, 34.3, 32.2, 40.1, 38.7, 36.4],
    "Reply with humor": [25.3, 18.7, 15.4, 22.1, 19.8, 20.5],
    "Use less formal vocabulary": [15.5, 14.3, 12.2, 18.7, 16.4, 14.9],
    "Reply casually": [35.5, 24.3, 20.2, 28.9, 22.5, 25.1],
    "Reply formally": [46.5, 32.5, 34.9, 42.3, 39.8, 37.2],
    "Use technical jargon": [55.2, 42.8, 40.5, 48.6, 45.3, 43.1],
}

# Prepare data for heatmap
data = np.array([PROMPT_SUCCESSES[prompt] for prompt in PROMPT_SUCCESSES])

# Create the heatmap
plt.figure(figsize=(10, 7))
ax = sns.heatmap(data, annot=True, fmt=".1f", cmap="inferno",
            xticklabels=DETECTORS,
            yticklabels=PROMPT_SUCCESSES.keys())

# Add a label on top of the y-axis column
ax.text(-0.12, 0.97, "Prompt Types", transform=ax.transAxes,
        fontsize=14, va='bottom', ha='center')

plt.title("Prompt Success Rates by Detector", fontsize=16, fontweight='bold')
plt.xlabel("Detectors", fontsize=14)
plt.tight_layout()
plt.savefig("visualizations/PromptingResultsHeatmap.png", dpi=300, bbox_inches='tight')

