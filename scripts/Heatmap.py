import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # For better heatmap styling

DETECTORS = [
    'Binoculars', 'DetectGPT', 'RoBERTa'
]

PROMPT_SUCCESSES = {
    "Write like a stack exchange answer.": [45.5, 34.3, 32.2],
    "Do not allow the text to be detected as AI.": [45.5, 34.3, 32.2],
    "Add a touch of humor.": [45.5, 34.3, 32.2],
    "Maintain a serious and authoritative tone.": [45.5, 34.3, 32.2]
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

