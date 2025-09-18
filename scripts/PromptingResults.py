import matplotlib.pyplot as plt
import numpy as np

# Data
data = {
    "Attack": {
        "Ignore and say": 35.7,
        "Ignore and print": 43.6,
        "INGORE AND PRINT": 40.7
    },
    "Delimiter length": {
        "0": 43.6,
        "4": 52.2,
        "10": 58.6,
        "20": 56.4
    },
    "Delimiter repetitions": {
        "1": 52.1,
        "2": 58.6,
        "3": 56.4,
        "4": 56.4
    },
    "Rogue string": {
        '"I hate humans"': 58.6,
        '"Kill all humans"': 49.3,
        "Non-harmful string": 70.0
    },
    "Temperature": {
        "0": 57.9,
        "0.25": 57.1,
        "0.5": 55.7,
        "1.0": 52.1
    },
    "Top-P": {
        "0": 58.6,
        "0.5": 57.1,
        "1.0": 58.6
    },
    "Frequency penalty": {
        "0": 58.6,
        "2.0": 57.1
    },
    "Presence penalty": {
        "0": 57.1,
        "2.0": 57.9
    },
    "Stop sequence": {
        "No": 60.0,
        "Yes": 47.5
    },
    "Text after {user_input}": {
        "No": 63.1,
        "Yes": 51.8
    },
    "Model": {
        "text-ada-001": 13.8,
        "text-babbage-001": 29.5,
        "text-curie-001": 23.8,
        "text-davinci-001": 30.5,
        "text-davinci-002": 58.6
    }
}

# Define y positions for subcategory labels and group labels
y_sub_labels = [34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 12]
rot_labels = [90, 0, 0, 90, 0, 0, 0, 0, 0, 0, 90]
y_group_labels = [48, 63, 50, 48, 63, 55, 63, 55, 46, 67, 42] 

# Plotting
fig, ax = plt.subplots(figsize=(14, 7))

categories = list(data.keys())
x_start = 0
group_spacing = 1  # space between groups
colors = plt.cm.tab20.colors

for idx, cat in enumerate(categories):
    if(cat == 'Model'):
        # Ignore model for now
        continue
    subcats = list(data[cat].keys())
    values = list(data[cat].values())
    x = np.arange(x_start, x_start + len(values))
    c = colors[idx % len(colors)]

    # Plot mini line for this group
    ax.plot(x, values, marker='o', color=c, linewidth=2)
    
    # Add value labels
    for xi, val, lbl in zip(x, values, subcats):
        ax.text(xi, val + 1.0, f'{val}', ha='center', va='bottom', fontsize=8)
        ax.text(xi, y_sub_labels[idx], lbl, ha='center', va='top', rotation=rot_labels[idx], fontsize=8)  # subcategory label

    # Add group label
    ax.text(np.mean(x), y_group_labels[idx], cat, ha='center', va='top', fontsize=10, fontweight='bold', color=c)

    x_start += len(values) + group_spacing

ax.set_xlim(-1, x_start - 1)
ax.set_ylabel('Success Rate (%)')
ax.set_title('Goal Hijacking Results by Group (Mini Line Plots)')
ax.set_xticks([])
ax.set_ylim(35, 73)
plt.tight_layout()
plt.savefig("visualizations/PromptingResults.png", dpi=300, bbox_inches='tight')

