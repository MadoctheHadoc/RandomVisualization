import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
import os
import matplotlib.cm as cm
import matplotlib.colors as mcolors



base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '..', 'data')
food_path = os.path.join(data_path, 'gov_spending.csv')

# Load CSV data
df = pd.read_csv(food_path)

# Automatically detect spending categories from the CSV
spending_categories = [col for col in df.columns if col not in ['Country', 'Revenue']]


# Generate a color map with as many distinct colors as needed
cmap = cm.get_cmap('plasma', len(spending_categories))  # or 'plasma', 'viridis', etc.
colors = [mcolors.to_hex(cmap(i)) for i in range(len(spending_categories))]

# Softer green for revenue
revenue_color = '#86efac'  # pastel green


# Set up the bar positions
x = np.arange(len(df))
width = 0.5

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot stacked bars for each country
for i, category in enumerate(spending_categories):
    bottoms = df[spending_categories[:i]].sum(axis=1) if i > 0 else 0
    ax.bar(x, df[category], width, bottom=bottoms, label=category, color=colors[i])

# Plot revenue (negative bars)
ax.bar(x, df['Revenue'], width, color=revenue_color, label='Revenue')

# Customize axes
ax.set_xticks(x)
ax.set_xticklabels(df['Country'])
ax.axhline(0, color='black', linewidth=1.2)
ax.set_ylabel('% of GDP')
ax.set_title('Government Spending and Revenue (% of GDP)')

# Legend
legend_elements = [Patch(facecolor=c, label=cat) for c, cat in zip(colors, spending_categories)]
legend_elements.append(Patch(facecolor=revenue_color, label='Revenue'))
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.show()
