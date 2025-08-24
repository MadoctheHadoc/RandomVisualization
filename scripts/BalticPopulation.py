import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np

# ==================== STYLE CONSTANTS ====================
BG_COLOR = '#2a2a2a'

# Define editable color palette
CITY_COLORS = [
    "#22689d", "#3e85ba",  # Estonia (Tallinn, rest)
    "#9d2235", "#ba3e3e",  # Latvia (Riga, rest)
    "#126508", "#36a22a", "#197c0e"  # Lithuania (Vilnius, Kaunas, rest)
]

# ==================== DATA ====================
BALTIC_POPULATIONS = [
    {'countries': 'Estonia', 'population': 1373101, 'cities': [
        ('Tallinn', 638076)
    ]},
    {'countries': 'Latvia', 'population': 1842226, 'cities': [
        ('Riga', 615764)
    ]},
    {'countries': 'Lithuania', 'population': 2897430, 'cities': [
        ('Vilnius', 747864),
        ('Kaunas', 410475)
    ]}
]

# ==================== FUNCTIONS ====================
def prepare_pie_data(data):
    """Prepare labels, sizes, and explode groups for pie chart."""
    labels, sizes, explode_groups = [], [], []
    group_id = 0

    for country in data:
        country_name = country['countries']
        country_pop = country['population']

        city_pops = [pop for city, pop in country['cities']]
        city_names = [city for city, pop in country['cities']]

        sum_city_pops = sum(city_pops)
        rest = country_pop - sum_city_pops

        # Add city slices
        for city, pop in zip(city_names, city_pops):
            labels.append(f"{city}\n{pop:,}")
            sizes.append(pop)
            explode_groups.append(group_id)

        # Add remaining country population slice
        if rest > 0:
            labels.append(f"Rest of {country_name}\n{rest:,}")
            sizes.append(rest)
            explode_groups.append(group_id)

        group_id += 1

    return labels, sizes, explode_groups

def create_base_plot():
    """Create figure and axis with consistent styling."""
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    return fig, ax

def plot_population_pie(ax, data, colors, explode_dist=0.01):
    """
    Layered pie chart:
    - Outer arc = full country
    - Inner arcs = cities inside country
    - Cities at smaller radius
    - Country label placed in remaining section
    - City labels formatted as 'Name - population'
    """
    total_pop = sum(c['population'] for c in data)
    start_angle = 140
    current_angle = start_angle

    wedges = []
    texts = []

    color_idx = 0  # for indexing colors

    for country_idx, country in enumerate(data):
        country_name = country['countries']
        country_pop = country['population']
        cities = country['cities']

        # Colors: country background + city colors
        country_color = colors[color_idx % len(colors)]
        city_colors = colors[color_idx + 1 : color_idx + 1 + len(cities)]
        color_idx += 1 + len(cities)

        # Country arc
        country_angle = (country_pop / total_pop) * 360
        country_mid_angle = current_angle + country_angle / 2
        offset_x = explode_dist * np.cos(np.radians(country_mid_angle))
        offset_y = explode_dist * np.sin(np.radians(country_mid_angle))

        country_wedge = Wedge(
            center=(offset_x, offset_y),
            r=1.0,
            theta1=current_angle,
            theta2=current_angle + country_angle,
            facecolor=country_color,
            edgecolor=BG_COLOR,
            linewidth=3
        )
        ax.add_patch(country_wedge)

        # Draw city wedges
        city_start_angle = current_angle
        inner_radius = 0.93
        total_city_angle = 0
        for city, pop, city_color in zip(cities, [p for _, p in cities], city_colors):
            city_angle = (pop / total_pop) * 360
            if city_angle < 1:
                city_angle = 1
            total_city_angle += city_angle
            city_wedge = Wedge(
                center=(offset_x, offset_y),
                r=inner_radius,
                theta1=city_start_angle,
                theta2=city_start_angle + city_angle,
                facecolor=city_color,
                edgecolor=BG_COLOR,
                linewidth=3
            )
            ax.add_patch(city_wedge)

            # City label formatted with commas
            mid_theta = (city_start_angle + city_start_angle + city_angle) / 2
            label_r = inner_radius * 0.63
            label_x = offset_x + label_r * np.cos(np.radians(mid_theta))
            label_y = offset_y + label_r * np.sin(np.radians(mid_theta))
            city_name, ignored = city
            ax.text(label_x, label_y, f"{city_name}\n{pop:,}", color="white",
                    ha="center", va="center", fontsize=17, fontweight='bold')

            city_start_angle += city_angle

        # Country label in remaining empty section
        remaining_angle = current_angle + country_angle - (current_angle + total_city_angle)
        if remaining_angle > 0:
            empty_mid_angle = current_angle + total_city_angle + remaining_angle / 2
            label_r_country = 0.6 # inside the empty portion of country
            label_x_country = offset_x + label_r_country * np.cos(np.radians(empty_mid_angle))
            label_y_country = offset_y + label_r_country * np.sin(np.radians(empty_mid_angle))
            country_pop = next(c['population'] for c in BALTIC_POPULATIONS if c['countries'] == country_name)
            ax.text(label_x_country, label_y_country, f"{country_name}\n{country_pop:,}",
                    color="white", ha="center", va="center", fontsize=22, fontweight='bold')

        current_angle += country_angle

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.set_aspect('equal')
    ax.axis('off')
    return wedges, texts

def get_group_mid_angle(explode_groups, sizes, group, start_angle, total):
    """Calculate the midpoint angle of a group for proper explode offset."""
    group_indices = [i for i, g in enumerate(explode_groups) if g == group]
    # Compute the start and end angles of the entire group
    start = start_angle + sum(sizes[:group_indices[0]]) / total * 360
    end = start_angle + sum(sizes[:group_indices[-1]+1]) / total * 360
    return (start + end) / 2

def style_chart(ax):
    """Apply chart title and style adjustments."""
    # Calculate total population
    total_population = sum(c['population'] for c in BALTIC_POPULATIONS)
    
    # Format with commas
    total_formatted = f"{total_population:,}"
    
    # Set title including total population and move it slightly down
    ax.set_title(f'Where do the {total_formatted} Baltic People Live?',
                 color='white', fontsize=24, y=0.97) 

    # Add credits in smaller text at bottom-right
    ax.text(1, -1.05, "Visualization by MadoctheHadoc\nUses 'Urban Area' populations\nSource: Wikipedia", 
            color='white', fontsize=10, ha='right', va='bottom', alpha=1.0)
    
    plt.axis('equal')

# ==================== MAIN FUNCTION ====================
def plot_baltic_population_chart():
    labels, sizes, explode_groups = prepare_pie_data(BALTIC_POPULATIONS)
    fig, ax = create_base_plot()
    plot_population_pie(ax, BALTIC_POPULATIONS, CITY_COLORS)
    style_chart(ax)
    plt.savefig("visualizations/BalticPopulation.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==================== EXECUTE ====================
if __name__ == "__main__":
    plot_baltic_population_chart()
