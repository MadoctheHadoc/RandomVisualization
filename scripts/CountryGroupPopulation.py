import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import numpy as np

# ==================== STYLE CONSTANTS ====================
BG_COLOR = '#2a2a2a'

# Collapse text
COLLAPSE = False

# Define editable color palette
BALTIC_COLORS = [
    "#22689d", "#3e85ba",  # Estonia (Tallinn, rest)
    "#9d2235", "#ba3e3e",  # Latvia (Riga, rest)
    "#126508", "#36a22a", "#197c0e"  # Lithuania (Vilnius, Kaunas, rest)
]

SCANDINAVIAN_COLORS = [
    "#22689d", "#3e85ba",  # Finland
    "#72750B", "#b6b238", "#9d9922",  # Sweden
    "#197c0e", "#36a22a",  # Norway
    "#9d2235", "#c54747",  # Denmark
]

BENELUX_COLORS = [
    "#22689d", "#65a4d5", "#3e85ba",  # Belgium
    "#9d7222", "#e2b67c", "#d7a061", "#c18e46",  # Netherlands
    "#29a22f"  # Luxembourg
]

YUGOSLAV_COLORS = [
    "#22689d", "#3e85ba",  # Serbia (Belgrade, rest)
    "#9d2235", "#ba3e3e",  # Croatia (Zagreb, rest)
    "#197c0e", "#36a22a",  # Bosnia & Herzegovina (Sarajevo, rest)
    "#84880F", "#afab30",  # Slovenia (Ljubljana, rest)
    "#8438ac", "#a75fcd",  # North Macedonia
    "#ac7a38", "#cd925f",  # Albania
    "#47c1c5",              # Montenegro
    "#8fbb4c"               # Kosovo
]

IRISH_COLORS = [
    "#126508", "#36a22a", "#197c0e",  # Rep. of Ireland
    "#653308", "#a2742a", "#7c520e",  # N. Ireland
]

# ==================== DATA ====================

YUGOSLAV_POPULATIONS = [
    {'countries': 'Serbia', 'population': 6617200, 'cities': [
        ('Belgrade', 1681405)  # Metro
    ]},
    {'countries': 'Croatia', 'population': 3861967, 'cities': [
        ('Zagreb', 1086528)  # Metro
    ]},
    {'countries': 'Bosnia and\nHerzegovina', 'population': 2904256, 'cities': [
        ('Sarajevo', 555210)  # Metro
    ]},
    {'countries': 'Slovenia', 'population': 2130850, 'cities': [
        ('Ljubljana', 569475)  # Metro
    ]},
    {'countries': 'North\nMacedonia', 'population': 1836713, 'cities': [
        ('Skopje', 526502)  # Metro
    ]},
    {'countries': 'Albania', 'population': 2402113, 'cities': [
        ('Tirana', 800986)
    ]},
    {'countries': 'Kosovo', 'population': 1585566, 'cities': [
    ]},
    {'countries': 'Mont.', 'population': 623327, 'cities': [
    ]}
]

BENELUX_POPULATIONS = [
    {'countries': 'Belgium', 'population': 11742796, 'cities': [
        ('Brussels', 3398857),
        ('Antwerp', 1172740)
    ]},
    {'countries': 'Netherlands', 'population': 17811291, 'cities': [
        ('Amsterdam', 2961252),
        ('Rotterdam', 1880019),
        ('Den Haag', 1150797)
    ]},
    {'countries': 'Lux.', 'population': 681973, 'cities': []}
]

BALTIC_POPULATIONS = [
    {'countries': 'Estonia', 'population': 1373101, 'cities': [
        ('Tallinn', 638076)
    ]},
    {'countries': 'Latvia', 'population': 1842226, 'cities': [
        ('Riga', 927953)
    ]},
    {'countries': 'Lithuania', 'population': 2897430, 'cities': [
        ('Vilnius', 747864),
        ('Kaunas', 403375)
    ]}
]

SCANDINAVIAN_POPULATIONS = [
    {'countries': 'Finland', 'population': 5635971, 'cities': [
        ('Helsinki', 1616656)
    ]},
    {'countries': 'Sweden', 'population': 10588230, 'cities': [
        ('Stockholm ', 2415139),
        ('Gothenburg', 1080980)
    ]},
    {'countries': 'Norway', 'population': 5601049, 'cities': [
        ('Oslo', 1588457)
    ]},
    {'countries': 'Denmark', 'population': 6001008, 'cities': [
        ('Copenhagen', 2135634)
    ]}
]

IRISH_POPULATIONS = [
    {'countries': 'Rep. of Ireland', 'population': 5380300, 'cities': [
        ('Dublin', 2082575), ('Cork', 406785)
    ]},
    {'countries': 'N. Ireland', 'population': 1910543, 'cities': [
        ('Belfast', 671559), ('Derry', 237000)
    ]},
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
    start_angle = 70
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
            label_r = inner_radius * 0.7
            label_x = offset_x + label_r * np.cos(np.radians(mid_theta))
            label_y = offset_y + label_r * np.sin(np.radians(mid_theta))
            city_name, ignored = city
            pop_text = f"{pop:,}" if COLLAPSE else f"{pop/1000:,.0f}K"
            ax.text(label_x, label_y, f"{city_name}\n{pop_text}", color="white",
                ha="center", va="center", fontsize=16, fontweight='bold')
            city_start_angle += city_angle

        # Country label in remaining empty section
        remaining_angle = current_angle + country_angle - (current_angle + total_city_angle)
        if remaining_angle > 0:
            empty_mid_angle = current_angle + total_city_angle + remaining_angle / 2
            label_r_country = 0.6 # inside the empty portion of country
            label_x_country = offset_x + label_r_country * np.cos(np.radians(empty_mid_angle))
            label_y_country = offset_y + label_r_country * np.sin(np.radians(empty_mid_angle))
            country_pop = next(c['population'] for c in data if c['countries'] == country_name)
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

def style_chart(ax, populations, people):
    """Apply chart title and style adjustments."""
    # Calculate total population
    total_population = sum(c['population'] for c in populations)
    
    # Format with commas
    total_formatted = f"{total_population:,}"
    
    # Set title including total population and move it slightly down
    ax.set_title(f'Where do the {total_formatted} {people} Live?',
                 color='white', fontsize=24, y=0.97) 

    # Add credits in smaller text at bottom-right
    ax.text(1, -1.07, "Visualization by MadoctheHadoc\nSource: Eurostat via Wikipedia\n" + 
            "Uses 'Metro Area' definitions which\nincludes neighbouring towns", 
            color='white', fontsize=10, ha='right', va='bottom', alpha=1.0)
    
    plt.axis('equal')

# ==================== MAIN FUNCTION ====================
def plot_population_chart(populations, colors, filename, people):
    labels, sizes, explode_groups = prepare_pie_data(populations)
    fig, ax = create_base_plot()
    plot_population_pie(ax, populations, colors)
    style_chart(ax, populations, people)
    plt.savefig(f"visualizations/{filename}Population.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

# ==================== EXECUTE ====================
if __name__ == "__main__":
    # plot_population_chart(
    #     SCANDINAVIAN_POPULATIONS, SCANDINAVIAN_COLORS, 'Scandinavia', 'Scandinavians')
    # plot_population_chart(
    #     BALTIC_POPULATIONS, BALTIC_COLORS, 'Baltic', 'Baltic People')
    # plot_population_chart(
    #     BENELUX_POPULATIONS, BENELUX_COLORS, 'Benelux', 'Benelux People')
    # plot_population_chart(
    #     YUGOSLAV_POPULATIONS, YUGOSLAV_COLORS, 'WestBalkan', 'West Balkan People')
    plot_population_chart(
        IRISH_POPULATIONS, IRISH_COLORS, 'Irish', 'Irish People')

