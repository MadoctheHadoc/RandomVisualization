import pandas as pd
import wbdata
import os
import matplotlib.pyplot as plt
import geopandas as gpd
import numpy as np
import pycountry
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Configuration
DATA_YEAR = 2021
INDICATORS = {
    'NV.IND.MANF.ZS': 'Manufacturing_pct',
    'NY.GDP.TOTL.RT.ZS': 'ResourceRents_pct'
}
CACHE_FILE = f"cache/resource_manufacturing_{DATA_YEAR}.pkl"
# COLORS = [
#     "#e8e8e8", "#e4acac", "#c85a5a",
#     "#b0d5df", "#ad9ea5", "#985356",
#     "#64acbe", "#627f8c", "#574249",
# ]
COLORS = [
    "#e8e8e8", "#ddb9b9", "#d38a8a", "#c85a5a",
    "#bdd4da", "#b2a8ae", "#aa7d82", "#a15256",
    "#91c0cc", "#8998a3", "#83717b", "#7c4a53",
    "#64acbe", "#608898", "#5b6574", "#574249",
]
ANNO_COLOR = "#464646"

EPS = 0.000000001
MAN_BINS = [-EPS, 10, 15, 20, 100 + EPS]
RES_BINS = [-EPS, 5, 10, 25, 100 + EPS]

OLDER_DATA = [  # Manufacturing, resources
    ("VEN", "Venezuela", 12.0, 11.8),
    ("ERI", "Eritrea", 5, 27),
    ("SSD", "South Sudan", 4, 13.1),
    ("TKM", "Turkmenistan", 20, 13.5),
    ("KWT", "Kuwait", 11, 29.3)
]

def fetch_world_bank_data():
    """Fetch World Bank data with proper ISO3 codes attached."""
    os.makedirs('cache', exist_ok=True)
    if os.path.exists(CACHE_FILE):
        return pd.read_pickle(CACHE_FILE)

    df_wb = wbdata.get_dataframe(INDICATORS).reset_index()
    df_wb = df_wb[df_wb['date'] == str(DATA_YEAR)]
    df_wb = df_wb.drop(columns=['date']).reset_index(drop=True)
    df_wb.rename(columns={'country': 'CountryWB'}, inplace=True)

    countries = wbdata.get_countries()
    meta_records = []
    for c in countries:
        meta_records.append({
            "ISO3": c["id"],
            "Country": c["name"]
        })
    df_meta = pd.DataFrame(meta_records)
    df_wb = df_wb.merge(df_meta, left_on='CountryWB', right_on='Country', how='left')
    df_wb = df_wb[['ISO3', 'Country', 'Manufacturing_pct', 'ResourceRents_pct']]
    df_wb['ISO3'] = df_wb['ISO3'].astype(str)
    
    # --- Manually add 2014 data for some countries ---
    df_wb = pd.concat([df_wb, pd.DataFrame(OLDER_DATA, columns=df_wb.columns)], ignore_index=True)   
    df_wb.to_pickle(CACHE_FILE)
    return df_wb

def clean_data(df):
    """Clean and prepare data for visualization, filling nulls with 0."""
    df_clean = df.copy().dropna(subset=['Manufacturing_pct', 'ResourceRents_pct'], how='all')
    df_clean = df_clean.fillna(0)
    return df_clean

def iso3_to_numeric(iso3):
    """Convert ISO3 code to numeric (UN M49)."""
    try:
        return int(pycountry.countries.get(alpha_3=iso3).numeric)
    except:
        return None

def assign_bivariate_class(row, man_bins, res_bins):
    """Assign a bivariate class (0-15) based on discretized values."""
    man_cat = np.digitize(row['Manufacturing_pct'], man_bins)
    res_cat = np.digitize(row['ResourceRents_pct'], res_bins)
    return (man_cat - 1) * 4 + (res_cat - 1)  # Classes 0-15

def create_legend(ax, colors, man_bins, res_bins):
    """Create a 3x3 bivariate legend grid with ranges labeled inside each cell."""
    # Coordinates for a 3x3 grid
    legend_ax = inset_axes(ax, width="20%", height="20%", loc='lower left')
    label_bins = (res_bins, man_bins)    
    labels = [[] for ignored in range(len(label_bins))]
    for i in range(len(label_bins)):
        bins = label_bins[i]
        labels[i].append(f"<{bins[1]}%")
        labels[i].extend([f"{bins[i]}-{bins[i+1]}%" for i in range(1, len(bins) - 2)])
        labels[i].append(f">{bins[len(bins) - 2]}%")

    res_length = len(res_bins) - 1
    man_length = len(man_bins) - 1
    # Draw colored squares
    for i in range(res_length):
        for j in range(man_length):
            color_idx = i * res_length + j
            legend_ax.add_patch(plt.Rectangle((j, 3-i), 1, 1, facecolor=colors[color_idx], edgecolor='white'))

    # Add column labels (top)
    for j, label in enumerate(labels[0]):
        legend_ax.text(j + 0.5, res_length + 0.05, label, ha='center', va='bottom', fontsize=10, color=ANNO_COLOR)

    # Add row labels (left)
    for i, label in enumerate(reversed(labels[1])):
        legend_ax.text(-0.05, i + 0.5, label, ha='right', va='center', fontsize=10, color=ANNO_COLOR)

    # Axis limits and styling
    legend_ax.set_xlim(0, res_length)
    legend_ax.set_ylim(0, man_length)
    legend_ax.set_aspect('equal')
    legend_ax.axis('off')

def annotate_map(ax):
    locations = [
        (-30, 45, False),
        (-8, 0, False),
        (123, 33, False),
        (66, 2, False),
        (84, 41, True),
        (115, -32, True),
        (-92, 6, False),
        (60, 70, True)
    ]  # x, y coordinates for annotations
    
    labels = [
        "Europe and Northern America\nare service-based economies\nwith only some manufacturing",
        "Africa has many very\nextractive economies",
        "The Indo-Pacific has very\nfew natural resources but\nremains the world's factory",
        "India & Pakistan lean more on\nservices and less on industry\nthan their neighbours",
        "China is the largest\nmanufacturer in the world",
        "Australia & Norway are\n(unsually) mature and\nextractive economies",
        "Latin America is varied\nwith some manufacturing\nand some mineral extraction",
        "Petrostates like Russia, Iran and\nthe Gulf states can leverage their\ncheap energy to expand manufacturing",
    ]
    
    multiplier = 100000
    for (x, y, white), text in zip(locations, labels):
        ax.text(multiplier * x, multiplier *  y, text,
                ha='center', va='center',
                fontsize=8,
                color=('white' if white else ANNO_COLOR))

def get_world_data(df_cleaned, projection_epsg):
    world_url = "https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson"
    world = gpd.read_file(world_url)

    # Merge World Bank data
    df_cleaned['iso_num'] = df_cleaned['ISO3'].apply(iso3_to_numeric)
    world_data = world.merge(df_cleaned, left_on='id', right_on='ISO3', how='left')

    # Fill NaN values
    world_data['Manufacturing_pct'] = world_data['Manufacturing_pct'].fillna(0)
    world_data['ResourceRents_pct'] = world_data['ResourceRents_pct'].fillna(0)

    # Discretize into 3 categories
    world_data['Manufacturing_cat'] = pd.cut(world_data['Manufacturing_pct'], bins=MAN_BINS, labels=False)
    world_data['ResourceRents_cat'] = pd.cut(world_data['ResourceRents_pct'], bins=RES_BINS, labels=False)

    # Assign bivariate class
    world_data['Bivariate_Class'] = world_data.apply(
        lambda row: assign_bivariate_class(row, MAN_BINS, RES_BINS), axis=1
    )

    # Reproject to desired projection
    world_data = world_data.to_crs(projection_epsg)
    return world_data

def visualize(df_cleaned, projection_epsg="ESRI:54042"):
    """Create a bivariate choropleth map with a different projection, legend, and bin labels."""
    world_data = get_world_data(df_cleaned, projection_epsg)

    # Map class to colors
    class_to_color = {i: COLORS[i] for i in range(len(COLORS))}

    # Plot with explicit colors
    fig, ax = plt.subplots(figsize=(16, 10))
    for cls in range(len(COLORS)):
        if cls in world_data['Bivariate_Class'].values:
            world_data[world_data['Bivariate_Class'] == cls].plot(
                color=class_to_color[cls],
                ax=ax,
                legend=False,
                alpha=1.0,
                edgecolor=ANNO_COLOR
            )

    # Add legend as a separate inset axis
    create_legend(ax, COLORS, MAN_BINS, RES_BINS)

    # Annotations
    annotate_map(ax)

    # Styling
    ax.set_title(
        f'Are countries Making or Taking?',
        fontsize=24
    )
    # Manufacturing & Resource Extraction in {DATA_YEAR}
    ax.axis('off')

    ax.set_xlim(-11000000, 15000000)  # meters in Robinson projection
    ax.set_ylim(-6500000, 9500000)    # cut off poles

    # Save
    output_path = f'visualizations/ManufacturingResourceExtraction{DATA_YEAR}.png'
    plt.savefig(output_path, dpi=500, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Map saved to: {output_path}")

if __name__ == "__main__":
    df = fetch_world_bank_data()
    df_cleaned = clean_data(df)
    visualize(df_cleaned)
