import wbdata
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import geopandas as gpd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

REGION_COLORS = plt.cm.tab20.colors  # Use a colormap with enough distinct colors

# World map configuration
WORLD_URL = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
MAP_PROJECTION = "ESRI:54042"  # Robinson projection
MAP_INSET_POSITION = (0.45, 0.5, 0.45, 0.30)  # (x, y, width, height) in figure coordinates

# ==================== CONFIG ====================
NUM_YEARS = 25  # Number of years to sum net migration
MOST_RECENT_YEAR = 2024  # Most recent year for population data
MOBILE = True

NET_MIGRATION_INDICATOR = 'SM.POP.NETM'  # Net migration indicator
POPULATION_INDICATOR = 'SP.POP.TOTL'     # Population indicator
CACHE_FILE = f"cache/net_migration_{NUM_YEARS}_{MOST_RECENT_YEAR}.pkl"

# Define color constants (matching your style)
BG_COLOR = '#2a2a2a'
ANNO_COLOR = "#848484"
COLORS = {
    'positive': "#99C86B",  # Green for positive net migration
    'negative': "#FF6B6B",  # Red for negative net migration
}

# Manual country code corrections
MANUALLY_ADD = [
    ('France', 'FRA'), ('Norway', 'NOR'), ('Kosovo', 'XKX'),
]

# ==================== REGIONS ====================
REGIONS = [
    {'name': 'Western\nEurope', 'countries': [
        'Austria', 'Belgium', 'Cyprus', 'Denmark', 'Finland', 'France', 'Germany',
        'Greece', 'Ireland', 'Italy', 'Luxembourg', 'Malta', 'Netherlands',
        'Portugal', 'Spain', 'Sweden', 'Norway', 'Switzerland', 'Liechtenstein',
        'Iceland'
    ]},
    {'name': 'Eastern\nEurope', 'countries': [
        'Czechia', 'Latvia', 'Lithuania', 'Estonia', 'Hungary',
        'Poland', 'Romania', 'Slovak Republic', 'Slovenia', 'Bulgaria', 'Croatia',
        'Russian Federation', 'Ukraine', 'Belarus', 'Armenia', 'Georgia', 'Azerbaijan',
        'Albania', 'Serbia', 'Bosnia and Herzegovina', 'North Macedonia', 'Montenegro', 'Moldova'
    ]},
    {'name': 'USA', 'countries': ['United States']},
    {'name': 'CANZUK', 'countries': ['Canada', 'United Kingdom', 'Australia', 'New Zealand']},
    {'name': 'China', 'countries': ['China', 'Hong Kong SAR, China', 'Macao SAR, China']},
    {'name': 'Other\nSouth Asia', 'countries': [
        'Bangladesh', 'Bhutan', 'Nepal', 'Pakistan', 'Sri Lanka', 'Maldives'
    ]},
    {'name': 'India', 'countries': [
        'India'
    ]},
    {'name': 'ASEAN', 'countries': [
        'Brunei Darussalam', 'Cambodia', 'Indonesia', 'Lao PDR', 'Malaysia', 'Myanmar',
        'Philippines', 'Thailand', 'Viet Nam'
    ]},
    {'name': 'Japan &\nS. Korea', 'countries': [
        'Korea, Rep.', 'Japan'
    ]},
    {'name': 'West\nAfrica', 'countries': [
        'Nigeria', 'Ghana', "Cote d'Ivoire", 'Niger', 'Burkina Faso', 'Mali', 'Senegal',
        'Benin', 'Chad', 'Togo', 'Sierra Leone', 'Liberia', 'Guinea', 'Mauritania',
        'Gambia, The', 'Guinea-Bissau', 'Cabo Verde'
    ]},
    {'name': 'North\nAfrica', 'countries': [
        'Egypt, Arab Rep.', 'Libya', 'Morocco', 'Algeria', 'Tunisia'
    ]},
    {'name': 'East\nAfrica', 'countries': [
        'Ethiopia', 'Kenya', 'Tanzania', 'Uganda', 'Rwanda', 'South Sudan',
        'Somalia, Fed. Rep.', 'Djibouti', 'Eritrea', 'Burundi', 'Comoros', 'Seychelles'
    ]},
    {'name': 'Central\nAfrica', 'countries': [
        'Angola', 'Cameroon', 'Chad', 'Congo, Dem. Rep.', 'Congo, Rep.',
        'Gabon', 'Equatorial Guinea', 'Central African Republic', 
        'Sao Tome and Principe'
    ]},
    {'name': 'South\nAfrica', 'countries': [
        'South Africa', 'Mozambique', 'Malawi', 'Zambia', 'Zimbabwe',
        'Botswana', 'Namibia', 'Lesotho', 'Eswatini'
    ]},
    {'name': 'Middle\nEast', 'countries': [
        'Saudi Arabia', 'Iran, Islamic Rep.', 'Iraq', 'Israel', 'Jordan', 'Lebanon', 'Oman', 'Qatar',
        'United Arab Emirates', 'Bahrain', 'Kuwait', 'Syrian Arab Republic', 'Yemen, Rep.',
        'West Bank and Gaza', 'Turkiye', 'Afghanistan'
    ]},
    {'name': 'South\nAmerica', 'countries': [
        'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 
        'Ecuador', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela, RB'
    ]},
    {'name': 'Central\nAmerica', 'countries': [
        'Costa Rica', 'Cuba', 'Dominican Republic', 'Ecuador', 'El Salvador', 'Guatemala',
        'Honduras', 'Mexico', 'Nicaragua', 'Panama', 
        'Haiti', 'Jamaica', 'Trinidad and Tobago', 'Belize'
    ]}
]

SIMPLIFIED_REGIONS = [
    {'name': 'Western\nEurope', 'countries': [
        'Austria', 'Belgium', 'Cyprus', 'Denmark', 'Finland', 'France', 'Germany',
        'Greece', 'Italy', 'Luxembourg', 'Malta', 'Netherlands',
        'Portugal', 'Spain', 'Sweden', 'Norway', 'Switzerland', 'Liechtenstein',
        'Iceland'
    ]},
    {'name': 'Eastern\nEurope', 'countries': [
        'Czechia', 'Latvia', 'Lithuania', 'Estonia', 'Hungary',
        'Poland', 'Romania', 'Slovak Republic', 'Slovenia', 'Bulgaria', 'Croatia',
        'Russian Federation', 'Ukraine', 'Belarus', 'Armenia', 'Georgia', 'Azerbaijan',
        'Albania', 'Serbia', 'Bosnia and Herzegovina', 'North Macedonia', 'Montenegro',
        'Moldova', 'Kosovo'
    ]},
    {'name': 'Anglo\nSphere', 'countries': ['United States', 'Canada', 'United Kingdom', 'Australia', 'New Zealand', 'Ireland']},
    {'name': 'China', 'countries': ['China', 'Hong Kong SAR, China', 'Macao SAR, China']},
    {'name': 'South Asia', 'countries': [
        'India', 'Bangladesh', 'Bhutan', 'Nepal', 'Pakistan', 'Sri Lanka', 'Maldives'
    ]},
    {'name': 'ASEAN', 'countries': [
        'Brunei Darussalam', 'Cambodia', 'Indonesia', 'Lao PDR', 'Malaysia', 'Myanmar',
        'Philippines', 'Thailand', 'Viet Nam'
    ]},
    {'name': 'Japan &\nS. Korea', 'countries': [
        'Korea, Rep.', 'Japan'
    ]},
    {'name': 'Africa', 'countries': [
        'Nigeria', 'South Africa', 'Ethiopia', 'Kenya', 'Tanzania', 'Uganda', 'Ghana', 'Angola',
        'Mozambique', "Cote d'Ivoire", 'Cameroon', 'Niger', 'Burkina Faso', 'Mali', 'Malawi',
        'Zambia', 'Senegal', 'Zimbabwe', 'Rwanda', 'Benin', 'Chad', 'South Sudan', 'Togo', 'Sudan'
        'Sierra Leone', 'Liberia', 'Botswana', 'Namibia', 'Somalia, Fed. Rep.', 'Congo, Dem. Rep.', 'Congo, Rep.',
        'Gabon', 'Guinea', 'Mauritania', 'Equatorial Guinea', 'Central African Republic', 'Lesotho',
        'Eswatini', 'Djibouti', 'Eritrea', 'Gambia, The', 'Guinea-Bissau', 'Burundi', 'Cabo Verde',
        'Comoros', 'Sao Tome and Principe', 'Seychelles', 'Madagascar',
        'Egypt, Arab Rep.', 'Libya', 'Morocco', 'Algeria', 'Tunisia'
    ]},
    {'name': 'Mid.\nEast', 'countries': [
        'Saudi Arabia', 'Iran, Islamic Rep.', 'Iraq', 'Israel', 'Jordan', 'Lebanon', 'Oman', 'Qatar',
        'United Arab Emirates', 'Bahrain', 'Kuwait', 'Syrian Arab Republic', 'Yemen, Rep.',
        'West Bank and Gaza', 'Turkiye', 'Afghanistan'
    ]},
    {'name': 'Latin\nAmerica', 'countries': [
        'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Costa Rica', 'Cuba',
        'Dominican Republic', 'Ecuador', 'El Salvador', 'Guatemala', 'Honduras',
        'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela, RB',
        'Haiti', 'Jamaica', 'Trinidad and Tobago', 'Belize', 'Guyana', 'Suriname'
    ]}
]

# ==================== FUNCTIONS ====================
def fetch_net_migration_data():
    """Fetch net migration and population data from World Bank."""
    if os.path.exists(CACHE_FILE):
        return pd.read_pickle(CACHE_FILE)

    indicators = {NET_MIGRATION_INDICATOR: 'NetMigration', POPULATION_INDICATOR: 'Population'}
    df = wbdata.get_dataframe(indicators).reset_index()
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year >= (MOST_RECENT_YEAR - NUM_YEARS)]
    df['Country'] = df['country'].str.strip()

    # Save cache
    os.makedirs("cache", exist_ok=True)
    df.to_pickle(CACHE_FILE)
    return df

def clean_data(df):
    """Clean and filter data."""
    # Only filter out aggregate entities, keep all years
    df = df[df['Country'].apply(is_country_entity)]
    return df

def is_country_entity(name):
    """Check if the name is a country (not an aggregate)."""
    aggregate_keywords = ['&', 'IDA', 'IBRD', 'income', 'demographic', 'OECD', 'HIPC', 'small states', 'classification', 'members', 'countries', 'excluding', '(US)']
    geographic_aggregates = ['World', 'Arab World', 'Euro area', 'European Union', 'Africa Eastern and Southern', 'Africa Western and Central', 'East Asia & Pacific', 'Europe & Central Asia', 'Latin America & Caribbean', 'Middle East, North Africa', 'North America', 'South Asia', 'Sub-Saharan Africa', 'Caribbean small states', 'Pacific island small states', 'Central Europe and the Baltics', 'Other small states', 'Fragile and conflict affected situations']
    return not any(name.startswith(geo) for geo in geographic_aggregates) and not any(k in name for k in aggregate_keywords)

def assign_regions(df, regions):
    """Assign each country to a region."""
    df = df.copy()
    df['Region'] = 'Other'
    for region in regions:
        df.loc[df['Country'].isin(region['countries']), 'Region'] = region['name']
    return df

def aggregate_net_migration(df, regions):
    """Aggregate net migration by region."""
    df = assign_regions(df, regions)
    
    # Get most recent population for each country
    df_recent = df[df['date'].dt.year == MOST_RECENT_YEAR].copy()
    
    # Sum net migration over all years, get most recent population per country
    df_country = df.groupby(['Region', 'Country']).agg({
        'NetMigration': 'sum',  # Sum migration over NUM_YEARS
    }).reset_index()
    
    # Merge with most recent population
    df_country = df_country.merge(
        df_recent[['Country', 'Population']], 
        on='Country', 
        how='left'
    )
    
    # Now aggregate by region (sum both migration AND population)
    df_sum = df_country.groupby('Region').agg({
        'NetMigration': 'sum',  # Total migration for region
        'Population': 'sum'      # Total population for region
    }).reset_index()
    
    # Calculate rate per 1000 people per year
    df_sum['NetMigrationRate'] = (df_sum['NetMigration'] / df_sum['Population'] / NUM_YEARS) * 1000
    
    return df_sum

def create_region_country_mapping(regions):
    """Create a dictionary mapping country names to region indices."""
    country_to_region = {}
    for i, region in enumerate(regions):
        for country in region['countries']:
            country_to_region[country] = i
    return country_to_region

def create_iso3_to_region_mapping(regions):
    """Create a dictionary mapping ISO3 codes to region indices using wbdata metadata."""
    # Get World Bank country metadata
    countries = wbdata.get_countries()
    
    # Create name to ISO3 mapping
    name_to_iso3 = {}
    for c in countries:
        name_to_iso3[c["name"]] = c["id"]
    
    # Create ISO3 to region mapping
    iso3_to_region = {}
    for i, region in enumerate(regions):
        for country_name in region['countries']:
            if country_name in name_to_iso3:
                iso3_to_region[name_to_iso3[country_name]] = i
    
    return iso3_to_region

def get_world_map_data(regions, projection_epsg):
    """Load world map and assign region colors."""
    world = gpd.read_file(WORLD_URL)
    
    # Add manual corrections
    for (name, code) in MANUALLY_ADD:
        world.loc[world['name'] == name, 'ISO3166-1-Alpha-3'] = code
    
    # Create mapping from ISO3 to region index
    iso3_to_region = create_iso3_to_region_mapping(regions)
    
    # Map ISO3 codes to region indices
    world['RegionIndex'] = -1  # Default for countries not in any region
    
    # Use ISO3 codes for matching
    if 'ISO3166-1-Alpha-3' in world.columns:
        for iso3, region_idx in iso3_to_region.items():
            world.loc[world['ISO3166-1-Alpha-3'] == iso3, 'RegionIndex'] = region_idx
    
    # Reproject to desired projection
    world = world.to_crs(projection_epsg)
    
    return world

def add_world_map_inset(fig, ax, df_sorted, regions, position=MAP_INSET_POSITION):
    """Add a world map inset showing regions in their assigned colors."""
    # Create inset axis
    map_ax = inset_axes(
        ax,
        width="100%",
        height="100%",
        bbox_to_anchor=position,
        bbox_transform=fig.transFigure,
    )
    
    # Load and prepare world map data
    world_data = get_world_map_data(regions, MAP_PROJECTION)
    
    # Create a mapping from region name to its sorted position (and thus color)
    region_to_color_idx = {}
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        region_to_color_idx[row['Region']] = i
    
    # Create a mapping from original region index to color index
    region_idx_to_color = {}
    for i, region in enumerate(regions):
        region_name = region['name']
        if region_name in region_to_color_idx:
            region_idx_to_color[i] = region_to_color_idx[region_name]
    
    # Plot each region with its corresponding color
    for region_idx in range(len(regions)):
        region_countries = world_data[world_data['RegionIndex'] == region_idx]
        if not region_countries.empty and region_idx in region_idx_to_color:
            color_idx = region_idx_to_color[region_idx]
            region_countries.plot(
                ax=map_ax,
                color=REGION_COLORS[color_idx % len(REGION_COLORS)],
                linewidth=0,  # No borders between countries
                edgecolor='none'
            )
    
    # Plot countries not in any region in gray
    other_countries = world_data[world_data['RegionIndex'] == -1]
    if not other_countries.empty:
        other_countries.plot(
            ax=map_ax,
            color='#404040',
            linewidth=0,
            edgecolor='none'
        )
    
    # Style the map
    map_ax.axis('off')
    map_ax.set_xlim(-11000000, 14300000)
    map_ax.set_ylim(-6500000, 9500000)
    
    # Add white border around the inset
    for spine in map_ax.spines.values():
        spine.set_visible(True)
        spine.set_color('white')
        spine.set_linewidth(2)
    
    # Ensure the inset is rendered on top
    map_ax.set_zorder(1000)
    
    return map_ax

def plot_net_migration(df, regions):
    """
    Vertical bars, width scaled by population, labels above/below bars.
    """
    os.makedirs("visualizations", exist_ok=True)

    # Sort regions by migration rate (highest first)
    df = df.sort_values("NetMigrationRate", ascending=False).reset_index(drop=True)

    # ---- Scale widths so the largest pop = width 1.0 ----
    max_pop = df["Population"].max()
    df["Width"] = df["Population"] / max_pop

    # ---- Compute left positions with fixed gap ----
    gap = 0.05
    positions = np.cumsum([0] + (df["Width"] + gap).tolist()[:-1])

    # ---- Figure ----
    if (MOBILE):
        fig, ax = plt.subplots(figsize=(12, 12))
    else:
        fig, ax = plt.subplots(figsize=(16, 10))
        
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # ---- Plot bars ----
    for i, (_, row) in enumerate(df.iterrows()):
        x0 = positions[i]
        height = row["NetMigrationRate"]
        # Use the region's index to select a color
        color = REGION_COLORS[i % len(REGION_COLORS)]  # Cycle through colors if needed
        ax.bar(
            x0,
            height,
            width=row["Width"],
            color=color,
            align="edge"
        )


    # ---- Add horizontal guideline at zero ----
    ax.axhline(y=0, color=ANNO_COLOR, linewidth=1, linestyle='-', alpha=1.0, zorder=1)
    
    # ---- Add labels on opposite side of x-axis ----
    
    # Add subtle horizontal grid
    ax.yaxis.grid(True, alpha=1.0, color=ANNO_COLOR, linewidth=0.5, zorder=-1)
    
    for i, (_, row) in enumerate(df.iterrows()):
        x0 = positions[i]
        width = row["Width"]
        center = x0 + width / 2
        height = row["NetMigrationRate"]
        pop_millions = row['Population'] / 1e6

        line_gap = 0.04
        text_gap = 0.02
        # Position label BELOW x-axis for positive bars, ABOVE x-axis for negative bars
        if height >= 0:
            # Positive bar goes up, so label goes below (negative y)
            line_y = -line_gap
            label_y = line_y - text_gap
            va = "top"
        else:
            # Negative bar goes down, so label goes above (positive y)
            line_y = line_gap
            label_y = line_y + text_gap
            va = "bottom"
            
        if(row['Region'] == 'Japan &\nS. Korea'):
            label_y = line_y - 0.3
        
        # Width line
        ax.plot(
            [x0+0.008, x0 + width-0.008],
            [line_y, line_y],
            color=ANNO_COLOR,
            linewidth=2,
            alpha=1.0
        )
        
        # Region name
        ax.text(
            center,
            label_y,
            f"{row['Region']}\n{pop_millions:.0f}M",
            ha="center",
            va=va,
            fontsize=11,
            color="white"
            # weight="bold"
        )

    # ---- Axis cleanup ----
    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.set_ylabel(f"Average Net Migration Rate (per 1000 people per year)", fontsize=20, color="white")
    ax.set_xlabel(f"Region Population {MOST_RECENT_YEAR}", fontsize=20, color="white")
    ax.xaxis.set_label_coords(0.5, 0.06)
    ax.set_title(f"Net Migration Rate by Region ({MOST_RECENT_YEAR-NUM_YEARS+1}-{MOST_RECENT_YEAR})",
                 fontsize=28, color="white", y=0.90)

    ax.tick_params(axis="y", colors="white", labelsize=12)
    
    # Format y-axis with one decimal place
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x)}'))

    # Remove all vertical/horizontal spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---- Add world map inset (pass the sorted dataframe) ----
    df_for_map = df.copy()  # df is already sorted at this point
    add_world_map_inset(fig, ax, df_for_map, regions, position=MAP_INSET_POSITION)

    plt.tight_layout()
    if (MOBILE):
        plt.savefig(f"visualizations/NetMigrationRate_{NUM_YEARS}Years_MOBILE.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(f"visualizations/NetMigrationRate_{NUM_YEARS}Years.png", dpi=300, bbox_inches='tight')


# ==================== MAIN ====================
if __name__ == "__main__":
    print('Fetching data...')
    df = fetch_net_migration_data()
    df = clean_data(df)
    
    print(f"\nTotal countries found: {len(df)}")
    if (MOBILE):
        df_agg = aggregate_net_migration(df, SIMPLIFIED_REGIONS)
    else:
        df_agg = aggregate_net_migration(df, REGIONS)
        
    print('\nRegion Summary:')
    print(df_agg[['Region', 'Population', 'NetMigration', 'NetMigrationRate']].to_string())
    
    print('\nRendering...')
    if (MOBILE):
        plot_net_migration(df_agg, SIMPLIFIED_REGIONS)
    else:
        plot_net_migration(df_agg, REGIONS)