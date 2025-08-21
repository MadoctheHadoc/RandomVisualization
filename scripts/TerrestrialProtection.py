import wbdata
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# ==================== CONFIG ====================
DATA_YEAR = 2022
CACHE_DIR = "cache"
VISUALIZATIONS_DIR = "visualizations"

# Create directories if they don't exist
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# World Bank indicators for protected land and related data
INDICATORS = {
    'ER.LND.PTLD.ZS': 'ProtectedLand_Pct',           # Terrestrial protected areas (% of total land area)
    'AG.LND.TOTL.K2': 'TotalLand_KM2',               # Land area (sq. km)
}

# Define color palette
BG_COLOR = "#ededed"
ANNO_COLOR = "#777777"
TEXT_COLOR = "#4D4D4D"

# Variables woah
USE_SIMPLIFIED_REGIONS = True

# ==================== REGIONS ====================
SIMPLIFIED_REGIONS = [
    {'name': 'North America', 'countries': [
        'United States', 'Canada', 'Greenland',
        'Mexico', 'Belize', 'Costa Rica', 'El Salvador', 'Guatemala', 'Honduras', 'Nicaragua', 'Panama',
        'Antigua and Barbuda', 'Bahamas, The', 'Barbados', 'Cuba', 'Dominica',
        'Dominican Republic', 'Grenada', 'Haiti', 'Jamaica', 'St. Kitts and Nevis',
        'St. Lucia', 'St. Vincent and the Grenadines', 'Trinidad and Tobago',
        'Aruba', 'British Virgin Islands', 'Cayman Islands', 'Curacao',
        'Puerto Rico (US)', 'Sint Maarten (Dutch part)', 'St. Martin (French part)',
        'Turks and Caicos Islands', 'Virgin Islands (U.S.)', 'Bermuda'
    ]},
    {'name': 'Europe', 'countries': [
        'Austria', 'Belgium', 'Cyprus', 'Denmark', 'Finland', 'France', 'Germany',
        'Greece', 'Ireland', 'Italy', 'Luxembourg', 'Malta', 'Netherlands',
        'Portugal', 'Spain', 'Sweden', 'Norway', 'Switzerland', 'Liechtenstein',
        'Iceland', 'Czechia', 'Latvia', 'Lithuania', 'Estonia', 'Hungary', 'San Marino',
        'Poland', 'Romania', 'Slovak Republic', 'Slovenia', 'Bulgaria', 'Croatia',
        'United Kingdom', 'Andorra', 'Monaco', 'Gibraltar', 'Isle of Man', 'Channel Islands',
        'Albania', 'Armenia', 'Azerbaijan', 'Belarus', 'Bosnia and Herzegovina',
        'Georgia', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia', 'Serbia', 'Ukraine', 'Faroe Islands', 'Small states', 'Not classified'
    ]},
    {'name': 'Russia', 'countries': [
        'Russian Federation'
    ]},
    {'name': 'Asia', 'countries': [
        'Kazakhstan', 'Uzbekistan', 'Turkmenistan', 'Tajikistan', 'Kyrgyz Republic', 'Afghanistan',
        'China', 'Korea, Rep.', 'Hong Kong SAR, China', 'Macao SAR, China', 'Japan', 'Mongolia', 'Korea, Dem. People\'s Rep.',
        'India', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Nepal', 'Bhutan', 'Maldives',
        'Brunei Darussalam', 'Cambodia', 'Indonesia', 'Lao PDR', 'Malaysia', 'Myanmar',
        'Philippines', 'Thailand', 'Viet Nam', 'Singapore', 'Timor-Leste'
    ]},
    {'name': 'South\nAmerica', 'countries': [
        'Argentina', 'Bolivia', 'Chile', 'Colombia', 'Ecuador', 'Guyana',
        'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela, RB', 'Brazil'
    ]},    
    {'name': 'MENA', 'countries': [
        'Saudi Arabia', 'Iran, Islamic Rep.', 'Turkiye', 'Iraq', 'Israel', 'Jordan', 'Lebanon',
        'Oman', 'Qatar', 'United Arab Emirates', 'Bahrain', 'Kuwait', 'Syrian Arab Republic',
        'Yemen, Rep.', 'West Bank and Gaza',
        'Egypt, Arab Rep.', 'Libya', 'Morocco', 'Algeria', 'Tunisia', 'Sudan'
    ]},
    {'name': 'Sub-Saharan Africa', 'countries': [
        'Nigeria', 'Ghana', "Cote d'Ivoire", 'Senegal', 'Mali', 'Burkina Faso', 'Niger', 'Mauritania',
        'Benin', 'Togo', 'Sierra Leone', 'Liberia', 'Guinea', 'Guinea-Bissau', 'Gambia, The',
        'Cabo Verde',
        'Ethiopia', 'Kenya', 'Tanzania', 'Uganda', 'Rwanda', 'Burundi', 'Somalia', 'Djibouti',
        'Eritrea', 'South Sudan', 'Seychelles', 'Comoros', 'Mauritius', 'Madagascar',
        'Cameroon', 'Central African Republic', 'Chad', 'Congo, Dem. Rep.', 'Congo, Rep.',
        'Equatorial Guinea', 'Gabon', 'Sao Tome and Principe',
        'South Africa', 'Botswana', 'Namibia', 'Zambia', 'Zimbabwe', 'Malawi', 'Mozambique',
        'Angola', 'Lesotho', 'Eswatini'
    ]},
    {'name': 'Oceania', 'countries': [
        'Australia', 'New Zealand', 'Papua New Guinea', 'Fiji', 'Solomon Islands', 'Vanuatu',
        'Samoa', 'Kiribati', 'Micronesia, Fed. Sts.', 'Tonga', 'Palau', 'Marshall Islands',
        'Tuvalu', 'Nauru', 'American Samoa', 'French Polynesia', 'New Caledonia',
        'Guam', 'Northern Mariana Islands'
    ]}
]

REGIONS = [
    {'name': 'United States', 'countries': [
        'United States'
    ]},
    {'name': 'Canada', 'countries': [
        'Canada'
    ]},
    {'name': 'Greenland', 'countries': [
        'Greenland'
    ]},
    {'name': 'Europe', 'countries': [
        'Austria', 'Belgium', 'Cyprus', 'Denmark', 'Finland', 'France', 'Germany',
        'Greece', 'Ireland', 'Italy', 'Luxembourg', 'Malta', 'Netherlands',
        'Portugal', 'Spain', 'Sweden', 'Norway', 'Switzerland', 'Liechtenstein',
        'Iceland', 'Czechia', 'Latvia', 'Lithuania', 'Estonia', 'Hungary', 'San Marino',
        'Poland', 'Romania', 'Slovak Republic', 'Slovenia', 'Bulgaria', 'Croatia',
        'United Kingdom', 'Andorra', 'Monaco', 'Gibraltar', 'Isle of Man', 'Channel Islands',
        'Albania', 'Armenia', 'Azerbaijan', 'Belarus', 'Bosnia and Herzegovina',
        'Georgia', 'Kosovo', 'Moldova', 'Montenegro', 'North Macedonia', 'Serbia', 'Ukraine'
    ]},
    {'name': 'Russia', 'countries': [
        'Russian Federation'
    ]},
    {'name': 'Central Asia', 'countries': [
        'Kazakhstan', 'Uzbekistan', 'Turkmenistan', 'Tajikistan', 'Kyrgyz Republic', 'Afghanistan'
    ]},
    {'name': 'East\nAsia', 'countries': [
        'China', 'Korea, Rep.', 'Hong Kong SAR, China', 'Macao SAR, China', 'Japan', 'Mongolia', 'Korea, Dem. People\'s Rep.',
    ]},
    {'name': 'South\nAsia', 'countries': [
        'India', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Nepal', 'Bhutan', 'Maldives'
    ]},
    {'name': 'Southeast\nAsia', 'countries': [
        'Brunei Darussalam', 'Cambodia', 'Indonesia', 'Lao PDR', 'Malaysia', 'Myanmar',
        'Philippines', 'Thailand', 'Viet Nam', 'Singapore', 'Timor-Leste'
    ]},
    {'name': 'South\nAmerica', 'countries': [
        'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador', 'Guyana',
        'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela, RB'
    ]},
    {'name': 'Middle\nEast', 'countries': [
        'Saudi Arabia', 'Iran, Islamic Rep.', 'Turkiye', 'Iraq', 'Israel', 'Jordan', 'Lebanon',
        'Oman', 'Qatar', 'United Arab Emirates', 'Bahrain', 'Kuwait', 'Syrian Arab Republic',
        'Yemen, Rep.', 'West Bank and Gaza'
    ]},
    {'name': 'North\nAfrica', 'countries': [
        'Egypt, Arab Rep.', 'Libya', 'Morocco', 'Algeria', 'Tunisia', 'Sudan'
    ]},
    {'name': 'West\nAfrica', 'countries': [
        'Nigeria', 'Ghana', "Cote d'Ivoire", 'Senegal', 'Mali', 'Burkina Faso', 'Niger', 'Mauritania',
        'Benin', 'Togo', 'Sierra Leone', 'Liberia', 'Guinea', 'Guinea-Bissau', 'Gambia, The',
        'Cabo Verde'
    ]},
    {'name': 'East\nAfrica', 'countries': [
        'Ethiopia', 'Kenya', 'Tanzania', 'Uganda', 'Rwanda', 'Burundi', 'Somalia', 'Djibouti',
        'Eritrea', 'South Sudan', 'Seychelles', 'Comoros', 'Mauritius', 'Madagascar'
    ]},
    {'name': 'Central\nAfrica', 'countries': [
        'Cameroon', 'Central African Republic', 'Chad', 'Congo, Dem. Rep.', 'Congo, Rep.',
        'Equatorial Guinea', 'Gabon', 'Sao Tome and Principe'
    ]},
    {'name': 'Southern\nAfrica', 'countries': [
        'South Africa', 'Botswana', 'Namibia', 'Zambia', 'Zimbabwe', 'Malawi', 'Mozambique',
        'Angola', 'Lesotho', 'Eswatini'
    ]},
    # {'name': 'Central America\n& Caribbean', 'countries': [
    #     'Mexico', 'Belize', 'Costa Rica', 'El Salvador', 'Guatemala', 'Honduras', 'Nicaragua', 'Panama',
    #     'Antigua and Barbuda', 'Bahamas, The', 'Barbados', 'Cuba', 'Dominica',
    #     'Dominican Republic', 'Grenada', 'Haiti', 'Jamaica', 'St. Kitts and Nevis',
    #     'St. Lucia', 'St. Vincent and the Grenadines', 'Trinidad and Tobago',
    #     'Aruba', 'British Virgin Islands', 'Cayman Islands', 'Curacao',
    #     'Puerto Rico (US)', 'Sint Maarten (Dutch part)', 'St. Martin (French part)',
    #     'Turks and Caicos Islands', 'Virgin Islands (U.S.)'
    # ]},
    {'name': 'Oceania', 'countries': [
        'Australia', 'New Zealand', 'Papua New Guinea', 'Fiji', 'Solomon Islands', 'Vanuatu',
        'Samoa', 'Kiribati', 'Micronesia, Fed. Sts.', 'Tonga', 'Palau', 'Marshall Islands',
        'Tuvalu', 'Nauru', 'American Samoa', 'French Polynesia', 'New Caledonia',
        'Guam', 'Northern Mariana Islands'
    ]},
]

# ==================== FUNCTIONS ====================
CACHE_FILE = f"{CACHE_DIR}/protected_land_{DATA_YEAR}.pkl"
BIODIVERSITY_FILE = "data/NationalSpeciesSummary.csv"

def fetch_world_bank_data(indicators, year):
    """Fetch data from World Bank API and merge with biodiversity data, using caching."""
    if os.path.exists(CACHE_FILE):
        print(f"Loading cached data from {CACHE_FILE}")
        return pd.read_pickle(CACHE_FILE)

    print(f"Fetching data from World Bank API for year {year}...")
    # Fetch World Bank data
    df_wb = wbdata.get_dataframe(indicators).reset_index()
    df_wb['date'] = pd.to_datetime(df_wb['date'])
    df_wb = df_wb[df_wb['date'].dt.year == year]
    df_wb.rename(columns={'country': 'Country'}, inplace=True)
    df_wb['Country'] = df_wb['Country'].str.strip()

    # Load biodiversity data
    df_diversity = pd.read_csv(BIODIVERSITY_FILE)
    print(df_diversity.head())
    df_diversity['Country'] = df_diversity[df_diversity.columns[0]].str.strip()  # Ensure consistency

    # Print biodiversity data for inspection
    print("Biodiversity Data:")
    print(df_diversity.head())

    # Merge datasets on 'Country'
    df_merged = pd.merge(df_wb, df_diversity, on='Country', how='left')

    # Print merged data for inspection
    print("\nMerged Data:")
    print(df_merged.head())

    # Save merged data to cache
    df_merged.to_pickle(CACHE_FILE)
    print(f"Merged data cached to {CACHE_FILE}")

    return df_merged

def is_country_entity(name):
    """Filter out regional aggregates and keep only country entities."""
    aggregate_keywords = [
        '&', 'IDA', 'IBRD', 'income', 'demographic', 'OECD', 'HIPC',
        'small states', 'classification', 'members', 'countries', 'excluding', '(US)'
    ]
    
    # Allow some specific entities
    if any(k in name for k in ['SAR', 'Puerto Rico']):
        return True
    
    if any(k in name for k in aggregate_keywords):
        return False

    geographic_aggregates = [
        'World', 'Arab World', 'Euro area', 'European Union',
        'Africa Eastern and Southern', 'Africa Western and Central',
        'East Asia & Pacific', 'Europe & Central Asia',
        'Latin America & Caribbean', 'Middle East, North Africa',
        'North America', 'South Asia', 'Sub-Saharan Africa',
        'Caribbean small states', 'Pacific island small states',
        'Central Europe and the Baltics', 'Other small states',
        'Fragile and conflict affected situations'
    ]
    return not any(name.startswith(geo) for geo in geographic_aggregates)

def clean_data(df):
    """Clean and prepare the data."""
    df = df.sort_values('date').drop_duplicates(subset=['Country'], keep='last')
    df = df[df['Country'].apply(is_country_entity)]
    
    # Convert to numeric and handle missing values
    numeric_columns = ['ProtectedLand_Pct', 'TotalLand_KM2']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Calculate derived metrics
    df['ProtectedLand_KM2'] = df['ProtectedLand_Pct'] * df['TotalLand_KM2'] / 100
    print(df['ProtectedLand_KM2'])
    
    return df

def assign_protection_category(protection_pct):
    """Assign protection category based on percentage of protected land."""
    if pd.isna(protection_pct):
        return 'No Data'
    elif protection_pct >= 30:
        return 'Very High'
    elif protection_pct >= 20:
        return 'High'
    elif protection_pct >= 10:
        return 'Medium'
    elif protection_pct >= 5:
        return 'Low'
    else:
        return 'Very Low'

def assign_regions(df, regions):
    """Assign countries to regions, aggregate land and species data, and calculate species density."""
    df = df.copy()
    df['Region'] = 'Other'

    # Get the set of unique countries in the dataset
    dataset_countries = set(df['Country'].unique())

    # Collect all countries defined in regions
    all_region_countries = set()
    for region in regions:
        print(region)
        all_region_countries.update(region['countries'])

    # Check for countries in regions not found in the dataset
    missing_in_dataset = all_region_countries - dataset_countries
    if missing_in_dataset:
        print("Warning: The following countries are defined in regions but not found in the dataset:")
        for country in sorted(missing_in_dataset):
            print(f"  - {country}")

    # Assign regions
    for region in regions:
        df.loc[df['Country'].isin(region['countries']), 'Region'] = region['name']

    # Check for countries in the dataset not assigned to any region
    unassigned_countries = dataset_countries - all_region_countries
    if unassigned_countries:
        print("\nWarning: The following countries in the dataset are not assigned to any region (will be 'Other'):")
        for country in sorted(unassigned_countries):
            print(f"  - {country}")

    df['EndemicSpecies'] = (
        df['Number of endemics at 100%'] * df['TotalLand_KM2']
    )
    # Group by region and aggregate data
    df_regions = df.groupby('Region').agg({
        'TotalLand_KM2': 'sum',
        'ProtectedLand_KM2': 'sum',
        'EndemicSpecies': 'sum',
    }).reset_index()

    # Calculate protected land percentage
    df_regions['ProtectedLand_Pct'] = (
        df_regions['ProtectedLand_KM2'] / df_regions['TotalLand_KM2'] * 100
    )

    # Calculate species density per km² for the entire region
    df_regions['EndemicSpecies'] = (
        df_regions['EndemicSpecies'] /  df_regions['TotalLand_KM2']
    )

    return df_regions

def plot_protected_land(df_regions):
    """Plot protected land area by region, with land area on the X-axis."""
    # Filter out the 'Other' region if desired
    df_regions = df_regions[df_regions['Region'] != 'Other']

    print(df_regions)
    # Sort regions by total land area (optional)
    df_regions = df_regions.sort_values('ProtectedLand_Pct', ascending=False)

    # Calculate width for each region (proportional to total land area)
    total_land = df_regions['TotalLand_KM2'].sum()
    df_regions['Width'] = df_regions['TotalLand_KM2'] / total_land

    # Create the plot
    fig, ax = create_protected_land_base_plot()
    left_positions = compute_left_positions(df_regions)
    plot_protected_land_bars(ax, df_regions, left_positions)
    configure_protected_land_axes(ax, df_regions, left_positions)
    add_protected_land_labels(ax, df_regions, left_positions)
    add_protected_land_source_label(ax)
    finalize_protected_land_plot(ax)

def create_protected_land_base_plot():
    """Create base plot with responsive sizing."""
    figsize = (16, 16)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    # Add y-axis grid lines
    ax.yaxis.grid(True, color=TEXT_COLOR, alpha=0.2)
    ax.set_axisbelow(True)

    # Remove black outline (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig, ax

def compute_left_positions(df_regions):
    """Compute left positions based on widths with gaps between regions."""
    gap = 0.01
    positions = [0]
    for i in range(len(df_regions) - 1):
        next_pos = positions[-1] + df_regions['Width'].iloc[i] + gap
        positions.append(next_pos)
    return np.array(positions)

def plot_protected_land_bars(ax, df_regions, left_positions):
    """Plot the bars for protected land area with color based on species density."""
    # Normalize species density for colormap
    norm = plt.Normalize(
        vmin=df_regions['EndemicSpecies'].min(),
        vmax=df_regions['EndemicSpecies'].max()
    )

    # Use a colormap (e.g., 'viridis', 'plasma', 'YlGn', etc.)
    cmap = plt.cm.YlGn  # Green gradient, but you can choose another

    # Create a color for each bar based on species density
    colors = cmap(norm(df_regions['EndemicSpecies'].values))

    # Plot the bars with the calculated colors
    ax.bar(
        left_positions,
        df_regions['ProtectedLand_Pct'],
        width=df_regions['Width'],
        color=colors,
        align='edge',
        edgecolor=BG_COLOR,
        linewidth=1.5
    )

    # Add a colorbar to show the scale
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Species Density per km²')

def configure_protected_land_axes(ax, df_regions, left_positions):
    """Configure axes with responsive styling."""
    total_width = left_positions[-1] + df_regions['Width'].iloc[-1]
    ax.set_xlim(left=0, right=total_width)

    ax.set_xticks([])
    ax.set_xticklabels([])

    ax.set_xlabel(
        'Total Land Area (sq. km)',
        color=TEXT_COLOR,
        fontsize=18,
        labelpad=18
    )
    ax.set_ylabel(
        'Protected Land Area (sq. km)',
        color=TEXT_COLOR,
        fontsize=18
    )
    ax.set_title(
        f'Terrestrial Protected Land by Region ({DATA_YEAR})',
        color=TEXT_COLOR,
        fontsize=24,
        pad=0,
        y=0.93,
        loc='center'
    )
    ax.tick_params(axis='y', colors=TEXT_COLOR, labelsize=12)

def add_protected_land_labels(ax, df_regions, left_positions):
    """Add region labels below the bars."""
    fontsize = 12
    ygap = -0.006 * ax.get_ylim()[1]

    for i, (_, region) in enumerate(df_regions.iterrows()):
        start = left_positions[i]
        width = region['Width']
        center = start + width / 2
        lw = 1.5
        eps = 0.001
        # Line
        ax.plot(
            [start + eps, start + width - eps],
            [ygap, ygap],
            color=ANNO_COLOR,
            linewidth=lw,
            alpha=1.0
        )
        # Label
        label = f"{region['Region']}\n{region['TotalLand_KM2'] / 1e6:.1f}M km²"
        ax.text(
            center,
            ygap * 2,
            label,
            ha='center',
            va='top',
            fontsize=fontsize,
            color=TEXT_COLOR
        )

def add_protected_land_source_label(ax):
    """Add source label with responsive positioning."""
    ax.text(
        0.99, 0.96,
        f"Source: World Bank ({DATA_YEAR})\nBy MadoctheHadoc",
        fontsize=12,
        color=TEXT_COLOR,
        ha='right',
        va='top',
        transform=ax.transAxes
    )

def finalize_protected_land_plot(ax):
    """Finalize the plot."""
    plt.tight_layout()
    plt.savefig(f"{VISUALIZATIONS_DIR}/GlobalLandProtection{DATA_YEAR}.png", dpi=300, facecolor=BG_COLOR)
    plt.close()

# ==================== MAIN ====================
if __name__ == "__main__":
    print(f"Starting terrestrial protected land analysis for {DATA_YEAR}")
    
    # Fetch and prepare data
    print('\nFetching data...')
    df = fetch_world_bank_data(INDICATORS, DATA_YEAR)
    df = clean_data(df)
    regions = SIMPLIFIED_REGIONS if USE_SIMPLIFIED_REGIONS else REGIONS
    df_regions = assign_regions(df, regions)
    plot_protected_land(df_regions)
    
    # Create visualizations
    print('\nCreating visualization')