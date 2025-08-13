import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# ======== CONSTANTS ======
# =========================
SCRIPT_DIR = os.path.dirname(__file__)
PRODUCTIVITY_PATH = os.path.join(SCRIPT_DIR, '../data/CountryProductivity.csv')
POPULATION_FILE_PATH = os.path.join(SCRIPT_DIR, '../data/CountryPopulation.txt')

ANNO_COLOR = "#A9A9A9"
GRID_COLOR = "#747474"
BACKGROUND = '#2a2a2a'

# Flags
USE_SIMPLIFIED_REGIONS = False
ANNOTATE_DESKTOP = False
ANNOTATE_MOBILE = False

REGIONS = [
    {
        'name': 'North America',
        'color': "#c17f29",
        'countries': [
            'United States', 'Canada', 'Mexico',
            'Bahamas', 'Cuba', 'Dominican Republic', 'Haiti', 'Jamaica',
            'Trinidad and Tobago', 'Belize', 'Costa Rica', 'El Salvador',
            'Guatemala', 'Honduras', 'Nicaragua', 'Panama'
        ]
    },
    {
        'name': 'East Asia',
        'color': "#c52d2d",
        'countries': [
            'China', 'Japan', 'South Korea', 'Taiwan', 'Hong Kong', 'Macau', 'Mongolia', 'North Korea'
        ]
    },
    {
        'name': 'Europe and Central Asia',
        'color': "#1f73b4",
        'countries': [
            'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark',
            'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy',
            'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal',
            'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Norway', 'Switzerland',
            'Liechtenstein', 'Iceland', 'United Kingdom', 'Andorra', 'Monaco', 'Russia',
            'Belarus', 'Ukraine', 'Moldova', 'Serbia', 'Montenegro', 'Albania',
            'Bosnia and Herzegovina', 'Kosovo', 'Kazakhstan', 'Kyrgyzstan', 'Tajikistan',
            'Turkmenistan', 'Uzbekistan'
        ]
    },
    {
        'name': 'South America',
        'color': "#1fb4ad",
        'countries': [
            'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Ecuador',
            'Guyana', 'Paraguay', 'Peru', 'Suriname', 'Uruguay', 'Venezuela'
        ]
    },
    {
        'name': 'Indo-Pacific',
        'color': "#b41fa8",
        'countries': [
            'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Malaysia', 'Myanmar',
            'Philippines', 'Singapore', 'Thailand', 'Vietnam', 'Timor-Leste',
            'Australia', 'New Zealand', 'Fiji', 'Papua New Guinea', 'Samoa', 'Tonga', 'Vanuatu'
        ]
    },
    {
        'name': 'South Asia',
        'color': "#8933cb",
        'countries': [
            'India', 'Pakistan', 'Nepal', 'Bangladesh', 'Sri Lanka', 'Bhutan', 'Maldives', 'Afghanistan'
        ]
    },
    {
        'name': 'Sub-Saharan Africa',
        'color': "#b4ad1f",
        'countries': [
            'Nigeria', 'South Africa', 'Ethiopia', 'Kenya', 'Tanzania', 'Uganda', 'Ghana', 'Angola',
            'Mozambique', 'Ivory Coast', 'Cameroon', 'Niger', 'Burkina Faso', 'Mali', 'Malawi',
            'Zambia', 'Senegal', 'Zimbabwe', 'Rwanda', 'Benin', 'Chad', 'South Sudan', 'Togo',
            'Sierra Leone', 'Liberia', 'Botswana', 'Namibia', 'Somalia', 'DR Congo', 'Congo',
            'Gabon', 'Guinea', 'Mauritania', 'Equatorial Guinea', 'Central African Republic', 'Lesotho',
            'Eswatini', 'Djibouti', 'Eritrea', 'Gambia', 'Guinea-Bissau', 'Burundi', 'Cape Verde',
            'Comoros', 'Sao Tome and Principe', 'Seychelles', 'Sudan'
        ]
    },
    {
        'name': 'MENA',
        'color': "#1daa1a",
        'countries': [
            'Saudi Arabia', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Lebanon', 'Oman', 'Qatar',
            'United Arab Emirates', 'Bahrain', 'Kuwait', 'Syria', 'Yemen', 'Palestine', 'Turkey',
            'Egypt', 'Libya', 'Morocco', 'Algeria', 'Tunisia'
        ]
    }
]


def load_data():
    # Load ILO productivity data with correct parsing
    df_prod = pd.read_csv(
        PRODUCTIVITY_PATH,
        sep=',',
        quotechar='"',
        engine='python'
    )

    # Rename columns to match expected names
    df_prod = df_prod.rename(columns={
        'ref_area.label': 'Location',
        'obs_value': 'Productivity'
    })
    df_prod['Location'] = df_prod['Location'].str.strip()
    df_prod = df_prod[['Location', 'Productivity']].copy()

    # Load population data
    df_population = pd.read_csv(POPULATION_FILE_PATH, sep='\t')
    df_population['Country'] = (
        df_population['Country']
        .str.strip()
        .str.replace(r'\s*\([^)]*\)', '', regex=True)
        .str.replace(r'\s*\[[^\]]*\]', '', regex=True)
    )
    df_population_clean = df_population[['Country', 'Population2025']].copy()
    df_population_clean.rename(columns={
        'Country': 'Location',
        'Population2025': 'Population'
    }, inplace=True)

    return df_prod, df_population_clean


def merge_data(df_prod, df_population):
    df = df_prod.merge(df_population, on='Location', how='left')

    # Clean population
    df['Population'] = (
        df['Population'].astype(str).str.replace(',', '')
    )
    df['Population'] = pd.to_numeric(df['Population'], errors='coerce')
    df = df.dropna(subset=['Population'])
    df = df[df['Population'] > 0].copy()

    return df


def assign_regions(df, regions):
    df['Region'] = 'Other'  # Default region for countries not in any specified region

    for region in regions:
        region_name = region['name']
        countries = region['countries']

        # Assign region to matching countries
        df.loc[df['Location'].isin(countries), 'Region'] = region_name

    return df


def plot_desktop(df, annotate=False):
    # Compression parameters
    threshold = 100
    compress_factor = 7  # higher = more compression beyond threshold

    def compress_x(x):
        if x <= threshold:
            return x
        else:
            return threshold + (x - threshold) / compress_factor

    # Apply transformation to plotting values
    df['ProductivityCompressed'] = df['Productivity'].apply(compress_x)

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    # Order regions by mean productivity
    region_avg_productivity = df.groupby('Region')['Productivity'].mean().sort_values(ascending=False)
    ordered_regions = region_avg_productivity.index.tolist()

    # Assign numerical Y positions
    y_pos = {region: i for i, region in enumerate(ordered_regions)}

    # Assign colors to regions
    region_colors = {region['name']: region.get('color', "#CBCBCB") for region in REGIONS}
    if 'Other' in ordered_regions and 'Other' not in region_colors:
        region_colors['Other'] = '#888888'

    # Normalize population for bubble size
    min_pop = df['Population'].min()
    max_pop = df['Population'].max()
    df['PopNorm'] = 950 * (np.sqrt(df['Population'] - min_pop) / np.sqrt(max_pop - min_pop))

    # Add random jiggle
    np.random.seed(42)
    df['Jiggle'] = np.random.uniform(-0.3, 0.3, size=len(df))

    # Plot bubbles
    for region in ordered_regions:
        region_df = df[df['Region'] == region]
        for _, row in region_df.iterrows():
            ax.scatter(
                row['ProductivityCompressed'], y_pos[region] + row['Jiggle'],
                s=row['PopNorm'],
                color=region_colors[region],
                alpha=0.6,
                edgecolor='white',
                linewidth=0.7
            )

    # Y-axis setup
    ax.set_yticks(range(len(ordered_regions)))
    ax.set_yticklabels(ordered_regions, color='white', fontsize=12)

    for i in range(len(ordered_regions) - 1):
        ax.axhline(i + 0.5, color=GRID_COLOR, linestyle='--', alpha=1.0, linewidth=0.8)

    # X-axis with custom ticks (inverse transform for labeling)
    def decompress_x(xc):
        if xc <= threshold:
            return xc
        else:
            return threshold + (xc - threshold) * compress_factor

    # Define desired fixed tick labels in real units
    tick_labels_real = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]

    # Transform them into compressed coordinates
    tick_positions_compressed = [compress_x(t) for t in tick_labels_real]

    # Apply to axis
    ax.set_xticks(tick_positions_compressed)
    ax.set_xticklabels([str(t) for t in tick_labels_real], color=GRID_COLOR)
    ax.set_xlim(0, compress_x(180))
    
    ax.set_xlabel('Labor Productivity (GDP per hour worked, 2021 int-$)', color='white', fontsize=14)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', which='both', length=0, colors='white')
    ax.xaxis.grid(True, color=GRID_COLOR, alpha=1.0)
    ax.set_axisbelow(True)

    # Title
    ax.set_title(
        'Labor Productivity by Region and Population (2025)',
        color='white',
        fontsize=20,
        pad=20
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        1.0, -0.05,
        "Sources: ILO (productivity) and World Population Review (population)",
        fontsize=9,
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes
    )

    # Annotate if requested
    if annotate:
        for _, row in df.iterrows():
            ax.annotate(
                row['Location'],
                (row['ProductivityCompressed'] + row['Jiggle'], y_pos[row['Region']]),
                textcoords="offset points",
                xytext=(0, 5),
                ha='center',
                fontsize=8,
                color='white'
            )

    plt.tight_layout()
    plt.savefig("visualizations/LaborProductivityByRegion.png", dpi=300, bbox_inches='tight')
    plt.show()


def main():
    df_prod, df_population = load_data()
    df = merge_data(df_prod, df_population)
    regions = [] if USE_SIMPLIFIED_REGIONS else REGIONS
    df = assign_regions(df, regions)

    if USE_SIMPLIFIED_REGIONS:
        # plot_mobile(df_final, ANNOTATE_MOBILE)
        pass
    else:
        plot_desktop(df, ANNOTATE_DESKTOP)
        pass

if __name__ == "__main__":
    main()