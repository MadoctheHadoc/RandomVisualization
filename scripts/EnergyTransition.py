import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# ======== CONSTANTS ======
# =========================
SCRIPT_DIR = os.path.dirname(__file__)
ENERGY_FILE_PATH = os.path.join(SCRIPT_DIR, '../data/EnergyGeneration.txt')
POPULATION_FILE_PATH = os.path.join(SCRIPT_DIR, '../data/CountryPopulation.txt')

ENERGY_COLUMNS = [
    'Total', 'Coal', 'Gas', 'Hydro', 'Nuclear', 'Wind', 'Solar',
    'Oil', 'Biomass', 'Geothermal'
]

OTHER_COLUMNS = [
    'Biomass', 'Geothermal'
]

FOSSIL_FUELS = ['Gas', 'Coal', 'Oil']

ORDERED_COLUMNS = [
    'Hydro', 'Solar', 'Wind', 'Nuclear',
    'Other', 'Gas', 'Oil', 'Coal'
]

COLORS = {
    'Hydro': "#535FB0",
    'Solar': "#BBB65C",
    'Wind': "#539CB0",
    'Nuclear': "#70B053",
    'Other': "#7D4AA6",
    'Gas': "#905F41",
    'Oil': "#7F4533",
    'Coal': "#562A2A"
}

ANNO_COLOR = "#929292"

BACKGROUND = '#2a2a2a'

# Flags
USE_SIMPLIFIED_REGIONS = True
ANNOTATE = True
GROUP_REMAINDER = True

SIMPLIFIED_REGIONS = [
    {
        'name': 'USA &\nCanada',
        'countries': [
            'United States', 'Canada'
        ]
    },    
    {
        'name': 'China',
        'countries': [
            'China'
        ]
    },
    {
        'name': 'Europe\n(ex. Russia)',
        'countries': [
            'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark',
            'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy',
            'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal',
            'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Ukraine', 'Belarus',
            'Armenia', 'Georgia', 'Azerbaijan', 'Bosnia and Herzegovina', 'Serbia', 'Montenegro',
            'Albania', 'United Kingdom', 'Norway', 'Switzerland', 'Liechtenstein'
        ]
    },
    {
        'name': 'Africa',
        'countries': [
            'Nigeria', 'South Africa', 'Ethiopia', 'Kenya', 'Tanzania', 'Uganda', 'Ghana', 'Angola',
            'Mozambique', 'Ivory Coast', 'Cameroon', 'Niger', 'Burkina Faso', 'Mali', 'Malawi',
            'Zambia', 'Senegal', 'Zimbabwe', 'Rwanda', 'Benin', 'Chad', 'South Sudan', 'Togo',
            'Sierra Leone', 'Liberia', 'Botswana', 'Namibia', 'Somalia', 'DR Congo', 'Congo',
            'Gabon', 'Guinea', 'Mauritania', 'Equatorial Guinea', 'Central African Republic', 'Lesotho',
            'Eswatini', 'Djibouti', 'Eritrea', 'Gambia', 'Guinea-Bissau', 'Burundi', 'Cape Verde',
            'Comoros', 'Sao Tome and Principe', 'Seychelles', 'Egypt', 'Libya', 'Morocco', 'Algeria', 'Tunisia'
        ]
    },
    {
        'name': 'Latin\nAmerica',
        'countries': [
            'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Costa Rica', 'Cuba',
            'Dominican Republic', 'Ecuador', 'El Salvador', 'Guatemala', 'Honduras',
            'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela',
            'Haiti', 'Jamaica', 'Trinidad and Tobago', 'Belize'
        ]
    },
    {
        'name': 'ASEAN',
        'countries': [
            'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Malaysia', 'Myanmar',
            'Philippines', 'Singapore', 'Thailand', 'Vietnam',
        ]
    },
    {
        'name': 'South Asia',
        'countries': [
            'India', 'Pakistan', 'Nepal', 'Bangladesh', 'Sri Lanka', 'Bhutan', 'Maldives'
        ]
    }
]

REGIONS = [
    {
        'name': 'USA &  \nCanada  ',
        'countries': [
            'United States', 'Canada'
        ]
    },
    {
        'name': 'China',
        'countries': [
            'China'
        ]
    },
    {
        'name': 'EEA\n& UK',
        'countries': [
            'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark',
            'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy',
            'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal',
            'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Norway', 'Switzerland',
            'Liechtenstein', 'Iceland', 'United Kingdom'
        ]
    },
    {
        'name': 'Non-EEA\nEurope',
        'countries': [
            'Russia', 'Ukraine', 'Belarus', 'Armenia', 'Georgia', 'Azerbaijan', 'Bosnia and Herzegovina', 'Serbia', 'Montenegro',
            'Albania'
        ]
    },
    {
        'name': 'ASEAN',
        'countries': [
            'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Malaysia', 'Myanmar',
            'Philippines', 'Singapore', 'Thailand', 'Vietnam'
        ]
    },
    {
        'name': '  Japan &\n  S. Korea',
        'countries': ['Japan', 'South Korea']
    },
    {
        'name': 'Other\nSouth Asia',
        'countries': [
            'Pakistan', 'Nepal', 'Bangladesh', 'Sri Lanka', 'Bhutan', 'Maldives'
        ]
    },
    {
        'name': 'India',
        'countries': [
            'India',
        ]
    },
    {
        'name': 'Sub-Saharan\nAfrica',
        'countries': [
            'Nigeria', 'South Africa', 'Ethiopia', 'Kenya', 'Tanzania', 'Uganda', 'Ghana', 'Angola',
            'Mozambique', 'Ivory Coast', 'Cameroon', 'Niger', 'Burkina Faso', 'Mali', 'Malawi',
            'Zambia', 'Senegal', 'Zimbabwe', 'Rwanda', 'Benin', 'Chad', 'South Sudan', 'Togo',
            'Sierra Leone', 'Liberia', 'Botswana', 'Namibia', 'Somalia', 'DR Congo', 'Congo',
            'Gabon', 'Guinea', 'Mauritania', 'Equatorial Guinea', 'Central African Republic', 'Lesotho',
            'Eswatini', 'Djibouti', 'Eritrea', 'Gambia', 'Guinea-Bissau', 'Burundi', 'Cape Verde',
            'Comoros', 'Sao Tome and Principe', 'Seychelles', 
        ]
    },
    {
        'name': 'MENA',
        'countries': [
            'Saudi Arabia', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Lebanon', 'Oman', 'Qatar', 
            'United Arab Emirates', 'Bahrain', 'Kuwait', 'Syria', 'Yemen', 'Palestine', 'Turkey',
            'Egypt', 'Libya', 'Morocco', 'Algeria', 'Tunisia'
        ]
    },
    {
        'name': 'Latin\nAmerica',
        'countries': [
            'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Costa Rica', 'Cuba',
            'Dominican Republic', 'Ecuador', 'El Salvador', 'Guatemala', 'Honduras',
            'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela',
            'Haiti', 'Jamaica', 'Trinidad and Tobago', 'Belize'
        ]
    }
]

EUROPEAN_REGIONS = [
    {
        'name': 'Scandinavia',
        'countries': [
            'Denmark', 'Finland', 'Norway', 'Iceland', 'Sweden'
        ]
    },
    {
        'name': 'Iberia',
        'countries': [
            'Spain', 'Portugal'
        ]
    },
    {
        'name': 'Union State',
        'countries': [
            'Russia', 'Belarus'
        ]
    },
    {
        'name': 'Caucasus',
        'countries': [
            'Armenia', 'Azerbaijan', 'Georgia'
        ]
    },
    {
        'name': 'Baltics',
        'countries': [
            'Lithuania', 'Latvia', 'Estonia'
        ]
    },
    {
        'name': 'British Isles',
        'countries': [
            'United Kingdom', 'Ireland'
        ]
    },
    {
        'name': 'Benelux',
        'countries': [
            'Netherlands', 'Belgium', 'Luxembourg'
        ]
    },
    {
        'name': 'Alps',
        'countries': [
            'Switzerland', 'Austria', 'Liechtenstein'
        ]
    },
    {
        'name': 'South Slavs',
        'countries': [
            'Slovenia', 'Croatia', 'Serbia', 'Bosnia and Herzegovina', 'Bulgaria', 'North Macedonia', 'Montenegro', 'Albania'
        ]
    },
    {
        'name': 'Italy',
        'countries': [
            'Italy', 'Malta'
        ]
    },
    {
        'name': 'Greece',
        'countries': [
            'Greece', 'Cyprus'
        ]
    },
    {
        'name': 'Romania',
        'countries': [
            'Romania', 'Moldova'
        ]
    },
]
# =========================
# ======== FUNCTIONS ======
# =========================

def load_data():
    df_energy = pd.read_csv(ENERGY_FILE_PATH, sep='\t')
    df_energy['Location'] = df_energy['Location'].str.strip()

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

    return df_energy, df_population_clean


def merge_data(df_energy, df_population):
    df = df_energy.merge(df_population, on='Location', how='left')

    # Clean population
    df['Population'] = (
        df['Population'].astype(str).str.replace(',', '')
    )
    df['Population'] = pd.to_numeric(df['Population'], errors='coerce')

    df = df.dropna(subset=['Population'])
    df = df[df['Population'] > 0].copy()

    # Convert energy columns
    for col in ENERGY_COLUMNS:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', ''),
            errors='coerce'
        ).fillna(0)
        
    df['Other'] = 0.0  # Initialize as scalar, pandas will broadcast to all rows
    for col in OTHER_COLUMNS:
        df['Other'] = df['Other'] + df[col]

    return df


def tag_regions(df, regions):
    df['Region'] = 'Other'
    for region in regions:
        df.loc[df['Location'].isin(region['countries']), 'Region'] = region['name']
    return df


def aggregate_region(df_region, name):
    df_sum = df_region[ORDERED_COLUMNS + ['Total']].sum()
    population_sum = df_region['Population'].sum()
    return pd.DataFrame({
        'Location': [name],
        **{col: [df_sum[col]] for col in (ORDERED_COLUMNS + ['Total'])},
        'Population': [population_sum]
    })


def prepare_final_dataframe(df, regions):
    flat_region_countries = [c for r in regions for c in r['countries']]
    df_filtered = df[~df['Location'].isin(flat_region_countries)].copy()

    aggregated_regions = []
    for region in regions:
        df_region = df[df['Region'] == region['name']]
        if not df_region.empty:
            aggregated_regions.append(aggregate_region(df_region, region['name']))

    if GROUP_REMAINDER:
        if not df_filtered.empty:
            df_other_sum = df_filtered[ORDERED_COLUMNS + ['Total']].sum()
            total_other_population = df_filtered['Population'].sum()
            remainderLabel = 'Rest of\nWorld' if USE_SIMPLIFIED_REGIONS else 'Rest of\nWorld'
            df_other_df = pd.DataFrame({
                'Location': [remainderLabel],
                **{col: [df_other_sum[col]] for col in (ORDERED_COLUMNS + ['Total'])},
                'Population': [total_other_population]
            })
            aggregated_regions.append(df_other_df)
        return pd.concat(aggregated_regions, ignore_index=True)

    else:
        # If not grouping, append the individual leftover countries as they are
        if not df_filtered.empty:
            aggregated_regions.append(df_filtered)
        return pd.concat(aggregated_regions, ignore_index=True)


def normalize_data(df):
    for col in (ORDERED_COLUMNS + ['Total']):
        df[col] = df[col] / df['Population']
    df['Width'] = df['Population'] / df['Population'].max()
    return df

def plot_electricity(df_final):
    """Plot electricity generation chart with responsive design based on USE_SIMPLIFIED_REGIONS."""
    fig, ax = create_electricity_base_plot()
    
    left_positions = compute_left_positions(df_final)
    bottom = plot_electricity_bars(ax, df_final, left_positions)
    configure_electricity_axes(ax, df_final, left_positions)
    add_electricity_labels(ax, df_final, left_positions)
    add_electricity_legend(ax)
    add_electricity_source_label(ax)
    if ANNOTATE:
        add_electricity_annotations(ax, df_final, left_positions, bottom)
    
    finalize_electricity_plot(ax)

def create_electricity_base_plot():
    """Create base plot with responsive sizing."""
    figsize = (12, 12) if USE_SIMPLIFIED_REGIONS else (13, 9)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)
    
    # Add y-axis grid lines
    ax.yaxis.grid(True, color='white', alpha=0.2)
    ax.set_axisbelow(True)
    
    # Remove black outline (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    return fig, ax

def compute_left_positions(df_final):
    """Compute left positions based on widths with gaps between countries."""
    gap = 0.013 if USE_SIMPLIFIED_REGIONS else 0.006
    positions = [0]
    for i in range(len(df_final) - 1):
        next_pos = positions[-1] + df_final['Width'].iloc[i] + gap
        positions.append(next_pos)
    return np.array(positions)

def plot_electricity_bars(ax, df_final, left_positions):
    """Plot the stacked bars for electricity generation."""
    bottom = np.zeros(len(df_final))
    line_width_multiplier = 1.0 if USE_SIMPLIFIED_REGIONS else 1.0
    
    for col in ORDERED_COLUMNS:
        lw = (1.0 if col in FOSSIL_FUELS else 1.0) * line_width_multiplier
        ax.bar(left_positions, df_final[col] * 1e6,  # convert TWh to MWh
               width=df_final['Width'], bottom=bottom,
               label=col, color=COLORS[col], align='edge',
               edgecolor=BACKGROUND, linewidth=lw)
        bottom += (df_final[col] * 1e6).values
        
    return bottom

def configure_electricity_axes(ax, df_final, left_positions):
    """Configure axes with responsive styling."""
    # Adjust x-axis limit
    total_width = left_positions[-1] + df_final['Width'].iloc[-1]
    ax.set_xlim(left=0, right=total_width)
    
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    smallsize = 24 if USE_SIMPLIFIED_REGIONS else 18
    titlesize = 28 if USE_SIMPLIFIED_REGIONS else 24
    pad = 15 if USE_SIMPLIFIED_REGIONS else 10
    # Larger fonts for mobile readability
    ax.set_xlabel(
        'Population (2025)',
        color='white',
        fontsize=smallsize,
        labelpad=pad
    )
    ax.set_ylabel(
        'Generation per Capita (MWh/person) 2024',
        color='white',
        fontsize=smallsize
    )
    y = 0.808 if USE_SIMPLIFIED_REGIONS else 0.93
    ax.set_title(
        'The State of Global Electricity Generation',
        color='white',
        fontsize=titlesize,
        pad=0,
        y=y,
        loc='center'
    )
    ax.tick_params(axis='y', colors='white', labelsize=12)

def add_electricity_labels(ax, df_final, left_positions):
    fontsize = 12 if USE_SIMPLIFIED_REGIONS else 9
    
    for i, (_, country) in enumerate(df_final.iterrows()):
        start = left_positions[i]
        width = country['Width']
        center = start + width / 2
        ygap = -0.006 * ax.get_ylim()[1]
        lw = 1.5
        eps = 0.01
        # Line
        ax.plot(
            [start + eps, start + width - eps],
            [ygap, ygap],
            color='white',
            linewidth=lw,
            alpha=1.0
        )
        
        # Label
        label = f"{country['Location']}\n{country['Population'] / 1e6:.0f}M"
        ax.text(
            center,
            ygap * 2,
            label,
            ha='center',
            va='top',
            fontsize=fontsize,
            color='white'
        )

def add_electricity_legend(ax):
    """Add legend with responsive sizing."""
    handles, labels = ax.get_legend_handles_labels()
    
    fontsize = 16 if USE_SIMPLIFIED_REGIONS else 16
    title_fontsize = 16 if USE_SIMPLIFIED_REGIONS else 16
    
    legend = ax.legend(
        handles[::-1], labels[::-1],  # reversed order
        title="Generation Source",
        loc='center right',
        facecolor=BACKGROUND,
        edgecolor='white',
        frameon=True,
        fontsize=fontsize,
        title_fontsize=title_fontsize,
        framealpha=1.0
    )
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

def add_electricity_source_label(ax):
    """Add source label with responsive positioning."""
    position = (0.99, 0.96) if USE_SIMPLIFIED_REGIONS else (0.99, 0.88)
    size = 14 if USE_SIMPLIFIED_REGIONS else 12
    
    ax.text(
        position[0], position[1],
        "Sources: Ember Energy (electricity)\nOur World in Data (population)\n By MadoctheHadoc",
        fontsize=size,
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes
    )

def add_electricity_annotations(ax, df_final, left_positions, bottom):
    simplified_annotations = [
        {
            'location': 'China',
            'text': "China generates the most electricity\nin the world but still produces much\nless than the US per capita",
            'text_pos': (0.25, 8.7),
        },
        {
            'location': 'Europe\n(ex. Russia)',
            'text': "Europe and Latin America are\nthe only regions with majority\nrenewable electricity",
            'text_pos': (1.1, 6.7),
        },
        {
            'location': 'South Asia',
            'text': "Per capita generation is very low\nacross the developing world and\nfossil fuels dominate",
            'text_pos': (2.5, 2.7),
        }
    ]
    
    annotations = [
        {
            'location': 'China',
            'text': "China is so big that it generates the most wind,\nsolar, hydro and coal power in the world!",
            'text_pos': (0.44, 8.7),
        },
        {
            'location': REGIONS[5]['name'],
            'text': "Japan and South Korea lag behind the rest of\nthe developed world in renewable roll out",
            'text_pos': (0.3, 11.0),
        },
        {
            'location': 'India',
            'text': "Per capita generation is very low across\nthe developing world and fossil fuels dominate",
            'text_pos': (3.1, 2.5),
        },
        {
            'location': 'MENA',
            'text': "The Middle East & North Africa\nis the only region to use significant\namounts of oil-generated electricity",
            'text_pos': (2.2, 4.6),
        },
        {
            'location': REGIONS[2]['name'],
            'text': "Europe runs the most decarbonised\nelectrical grid of any major region\nwith >70% of low carbon sources",
            'text_pos': (1.6, 6.6),
        }
    ]
    
    annotations = simplified_annotations if USE_SIMPLIFIED_REGIONS else annotations
    add_electricity_annotation_arrows(ax, df_final, left_positions, bottom, annotations)

def add_electricity_annotation_arrows(ax, df_final, left_positions, bottom, annotations):
    """Add annotation arrows to the plot."""
    for annotation in annotations:
        location = annotation['location']
        text = annotation['text']
        text_pos = annotation['text_pos']
        location_idx = df_final.index[df_final['Location'] == location][0]
        
        # Calculate arrow target position
        target_x = left_positions[location_idx] + df_final['Width'].iloc[location_idx] / 2
        target_y = bottom[location_idx]
        size = 14 if USE_SIMPLIFIED_REGIONS else 12
        
        ax.annotate(
            text,
            xy=(target_x, target_y),
            xytext=text_pos,
            textcoords='data',
            arrowprops=dict(arrowstyle='->', color=ANNO_COLOR),
            color=ANNO_COLOR,
            fontsize=size,
            ha='left'
        )

def europe_regionalize(df):
    europe_countries = [
        'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
        'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece',
        'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg',
        'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia',
        'Slovenia', 'Spain', 'Sweden', 'Norway', 'Switzerland', 'Liechtenstein',
        'Iceland', 'United Kingdom', 'Russia', 'Ukraine', 'Belarus', 'Armenia',
        'Georgia', 'Azerbaijan', 'Bosnia and Herzegovina', 'Serbia',
        'Montenegro', 'Albania'
    ]
    df = df[df['Location'].isin(europe_countries)].copy()

    return regionalize(df, EUROPEAN_REGIONS)

def regionalize(df, regions):
    df = tag_regions(df, regions)
    df_final = prepare_final_dataframe(df, regions)
    df_final = normalize_data(df_final)
    df_final = df_final.sort_values(by='Total', ascending=False).reset_index(drop=True)
    return df_final

def finalize_electricity_plot(ax):
    """Finalize and save the plot with appropriate filename."""
    suffix = "Mobile" if USE_SIMPLIFIED_REGIONS else "Desktop"
    filename = f"ElectricityGeneration{suffix}.png"
    
    plt.tight_layout()
    plt.savefig(f"visualizations/{filename}", dpi=300, bbox_inches='tight')

def main():
    df_energy, df_population = load_data()
    df = merge_data(df_energy, df_population)
    df = df[df['Location'] != 'World'].copy()
    
    regions = SIMPLIFIED_REGIONS if USE_SIMPLIFIED_REGIONS else REGIONS
    df_final = regionalize(df, regions)
    plot_electricity(df_final)

if __name__ == "__main__":
    main()