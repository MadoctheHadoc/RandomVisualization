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

FOSSIL_FUELS = ['Gas', 'Coal', 'Oil']

ORDERED_COLUMNS = [
    'Hydro', 'Solar', 'Wind', 'Nuclear', 'Biomass', 'Geothermal',
    'Gas', 'Oil', 'Coal'
]

COLORS = {
    'Hydro': "#8d94cb",
    'Solar': "#ffd92f",
    'Wind': "#66c2b9",
    'Nuclear': "#66d854",
    'Biomass': "#e78ac3",
    'Geothermal': "#b08dcb",
    'Gas': "#e0754a",
    'Oil': "#ac341f",
    'Coal': "#570F0F"
}

ANNO_COLOR = '#888888'

BACKGROUND = '#2a2a2a'

# Flags
USE_SIMPLIFIED_REGIONS = False
ANNOTATE_DESKTOP = False
ANNOTATE_MOBILE = False
GROUP_REMAINDER = False

# Regions definitions (simplified and full) — omitted here for brevity
SIMPLIFIED_REGIONS = [
    {
        'name': 'USA',
        'countries': [
            'United States'
        ]
    },    
    {
        'name': 'China',
        'countries': [
            'China'
        ]
    },
    {
        'name': 'Europe (ex. Russia)',
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
        'name': 'Latin America',
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
        'name': 'USA',
        'countries': [
            'United States'
        ]
    },    
    {
        'name': 'China',
        'countries': [
            'China'
        ]
    },
    {
        'name': 'EEA',
        'countries': [
            'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark',
            'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy',
            'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal',
            'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Norway', 'Switzerland',
            'Liechtenstein', 'Iceland'
        ]
    },
    {
        'name': 'Other Europe',
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
        'name': 'Developed Asia',
        'countries': ['Japan', 'South Korea', 'Taiwan', 'Hong Kong', 'Macau']
    },
    {
        'name': 'South Asia',
        'countries': [
            'India', 'Pakistan', 'Nepal', 'Bangladesh', 'Sri Lanka', 'Bhutan', 'Maldives'
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
        'name': 'Middle East',
        'countries': [
            'Saudi Arabia', 'Iran', 'Iraq', 'Israel', 'Jordan', 'Lebanon', 'Oman', 'Qatar', 
            'United Arab Emirates', 'Bahrain', 'Kuwait', 'Syria', 'Yemen', 'Palestine', 'Turkey'
        ]
    },
    {
        'name': 'Latin America',
        'countries': [
            'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Costa Rica', 'Cuba',
            'Dominican Republic', 'Ecuador', 'El Salvador', 'Guatemala', 'Honduras',
            'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela',
            'Haiti', 'Jamaica', 'Trinidad and Tobago', 'Belize'
        ]
    },
    {
        'name': 'CANZUK',
        'countries': [
            'Canada', 'Australia', 'New Zealand', 'United Kingdom'
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

    return df


def tag_regions(df, regions):
    df['Region'] = 'Other'
    for region in regions:
        df.loc[df['Location'].isin(region['countries']), 'Region'] = region['name']
    return df


def aggregate_region(df_region, name):
    df_sum = df_region[ENERGY_COLUMNS].sum()
    population_sum = df_region['Population'].sum()
    return pd.DataFrame({
        'Location': [name],
        **{col: [df_sum[col]] for col in ENERGY_COLUMNS},
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
            df_other_sum = df_filtered[ENERGY_COLUMNS].sum()
            total_other_population = df_filtered['Population'].sum()
            df_other_df = pd.DataFrame({
                'Location': ['Rest of World'],
                **{col: [df_other_sum[col]] for col in ENERGY_COLUMNS},
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
    for col in ENERGY_COLUMNS:
        df[col] = df[col] / df['Population']
    df['Width'] = df['Population'] / df['Population'].max()
    return df


def plot_desktop(df_final, annotate):
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(BACKGROUND)
    ax.set_facecolor(BACKGROUND)

    # Compute left positions based on widths
    left_positions = np.cumsum([0] + df_final['Width'].tolist()[:-1])

    # Draw stacked bars
    bottom = np.zeros(len(df_final))
    for col in ORDERED_COLUMNS:
        lw = 1.3 if col in FOSSIL_FUELS else 1.0
        ax.bar(left_positions, df_final[col] * 1e6,  # convert TWh to MWh
               width=df_final['Width'], bottom=bottom,
               label=col, color=COLORS[col], align='edge',
               edgecolor='black', linewidth=lw)
        bottom += (df_final[col] * 1e6).values

    # Set x-axis labels below bars
    ax.set_xticks(left_positions + df_final['Width'] / 2)
    ax.set_xticklabels(df_final['Location'], rotation=-27, ha='left', fontsize=10, color='white')

    # Add y-axis grid lines
    ax.yaxis.grid(True, color='white', alpha=0.2)
    ax.set_axisbelow(True)

    # Axis and label styling
    ax.set_xlabel('Population 2025', color='white', fontsize=16)
    ax.set_ylabel(
        'Electricity Generation per Capita\n(MWh/person, 2024)',
        color='white',
        fontsize=16
    )

    ax.set_title(
        'Electricity Generation by Population and Source',
        color='white',
        fontsize=30,
        pad=0,
        y=0.93
    )

    # White y-ticks
    ax.tick_params(axis='y', colors='white')
    # White x-ticks
    ax.tick_params(axis='x', colors='white')

    # Adjust x-axis limit to avoid trailing whitespace
    total_width = left_positions[-1] + df_final['Width'].iloc[-1]
    ax.set_xlim(left=0, right=total_width)

    # Legend with dark background and white text
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles[::-1], labels[::-1],  # reversed order
        title="Generation Source",
        loc='center right',
        facecolor=BACKGROUND,
        edgecolor='white',
        frameon=True,
        fontsize=12,         # Legend text size
        title_fontsize=13    # Legend title size
    )
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    # Add source label in bottom-right corner
    ax.text(
        1.0, -0.12,
        "Sources: Ember Energy (electricity) and Our World in Data (population)\n By MadoctheHadoc",
        fontsize=9,
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes
    )
    
    if (annotate):
        # Get index positions for each location
        china_idx = df_final.index[df_final['Location'] == 'China'][0]
        middle_east_idx = df_final.index[df_final['Location'] == 'Middle East'][0]
        south_asia_idx = df_final.index[df_final['Location'] == 'South Asia'][0]
        asia_idx = df_final.index[df_final['Location'] == 'Developed Asia'][0]
        
        # Annotation: China
        ax.annotate(
            "China is so big that it generates the most wind,\nsolar, hydro and coal power in the world!",
            xy=(left_positions[china_idx] + df_final['Width'].iloc[china_idx] / 2,
                bottom[china_idx]),  # arrow target
            xytext=(0.5, 8.7),  # text position
            textcoords='data',
            arrowprops=dict(arrowstyle='->', color=ANNO_COLOR),
            color=ANNO_COLOR,
            fontsize=10,
            ha='left'
        )
        
        # Annotation: Japan and Korea
        ax.annotate(
            "Japan and South Korea lag behind the rest of\nthe developed world in renewable roll out",
            xy=(left_positions[asia_idx] + df_final['Width'].iloc[asia_idx] / 2,
                bottom[asia_idx]),  # arrow target
            xytext=(0.3, 10.7),  # text position
            textcoords='data',
            arrowprops=dict(arrowstyle='->', color=ANNO_COLOR),
            color=ANNO_COLOR,
            fontsize=10,
            ha='left'
        )

        # Annotation: Developing world
        ax.annotate(
            "Per capita generation is very low across\nthe developing world and fossil fuels dominate",
            xy=(left_positions[south_asia_idx] + df_final['Width'].iloc[south_asia_idx] / 2,
                bottom[south_asia_idx]),
            xytext=(2.8, 2.7),  # text position
            textcoords='data',
            arrowprops=dict(arrowstyle='->', color=ANNO_COLOR),
            color=ANNO_COLOR,
            fontsize=10,
            ha='left'
        )
        
        # Annotation: Middle East
        ax.annotate(
            "The Middle East is the only region to use\nsignificant amounts of oil-generated electricity",
            xy=(left_positions[middle_east_idx] + df_final['Width'].iloc[middle_east_idx] / 2,
                bottom[middle_east_idx]),
            xytext=(1.6, 6.7),  # text position
            textcoords='data',
            arrowprops=dict(arrowstyle='->', color=ANNO_COLOR),
            color=ANNO_COLOR,
            fontsize=10,
            ha='left'
        )

    # Remove black outline (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("visualizations/ElectricityGenerationDesktop.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_mobile(df_final, annotate):
    abbreviations = {
        'United States': 'USA',
        'Europe (ex. Russia)': 'Europe',
        'Latin America': 'Latin\nAmerica',
    }
    df_final['Location'] = df_final['Location'].replace(abbreviations)
    # Plot with portrait layout for mobile
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor(BACKGROUND)  # Dark background
    ax.set_facecolor(BACKGROUND)

    # Compute left positions based on widths
    left_positions = np.cumsum([0] + df_final['Width'].tolist()[:-1])

    fossil_fuels = ['Gas', 'Coal', 'Oil']
    # Draw stacked bars
    bottom = np.zeros(len(df_final))
    for col in ORDERED_COLUMNS:
        lw = 1.3 if col in fossil_fuels else 1.0
        ax.bar(left_positions, df_final[col] * 1e6,  # convert TWh to MWh
               width=df_final['Width'], bottom=bottom,
               label=col, color=COLORS[col], align='edge',
               edgecolor='black', linewidth=lw)
        bottom += (df_final[col] * 1e6).values


    # Set x-axis labels below bars
    ax.set_xticks(left_positions + df_final['Width'] / 2)
    ax.set_xticklabels(df_final['Location'], rotation=0, ha='center', fontsize=12, color='white')

    # Add y-axis grid lines
    ax.yaxis.grid(True, color='white', alpha=0.2)
    ax.set_axisbelow(True)

    # Axis and label styling (larger for mobile readability)
    ax.set_xlabel('Population (2025)', color='white', fontsize=18, labelpad=10)
    ax.set_ylabel(
        'Electricity Generation per Capita\n(MWh/person, 2024)',
        color='white',
        fontsize=18,
        labelpad=12
    )
    ax.set_title(
        'Electricity Generation by Population and Source',
        color='white',
        fontsize=30,
        pad=0,
        y=0.93,
        loc='right'
    )

    # White ticks
    ax.tick_params(axis='x', colors='white', labelsize=12)
    ax.tick_params(axis='y', colors='white', labelsize=12)

    # Adjust x-axis limit to avoid trailing whitespace
    total_width = left_positions[-1] + df_final['Width'].iloc[-1]
    ax.set_xlim(left=0, right=total_width)

    # Legend with larger font and dark background
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(
        handles[::-1], labels[::-1],  # reversed order
        title="Generation Source",
        loc='center right',
        facecolor=BACKGROUND,
        edgecolor='white',
        frameon=True,
        fontsize=14,
        title_fontsize=15
    )
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    # Source label in bottom-right corner
    ax.text(
        1.0, -0.035,
        "Sources: Ember Energy (electricity) and Our World in Data (population)\n By MadoctheHadoc",
        fontsize=10,
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes
    )
    
    if (annotate):
        # Get index positions for each location
        china_idx = df_final.index[df_final['Location'] == 'China'][0]
        europe_idx = df_final.index[df_final['Location'] == 'Europe'][0]
        south_asia_idx = df_final.index[df_final['Location'] == 'South Asia'][0]

        # Annotation: China and USA
        ax.annotate(
            "China generates the most electricity in the world\nbut still produces much less than the US per capita",
            xy=(left_positions[china_idx] + df_final['Width'].iloc[china_idx] / 2,
                bottom[china_idx]),  # arrow target
            xytext=(0.3, 8.7),  # text position
            textcoords='data',
            arrowprops=dict(arrowstyle='->', color=ANNO_COLOR),
            color=ANNO_COLOR,
            fontsize=10,
            ha='left'
        )

        # Annotation: Europe and Latin America
        ax.annotate(
            "Europe and Latin America are only major\nregions with majority renewable energy",
            xy=(left_positions[europe_idx] + df_final['Width'].iloc[europe_idx] / 2,
                bottom[europe_idx]),
            xytext=(1.2, 6.7),  # text position
            textcoords='data',
            arrowprops=dict(arrowstyle='->', color=ANNO_COLOR),
            color=ANNO_COLOR,
            fontsize=10,
            ha='left'
        )
        
        # Annotation: Developing World
        ax.annotate(
            "Per capita generation is very low across\nthe developing world and fossil fuels dominate",
            xy=(left_positions[south_asia_idx] + df_final['Width'].iloc[south_asia_idx] / 2,
                bottom[south_asia_idx]),
            xytext=(2.6, 2.7),  # text position
            textcoords='data',
            arrowprops=dict(arrowstyle='->', color=ANNO_COLOR),
            color=ANNO_COLOR,
            fontsize=10,
            ha='left'
        )

    # Remove black outline (spines)
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("visualizations/ElectricityGenerationMobile.png", dpi=300, bbox_inches='tight')
    plt.show()


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


def main():
    df_energy, df_population = load_data()
    df = merge_data(df_energy, df_population)
    df = df[df['Location'] != 'World'].copy()
    
    # regions = REGIONS if USE_SIMPLIFIED_REGIONS else SIMPLIFIED_REGIONS
    # df_final = regionalize(df, regions)

    df_final = europe_regionalize(df)

    if USE_SIMPLIFIED_REGIONS:
        plot_mobile(df_final, ANNOTATE_MOBILE)
    else:
        plot_desktop(df_final, ANNOTATE_DESKTOP)

if __name__ == "__main__":
    main()