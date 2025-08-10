import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- FILE PATHS ---
script_dir = os.path.dirname(__file__)
gdp_file_path = os.path.join(script_dir, '../data/CountryGDPBySector.txt')
population_file_path = os.path.join(script_dir, '../data/CountryPopulation.txt')

# --- READ AND CLEAN GDP DATA ---
df_gdp = pd.read_csv(gdp_file_path, sep='\t')
df_gdp.columns = df_gdp.columns.str.strip()
df_gdp['Country/Economy'] = df_gdp['Country/Economy'].str.strip()

# Convert GDP columns to numeric
gdp_cols = ['Total GDP (US$MM)', 'Agricultural (US$MM)', 'Industrial (US$MM)', 'Service (US$MM)']
for col in gdp_cols:
    df_gdp[col] = pd.to_numeric(df_gdp[col].astype(str).str.replace(',', ''), errors='coerce')

# --- READ AND CLEAN POPULATION DATA ---
df_pop = pd.read_csv(population_file_path, sep='\t')
df_pop['Country'] = df_pop['Country'].str.strip()
df_pop['Country'] = df_pop['Country'].str.replace(r'\s*\([^)]*\)', '', regex=True)
df_pop['Country'] = df_pop['Country'].str.replace(r'\s*\[[^\]]*\]', '', regex=True)

# Extract population column
df_pop_clean = df_pop[['Country', 'Population2025']].copy()
df_pop_clean.rename(columns={'Country': 'Country/Economy', 'Population2025': 'Population'}, inplace=True)
df_pop_clean['Population'] = pd.to_numeric(df_pop_clean['Population'].astype(str).str.replace(',', ''), errors='coerce')

# --- MERGE GDP WITH POPULATION ---
df = df_gdp.merge(df_pop_clean, on='Country/Economy', how='left')
df = df.dropna(subset=['Population'])
df = df[df['Population'] > 0].copy()

# --- CALCULATE PER CAPITA GDP ---
df['Agriculture_PC'] = df['Agricultural (US$MM)'] * 1e6 / df['Population']
df['Industry_PC'] = df['Industrial (US$MM)'] * 1e6 / df['Population']
df['Services_PC'] = df['Service (US$MM)'] * 1e6 / df['Population']

# --- REGION GROUPING ---
simplified_regions = [
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
        'name': 'India',
        'countries': [
            'India'
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
        'name': 'Other Developed',
        'countries': [
            'Canada', 'Australia', 'New Zealand', 'Japan', 'South Korea', 'Taiwan',
            'Hong Kong', 'Macau'
        ]
    }
]

regions = [
    {
        'name': 'Western Europe',
        'countries': [
            'Austria', 'Belgium', 'Cyprus', 'Denmark', 'Finland', 'France', 'Germany',
            'Greece', 'Ireland', 'Italy', 'Luxembourg', 'Malta', 'Netherlands',
            'Portugal', 'Spain', 'Sweden', 'Norway', 'Switzerland', 'Liechtenstein',
            'Iceland'
        ]
    },
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
        'name': 'India',
        'countries': [
            'India'
        ]
    },
    {
        'name': 'Eastern Europe',
        'countries': [
            'Russia', 'Ukraine', 'Belarus', 'Armenia', 'Georgia', 'Azerbaijan',
            'Albania', 'Bulgaria', 'Croatia', 'Serbia', 'Bosnia and Herzegovina',
            'Czech Republic', 'Latvia', 'Lithuania', 'Estonia', 'Hungary', 'Poland',
            'Romania', 'Slovakia', 'Slovenia', 'North Macedonia', 'Montenegro'
        ]
    },
    {
        'name': 'ASEAN',
        'countries': [
            'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Malaysia', 'Myanmar',
            'Philippines', 'Thailand', 'Vietnam'
        ]
    },
    {
        'name': 'Asian Tigers & Japan',
        'countries': ['South Korea', 'Taiwan', 'Hong Kong', 'Macau', 'Singapore', 'Japan']
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

# regions = simplified_regions

df['Region'] = 'Other'
for region in regions:
    df.loc[df['Country/Economy'].isin(region['countries']), 'Region'] = region['name']

# --- AGGREGATE BY REGION ---
def aggregate_region(df_region, name):
    total_pop = df_region['Population'].sum()
    agg = pd.Series({
        'Country/Economy': name,
        'Agriculture_PC': (df_region['Agricultural (US$MM)'].sum() * 1e6) / total_pop,
        'Industry_PC':    (df_region['Industrial (US$MM)'].sum() * 1e6) / total_pop,
        'Services_PC':    (df_region['Service (US$MM)'].sum() * 1e6) / total_pop,
        'Population': total_pop
    })
    return agg

aggregated = [aggregate_region(df[df['Region'] == r['name']], r['name']) for r in regions]
aggregated.append(aggregate_region(df[df['Region'] == 'Other'], 'Rest of World'))
df_agg = pd.DataFrame(aggregated)

# --- CALCULATE BAR WIDTH ---
df_agg['Width'] = df_agg['Population'] / df_agg['Population'].max()  # scaled for plotting

# --- SORT FOR PLOTTING ---
df_agg['TotalGDP'] = df_agg[['Agriculture_PC', 'Industry_PC', 'Services_PC']].sum(axis=1)
df_agg = df_agg.sort_values(by='TotalGDP', ascending=False).reset_index(drop=True)

# --- PLOT STACKED BAR CHART ---
colors = {
    'Agriculture_PC': '#64c080',
    'Industry_PC': '#6490a6',
    'Services_PC': '#b04444'
}

def plot_gdp_stacked(df_final):
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#2a2a2a')
    ax.set_facecolor('#2a2a2a')

    # Compute left positions based on widths
    left_positions = np.cumsum([0] + df_final['Width'].tolist()[:-1])

    # Draw stacked bars
    bottom = np.zeros(len(df_final))
    for col in ['Agriculture_PC', 'Industry_PC', 'Services_PC']:
        ax.bar(
            left_positions, df_final[col],
            width=df_final['Width'],
            bottom=bottom,
            label=col.replace('_PC', ''),
            color=colors[col],
            edgecolor='#2a2a2a',
            linewidth=1.0,
            align='edge'  # Important for correct alignment
        )
        bottom += df_final[col].values

    # Set x-ticks at center of each bar
    ax.set_xticks(left_positions + df_final['Width'] / 2)
    ax.set_xticklabels(df_final['Country/Economy'], rotation=-25, ha='left', fontsize=10, color='white')

    # Axis labels and title
    ax.set_xlabel('Population 2025 (bar width)', color='white', fontsize=16)
    ax.set_ylabel('GDP per Capita (US$)', color='white', fontsize=16)
    ax.set_title('GDP by Sector Scaled by Population (2025)', color='white', fontsize=19)

    # Grid, ticks, legend styling
    ax.yaxis.grid(True, color='white', alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')

    # Limit x-axis to total width
    total_width = left_positions[-1] + df_final['Width'].iloc[-1]
    ax.set_xlim(left=0, right=total_width)

    # Legend
    legend = ax.legend(
        title='Sector',
        loc='upper right',
        facecolor='#2a2a2a',
        edgecolor='white',
        frameon=True,
        fontsize=12,
        title_fontsize=13
    )
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    # Source annotation
    ax.text(
        1.0, -0.12,
        "Source: CountryGDPBySector.txt & CountryPopulation.txt\nVisualization by u/MadoctheHadoc",
        fontsize=9,
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes
    )

    # Remove black spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("gdp_by_sector_scaled.png", dpi=300, bbox_inches='tight')
    plt.show()


# Plot the final result
plot_gdp_stacked(df_agg)
