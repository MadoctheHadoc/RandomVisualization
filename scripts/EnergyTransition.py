import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Define the file path relative to the script's location
script_dir = os.path.dirname(__file__)
energy_file_path = os.path.join(script_dir, '../data/EnergyGeneration.txt')
population_file_path = os.path.join(script_dir, '../data/CountryPopulation.txt')

# Read the energy data into a dataframe
df = pd.read_csv(energy_file_path, sep='\t')

# Clean up location names
df['Location'] = df['Location'].str.strip()

# Read and clean population data
df_population = pd.read_csv(population_file_path, sep='\t')
df_population['Country'] = df_population['Country'].str.strip()
df_population['Country'] = df_population['Country'].str.replace(r'\s*\([^)]*\)', '', regex=True)
df_population['Country'] = df_population['Country'].str.replace(r'\s*\[[^\]]*\]', '', regex=True)

# Extract and rename population column
df_population_clean = df_population[['Country', 'Population2025']].copy()
df_population_clean.rename(columns={
    'Country': 'Location',
    'Population2025': 'Population'
}, inplace=True)

# Merge energy data with population
df = df.merge(df_population_clean, on='Location', how='left')

# Convert Population column to numeric
df['Population'] = df['Population'].astype(str).str.replace(',', '')
df['Population'] = pd.to_numeric(df['Population'], errors='coerce')

# Drop countries with missing or zero population
df = df.dropna(subset=['Population'])
df = df[df['Population'] > 0].copy()

# Convert energy columns to numeric
energy_columns = ['Total', 'Coal', 'Gas', 'Hydro', 'Nuclear', 'Wind', 'Solar', 'Oil', 'Biomass', 'Geothermal']
for col in energy_columns:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

# Define countries to show individually
selected_countries = ['United States', 'China', 'India']

# Filter out the 'World' row
df_data = df[df['Location'] != 'World'].copy()

# Define regions and member countries
regions = [
    {
        'name': 'EU',
        'countries': [
            'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic', 'Denmark',
            'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy',
            'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal',
            'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden'
        ]
    },
    {
        'name': 'Other Europe',
        'countries': [
            'Russia', 'Ukraine', 'Belarus', 'Armenia', 'Georgia', 'Azerbaijan', 'Bosnia and Herzegovina', 'Serbia', 'Montenegro',
            'Albania', 'Kazakhstan', 'Kyrgyzstan', 'Tajikistan', 'Turkmenistan', 'Uzbekistan'
        ]
    },
    {
        'name': 'Other Asia',
        'countries': [
            'Brunei', 'Cambodia', 'Indonesia', 'Laos', 'Malaysia', 'Myanmar',
            'Philippines', 'Singapore', 'Thailand', 'Vietnam', 'Bangladesh', 'Pakistan', 'Nepal', 'Mongolia', 'North Korea'
        ]
    },
    {
        'name': 'Developed Asia',
        'countries': ['Japan', 'South Korea', 'Taiwan', 'Hong Kong', 'Macau']
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
    'name': 'Other Developed',
    'countries': [
        'Canada', 'Australia', 'New Zealand', 'United Kingdom', 'Norway', 'Switzerland', 'Liechtenstein'
    ]
    }
]

# Tag regions
df_data['Region'] = 'Other'
for region in regions:
    df_data.loc[df_data['Location'].isin(region['countries']), 'Region'] = region['name']

# Track selected country list update
flat_region_countries = [country for r in regions for country in r['countries']]
selected_countries = [c for c in selected_countries if c not in flat_region_countries]

# Remove region countries from individual display
df_data_filtered = df_data[~df_data['Location'].isin(flat_region_countries)].copy()

# Function to aggregate regions
def aggregate_region(df_region, name):
    df_sum = df_region[energy_columns].sum()
    population_sum = df_region['Population'].sum()
    return pd.DataFrame({
        'Location': [name],
        'Total': [df_sum['Total']],
        'Coal': [df_sum['Coal']],
        'Gas': [df_sum['Gas']],
        'Hydro': [df_sum['Hydro']],
        'Nuclear': [df_sum['Nuclear']],
        'Wind': [df_sum['Wind']],
        'Solar': [df_sum['Solar']],
        'Oil': [df_sum['Oil']],
        'Biomass': [df_sum['Biomass']],
        'Geothermal': [df_sum['Geothermal']],
        'Population': [population_sum]
    })

# Create aggregated region rows
aggregated_regions = []
for region in regions:
    df_region = df_data[df_data['Region'] == region['name']]
    if not df_region.empty:
        aggregated_regions.append(aggregate_region(df_region, region['name']))

# Re-separate selected and other
df_selected = df_data_filtered[df_data_filtered['Location'].isin(selected_countries)].copy()
df_other = df_data_filtered[~df_data_filtered['Location'].isin(selected_countries)].copy()

# "Other" region aggregation
df_other_sum = df_other[energy_columns].sum()
total_other_population = df_other['Population'].sum()
df_other_df = pd.DataFrame({
    'Location': ['Other'],
    'Total': [df_other_sum['Total']],
    'Coal': [df_other_sum['Coal']],
    'Gas': [df_other_sum['Gas']],
    'Hydro': [df_other_sum['Hydro']],
    'Nuclear': [df_other_sum['Nuclear']],
    'Wind': [df_other_sum['Wind']],
    'Solar': [df_other_sum['Solar']],
    'Oil': [df_other_sum['Oil']],
    'Biomass': [df_other_sum['Biomass']],
    'Geothermal': [df_other_sum['Geothermal']],
    'Population': [total_other_population]
})

# Final DataFrame: selected + regions + other
df_final = pd.concat([df_selected] + aggregated_regions + [df_other_df], ignore_index=True)


# Normalize energy values by population (TWh per person)
for col in energy_columns:
    df_final[col] = df_final[col] / df_final['Population']

# Normalize population to use for bar width (scaling for visibility)
max_width = 1.0
df_final['Width'] = df_final['Population'] / df_final['Population'].max() * max_width

# Plot
# Sort by Total per capita energy generation (descending)s
df_final = df_final.sort_values(by='Total', ascending=False).reset_index(drop=True)

# Define new energy order and pastel color palette
ordered_columns = ['Hydro', 'Wind', 'Solar', 'Geothermal', 'Biomass', 'Nuclear', 'Gas', 'Oil', 'Coal']
pastel_colors = {
    'Hydro':     '#6470c0',
    'Wind':      '#6480b3',
    'Solar':     '#6490a6',
    'Geothermal':'#64a09a',
    'Biomass':   '#64af8e',
    'Nuclear':   '#64c080',
    'Gas': '#fcad9c',
    'Oil': '#fc9882',
    'Coal': '#fc8469'
}

# Plot
# Plot
fig, ax = plt.subplots(figsize=(14, 8))
fig.patch.set_facecolor('#2a2a2a')  # Dark background
ax.set_facecolor('#2a2a2a')

# Compute left positions based on widths
left_positions = np.cumsum([0] + df_final['Width'].tolist()[:-1])

# Draw stacked bars
bottom = np.zeros(len(df_final))
for col in ordered_columns:
    ax.bar(left_positions, df_final[col],
           width=df_final['Width'], bottom=bottom,
           label=col, color=pastel_colors[col], align='edge',
           edgecolor='black', linewidth=0.2)
    bottom += df_final[col].values

# Set x-axis labels below bars
ax.set_xticks(left_positions + df_final['Width'] / 2)
ax.set_xticklabels(df_final['Location'], rotation=-25, ha='left', fontsize=10, color='white')

# Add y-axis grid lines
ax.yaxis.grid(True, color='white', alpha=0.2)
ax.set_axisbelow(True)

# Axis and label styling
ax.set_xlabel('Population 2025', color='white', fontsize=16)
ax.set_ylabel('Per Capita Electricity Generation (TWh per person) in 2024', color='white', fontsize=16)
ax.set_title('Electricity Generation by Population and Source', color='white', fontsize=19)

# White y-ticks
ax.tick_params(axis='y', colors='white')
# White x-ticks
ax.tick_params(axis='x', colors='white')

# Adjust x-axis limit to avoid trailing whitespace
total_width = left_positions[-1] + df_final['Width'].iloc[-1]
ax.set_xlim(left=0, right=total_width)

# Legend with dark background and white text
legend = ax.legend(
    title="Generation Source",
    loc='upper right',
    facecolor='#2a2a2a',
    edgecolor='white',
    frameon=True,
    fontsize=12,         # Legend text size
    title_fontsize=13    # Legend title size
)
plt.setp(legend.get_texts(), color='white')
plt.setp(legend.get_title(), color='white')

# Remove black outline (spines)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.show()
