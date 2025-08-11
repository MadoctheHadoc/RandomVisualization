import wbdata
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- World Bank indicators ---
# Choose between nominal GDP or PPP-adjusted GDP
USE_PPP_ADJUSTED = True  # Set to False for nominal GDP in current USD

if USE_PPP_ADJUSTED:
    indicators = {
        'NY.GDP.MKTP.PP.CD': 'TotalGDP_PPP',        # GDP, PPP (current international $)
        'NV.AGR.TOTL.ZS': 'Agriculture_pct',        # Agriculture value added (% of GDP)
        'NV.IND.TOTL.ZS': 'Industry_pct',           # Industry value added (% of GDP)
        'NV.SRV.TOTL.ZS': 'Services_pct',           # Services value added (% of GDP)
        'SL.AGR.EMPL.ZS': 'Agriculture_emp_pct',    # Employment in agriculture (% of total employment)
        'SL.IND.EMPL.ZS': 'Industry_emp_pct',       # Employment in industry (% of total employment)  
        'SL.SRV.EMPL.ZS': 'Services_emp_pct'        # Employment in services (% of total employment)
    }
    gdp_column = 'TotalGDP_PPP'
    gdp_label = 'PPP-adjusted'
else:
    indicators = {
        'NY.GDP.MKTP.CD': 'TotalGDP_USD',           # GDP (current US$)
        'NV.AGR.TOTL.ZS': 'Agriculture_pct',        # Agriculture value added (% of GDP)
        'NV.IND.TOTL.ZS': 'Industry_pct',           # Industry value added (% of GDP)
        'NV.SRV.TOTL.ZS': 'Services_pct',           # Services value added (% of GDP)
        'SL.AGR.EMPL.ZS': 'Agriculture_emp_pct',    # Employment in agriculture (% of total employment)
        'SL.IND.EMPL.ZS': 'Industry_emp_pct',       # Employment in industry (% of total employment)
        'SL.SRV.EMPL.ZS': 'Services_emp_pct'        # Employment in services (% of total employment)
    }
    gdp_column = 'TotalGDP_USD'
    gdp_label = 'nominal'

pop_indicator = {'SP.POP.TOTL': 'Population'}

# --- Fetch data for recent year (e.g. 2021) ---
data_year = 2022

# Fetch GDP & sector data
df_gdp = wbdata.get_dataframe(indicators)
df_gdp = df_gdp.reset_index()
df_gdp['date'] = pd.to_datetime(df_gdp['date'])
df_gdp = df_gdp[df_gdp['date'].dt.year == data_year]

# Fetch population data
df_pop = wbdata.get_dataframe(pop_indicator)
df_pop = df_pop.reset_index()
df_pop['date'] = pd.to_datetime(df_pop['date'])
df_pop = df_pop[df_pop['date'].dt.year == data_year]

# Merge GDP and population data
df = pd.merge(df_gdp, df_pop, on=['country', 'date'], how='inner')

# Clean country names column - rename to 'Country'
df.rename(columns={'country': 'Country'}, inplace=True)
df['Country'] = df['Country'].str.strip()

print(f"Initial dataset: {len(df)} rows")
print(f"Countries before deduplication: {df['Country'].nunique()} unique")

# Check for duplicates BEFORE removing them
duplicates = df[df.duplicated(subset=['Country'], keep=False)]
if len(duplicates) > 0:
    print(f"\nFound {len(duplicates)} duplicate rows:")
    print(duplicates[['Country', 'date', 'Population']].sort_values(['Country', 'date']))

# Remove duplicates - keep latest year rows
df = df.sort_values('date').drop_duplicates(subset=['Country'], keep='last')
print(f"After deduplication: {len(df)} rows, {df['Country'].nunique()} unique countries")

# --- Filter out World Bank regional/income/development aggregates ---
# More comprehensive approach: filter out anything that looks like an aggregate
def is_country_entity(country_name):
    """Return True if this looks like an actual country, False if it's an aggregate"""
    
    # Direct aggregate patterns
    aggregate_keywords = [
        '&', 'IDA', 'IBRD', 'income', 'demographic', 'OECD', 'HIPC',
        'small states', 'classification', 'members', 'countries',
        'excluding', '(US)', 'SAR'  # Special Administrative Region entries like Hong Kong should be kept
    ]
    
    # Remove SAR from the list since we want to keep those
    aggregate_keywords = [k for k in aggregate_keywords if k != 'SAR']
    
    # Check for aggregate keywords (but keep SAR entities)
    if any(keyword in country_name for keyword in aggregate_keywords):
        # Exception: keep SAR entities (Hong Kong, Macau)
        if 'SAR' in country_name:
            return True
        # Exception: keep Puerto Rico
        if 'Puerto Rico' in country_name:
            return True
        return False
    
    # Geographic aggregate regions
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
    
    if any(country_name.startswith(geo) for geo in geographic_aggregates):
        return False
    
    return True

# Filter out aggregates
df_before_filter = len(df)
df = df[df['Country'].apply(is_country_entity)]
print(f"After filtering out {df_before_filter - len(df)} aggregate entities: {len(df)} countries")

# --- Calculate sector GDP in PPP or USD ---
df['Agricultural (PPP/USD)'] = df['Agriculture_pct'] / 100 * df[gdp_column]
df['Industrial (PPP/USD)'] = df['Industry_pct'] / 100 * df[gdp_column]
df['Service (PPP/USD)'] = df['Services_pct'] / 100 * df[gdp_column]

# --- Calculate per capita GDP by sector ---
df['Population'] = df['Population'].astype(float)
df['Agriculture_PC'] = df['Agricultural (PPP/USD)'] / df['Population']
df['Industry_PC'] = df['Industrial (PPP/USD)'] / df['Population']
df['Services_PC'] = df['Service (PPP/USD)'] / df['Population']

# --- Calculate GDP per worker by sector (productivity) ---
# This requires employment data - filter out countries without employment data
df = df.dropna(subset=['Agriculture_emp_pct', 'Industry_emp_pct', 'Services_emp_pct'])
print(f"After filtering for employment data: {len(df)} countries")

# Calculate total employment (assuming total labor force)
# We'll use a proxy: assume total employment is proportional to population aged 15-64
# For simplicity, we'll use total population * 0.6 as rough employed population estimate
df['Total_Employment_Est'] = df['Population'] * 0.6  # Rough estimate

# Calculate employment by sector
df['Agriculture_Employment'] = df['Agriculture_emp_pct'] / 100 * df['Total_Employment_Est']
df['Industry_Employment'] = df['Industry_emp_pct'] / 100 * df['Total_Employment_Est']  
df['Services_Employment'] = df['Services_emp_pct'] / 100 * df['Total_Employment_Est']

# Calculate GDP per worker (productivity) by sector
df['Agriculture_Productivity'] = df['Agricultural (PPP/USD)'] / df['Agriculture_Employment']
df['Industry_Productivity'] = df['Industrial (PPP/USD)'] / df['Industry_Employment']
df['Services_Productivity'] = df['Service (PPP/USD)'] / df['Services_Employment']

# Handle division by zero (when employment is 0%)
df['Agriculture_Productivity'] = df['Agriculture_Productivity'].replace([np.inf, -np.inf], 0)
df['Industry_Productivity'] = df['Industry_Productivity'].replace([np.inf, -np.inf], 0)
df['Services_Productivity'] = df['Services_Productivity'].replace([np.inf, -np.inf], 0)

# Use your original manual regional groupings
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
        'countries': ['United States']
    },
    {
        'name': 'China',
        'countries': ['China']
    },
    {
        'name': 'India',
        'countries': ['India']
    },
    {
        'name': 'Eastern Europe',
        'countries': [
            'Russian Federation', 'Ukraine', 'Belarus', 'Armenia', 'Georgia', 'Azerbaijan',
            'Albania', 'Bulgaria', 'Croatia', 'Serbia', 'Bosnia and Herzegovina',
            'Czechia', 'Latvia', 'Lithuania', 'Estonia', 'Hungary', 'Poland',
            'Romania', 'Slovak Republic', 'Slovenia', 'North Macedonia', 'Montenegro'
        ]
    },
    {
        'name': 'ASEAN',
        'countries': [
            'Brunei Darussalam', 'Cambodia', 'Indonesia', 'Lao PDR', 'Malaysia', 'Myanmar',
            'Philippines', 'Thailand', 'Viet Nam'
        ]
    },
    {
        'name': 'Asian Tigers & Japan',
        'countries': ['Korea, Rep.', 'Taiwan', 'Hong Kong SAR, China', 'Macao SAR, China', 'Singapore', 'Japan']
    },
    {
        'name': 'Africa',
        'countries': [
            'Nigeria', 'South Africa', 'Ethiopia', 'Kenya', 'Tanzania', 'Uganda', 'Ghana', 'Angola',
            'Mozambique', 'Cote d\'Ivoire', 'Cameroon', 'Niger', 'Burkina Faso', 'Mali', 'Malawi',
            'Zambia', 'Senegal', 'Zimbabwe', 'Rwanda', 'Benin', 'Chad', 'South Sudan', 'Togo',
            'Sierra Leone', 'Liberia', 'Botswana', 'Namibia', 'Somalia', 'Congo, Dem. Rep.', 'Congo, Rep.',
            'Gabon', 'Guinea', 'Mauritania', 'Equatorial Guinea', 'Central African Republic', 'Lesotho',
            'Eswatini', 'Djibouti', 'Eritrea', 'Gambia, The', 'Guinea-Bissau', 'Burundi', 'Cabo Verde',
            'Comoros', 'Sao Tome and Principe', 'Seychelles', 'Egypt, Arab Rep.', 'Libya', 'Morocco', 'Algeria', 'Tunisia'
        ]
    },
    {
        'name': 'Middle East',
        'countries': [
            'Saudi Arabia', 'Iran, Islamic Rep.', 'Iraq', 'Israel', 'Jordan', 'Lebanon', 'Oman', 'Qatar', 
            'United Arab Emirates', 'Bahrain', 'Kuwait', 'Syrian Arab Republic', 'Yemen, Rep.', 'West Bank and Gaza', 'Turkiye'
        ]
    },
    {
        'name': 'Latin America',
        'countries': [
            'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Costa Rica', 'Cuba',
            'Dominican Republic', 'Ecuador', 'El Salvador', 'Guatemala', 'Honduras',
            'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela, RB',
            'Haiti', 'Jamaica', 'Trinidad and Tobago', 'Belize'
        ]
    },
    {
        'name': 'CANZUK',
        'countries': ['Canada', 'Australia', 'New Zealand', 'United Kingdom']
    }
]


# --- Assign regions to countries ---
df['Region'] = 'Other'
total_assigned = 0
for region in regions:
    matches = df['Country'].isin(region['countries'])
    df.loc[matches, 'Region'] = region['name']
    assigned_count = matches.sum()
    total_assigned += assigned_count
    
    if assigned_count > 0:
        assigned_pop = df[df['Region'] == region['name']]['Population'].sum()

other_count = len(df[df['Region'] == 'Other'])
other_pop = df[df['Region'] == 'Other']['Population'].sum()

# --- Aggregate by region ---
def aggregate_region(df_region, name):
    if len(df_region) == 0:
        return pd.Series({
            'Country': name,
            'Agriculture_PC': 0,
            'Industry_PC': 0,
            'Services_PC': 0,
            'Agriculture_emp_pct': 0,
            'Industry_emp_pct': 0,
            'Services_emp_pct': 0,
            'Agriculture_Productivity': 0,
            'Industry_Productivity': 0,
            'Services_Productivity': 0,
            'Population': 0
        })
    
    total_pop = df_region['Population'].sum()
    total_employment = df_region['Total_Employment_Est'].sum()
    
    # Aggregate employment percentages (weighted by employment)
    agr_emp_total = df_region['Agriculture_Employment'].sum()
    ind_emp_total = df_region['Industry_Employment'].sum()
    srv_emp_total = df_region['Services_Employment'].sum()
    
    agg = pd.Series({
        'Country': name,
        'Agriculture_PC': (df_region['Agricultural (PPP/USD)'].sum()) / total_pop if total_pop > 0 else 0,
        'Industry_PC': (df_region['Industrial (PPP/USD)'].sum()) / total_pop if total_pop > 0 else 0,
        'Services_PC': (df_region['Service (PPP/USD)'].sum()) / total_pop if total_pop > 0 else 0,
        'Agriculture_emp_pct': (agr_emp_total / total_employment * 100) if total_employment > 0 else 0,
        'Industry_emp_pct': (ind_emp_total / total_employment * 100) if total_employment > 0 else 0,
        'Services_emp_pct': (srv_emp_total / total_employment * 100) if total_employment > 0 else 0,
        'Agriculture_Productivity': (df_region['Agricultural (PPP/USD)'].sum()) / agr_emp_total if agr_emp_total > 0 else 0,
        'Industry_Productivity': (df_region['Industrial (PPP/USD)'].sum()) / ind_emp_total if ind_emp_total > 0 else 0,
        'Services_Productivity': (df_region['Service (PPP/USD)'].sum()) / srv_emp_total if srv_emp_total > 0 else 0,
        'Population': total_pop
    })
    return agg

# Aggregate defined regions
aggregated = []

# Use manual regions
for region in regions:
    region_df = df[df['Region'] == region['name']]
    if len(region_df) > 0:  # Only add if region has countries
        aggregated.append(aggregate_region(region_df, region['name']))

# Handle "Rest of World" - use countries marked as 'Other'
rest_of_world_df = df[df['Region'] == 'Other']
if len(rest_of_world_df) > 0:
    aggregated.append(aggregate_region(rest_of_world_df, 'Rest of World'))

df_agg = pd.DataFrame(aggregated)

# Remove empty regions (population = 0)
df_agg = df_agg[df_agg['Population'] > 0]

# --- Calculate bar widths based on employment ---
# Each sector's width is proportional to its employment percentage
# Total width is still proportional to population for country comparison
df_agg['Base_Width'] = df_agg['Population'] / df_agg['Population'].max()

# Calculate sector widths within each country's total width
df_agg['Agriculture_Width'] = df_agg['Base_Width'] * (df_agg['Agriculture_emp_pct'] / 100)
df_agg['Industry_Width'] = df_agg['Base_Width'] * (df_agg['Industry_emp_pct'] / 100)
df_agg['Services_Width'] = df_agg['Base_Width'] * (df_agg['Services_emp_pct'] / 100)

# Ensure widths don't exceed base width due to rounding
total_sector_width = df_agg['Agriculture_Width'] + df_agg['Industry_Width'] + df_agg['Services_Width']
scaling_factor = df_agg['Base_Width'] / total_sector_width
df_agg['Agriculture_Width'] *= scaling_factor
df_agg['Industry_Width'] *= scaling_factor
df_agg['Services_Width'] *= scaling_factor

# --- Sort for plotting ---
df_agg['TotalGDP'] = df_agg[['Agriculture_PC', 'Industry_PC', 'Services_PC']].sum(axis=1)
df_agg = df_agg.sort_values(by='TotalGDP', ascending=False).reset_index(drop=True)

# --- Plotting colors ---
colors = {
    'Agriculture_PC': '#64c080',
    'Industry_PC': '#6490a6',
    'Services_PC': '#b04444'
}

def plot_gdp_employment_stacked(df_final):
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('#2a2a2a')
    ax.set_facecolor('#2a2a2a')

    # Calculate the maximum width for each country
    df_final['Max_Width'] = df_final[['Agriculture_Width', 'Industry_Width', 'Services_Width']].max(axis=1)

    # Calculate positions for each country
    country_positions = np.cumsum([0] + df_final['Max_Width'].tolist()[:-1])

    # Plot each sector as stacked bars with widths proportional to employment
    for i, (_, country) in enumerate(df_final.iterrows()):
        country_start = country_positions[i]
        max_width = country['Max_Width']

        # Calculate the offset to center each bar
        def get_offset(sector_width):
            return (max_width - sector_width) / 2

        # Plot services bar at the bottom
        if country['Services_Width'] > 0:
            offset = get_offset(country['Services_Width'])
            ax.bar(country_start + offset, country['Services_PC'],
                  width=country['Services_Width'],
                  color=colors['Services_PC'],
                  edgecolor='#2a2a2a', linewidth=0.5,
                  align='edge', label='Services' if i == 0 else "")

        # Plot industry bar on top of services bar
        if country['Industry_Width'] > 0:
            offset = get_offset(country['Industry_Width'])
            ax.bar(country_start + offset, country['Industry_PC'],
                  width=country['Industry_Width'],
                  color=colors['Industry_PC'],
                  edgecolor='#2a2a2a', linewidth=0.5,
                  align='edge', bottom=country['Services_PC'],
                  label='Industry' if i == 0 else "")

        # Plot agriculture bar on top of industry bar
        if country['Agriculture_Width'] > 0:
            offset = get_offset(country['Agriculture_Width'])
            ax.bar(country_start + offset, country['Agriculture_PC'],
                  width=country['Agriculture_Width'],
                  color=colors['Agriculture_PC'],
                  edgecolor='#2a2a2a', linewidth=0.5,
                  align='edge', bottom=country['Services_PC'] + country['Industry_PC'],
                  label='Agriculture' if i == 0 else "")

    # Set country labels at center of each country's total width
    country_centers = country_positions + df_final['Max_Width'] / 2
    ax.set_xticks(country_centers)
    ax.set_xticklabels(df_final['Country'], rotation=-25, ha='left', fontsize=10, color='white')

    ax.set_xlabel('Population (total width) × Employment % (sector width)', color='white', fontsize=14)
    ax.set_ylabel(f'GDP per Capita by Sector ({gdp_label}, international $)', color='white', fontsize=14)
    ax.set_title(f'GDP per Capita & Employment by Sector ({gdp_label} GDP, {data_year})', color='white', fontsize=17)

    ax.yaxis.grid(True, color='white', alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')

    total_width = country_positions[-1] + df_final['Max_Width'].iloc[-1]
    ax.set_xlim(left=0, right=total_width)

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

    # Add explanatory text
    ax.text(
        1.0, -0.15,
        f"Source: World Bank API (GDP & Employment data, {data_year})\n" +
        f"Bar width ∝ Population × Employment %. Bar height = GDP per capita by sector.\n" +
        f"Visualization by u/MadoctheHadoc",
        fontsize=9,
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("gdp_employment_by_sector_scaled.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print productivity insights
    print(f"\n=== PRODUCTIVITY INSIGHTS ({gdp_label} GDP per worker) ===")
    for _, country in df_final.iterrows():
        if country['Population'] > 0:
            print(f"\n{country['Country']}:")
            print(f"  Agriculture: {country['Agriculture_Productivity']:,.0f} $/worker ({country['Agriculture_emp_pct']:.1f}% employment)")
            print(f"  Industry:    {country['Industry_Productivity']:,.0f} $/worker ({country['Industry_emp_pct']:.1f}% employment)")
            print(f"  Services:    {country['Services_Productivity']:,.0f} $/worker ({country['Services_emp_pct']:.1f}% employment)")

def plot_productivity_stacked(df_final):
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.patch.set_facecolor('#2a2a2a')
    ax.set_facecolor('#2a2a2a')

    # Calculate the maximum width for each country
    df_final['Max_Width'] = df_final[['Agriculture_Width', 'Industry_Width', 'Services_Width']].max(axis=1)

    # Calculate positions for each country
    country_positions = np.cumsum([0] + df_final['Max_Width'].tolist()[:-1])

    # Plot each sector as stacked bars with widths proportional to employment
    for i, (_, country) in enumerate(df_final.iterrows()):
        country_start = country_positions[i]
        max_width = country['Max_Width']

        # Calculate the offset to center each bar
        def get_offset(sector_width):
            return (max_width - sector_width) / 2

        # Plot agriculture bar on top of industry bar
        if country['Agriculture_Width'] > 0:
            offset = get_offset(country['Agriculture_Width'])
            ax.bar(country_start + offset, country['Agriculture_Productivity'],
                  width=country['Agriculture_Width'],
                  color=colors['Agriculture_PC'],
                  edgecolor='#2a2a2a', linewidth=0.2,
                  align='edge', bottom=country['Services_Productivity'] + country['Industry_Productivity'],
                  label='Agriculture' if i == 0 else "")

        # Plot industry bar on top of services bar
        if country['Industry_Width'] > 0:
            offset = get_offset(country['Industry_Width'])
            ax.bar(country_start + offset, country['Industry_Productivity'],
                  width=country['Industry_Width'],
                  color=colors['Industry_PC'],
                  edgecolor='#2a2a2a', linewidth=0.2,
                  align='edge', bottom=country['Services_Productivity'],
                  label='Industry' if i == 0 else "")

        # Plot services bar at the bottom
        if country['Services_Width'] > 0:
            offset = get_offset(country['Services_Width'])
            ax.bar(country_start + offset, country['Services_Productivity'],
                  width=country['Services_Width'],
                  color=colors['Services_PC'],
                  edgecolor='#2a2a2a', linewidth=0.2,
                  align='edge', label='Services' if i == 0 else "")

    # Set country labels at center of each country's total width
    country_centers = country_positions + df_final['Max_Width'] / 2
    ax.set_xticks(country_centers)
    ax.set_xticklabels(df_final['Country'], rotation=-25, ha='left', fontsize=10, color='white')

    ax.set_xlabel('Population (total width) × Employment % (sector width)', color='white', fontsize=14)
    ax.set_ylabel('Productivity by Sector (international $)', color='white', fontsize=14)
    ax.set_title(f'Productivity by Sector ({gdp_label} GDP, {data_year})', color='white', fontsize=17)

    ax.yaxis.grid(True, color='white', alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')

    total_width = country_positions[-1] + df_final['Max_Width'].iloc[-1]
    ax.set_xlim(left=0, right=total_width)

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

    # Add explanatory text
    ax.text(
        1.0, -0.15,
        f"Source: World Bank API (GDP & Employment data, {data_year})\n" +
        f"Bar width ∝ Population × Employment %. Bar height = Productivity by sector.\n" +
        f"Visualization by u/MadoctheHadoc",
        fontsize=9,
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig("productivity_by_sector_scaled.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Print productivity insights
    print(f"\n=== PRODUCTIVITY INSIGHTS ({gdp_label} GDP per worker) ===")
    for _, country in df_final.iterrows():
        if country['Population'] > 0:
            print(f"\n{country['Country']}:")
            print(f"  Agriculture: {country['Agriculture_Productivity']:,.0f} $/worker ({country['Agriculture_emp_pct']:.1f}% employment)")
            print(f"  Industry:    {country['Industry_Productivity']:,.0f} $/worker ({country['Industry_emp_pct']:.1f}% employment)")
            print(f"  Services:    {country['Services_Productivity']:,.0f} $/worker ({country['Services_emp_pct']:.1f}% employment)")


# --- Plot ---
plot_productivity_stacked(df_agg)