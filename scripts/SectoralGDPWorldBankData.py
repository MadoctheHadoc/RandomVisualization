import wbdata
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ==================== CONFIG ====================
USE_PPP_ADJUSTED = True  # False for nominal
USE_SIMPLIFIED_REGIONS = True
DATA_YEAR = 2001

# Update the indicators dictionaries to include resource extraction
GDP_INDICATORS_PPP = {
    'NY.GDP.MKTP.PP.CD': 'TotalGDP_PPP',
    'NV.AGR.TOTL.ZS': 'Agriculture_pct',
    'NV.IND.TOTL.ZS': 'Industry_pct',
    'NV.SRV.TOTL.ZS': 'Services_pct',
    'NV.IND.MANF.ZS': 'Manufacturing_pct',
    'NY.GDP.TOTL.RT.ZS': 'ResourceExtraction_pct',
    'SL.AGR.EMPL.ZS': 'Agriculture_emp_pct',
    'SL.IND.EMPL.ZS': 'Industry_emp_pct',
    'SL.SRV.EMPL.ZS': 'Services_emp_pct'
}

GDP_INDICATORS_NOMINAL = {
    'NY.GDP.MKTP.CD': 'TotalGDP_USD',
    'NV.AGR.TOTL.ZS': 'Agriculture_pct',
    'NV.IND.TOTL.ZS': 'Industry_pct',
    'NV.SRV.TOTL.ZS': 'Services_pct',
    'NV.IND.MANF.ZS': 'Manufacturing_pct',
    'NY.GDP.TOTL.RT.ZS': 'ResourceExtraction_pct',
    'SL.AGR.EMPL.ZS': 'Agriculture_emp_pct',
    'SL.IND.EMPL.ZS': 'Industry_emp_pct',
    'SL.SRV.EMPL.ZS': 'Services_emp_pct'
}

LABOR_FORCE_INDICATOR = {'SL.TLF.TOTL.IN': 'LaborForce'}

GDP_COLUMN = 'TotalGDP_PPP' if USE_PPP_ADJUSTED else 'TotalGDP_USD'
GDP_LABEL = 'PPP-adjusted' if USE_PPP_ADJUSTED else 'nominal'

COLORS = {
    'Agriculture_PC': "#63c652",
    'ResourceExtraction_PC': "#6dbcfc",
    'OtherIndustry_PC': "#6d7bfc", 
    'Manufacturing_PC': "#9961d4",
    'Services_PC': "#c65252"
}


# ==================== REGIONS ====================
REGIONS = [
    {'name': 'Western\nEurope', 'countries': [
        'Austria', 'Belgium', 'Cyprus', 'Denmark', 'Finland', 'France', 'Germany',
        'Greece', 'Ireland', 'Italy', 'Luxembourg', 'Malta', 'Netherlands',
        'Portugal', 'Spain', 'Sweden', 'Norway', 'Switzerland', 'Liechtenstein',
        'Iceland'
    ]},
    {'name': 'USA', 'countries': ['United States']},
    {'name': 'China', 'countries': ['China']},
    {'name': 'India', 'countries': ['India']},
    {'name': 'Eastern\nEurope', 'countries': [
        'Russian Federation', 'Ukraine', 'Belarus', 'Armenia', 'Georgia', 'Azerbaijan',
        'Albania', 'Bulgaria', 'Croatia', 'Serbia', 'Bosnia and Herzegovina',
        'Czechia', 'Latvia', 'Lithuania', 'Estonia', 'Hungary', 'Poland',
        'Romania', 'Slovak Republic', 'Slovenia', 'North Macedonia', 'Montenegro'
    ]},
    {'name': 'ASEAN', 'countries': [
        'Brunei Darussalam', 'Cambodia', 'Indonesia', 'Lao PDR', 'Malaysia', 'Myanmar',
        'Philippines', 'Thailand', 'Viet Nam'
    ]},
    {'name': 'Asian Tigers\n& Japan', 'countries': [
        'Korea, Rep.', 'Taiwan', 'Hong Kong SAR, China', 'Macao SAR, China', 'Singapore', 'Japan'
    ]},
    {'name': 'Africa', 'countries': [
        'Nigeria', 'South Africa', 'Ethiopia', 'Kenya', 'Tanzania', 'Uganda', 'Ghana', 'Angola',
        'Mozambique', "Cote d'Ivoire", 'Cameroon', 'Niger', 'Burkina Faso', 'Mali', 'Malawi',
        'Zambia', 'Senegal', 'Zimbabwe', 'Rwanda', 'Benin', 'Chad', 'South Sudan', 'Togo',
        'Sierra Leone', 'Liberia', 'Botswana', 'Namibia', 'Somalia', 'Congo, Dem. Rep.', 'Congo, Rep.',
        'Gabon', 'Guinea', 'Mauritania', 'Equatorial Guinea', 'Central African Republic', 'Lesotho',
        'Eswatini', 'Djibouti', 'Eritrea', 'Gambia, The', 'Guinea-Bissau', 'Burundi', 'Cabo Verde',
        'Comoros', 'Sao Tome and Principe', 'Seychelles', 'Egypt, Arab Rep.', 'Libya', 'Morocco',
        'Algeria', 'Tunisia'
    ]},
    {'name': 'Middle\nEast', 'countries': [
        'Saudi Arabia', 'Iran, Islamic Rep.', 'Iraq', 'Israel', 'Jordan', 'Lebanon', 'Oman', 'Qatar',
        'United Arab Emirates', 'Bahrain', 'Kuwait', 'Syrian Arab Republic', 'Yemen, Rep.',
        'West Bank and Gaza', 'Turkiye'
    ]},
    {'name': 'Latin\nAmerica', 'countries': [
        'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Costa Rica', 'Cuba',
        'Dominican Republic', 'Ecuador', 'El Salvador', 'Guatemala', 'Honduras',
        'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela, RB',
        'Haiti', 'Jamaica', 'Trinidad and Tobago', 'Belize'
    ]},
    {'name': 'CANZUK', 'countries': ['Canada', 'Australia', 'New Zealand', 'United Kingdom']},
    {'name': 'Rest of Asia', 'countries': [
        'Afghanistan', 'Bangladesh', 'Bhutan', 'Nepal', 'Pakistan',  # South Asia not in India
        'Kazakhstan', 'Kyrgyz Republic', 'Tajikistan', 'Turkmenistan', 'Uzbekistan',  # Central Asia
        'Mongolia', 'Sri Lanka', 'Maldives'  # Other Asian states
    ]}
]

SIMPLIFIED_REGIONS = [
    {'name': 'USA', 'countries': ['United States']},
    {'name': 'China', 'countries': ['China']},
    {'name': 'South Asia', 'countries': [
        'India', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Nepal', 'Bhutan', 'Maldives', 'Afghanistan'
    ]},
    {'name': 'EEA', 'countries': [
        'Austria', 'Belgium', 'Cyprus', 'Denmark', 'Finland', 'France', 'Germany',
        'Greece', 'Ireland', 'Italy', 'Luxembourg', 'Malta', 'Netherlands',
        'Portugal', 'Spain', 'Sweden', 'Norway', 'Switzerland', 'Liechtenstein',
        'Iceland', 'Bulgaria', 'Croatia', 'Czechia', 'Latvia', 'Lithuania', 'Estonia',
        'Hungary', 'Poland', 'Romania', 'Slovak Republic', 'Slovenia'
    ]},
    {'name': 'ASEAN', 'countries': [
        'Brunei Darussalam', 'Cambodia', 'Indonesia', 'Lao PDR', 'Malaysia', 'Myanmar',
        'Philippines', 'Thailand', 'Viet Nam', 'Singapore'
    ]},
    {'name': 'Africa', 'countries': [
        'Nigeria', 'South Africa', 'Ethiopia', 'Kenya', 'Tanzania', 'Uganda', 'Ghana', 'Angola',
        'Mozambique', "Cote d'Ivoire", 'Cameroon', 'Niger', 'Burkina Faso', 'Mali', 'Malawi',
        'Zambia', 'Senegal', 'Zimbabwe', 'Rwanda', 'Benin', 'Chad', 'South Sudan', 'Togo',
        'Sierra Leone', 'Liberia', 'Botswana', 'Namibia', 'Somalia', 'Congo, Dem. Rep.', 'Congo, Rep.',
        'Gabon', 'Guinea', 'Mauritania', 'Equatorial Guinea', 'Central African Republic', 'Lesotho',
        'Eswatini', 'Djibouti', 'Eritrea', 'Gambia, The', 'Guinea-Bissau', 'Burundi', 'Cabo Verde',
        'Comoros', 'Sao Tome and Principe', 'Seychelles', 'Egypt, Arab Rep.', 'Libya', 'Morocco',
        'Algeria', 'Tunisia'
    ]},
    {'name': 'Middle\nEast', 'countries': [
        'Saudi Arabia', 'Iran, Islamic Rep.', 'Iraq', 'Israel', 'Jordan', 'Lebanon', 'Oman', 'Qatar',
        'United Arab Emirates', 'Bahrain', 'Kuwait', 'Syrian Arab Republic', 'Yemen, Rep.',
        'West Bank and Gaza', 'Turkiye'
    ]},
    {'name': 'Other\nDeveloped', 'countries': [
        'Japan', 'Korea, Rep.', 'Canada', 'Australia', 'New Zealand', 'Hong Kong SAR, China',
        'Macao SAR, China', 'Taiwan', 'Israel', 'United Kingdom'
    ]}
]

# ==================== FUNCTIONS ====================

def fetch_world_bank_data(indicators, year):
    """Fetch GDP/sector data and population for a given year."""
    df_gdp = wbdata.get_dataframe(indicators).reset_index()
    df_gdp['date'] = pd.to_datetime(df_gdp['date'])
    df_gdp = df_gdp[df_gdp['date'].dt.year == year]

    df_lab = wbdata.get_dataframe(LABOR_FORCE_INDICATOR).reset_index()
    df_lab['date'] = pd.to_datetime(df_lab['date'])
    df_lab = df_lab[df_lab['date'].dt.year == year]

    df = pd.merge(df_gdp, df_lab, on=['country', 'date'], how='inner')
    df.rename(columns={'country': 'Country'}, inplace=True)
    df['Country'] = df['Country'].str.strip()
    return df

def is_country_entity(name):
    aggregate_keywords = [
        '&', 'IDA', 'IBRD', 'income', 'demographic', 'OECD', 'HIPC',
        'small states', 'classification', 'members', 'countries', 'excluding', '(US)'
    ]
    if any(k in name for k in aggregate_keywords):
        if 'SAR' in name or 'Puerto Rico' in name:
            return True
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
    df = df.sort_values('date').drop_duplicates(subset=['Country'], keep='last')
    df = df[df['Country'].apply(is_country_entity)]
    return df

# Also need to fix the calculate_gdp_metrics function to avoid the pandas warning
def calculate_gdp_metrics(df):
    df = df.copy()  # Create a copy to avoid the warning
    
    df['Agricultural (PPP/USD)'] = df['Agriculture_pct'] / 100 * df[GDP_COLUMN]
    df['Industrial (PPP/USD)'] = df['Industry_pct'] / 100 * df[GDP_COLUMN]
    df['Service (PPP/USD)'] = df['Services_pct'] / 100 * df[GDP_COLUMN]

    df['LaborForce'] = df['LaborForce'].astype(float)
    df['Agriculture_PC'] = df['Agricultural (PPP/USD)'] / df['LaborForce']
    df['Industry_PC'] = df['Industrial (PPP/USD)'] / df['LaborForce']
    df['Services_PC'] = df['Service (PPP/USD)'] / df['LaborForce']

    df = df.dropna(subset=['Agriculture_emp_pct', 'Industry_emp_pct', 'Services_emp_pct'])
    df['Agriculture_Employment'] = df['Agriculture_emp_pct'] / 100 * df['LaborForce']
    df['Industry_Employment'] = df['Industry_emp_pct'] / 100 * df['LaborForce']
    df['Services_Employment'] = df['Services_emp_pct'] / 100 * df['LaborForce']

    df['Agriculture_Productivity'] = df['Agricultural (PPP/USD)'] / df['Agriculture_Employment']
    df['Industry_Productivity'] = df['Industrial (PPP/USD)'] / df['Industry_Employment']
    df['Services_Productivity'] = df['Service (PPP/USD)'] / df['Services_Employment']

    for col in ['Agriculture_Productivity', 'Industry_Productivity', 'Services_Productivity']:
        df[col] = df[col].replace([np.inf, -np.inf], 0)

    # Calculate manufacturing, resource extraction, and other industry
    df['Manufacturing (PPP/USD)'] = df['Manufacturing_pct'] / 100 * df[GDP_COLUMN]
    df['ResourceExtraction (PPP/USD)'] = df['ResourceExtraction_pct'] / 100 * df[GDP_COLUMN]
    df['OtherIndustry (PPP/USD)'] = df['Industrial (PPP/USD)'] - df['Manufacturing (PPP/USD)'] - df['ResourceExtraction (PPP/USD)']

    # Per capita values (now per labor force member)
    df['Manufacturing_PC'] = df['Manufacturing (PPP/USD)'] / df['LaborForce']
    df['ResourceExtraction_PC'] = df['ResourceExtraction (PPP/USD)'] / df['LaborForce']
    df['OtherIndustry_PC'] = df['OtherIndustry (PPP/USD)'] / df['LaborForce']

    # Productivity (assuming employment breakdown is only total industry)
    df['Manufacturing_Productivity'] = df['Manufacturing (PPP/USD)'] / df['Industry_Employment']
    df['ResourceExtraction_Productivity'] = df['ResourceExtraction (PPP/USD)'] / df['Industry_Employment']
    df['OtherIndustry_Productivity'] = df['OtherIndustry (PPP/USD)'] / df['Industry_Employment']

    # Replace inf values
    for col in ['Manufacturing_Productivity', 'ResourceExtraction_Productivity', 'OtherIndustry_Productivity']:
        df[col] = df[col].replace([np.inf, -np.inf], 0)

    return df

def assign_regions(df, regions):
    df['Region'] = 'Other'
    for region in regions:
        df.loc[df['Country'].isin(region['countries']), 'Region'] = region['name']
    return df

def aggregate_region(df_region, name):
    if len(df_region) == 0:
        return pd.Series({'Country': name, 'LaborForce': 0, **{k: 0 for k in [
            'Agriculture_PC', 'Industry_PC', 'Services_PC',
            'Agriculture_emp_pct', 'Industry_emp_pct', 'Services_emp_pct',
            'Agriculture_Productivity', 'Industry_Productivity', 'Services_Productivity',
            'Manufacturing_Productivity', 'ResourceExtraction_Productivity', 'OtherIndustry_Productivity',
            'Agriculture_Width', 'Industry_Width', 'Services_Width',
            'Agriculture_Employment', 'Industry_Employment', 'Services_Employment',
            'Manufacturing_Employment', 'ResourceExtraction_Employment', 'OtherIndustry_Employment'
        ]}})
    
    total_labor_force = df_region['LaborForce'].sum()
    agr_emp = df_region['Agriculture_Employment'].sum()
    ind_emp = df_region['Industry_Employment'].sum()
    srv_emp = df_region['Services_Employment'].sum()
    
    # Calculate employment for each industry subsector
    total_industrial_gdp = df_region['Industrial (PPP/USD)'].sum()
    total_manufacturing_gdp = df_region['Manufacturing (PPP/USD)'].sum()
    total_resource_gdp = df_region['ResourceExtraction (PPP/USD)'].sum()
    total_other_industry_gdp = total_industrial_gdp - total_manufacturing_gdp - total_resource_gdp
    
    # Calculate employment ratios based on GDP shares
    manufacturing_ratio = total_manufacturing_gdp / total_industrial_gdp if total_industrial_gdp > 0 else 0
    resource_ratio = total_resource_gdp / total_industrial_gdp if total_industrial_gdp > 0 else 0
    other_industry_ratio = max(0, 1 - manufacturing_ratio - resource_ratio)  # Ensure non-negative
    
    manufacturing_emp = ind_emp * manufacturing_ratio
    resource_emp = ind_emp * resource_ratio
    other_industry_emp = ind_emp * other_industry_ratio
    
    # Calculate width proportional to labor force * employment percentage
    agriculture_width = total_labor_force * (agr_emp / total_labor_force) if total_labor_force > 0 else 0
    industry_width = total_labor_force * (ind_emp / total_labor_force) if total_labor_force > 0 else 0
    services_width = total_labor_force * (srv_emp / total_labor_force) if total_labor_force > 0 else 0

    return pd.Series({
        'Country': name,
        'Agriculture_PC': df_region['Agricultural (PPP/USD)'].sum() / total_labor_force,
        'Industry_PC': df_region['Industrial (PPP/USD)'].sum() / total_labor_force,
        'Services_PC': df_region['Service (PPP/USD)'].sum() / total_labor_force,
        'Agriculture_emp_pct': agr_emp / total_labor_force * 100 if total_labor_force > 0 else 0,
        'Industry_emp_pct': ind_emp / total_labor_force * 100 if total_labor_force > 0 else 0,
        'Services_emp_pct': srv_emp / total_labor_force * 100 if total_labor_force > 0 else 0,
        'Agriculture_Productivity': df_region['Agricultural (PPP/USD)'].sum() / agr_emp if agr_emp > 0 else 0,
        'Industry_Productivity': df_region['Industrial (PPP/USD)'].sum() / ind_emp if ind_emp > 0 else 0,
        'Services_Productivity': df_region['Service (PPP/USD)'].sum() / srv_emp if srv_emp > 0 else 0,
        'Manufacturing_Productivity': total_manufacturing_gdp / manufacturing_emp if manufacturing_emp > 0 else 0,
        'ResourceExtraction_Productivity': total_resource_gdp / resource_emp if resource_emp > 0 else 0,
        'OtherIndustry_Productivity': total_other_industry_gdp / other_industry_emp if other_industry_emp > 0 else 0,
        'LaborForce': total_labor_force,
        'Agriculture_Width': agriculture_width,
        'Industry_Width': industry_width,
        'Services_Width': services_width,
        'Agriculture_Employment': agr_emp,
        'Industry_Employment': ind_emp,
        'Services_Employment': srv_emp,
        'Manufacturing_Employment': manufacturing_emp,
        'ResourceExtraction_Employment': resource_emp,
        'OtherIndustry_Employment': other_industry_emp
    })

def aggregate_all_regions(df, regions):
    aggregated = []
    for region in regions:
        aggregated.append(aggregate_region(df[df['Region'] == region['name']], region['name']))
    aggregated.append(aggregate_region(df[df['Region'] == 'Other'], 'Rest of World'))
    df_agg = pd.DataFrame(aggregated)
    df_agg = df_agg[df_agg['LaborForce'] > 0]
    return df_agg

def plot_productivity(df_final):    
    # Sort by GDP per capita (PPP)
    df_final = df_final.copy()  # Avoid the pandas warning
    df_final['GDP_per_capita'] = (df_final['Agriculture_PC'] + df_final['Industry_PC'] + df_final['Services_PC'])
    df_final = df_final.sort_values('GDP_per_capita', ascending=False)
    
    fig, ax = plt.subplots(figsize=(16, 16))
    if not USE_SIMPLIFIED_REGIONS:
        fig, ax = plt.subplots(figsize=(22, 16))
    
    fig.patch.set_facecolor('#2a2a2a')
    ax.set_facecolor('#2a2a2a')

    # Calculate total width (Agriculture + Industry + Services) - industry is combined
    df_final['Total_Width'] = (
        df_final['Agriculture_Width'] +
        df_final['Industry_Width'] +
        df_final['Services_Width']
    )
    
    # Normalize widths to make them reasonable for plotting
    max_width = df_final['Total_Width'].max()
    scale_factor = 1.0 / max_width if max_width > 0 else 1.0
    
    df_final['Agriculture_Width'] *= scale_factor
    df_final['Industry_Width'] *= scale_factor
    df_final['Services_Width'] *= scale_factor
    df_final['Total_Width'] *= scale_factor

    # Enforce minimum width after scaling
    min_width = 0.02
    df_final['Agriculture_Width'] = df_final['Agriculture_Width'].clip(lower=min_width)
    df_final['Industry_Width']    = df_final['Industry_Width'].clip(lower=min_width)
    df_final['Services_Width']    = df_final['Services_Width'].clip(lower=min_width)
    
    df_final['Total_Width']       = (
        df_final['Agriculture_Width'] +
        df_final['Industry_Width'] +
        df_final['Services_Width']
    )

    gap = 0.03
    country_positions = np.cumsum([0] + (df_final['Total_Width'] + gap).tolist()[:-1])

    linewidth = 2
    # Plot bars
    for i, (_, country) in enumerate(df_final.iterrows()):
        country_start = country_positions[i]
        # Agriculture
        if country['Agriculture_Width'] > 0:
            ax.bar(
                country_start, country['Agriculture_Productivity'],
                width=country['Agriculture_Width'],
                color=COLORS['Agriculture_PC'],
                edgecolor='#2a2a2a', linewidth=linewidth,
                align='edge', label='Agriculture' if i == 0 else ""
            )
            country_start += country['Agriculture_Width']

        # Industry (stacked: Manufacturing, Resource Extraction, Other Industry)

        # Calculate the employment ratios to determine stacking proportions
        total_ind_employment = country['Industry_Employment'] if country['Industry_Employment'] > 0 else 1
        manufacturing_employment = country.get('Manufacturing_Employment', total_ind_employment * 0.33)
        resource_employment = country.get('ResourceExtraction_Employment', total_ind_employment * 0.33)
        other_employment = country.get('OtherIndustry_Employment', total_ind_employment * 0.34)
        
        manufacturing_ratio = manufacturing_employment / total_ind_employment
        resource_ratio = resource_employment / total_ind_employment
        other_ratio = other_employment / total_ind_employment
        
        # Get productivities
        manufacturing_productivity = country.get('Manufacturing_Productivity', country['Industry_Productivity'])
        resource_productivity = country.get('ResourceExtraction_Productivity', country['Industry_Productivity'])
        other_productivity = country.get('OtherIndustry_Productivity', country['Industry_Productivity'])
        current_bottom = 0
        
        # Industry
        resource_height = resource_productivity * resource_ratio
        ax.bar(
            country_start, resource_height,
            width=country['Industry_Width'],
            bottom=current_bottom,
            color=COLORS['ResourceExtraction_PC'],
            edgecolor='#2a2a2a', linewidth=linewidth,
            align='edge', label='Resource Extraction' if i == 0 else ""
        )
        current_bottom += resource_height
        
        other_height = other_productivity * other_ratio
        ax.bar(
            country_start, other_height,
            width=country['Industry_Width'],
            bottom=current_bottom,
            color=COLORS['OtherIndustry_PC'],
            edgecolor='#2a2a2a', linewidth=linewidth,
            align='edge', label='Other Industry' if i == 0 else ""
        )
        current_bottom += other_height
            
        manufacturing_height = manufacturing_productivity * manufacturing_ratio
        ax.bar(
            country_start, manufacturing_height,
            width=country['Industry_Width'],
            bottom=current_bottom,
            color=COLORS['Manufacturing_PC'],
            edgecolor='#2a2a2a', linewidth=linewidth,
            align='edge', label='Manufacturing' if i == 0 else ""
        )
        
        country_start += country['Industry_Width']

        # Services
        ax.bar(
            country_start, country['Services_Productivity'],
            width=country['Services_Width'],
            color=COLORS['Services_PC'],
            edgecolor='#2a2a2a', linewidth=linewidth,
            align='edge', label='Services' if i == 0 else ""
        )

    # X-ticks in the middle of each country's total width
    country_centers = country_positions + df_final['Total_Width'] / 2
    ax.set_xticks(country_centers)
    ax.set_xticklabels(df_final['Country'])
    # ax.set_xticklabels(df_final['Country'], rotation=-30, ha='left')

    ax.set_xlabel('Labor Force', color='white', fontsize=24)
    ax.set_ylabel('GDP per Employed Person (international $)', color='white', fontsize=24)
    ax.set_title(f'GDP by Sector ({GDP_LABEL}, {DATA_YEAR})', color='white', fontsize=30)

    ax.yaxis.grid(True, color='white', alpha=0.2)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(labelsize=13)

    total_width = country_positions[-1] + df_final['Total_Width'].iloc[-1] if len(country_positions) > 0 else 1
    ax.set_xlim(left=0, right=total_width)

    legend = ax.legend(
        title='Sector',
        loc='center right',
        facecolor='#2a2a2a',
        edgecolor='white',
        frameon=True,
        fontsize=20,
        title_fontsize=18
    )
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    # Add explanatory text
    ax.text(
        1.0, -0.03,
        f"Source: World Bank API (GDP & Employment data, {DATA_YEAR})\n" +
        "Visualization by MadoctheHadoc",
        fontsize=14,
        color='white',
        ha='right',
        va='top',
        transform=ax.transAxes
    )
    
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(f"visualizations/SectorProductivity{DATA_YEAR}.png", dpi=300, bbox_inches='tight')
    plt.show()

# ==================== MAIN ====================
if __name__ == "__main__":
    indicators = GDP_INDICATORS_PPP if USE_PPP_ADJUSTED else GDP_INDICATORS_NOMINAL
    regions = SIMPLIFIED_REGIONS if USE_SIMPLIFIED_REGIONS else REGIONS
    
    print('Fetching...')
    df = fetch_world_bank_data(indicators, DATA_YEAR)
    df = clean_data(df)
    
    print('Aggregating...')
    df = calculate_gdp_metrics(df)
    df = assign_regions(df, regions)
    df_agg = aggregate_all_regions(df, regions)
    
    print('Rendering...')
    plot_productivity(df_agg)