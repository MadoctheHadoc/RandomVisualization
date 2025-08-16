import wbdata
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ==================== CONFIG ====================
USE_PPP_ADJUSTED = True
USE_SIMPLIFIED_REGIONS = False
DATA_YEAR = 2021

GDP_INDICATORS_PPP = {
    'NY.GDP.MKTP.PP.CD': 'TotalGDP_PPP',
    'NV.AGR.TOTL.ZS': 'Agriculture_pct',
    'NV.IND.TOTL.ZS': 'Industry_pct',
    'NV.SRV.TOTL.ZS': 'Services_pct',
    'NV.IND.MANF.ZS': 'Manufacturing_pct',
    'NY.GDP.TOTL.RT.ZS': 'ResourceExtraction_pct',
    'SL.AGR.EMPL.ZS': 'Agriculture_emp_pct',
    'SL.IND.EMPL.ZS': 'Industry_emp_pct',
    'SL.SRV.EMPL.ZS': 'Services_emp_pct',
    'PA.NUS.PPPC.RF': 'PPP_Conversion_Factor'
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

# Define color constants
BG_COLOR = '#2a2a2a'
ANNO_COLOR = "#848484"

COLORS = {
    # Agriculture (muted green)
    'Agriculture': "#99C86B",
    # Industry subsectors (cool tones)
    'Manufacturing': "#00B4D8",
    'OtherIndustry': "#5C89C7",
    'ResourceExtraction': "#7B6BC8",
    # Services subsectors (warm tones)
    'InformationCommunication': "#FF6B6B",
    'FinancialInsurance':      "#FF914D",
    'RealEstate':              "#FFB347",
    'PublicServices':          "#E68AB8",
    'ProfessionalServices':    "#CD629B",
    'WholesaleRetail':         "#D1826F",
    'OtherServices':           "#CD6B62",
}

OECD_SUBSECTORS = [
     'WholesaleRetail', 'ProfessionalServices', 'PublicServices', 
     'RealEstate', 'FinancialInsurance', 'InformationCommunication',
]

SECTOR_SPLITS = [
    ('Agriculture', ['Agriculture']),
    ('Industry', ['ResourceExtraction', 'OtherIndustry', 'Manufacturing']),
    ('Services', (['OtherServices'] + OECD_SUBSECTORS))
]

SUBSECTOR_DISPLAY_NAMES = {
    'Agriculture': 'Agriculture',
    'ResourceExtraction': 'Resource Extraction',
    'OtherIndustry': 'Other Industry',
    'Manufacturing': 'Manufacturing',
    'WholesaleRetail': 'Wholesale & Retail Trade',
    'RealEstate': 'Real Estate',
    'ProfessionalServices': 'Professional Services',
    'PublicServices': 'Public Services',
    'InformationCommunication': 'IT',
    'FinancialInsurance': 'Finance & Insurance',
    'OtherServices': 'Other Services'
}

# ==================== REGIONS ====================
REGIONS = [
    {'name': 'Western\nEurope', 'countries': [
        'Austria', 'Belgium', 'Cyprus', 'Denmark', 'Finland', 'France', 'Germany',
        'Greece', 'Ireland', 'Italy', 'Luxembourg', 'Malta', 'Netherlands',
        'Portugal', 'Spain', 'Sweden', 'Norway', 'Switzerland', 'Liechtenstein',
        'Iceland', 'United Kingdom'
    ]},
    {'name': 'USA &\nCanada', 'countries': ['United States', 'Canada']},
    {'name': 'China', 'countries': ['China']},
    {'name': 'South Asia', 'countries': ['India', 'Bangladesh', 'Bhutan', 'Nepal', 'Pakistan', 'Sri Lanka', 'Maldives']},
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
    {'name': 'Asia\nDeveloped', 'countries': [
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
        'West Bank and Gaza', 'Turkiye', 'Afghanistan'
    ]},
    {'name': 'Latin\nAmerica', 'countries': [
        'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Costa Rica', 'Cuba',
        'Dominican Republic', 'Ecuador', 'El Salvador', 'Guatemala', 'Honduras',
        'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela, RB',
        'Haiti', 'Jamaica', 'Trinidad and Tobago', 'Belize'
    ]},
    # {'name': 'Rest of Asia', 'countries': [
    #     'Bangladesh', 'Bhutan', 'Nepal', 'Pakistan', 'Sri Lanka', 'Maldives'  # South Asia not in India
    #     'Kazakhstan', 'Kyrgyz Republic', 'Tajikistan', 'Turkmenistan', 'Uzbekistan',  # Central Asia
    #     'Mongolia',  # Other Asian states
    # ]}
]

SIMPLIFIED_REGIONS = [
    {'name': 'USA &\nCanada', 'countries': ['United States', 'Canada']},
    {'name': 'China', 'countries': ['China']},
    {'name': 'South Asia', 'countries': [
        'India', 'Pakistan', 'Bangladesh', 'Sri Lanka', 'Nepal', 'Bhutan', 'Maldives', 'Afghanistan'
    ]},
    {'name': 'Other\nDeveloped', 'countries': [
        'Austria', 'Belgium', 'Cyprus', 'Denmark', 'Finland', 'France', 'Germany',
        'Greece', 'Ireland', 'Italy', 'Luxembourg', 'Malta', 'Netherlands',
        'Portugal', 'Spain', 'Sweden', 'Norway', 'Switzerland', 'Liechtenstein',
        'Iceland', 'Bulgaria', 'Croatia', 'Czechia', 'Latvia', 'Lithuania', 'Estonia',
        'Hungary', 'Poland', 'Romania', 'Slovak Republic', 'Slovenia', 'Japan',
        'Korea, Rep.', 'Australia', 'New Zealand', 'Hong Kong SAR, China',
        'Macao SAR, China', 'Taiwan', 'Israel', 'United Kingdom', 'Singapore'
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
    {'name': 'Latin America', 'countries': [
        'Argentina', 'Bolivia', 'Brazil', 'Chile', 'Colombia', 'Costa Rica', 'Cuba',
        'Dominican Republic', 'Ecuador', 'El Salvador', 'Guatemala', 'Honduras',
        'Mexico', 'Nicaragua', 'Panama', 'Paraguay', 'Peru', 'Uruguay', 'Venezuela, RB',
        'Haiti', 'Jamaica', 'Trinidad and Tobago', 'Belize'
    ]},
]

# ==================== FUNCTIONS ====================
CACHE_FILE = f"cache/worldbank_{DATA_YEAR}_{'ppp' if USE_PPP_ADJUSTED else 'nominal'}.pkl"
OECD_FILE = '../data/OECDSectoralData.csv'

def fetch_world_bank_data(indicators, year):
    if os.path.exists(CACHE_FILE):
        return pd.read_pickle(CACHE_FILE)

    all_indicators = indicators.copy()
    all_indicators.update(LABOR_FORCE_INDICATOR)

    # Fetch data from World Bank API
    df_wb = wbdata.get_dataframe(all_indicators).reset_index()
    df_wb['date'] = pd.to_datetime(df_wb['date'])
    df_wb = df_wb[df_wb['date'].dt.year == year]
    df_wb.rename(columns={'country': 'Country'}, inplace=True)
    df_wb['Country'] = df_wb['Country'].str.strip()

    # Load and process OECD data
    script_dir = os.path.dirname(__file__)
    service_dir = os.path.join(script_dir, OECD_FILE)
    df_oecd = pd.read_csv(service_dir, sep=',')

    # Filter relevant columns and rows
    sector_mapping = {
        'Information and communication': 'InformationCommunication',
        'Public administration, defence, education, human health and social work activities': 'PublicServices',
        'Real estate activities': 'RealEstate',
        'Professional, scientific and technical activities; administrative and support service activities': 'ProfessionalServices',
        'Financial and insurance activities': 'FinancialInsurance',
        'Wholesale and retail trade; repair of motor vehicles and motorcycles; transportation and storage; accommodation and food service activities': 'WholesaleRetail',
        # 'Arts, entertainment and recreation; other service activities; activities of household and extra-territorial organizations and bodies': 'ArtsOtherServices'
    }

    # Filter relevant rows and pivot the data
    df_oecd_filtered = df_oecd[df_oecd['Economic activity'].isin(sector_mapping.keys())]
    df_oecd_pivoted = df_oecd_filtered.pivot_table(
        index='Reference area',
        columns='Economic activity',
        values='OBS_VALUE',
        aggfunc='sum'
    ).reset_index()
    # Rename columns according to sector_mapping
    df_oecd_pivoted.rename(columns=sector_mapping, inplace=True)
    df_oecd_pivoted.rename(columns={'Reference area': 'Country'}, inplace=True)
    df_oecd_pivoted[OECD_SUBSECTORS] *= 1e6
    

    # Merge OECD data with World Bank data
    df = df_wb.merge(df_oecd_pivoted, on='Country', how='left')

    # Save cache
    df.to_pickle(CACHE_FILE)
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
    df['LaborForce'] = df['LaborForce'].astype(float)
    return df

def calculate_gdp_metrics(df):
    df = df.copy()

    # --- Helper to calculat
    # e per capita and productivity ---
    def compute_metrics(gdp_col, emp_pct_col, emp_col_name, pc_col_name, prod_col_name):
        df[emp_col_name] = df[emp_pct_col] / 100 * df['LaborForce']
        df[pc_col_name] = df[gdp_col] / df['LaborForce']
        df[prod_col_name] = df[gdp_col] / df[emp_col_name]
        df[prod_col_name] = df[prod_col_name].replace([np.inf, -np.inf], 0)

    # --- Convert labor force to numeric ---
    df['LaborForce'] = df['LaborForce'].astype(float)

    # --- Major sectors ---
    df['Agricultural (PPP/USD)'] = df['Agriculture_pct'] / 100 * df[GDP_COLUMN]
    df['Industrial (PPP/USD)']   = df['Industry_pct'] / 100 * df[GDP_COLUMN]
    df['Service (PPP/USD)']      = df['Services_pct'] / 100 * df[GDP_COLUMN]
    
    compute_metrics('Agricultural (PPP/USD)', 'Agriculture_emp_pct',
                    'Agriculture_Employment', 'Agriculture_PC', 'Agriculture_Productivity')
    
    compute_metrics('Industrial (PPP/USD)', 'Industry_emp_pct',
                    'Industry_Employment', 'Industry_PC', 'Industry_Productivity')
    
    compute_metrics('Service (PPP/USD)', 'Services_emp_pct',
                    'Services_Employment', 'Services_PC', 'Services_Productivity')

    # --- Industry subsectors ---
    df['Manufacturing (PPP/USD)']      = df['Manufacturing_pct'] / 100 * df[GDP_COLUMN]
    df['ResourceExtraction (PPP/USD)'] = df['ResourceExtraction_pct'] / 100 * df[GDP_COLUMN]
    df['OtherIndustry (PPP/USD)']      = df['Industrial (PPP/USD)'] - df['Manufacturing (PPP/USD)'] - df['ResourceExtraction (PPP/USD)']
 
    # Convert OECD USD values to PPP if needed and calculate percentages
    for service_name in OECD_SUBSECTORS:
        pct_col = f'{service_name}_pct'
        gdp_col = f'{service_name} (PPP/USD)'
        
        # Convert USD to PPP if needed
        if USE_PPP_ADJUSTED:
            df[gdp_col] = df[service_name] / df['PPP_Conversion_Factor'].copy()
        else:
            df[gdp_col] = df[service_name].copy()
        
        # Calculate percentage of total GDP
        df[pct_col] = (df[gdp_col] / df[GDP_COLUMN] * 100).fillna(0)
        
        # Ensure no negative values or values > 100%
        df[pct_col] = df[pct_col].clip(0, 100)
    
    # Calculate total OECD services percentage
    oecd_service_pct_columns = [col + '_pct' for col in OECD_SUBSECTORS]
    total_oecd_services_pct = df[oecd_service_pct_columns].sum(axis=1)
    
    # Calculate remaining services as "Other Services"
    df['OtherServices_pct'] = (df['Services_pct'] - total_oecd_services_pct).clip(lower=0)
    df['OtherServices (PPP/USD)'] = df['OtherServices_pct'] / 100 * df[GDP_COLUMN]

    # --- Use SECTOR_SPLITS to calculate subsector metrics ---
    for sector, subsectors in SECTOR_SPLITS:
        sector_employment_col = f'{sector}_Employment'
        
        for subsector in subsectors:
            if subsector == 'Agriculture':
                continue  # Already handled above
                
            gdp_col = f'{subsector} (PPP/USD)'
            pc_col = f'{subsector}_PC'
            prod_col = f'{subsector}_Productivity'
            
            # Ensure GDP column exists (it should from above calculations)
            if gdp_col not in df.columns:
                df[gdp_col] = 0
            
            df[pc_col] = df[gdp_col] / df['LaborForce']
            df[prod_col] = df[gdp_col] / df[sector_employment_col]
            df[prod_col] = df[prod_col].replace([np.inf, -np.inf], 0)

    return df

def aggregate_region(df_region, name):
    if len(df_region) == 0:
        # Generate zero values for all metrics using SECTOR_SPLITS
        zero_metrics = {'Country': name, 'LaborForce': 0}
        
        # Add main sector metrics
        for sector, subsectors in SECTOR_SPLITS:
            zero_metrics.update({
                f'{sector}_PC': 0,
                f'{sector}_emp_pct': 0,
                f'{sector}_Productivity': 0,
                f'{sector}_Width': 0,
                f'{sector}_Employment': 0
            })
            # Add subsector metrics
            for subsector in subsectors:
                zero_metrics.update({
                    f'{subsector}_PC': 0,
                    f'{subsector}_Productivity': 0,
                    f'{subsector}_Employment': 0
                })
        
        return pd.Series(zero_metrics)

    total_labor_force = df_region['LaborForce'].sum()
    
    # Calculate employment and GDP totals for main sectors
    sector_data = {}
    for sector, subsectors in SECTOR_SPLITS:
        # Map sector names to their GDP column names
        if sector == 'Agriculture':
            sector_key = 'Agricultural'
        elif sector == 'Industry':
            sector_key = 'Industrial'
        elif sector == 'Services':
            sector_key = 'Service'
        else:
            sector_key = sector
            
        # Handle subsector GDP columns - Agriculture uses 'Agricultural' in column name
        subsector_gdps = {}
        for subsector in subsectors:
            if subsector == 'Agriculture':
                subsector_gdps[subsector] = df_region['Agricultural (PPP/USD)'].sum()
            else:
                subsector_gdps[subsector] = df_region[f'{subsector} (PPP/USD)'].sum()
            
        sector_data[sector] = {
            'employment': df_region[f'{sector}_Employment'].sum(),
            'gdp': df_region[f'{sector_key} (PPP/USD)'].sum(),
            'subsector_gdps': subsector_gdps
        }

    # Calculate employment allocation and metrics for each sector
    result = {'Country': name, 'LaborForce': total_labor_force}
    
    for sector, subsectors in SECTOR_SPLITS:
        sector_emp = sector_data[sector]['employment']
        sector_gdp = sector_data[sector]['gdp']
        
        # Main sector metrics
        result[f'{sector}_PC'] = sector_gdp / total_labor_force
        result[f'{sector}_emp_pct'] = sector_emp / total_labor_force * 100 if total_labor_force > 0 else 0
        result[f'{sector}_Productivity'] = sector_gdp / sector_emp if sector_emp > 0 else 0
        result[f'{sector}_Width'] = sector_emp
        result[f'{sector}_Employment'] = sector_emp
        
        # Subsector metrics
        for subsector in subsectors:
            subsector_gdp = sector_data[sector]['subsector_gdps'][subsector]
            
            # Calculate employment share based on GDP share
            if sector_gdp > 0:
                emp_share = subsector_gdp / sector_gdp
            else:
                emp_share = 1.0 / len(subsectors)
            
            subsector_emp = sector_emp * emp_share
            
            result[f'{subsector}_PC'] = subsector_gdp / total_labor_force
            result[f'{subsector}_Productivity'] = subsector_gdp / subsector_emp if subsector_emp > 0 else 0
            result[f'{subsector}_Employment'] = subsector_emp

    return pd.Series(result)

def assign_regions(df, regions):
    df = df.copy()
    df['Region'] = 'Other'
    for region in regions:
        df.loc[df['Country'].isin(region['countries']), 'Region'] = region['name']
    return df

def aggregate_all_regions(df, regions):
    aggregated = []
    for region in regions:
        aggregated.append(aggregate_region(df[df['Region'] == region['name']], region['name']))
    aggregated.append(aggregate_region(df[df['Region'] == 'Other'], 'Rest of World'))
    df_agg = pd.DataFrame(aggregated)
    df_agg = df_agg[df_agg['LaborForce'] > 0]
    return df_agg


def plot_productivity(df_final):
    df_final, scale_factor = prepare_dataframe(df_final)
    fig, ax = create_base_plot()

    country_positions = compute_country_positions(df_final)
    plot_sector_bars(ax, df_final, country_positions)
    add_country_labels(ax, df_final, country_positions)

    configure_axes(ax, df_final, country_positions)
    draw_legend(ax)
    draw_reference_box(ax, scale_factor)

    for spine in ax.spines.values():
        spine.set_visible(False)
    
    annotate_function(ax)

    plt.tight_layout()
    plt.savefig(f"visualizations/SectorProductivity{DATA_YEAR}.png", dpi=300, bbox_inches='tight')

def prepare_dataframe(df):
    df = df.copy()
    df['GDP_per_capita'] = sum(df[f'{sector}_PC'] for sector, _ in SECTOR_SPLITS)
    df = df.sort_values('GDP_per_capita', ascending=False)

    df['Total_Width'] = sum(df[f'{sector}_Width'] for sector, _ in SECTOR_SPLITS)
    max_width = df['Total_Width'].max()
    scale_factor = 1.0 / max_width if max_width > 0 else 1.0

    min_width = 0.02 if USE_SIMPLIFIED_REGIONS else 0.012
    for sector, _ in SECTOR_SPLITS:
        df[f'{sector}_Width'] *= scale_factor
        df[f'{sector}_Width'] = df[f'{sector}_Width'].clip(lower=min_width)

    df['Total_Width'] = sum(df[f'{sector}_Width'] for sector, _ in SECTOR_SPLITS)

    return df, scale_factor

def create_base_plot():
    figsize = (16, 16) if USE_SIMPLIFIED_REGIONS else (22, 16)
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    return fig, ax

def compute_country_positions(df):
    gap = 0.03
    return np.cumsum([0] + (df['Total_Width'] + gap).tolist()[:-1])

def plot_sector_bars(ax, df, positions):
    for i, (_, country) in enumerate(df.iterrows()):
        country_start = positions[i]
        for sector, subsectors in SECTOR_SPLITS:
            current_bottom = 0
            for subsector in subsectors:
                height = country[f'{sector}_Productivity'] * country[f'{subsector}_PC'] / country[f'{sector}_PC']
                ax.bar(
                    country_start, height,
                    width=country[f'{sector}_Width'],
                    bottom=current_bottom,
                    color=COLORS[subsector],
                    edgecolor=BG_COLOR,
                    linewidth=2.0,
                    align='edge',
                    label=subsector if i == 0 else ""
                )
                current_bottom += height
            country_start += country[f'{sector}_Width']

def add_country_labels(ax, df, positions):
    for i, (_, country) in enumerate(df.iterrows()):
        start = positions[i]
        width = country['Total_Width']
        center = start + width / 2
        line_y = -0.008 * ax.get_ylim()[1]

        # Line
        ax.plot(
            [start, start + width],
            [line_y, line_y],
            color=ANNO_COLOR,
            linewidth=2,
            alpha=1.0
        )

        size = 16 if USE_SIMPLIFIED_REGIONS else 13
        # Label
        label = f"{country['Country']}\n{country['LaborForce'] / 1e6:.0f}M"
        ax.text(
            center,
            line_y - ax.get_ylim()[1] * 0.01,
            label,
            ha='center',
            va='top',
            fontsize=size,
            color='white'
        )

def configure_axes(ax, df, positions):
    total_width = positions[-1] + df['Total_Width'].iloc[-1] if len(positions) > 0 else 1
    ax.set_xlim(left=0, right=total_width)

    ax.set_xlabel('Labor Force (millions)', color='white', fontsize=24)
    ax.set_ylabel(f'GDP per Employed Person ({GDP_LABEL} international $ in {DATA_YEAR})', color='white', fontsize=24)
    ax.set_title(f'The Global Economy by Sector and Country {DATA_YEAR}', color='white', fontsize=36)

    ax.yaxis.grid(True, color=ANNO_COLOR, alpha=1.0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(labelsize=13)

    ax.set_xticks([])
    ax.set_xticklabels([])

def draw_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    col1_handles, col1_labels = [], []
    col2_handles, col2_labels = [], []

    def get_handle(label):
        if label in labels:
            return handles[labels.index(label)]
        return None

    # --- Column 1: Agriculture + Industry (Manufacturing group) ---
    col1_labels.append("— Agriculture —")
    col1_handles.append(plt.Line2D([0], [0], linestyle='none'))
    for _, subsectors in SECTOR_SPLITS[:1]:
        for subsector in reversed(subsectors):
            h = get_handle(subsector)
            if h:
                col1_handles.append(h)
                col1_labels.append(SUBSECTOR_DISPLAY_NAMES.get(subsector, subsector))

    
    col1_labels.append("— Industry —")
    col1_handles.append(plt.Line2D([0], [0], linestyle='none'))
    for _, subsectors in SECTOR_SPLITS[1:2]:
        for subsector in reversed(subsectors):
            h = get_handle(subsector)
            if h:
                col1_handles.append(h)
                col1_labels.append(SUBSECTOR_DISPLAY_NAMES.get(subsector, subsector))

    # --- Column 2: Services group ---
    col2_labels.append("— Services —")
    col2_handles.append(plt.Line2D([0], [0], linestyle='none'))
    for subsector in reversed(SECTOR_SPLITS[2][1]):
        h = get_handle(subsector)
        if h:
            col2_handles.append(h)
            col2_labels.append(SUBSECTOR_DISPLAY_NAMES.get(subsector, subsector))

    # Pad so both columns align
    max_len = max(len(col1_handles), len(col2_handles))
    for lst, lbl in [(col1_handles, col1_labels), (col2_handles, col2_labels)]:
        while len(lst) < max_len:
            lst.append(plt.Line2D([0], [0], linestyle='none'))
            lbl.append("")

    final_handles = col1_handles + col2_handles
    final_labels = col1_labels + col2_labels

    legend = ax.legend(
        final_handles, final_labels,
        title='Subsector',
        loc='upper right',
        ncol=2,
        columnspacing=1.5,
        handletextpad=0.8,
        labelspacing=0.5,
        facecolor=BG_COLOR,
        edgecolor='white',
        frameon=True,
        fontsize=14,
        title_fontsize=0,
        borderpad=1.0,
        framealpha=1.0
    )

    # Styling
    plt.setp(legend.get_texts(), color='white')
    plt.setp(legend.get_title(), color='white')

    # Bold the "section headers"
    for text in legend.get_texts():
        if text.get_text().startswith("—"):
            text.set_fontweight("bold")
            text.set_fontsize(16)

def draw_reference_box(ax, scale_factor):
    ref_workers = 240_000_000.0
    ref_gdp = 12_500.0
    gdp_share = 0.35
    ref_width = ref_workers * scale_factor
    ref_height = ref_gdp

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xpos = 0.3 if USE_SIMPLIFIED_REGIONS else 0.5
    x_mid = xlim[0] + (xlim[1] - xlim[0]) * xpos
    x_rect = x_mid - ref_width / 2
    y_rect = ylim[0] + (ylim[1] - ylim[0]) * 0.873

    # --- main rectangle ---
    rect = plt.Rectangle(
        (x_rect, y_rect), ref_width, ref_height,
        linewidth=1.5, edgecolor="white", facecolor="none"
    )
    ax.add_patch(rect)

    # --- horizontal reference line for workers ---
    ygap = ylim[1] * 0.005
    ax.plot(
        [x_rect, x_rect + ref_width],
        [y_rect - ygap, y_rect - ygap],
        color="white", linewidth=2,
        linestyle="--"
    )
    ax.text(
        x_rect + ref_width / 2,
        y_rect - 2 * ygap,
        "240M workers",
        color="white",
        ha="center", va="top",
        fontsize=12
    )

    # --- vertical reference line for GDP per worker ---
    xgap = xlim[1] * 0.005
    ax.plot(
        [x_rect - xgap, x_rect - xgap],
        [y_rect, y_rect + ref_height],
        color="white",
        linewidth=2,
        linestyle="--"
    )
    ax.text(
        x_rect - 2 * xgap,
        y_rect + ref_height / 2,
        "$12,500\nper worker",
        color="white",
        ha="right", va="center",
        fontsize=12
    )

    # --- label for area (total GDP) ---
    ax.text(
        x_mid, y_rect + ref_height + ygap,
        "Area = $3T GDP",
        color="white",
        ha="center", va="bottom",
        fontsize=14
    )

    sector_y = y_rect + ref_height * gdp_share
    ax.plot(
        [x_rect, x_rect + ref_width],
        [sector_y, sector_y],
        color="white", linewidth=2
    )
    mid_sector_y = (y_rect + ref_height * gdp_share * 0.5)
    mid_other_sector_y = sector_y + 0.5 * (1 - gdp_share) * ref_height

    for (share, mid_y) in [(gdp_share, mid_sector_y), (1-gdp_share, mid_other_sector_y)]:
        ax.text(
            x_rect + ref_width + xgap, mid_y,
            f"{int(share * 100)}% of GDP in\nthis subsector",
            color="white",
            ha="left", va="center",
            fontsize=12, rotation=0
        )
    
def annotate_function(ax):
    simple_annotations = [
        ('Developed economies are mostly\n' + 
         'based on services with large public\n' + 
         'and professional service sectors',
        (0.32, 108000), (0.6, 93000)),
        ('China has the largest\n' +
         'manufacturing base in\n' +
         'the world, constituting\n' +
         '25% of its 29T economy',
        (1.0, 63000), (1.2, 45000)),
        ('South Asia has a young population and few\n' +
         'women work turning a 2000M population into\n' +
         'just a 729M labor force; this is now the\n' + 
         'fastest growing part of the global economy',
        (3.0, 45000), (3.8, 29000)),
        ('Industry in Africa, Latin America, Russia\n' +
         'and the Gulf states depends much more on\n' +
         'resource extraction then the rest of the world',
        (2.1, 70000), (2.25, 49500)),
    ]
    
    annotations = [
        ('Northern America has the most productive service sector\n' + 
         'in the world, employing most of the population and\n' +
         'constituting 80% of the US\'s 24T GDP; the country has\n' +
         'a much larger tech & finance sector compared to other\n' + 
         'developed countries',
        (0.32, 125000), (0.25, 115000)),
        ('Japanese, Korean and Taiwanese industry leads the\n' +
         'world in productivity despite having no natural\n' + 
         'resources while services and agriculture have lagged behind',
        (0.5, 111000), (0.63, 95000)),
        ('The Gulf and Russian economies are driven mostly\n' + 
         'by highly productive resource extraction of oil & gas\n' +
         'to the exclusion of other industries', 
        (0.9, 87000), (1.04, 70000)),
        ('China has grown rapidly into the largest economy in\n' + 
         'the world (29T) through manufacturing-led export but\n' +
         'the country will face challenges transitioning to a\n' +
         'service-based economy as the population ages and\n' +
         'and declines while growth has slowed',
        (1.4, 63000), (1.6, 45500)),
        ('South Asian services are oddly productive in comparison to\n' +
         'industry, indicating potential for a \'service-led export\'\n' +
         'boom, however it isn\'t clear whether there is enough\n' +
         'demand for service export to be able to absorb the agrarian\n' +
         'labor force the way manufacturing-led export did in China',
        (2.9, 46000), (4.0, 29500)),
        ('Africa is a diverse continent\n' +
         'but broadly, it is agrarian and\n' + 
         'extractive, making up >20%\n' +
         'of global natural resource rents',
        (4.1, 65000), (4.55, 31500))
    ]
    
    fs = 16 if USE_SIMPLIFIED_REGIONS else 14
    annotations = simple_annotations if USE_SIMPLIFIED_REGIONS else annotations
    for annotation in annotations:
        # Unpack annotation details
        text, text_position, target_position = annotation
        
        # Annotate with an arrow
        ax.annotate(
            text,
            xy=target_position,
            xytext=text_position,
            textcoords='data',
            arrowprops=dict(arrowstyle='->', color=ANNO_COLOR, linewidth=2.0),
            color=ANNO_COLOR,
            fontsize=fs,
            ha='left'
        )

    context = True
    if (context):
        x_text_loc = 2.7 if USE_SIMPLIFIED_REGIONS else 3.6
        oecd_text = '*Service subsector data comes from the OECD but\nwe have less and less of it in developing countries\nand more services are labelled "other"'
        ax.annotate(
            text=oecd_text,
            xy=(x_text_loc, 90000),
            textcoords='data',
            color=ANNO_COLOR,
            fontsize=fs,
            ha='left'
        )
        
        credit_text = 'Visualization by MadoctheHadoc; Sources:\nWorld Bank - employment, sector and GDP data\nOECD - service subsector breakdown'
        ax.annotate(
            text=credit_text,
            xy=(x_text_loc, 105000),
            textcoords='data',
            color=ANNO_COLOR,
            fontsize=fs,
            ha='left'
        )

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