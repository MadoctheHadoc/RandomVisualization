import os
import pandas as pd
import matplotlib.pyplot as plt

# WB PPP Current Price for productivity weighting
W_EUROPE_GDP = {
    'Austria':      678_357.86,
    'Belgium':      871_779.58,
    'Denmark':      489_385.56,
    'Finland':      367_420.70,
    'France':     4_288_380.14,
    'Germany':    6_142_807.00,
    # 'Ireland':      720_000.57, # Exlcuded due to bad data
    'Italy':      3_655_908.76,
    'Netherlands':1_550_563.90,
    'Norway':       568_581.57,
    'Portugal':     552_699.92,
    'Spain':      2_831_537.41,
    'Sweden':       759_379.82,
    'Switzerland':  869_017.90,
    'United Kingdom': 4_292_669.10,
}

E_EUROPE_GDP = {
    'Bulgaria': 270_342.91,	
    'Croatia': 191_573.27,
    'Czechia':	624_699.15,
    'Estonia'	: 68_574.11,
    'Greece'	: 461_230.56,
    'Hungary'	: 464_261.88,
    'Latvia'	: 80_979.09,
    'Lithuania'	: 159_681.05,
    'Poland'	: 1_874_118.07,
    'Romania'	: 934_996.62,
    'Slovenia'	: 121_658.29,
}





ALSO_INCLUDE = [
    'United States',
    # 'United Kingdom',
    # 'Japan',
    # 'Australia',
    # 'Canada',
    # 'Spain',
]

COUNTRY_COLOURS = {
    'United States': "#C22222",
    # 'United Kingdom': "#53C93C",
    # 'Japan':          "#A251D8",
    # 'Australia': "#51D6D0",
    # 'Canada': "#C22222",
    # 'Spain':"#E36F2D",
}

WE_colour = '#2271B2'
EE_colour = "#41BE28"


def load_data():
    SCRIPT_DIR = os.path.dirname(__file__)
    FILE_PATH = os.path.join(SCRIPT_DIR, '../data/PWT-labour-productivity.csv')
    
    # Read with correct header and skip the unwanted row
    df = pd.read_csv(FILE_PATH, header=0, skiprows=[1])    
    return df

def format_data(df):
    # Filter to Western Europe, then compute weighted mean productivity per year
    df_we = df[df['Entity'].isin(W_EUROPE_GDP)].copy()
    df_we['weight'] = df_we['Entity'].map(W_EUROPE_GDP)

    df_we = (
        df_we.groupby('Year', as_index=False)
        .apply(
            lambda g: pd.Series({
                'WE_Productivity': (
                    g['Productivity: output per hour worked'] * g['weight']
                ).sum() / g['weight'].sum()
            }),
            include_groups=False
        )
        .reset_index(drop=True)
    )
    
    df_ee = df[df['Entity'].isin(E_EUROPE_GDP)].copy()
    df_ee['weight'] = df_ee['Entity'].map(E_EUROPE_GDP)
    df_ee = (
        df_ee.groupby('Year', as_index=False)
        .apply(
            lambda g: pd.Series({
                'EE_Productivity': (
                    g['Productivity: output per hour worked'] * g['weight']
                ).sum() / g['weight'].sum()
            }),
            include_groups=False
        )
        .reset_index(drop=True)
    )

    dfs = [df_we, df_ee]

    # Also pull additional countries for comparison
    for country in ALSO_INCLUDE:
        col_name = f"{country}_Productivity"
        dfs.append(
            df[df['Entity'] == country][['Year', 'Productivity: output per hour worked']]
            .rename(columns={'Productivity: output per hour worked': col_name})
            .reset_index(drop=True)
        )

    # Merge all series into a single time-series dataframe
    df_merged = dfs[0]
    for other in dfs[1:]:
        df_merged = pd.merge(df_merged, other, on='Year', how='inner')

    return df_merged

def plot_data(df):
    fig, ax = plt.subplots(figsize=(12, 12))

    ax.plot(df['Year'], df['WE_Productivity'], linewidth=2.5,
            color=WE_colour, label='Western Europe (GDP-weighted avg)')
    ax.plot(df['Year'], df['EE_Productivity'], linewidth=2.5,
            color=EE_colour, label='Eastern Europe (GDP-weighted avg)')

    for i, country in enumerate(ALSO_INCLUDE):
        col  = f"{country}_Productivity"
        color = COUNTRY_COLOURS[country]
        ax.plot(df['Year'], df[col], linewidth=2.5,
                color=color, label=country, linestyle='--')

        # Shade gap between WE and this country
        # ax.fill_between(df['Year'], df['WE_Productivity'], df[col],
        #                 alpha=0.08, color=color)

    ax.set_title('Labour Productivity: Western Europe vs. Comparators',
                 fontsize=15, fontweight='bold', pad=14)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Output per Hour Worked (USD, 2017 PPP)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0)

    plt.tight_layout()
    plt.savefig("visualizations/WesternEuropeProductivity.png",
                dpi=300, bbox_inches="tight", facecolor='white')


df = load_data()
df_formatted = format_data(df)
plot_data(df_formatted)