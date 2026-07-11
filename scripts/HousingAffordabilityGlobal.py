import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import geopandas as gpd
import numpy as np
import requests
import os
import time
from bs4 import BeautifulSoup

SCRIPT_DIR = os.path.dirname(__file__)
WORLDCITIES_PATH = os.path.join(SCRIPT_DIR, '../data/worldcities.csv')
NUMBEO_CACHE = os.path.join(SCRIPT_DIR, '../cache/numbeo_global.pkl')

BG_COLOR = '#2a2a2a'
ANNO_COLOR = "#848484"

NUMBEO_URL = "https://www.numbeo.com/cost-of-living/rankings_current.jsp"

NE_110M_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
MAP_CRS = "ESRI:54029"

COUNTRY_NAME_MAP = {
    'Czech Republic': 'Czechia',
    'Slovak Republic': 'Slovakia',
    'Russian Federation': 'Russia',
    'Republic of Serbia': 'Serbia',
    'Moldova, Republic of': 'Moldova',
    'Kosovo (under UNSCR 1244)': 'Kosovo',
    'Korea, South': 'South Korea',
}

US_STATE_ABBREVS = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
    'DC', 'PR', 'VI', 'GU', 'AS', 'MP',
}

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}


def scrape_numbeo():
    os.makedirs(os.path.dirname(NUMBEO_CACHE), exist_ok=True)
    if os.path.exists(NUMBEO_CACHE):
        print("  Loading cached Numbeo data...")
        return pd.read_pickle(NUMBEO_CACHE)

    print("  Scraping Numbeo global rankings...")
    time.sleep(2)
    resp = requests.get(NUMBEO_URL, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, 'html.parser')
    table = None
    for t in soup.find_all('table'):
        first_row = t.find('tr')
        if first_row:
            header_texts = [c.get_text(strip=True).lower() for c in first_row.find_all('th')]
            if 'city' in header_texts:
                table = t
                break

    if table is None:
        raise RuntimeError("Could not find Numbeo data table")

    header_cells = table.find('tr').find_all('th')
    headers = [c.get_text(strip=True) for c in header_cells]

    rent_idx = None
    ppp_idx = None
    for i, h in enumerate(headers):
        h_lower = h.lower()
        if 'rent' in h_lower and 'index' in h_lower and 'plus' not in h_lower:
            rent_idx = i
        if 'purchasing' in h_lower:
            ppp_idx = i

    rows = []
    rank_counter = 0
    for tr in table.find_all('tr'):
        cells = tr.find_all(['td', 'th'])
        if len(cells) >= max(rent_idx, ppp_idx) + 1:
            cell_texts = [c.get_text(strip=True) for c in cells]
            if 'rank' in cell_texts[0].lower() or 'city' in cell_texts[1].lower():
                continue
            city_text = cell_texts[1]
            try:
                rent_val = float(cell_texts[rent_idx]) if rent_idx else None
                ppp_val = float(cell_texts[ppp_idx]) if ppp_idx else None
                if rent_val is not None and ppp_val is not None:
                    rank_counter += 1
                    city, country = parse_city_country(city_text)
                    rows.append({
                        'Rank': rank_counter,
                        'City': city,
                        'Country': country,
                        'RentIndex': rent_val,
                        'PurchasingPowerIndex': ppp_val,
                    })
            except (ValueError, IndexError):
                continue

    df = pd.DataFrame(rows)
    df.to_pickle(NUMBEO_CACHE)
    print(f"  Found {len(df)} cities globally")
    return df


def parse_city_country(text):
    parts = text.rsplit(',', 1)
    if len(parts) == 2:
        city = parts[0].strip()
        country = parts[1].strip()
        if country.upper() in US_STATE_ABBREVS:
            country = 'United States'
        return city, country
    return text.strip(), ''


def normalize_city(name):
    name = name.lower().strip()
    name = name.replace('ü', 'u').replace('é', 'e').replace('è', 'e')
    name = name.replace('ã', 'a').replace('á', 'a').replace('ó', 'o')
    name = name.replace('ñ', 'n').replace('ö', 'o').replace('ä', 'a')
    name = name.replace('ś', 's').replace('ż', 'z').replace('ź', 'z')
    return name


def load_worldcities():
    df = pd.read_csv(WORLDCITIES_PATH)
    df = df.drop_duplicates(subset=['city', 'country'], keep='first')
    return df


def match_cities(cities_df, numbeo_df):
    from collections import defaultdict
    numbeo_lookup = defaultdict(list)
    for _, row in numbeo_df.iterrows():
        city_clean = row['City']
        if row['Country'] == 'United States' and ',' in city_clean:
            parts = city_clean.rsplit(',', 1)
            if parts[1].strip().upper() in US_STATE_ABBREVS:
                city_clean = parts[0].strip()
        key = normalize_city(city_clean)
        numbeo_lookup[key].append(row)

    matched = []
    for _, city_row in cities_df.iterrows():
        city_name = str(city_row['city']).strip()
        country_name = str(city_row['country']).strip()
        key = normalize_city(city_name)

        best = None
        candidates = numbeo_lookup.get(key, [])

        for row in candidates:
            n_country = row['Country']
            if country_name == n_country or country_name.startswith(n_country) or n_country.startswith(country_name):
                best = row
                break

        if best is None:
            canonical = COUNTRY_NAME_MAP.get(country_name, country_name)
            for row in candidates:
                if row['Country'] == canonical:
                    best = row
                    break

        if best is not None:
            matched.append({
                'city': city_name,
                'country': country_name,
                'lat': city_row['lat'],
                'lng': city_row['lng'],
                'population': city_row['population'],
                'RentIndex': best['RentIndex'],
                'PurchasingPowerIndex': best['PurchasingPowerIndex'],
            })

    df = pd.DataFrame(matched)
    df['AffordabilityRatio'] = df['RentIndex'] / df['PurchasingPowerIndex']
    df = df[df['AffordabilityRatio'] > 0].copy()
    df = df[df['population'] >= 100_000].copy()
    df = df.sort_values('AffordabilityRatio', ascending=False).reset_index(drop=True)

    print(f"\nMatched {len(df)} cities globally")
    return df


def plot_map(df):
    os.makedirs("visualizations", exist_ok=True)

    fig, ax = plt.subplots(figsize=(24, 14))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    world = gpd.read_file(NE_110M_URL)
    world = world[world['NAME'] != 'Antarctica']
    world = world.to_crs(MAP_CRS)
    world.plot(
        ax=ax,
        color='#3a3a3a',
        edgecolor='#555555',
        linewidth=0.3,
        zorder=1,
    )

    from pyproj import Transformer
    transformer = Transformer.from_crs("EPSG:4326", MAP_CRS, always_xy=True)

    for lng in range(-180, 181, 15):
        lats = np.linspace(-80, 84, 200)
        gx, gy = transformer.transform(np.full_like(lats, lng), lats)
        ax.plot(gx, gy, color='#444444', linewidth=0.3, zorder=2)
    for lat in range(-75, 76, 15):
        lngs = np.linspace(-180, 180, 400)
        gx, gy = transformer.transform(lngs, np.full_like(lngs, lat))
        ax.plot(gx, gy, color='#444444', linewidth=0.3, zorder=2)
    x, y = transformer.transform(df['lng'].values, df['lat'].values)

    norm = mcolors.Normalize(
        vmin=df['AffordabilityRatio'].quantile(0.05),
        vmax=df['AffordabilityRatio'].quantile(0.95)
    )
    cmap = plt.cm.RdYlGn_r

    max_pop = df['population'].max()
    min_size = 10
    max_size = 500
    sizes = min_size + (df['population'] / max_pop) * (max_size - min_size)

    ax.scatter(
        x, y,
        s=sizes,
        c=df['AffordabilityRatio'],
        cmap=cmap,
        norm=norm,
        alpha=0.85,
        edgecolors='white',
        linewidths=0.4,
        zorder=5,
    )

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax, shrink=0.4, aspect=20, pad=0.02,
    )
    cbar.set_label('Affordability Ratio\n(Rent Index / Purchasing Power Index)', color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    legend_sizes = [100_000, 1_000_000, 10_000_000]
    legend_labels = ['100K', '1M', '10M']
    legend_handles = []
    for s in legend_sizes:
        sz = min_size + (s / max_pop) * (max_size - min_size)
        handle = ax.scatter([], [], s=sz, c=ANNO_COLOR, alpha=0.6, edgecolors='white', linewidths=0.4)
        legend_handles.append(handle)

    legend = ax.legend(
        legend_handles, legend_labels,
        title='Population',
        loc='lower left',
        fontsize=9,
        title_fontsize=10,
        frameon=True,
        facecolor=BG_COLOR,
        edgecolor=ANNO_COLOR,
        labelcolor='white',
        scatterpoints=1,
    )
    legend.get_title().set_color('white')

    bounds = world.total_bounds
    x_margin = (bounds[2] - bounds[0]) * 0.02
    y_margin = (bounds[3] - bounds[1]) * 0.02
    ax.set_xlim(bounds[0] - x_margin, bounds[2] + x_margin)
    ax.set_ylim(bounds[1] - y_margin, bounds[3] + y_margin)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(
        'Housing Affordability in Cities Worldwide',
        fontsize=28, color='white', fontweight='bold', pad=20
    )
    ax.text(
        0.5, 1.02,
        'Rent Index relative to Purchasing Power (Numbeo, 2025) | Red = Less Affordable, Green = More Affordable',
        transform=ax.transAxes, ha='center', fontsize=13, color=ANNO_COLOR
    )

    ax.text(
        0.99, 0.01,
        'Visualization by MadoctheHadoc\nData: Numbeo (2025), Population: GeoNames',
        transform=ax.transAxes, ha='right', va='bottom',
        fontsize=8, color=ANNO_COLOR
    )

    plt.tight_layout()
    plt.savefig("visualizations/HousingAffordabilityGlobal.png", dpi=300, bbox_inches='tight')
    print("\nSaved: visualizations/HousingAffordabilityGlobal.png")


if __name__ == "__main__":
    print("Loading city data...")
    cities_df = load_worldcities()
    print(f"  Found {len(cities_df)} cities in worldcities.csv")

    numbeo_df = scrape_numbeo()

    df = match_cities(cities_df, numbeo_df)

    print("\nTop 10 least affordable:")
    for _, row in df.head(10).iterrows():
        print(f"  {row['city']}, {row['country']}: {row['AffordabilityRatio']:.2f}")

    print("\nTop 10 most affordable:")
    for _, row in df.tail(10).iterrows():
        print(f"  {row['city']}, {row['country']}: {row['AffordabilityRatio']:.2f}")

    print("\nRendering...")
    plot_map(df)
