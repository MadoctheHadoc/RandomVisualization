import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from scipy.stats import rankdata

def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = array.flatten()
    if np.amin(array) < 0:
        array -= np.amin(array)  # zero minimum
    array = np.sort(array)
    n = array.size
    index = np.arange(1, n+1)
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

# Load current S&P 500 company data with price and market cap
df_companies = pd.read_csv('data/HistoricalStocks/sp500_companies.csv')

# Calculate shares outstanding = MarketCap / Currentprice
df_companies['SharesOutstanding'] = df_companies['Marketcap'] / df_companies['Currentprice']

# We’ll store yearly market caps in this dict: {year: [market_caps]}
yearly_market_caps = {}

# Define years of interest
start_year = 2000
end_year = datetime.now().year

symbols = df_companies['Symbol'].tolist()

for symbol in symbols:
    try:
        print(f"Fetching data for {symbol}...")
        stock = yf.Ticker(symbol)
        hist = stock.history(start=f'{start_year}-01-01', end=f'{end_year + 1}-01-01')
        print(f"{symbol} columns:", hist.columns)

        if 'Adj Close' not in hist.columns:
            # Estimate adjusted close by adjusting Close for splits and dividends
            hist['Adj Close'] = hist['Close'] * hist['Stock Splits'].replace(0,1).cumprod()
            # Dividends usually require more complex adjustments over time;
            # if needed, could apply dividend adjustments too but it's complex.

        # Proceed with hist['Adj Close']


        shares_outstanding = df_companies.loc[df_companies['Symbol'] == symbol, 'SharesOutstanding'].values[0]

        # Extract year-end prices from monthly data
        hist['Year'] = hist.index.year
        year_end_prices = hist.groupby('Year')['Adj Close'].last()  # last trading day price per year

        for year, adj_close_price in year_end_prices.items():
            if year < start_year or year > end_year:
                continue
            market_cap_estimate = adj_close_price * shares_outstanding

            if year not in yearly_market_caps:
                yearly_market_caps[year] = []
            yearly_market_caps[year].append(market_cap_estimate)

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

# Calculate yearly Gini coefficients
gini_results = []
for year in sorted(yearly_market_caps.keys()):
    market_caps = np.array(yearly_market_caps[year])
    if len(market_caps) > 0:
        g = gini(market_caps)
        gini_results.append({'Year': year, 'GiniCoefficient': g})

df_gini = pd.DataFrame(gini_results)
print(df_gini)
