import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, '..', 'data')
food_path = os.path.join(data_path, 'food_security_scores.csv')
gdp_path = os.path.join(data_path, 'country_GDP_PPP.csv')

# Load food security data
food_df = pd.read_csv(food_path)
food_df = food_df[['Country', 'Availability']].dropna()

# Load GDP data and skip metadata
raw_gdp_df = pd.read_csv(gdp_path, skiprows=4)

# Strip leading/trailing spaces from column names
raw_gdp_df.columns = raw_gdp_df.columns.str.strip()

# Ensure the year column exists
if '2022' not in raw_gdp_df.columns:
    print("Column '2022' not found. Check CSV formatting.")
    print("Available columns:", raw_gdp_df.columns.tolist())
    raise SystemExit

# Filter GDP per capita indicator only
gdp_df = raw_gdp_df[raw_gdp_df['Indicator Name'] == 'GDP per capita (current US$)']

# Clean up GDP data
gdp_df = gdp_df[['Country Name', '2022']].rename(columns={'Country Name': 'Country', '2022': 'GDP per capita'})
gdp_df['GDP per capita'] = pd.to_numeric(gdp_df['GDP per capita'], errors='coerce')
gdp_df['Country'] = gdp_df['Country'].str.strip()

# Clean food country names
food_df['Country'] = food_df['Country'].str.strip()

# Merge
merged_df = pd.merge(food_df, gdp_df, on='Country', how='inner')
merged_df = merged_df.dropna()

# Plotting
plt.figure(figsize=(12, 7))
plt.scatter(merged_df['GDP per capita'], merged_df['Availability'], alpha=0.7)
plt.xlabel('GDP per Capita (Current US$, 2022)')
plt.ylabel('Food Availability Score')
plt.title('Food Availability vs GDP per Capita (2022)')
plt.grid(True)

# Optional: annotate low or high outliers
for _, row in merged_df.iterrows():
    plt.annotate(row['Country'], (row['GDP per capita'], row['Availability']), fontsize=8)

plt.tight_layout()
plt.show()
