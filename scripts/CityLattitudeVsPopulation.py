import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Define the file path relative to the script's location
script_dir = os.path.dirname(__file__)
file_path = os.path.join(script_dir, '../data/worldcities.csv')

# Load the dataset
df = pd.read_csv(file_path)

# Convert latitude and population columns to numeric
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lng'] = pd.to_numeric(df['lng'], errors='coerce')
df['population'] = pd.to_numeric(df['population'], errors='coerce')

# Drop rows with missing values in latitude, longitude, or population
df = df.dropna(subset=['lat', 'lng', 'population'])

# Calculate logarithm of population
df['log_population'] = np.log10(df['population'])

# Create latitude bands (e.g., every 10 degrees)
lat_bins = np.arange(-90, 100, 10)
df['lat_band'] = pd.cut(df['lat'], bins=lat_bins, right=False)

# Find the city with the maximum log population for each latitude band
max_cities = df.loc[df.groupby('lat_band', observed=True)['log_population'].idxmax()]

# Assign unique colors to each country
unique_countries = max_cities['country'].dropna().unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(unique_countries)))
country_color_map = {country: color for country, color in zip(unique_countries, colors)}

# Plot the map
plt.figure(figsize=(12, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.set_global()
ax.add_feature(cfeature.LAND, zorder=0, edgecolor='black')
ax.add_feature(cfeature.OCEAN, zorder=0, color='lightblue')
ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=1)
ax.add_feature(cfeature.COASTLINE, zorder=1)

# Plot each city
for _, row in max_cities.iterrows():
    plt.plot(
        row['lng'], row['lat'], 
        marker='o', color=country_color_map.get(row['country'], 'gray'), 
        transform=ccrs.PlateCarree()
    )
    plt.text(
        row['lng'], row['lat'], row['city'], 
        transform=ccrs.PlateCarree(), fontsize=8,
        ha='right'
    )

# Add legend
legend_patches = [plt.Line2D([0], [0], marker='o', color=color, lw=0) for color in country_color_map.values()]
plt.legend(legend_patches, country_color_map.keys(), title="Country", loc='lower left', bbox_to_anchor=(1, 0))

plt.title("Maximum Population Cities Across Latitude Bands")
plt.show()
