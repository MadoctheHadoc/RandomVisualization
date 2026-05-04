import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Params - logarithmic spacing for better coverage
size_min, size_max = 0.1, 10000  # m^2 (from tiny to massive)
altitude_min, altitude_max = 10, 2000  # meters
resolution = 150  # higher resolution for log scale

# Use logarithmic spacing for sizes to better represent scale differences
sizes = np.logspace(np.log10(size_min), np.log10(size_max), resolution)
altitudes = np.linspace(altitude_min, altitude_max, resolution)
S, A = np.meshgrid(sizes, altitudes)

def get_wind_speed(altitude, v0=5, h0=10, alpha=0.14):
    """Calculate wind speed at a given altitude using the power law."""
    return v0 * (altitude / h0) ** alpha

def get_turbine_mass(area, mass_per_m2=50):
    """Estimate turbine mass based on swept area."""
    return mass_per_m2 * area

def get_floating_cost(area, altitude, turbine_base_cost=2000, turbine_area_cost=500,
                      cable_cost=50, helium_cost_per_kg=2, mass_per_m2=50,
                      factory_threshold=10):
    """
    Calculate the total cost of a floating wind turbine with economies of scale.

    Parameters:
    - area: Swept area (m²)
    - altitude: Operating height (m)
    - turbine_base_cost: Base cost for custom turbines (USD)
    - turbine_area_cost: Cost per m² of turbine area (USD/m²)
    - cable_cost: Cost per meter of tether cable (USD/m)
    - helium_cost_per_kg: Cost per kg of lift capacity per year (USD/kg/year)
    - mass_per_m2: Turbine mass per unit swept area (kg/m²)
    - factory_threshold: Area threshold below which factory production applies (m²)

    Returns:
    - Total annual cost (USD/year)
    """
    # Factory-produced small turbines have lower unit costs
    if area < factory_threshold:
        # Mass production: lower cost per unit area, no base cost
        turbine_annual = (turbine_area_cost * 0.3) * area / 20  # 70% discount
    else:
        # Custom large turbines: economies of scale with area, but have base cost
        turbine_annual = (turbine_base_cost + turbine_area_cost * area ** 0.85) / 20
    
    # Tether cable cost (proportional to altitude)
    cable_annual = cable_cost * altitude / 20
    
    # Helium/buoyancy cost (proportional to weight being lifted)
    turbine_mass = get_turbine_mass(area, mass_per_m2)
    helium_annual = helium_cost_per_kg * turbine_mass
    
    return turbine_annual + cable_annual + helium_annual

def calculate_cost_per_mwh(area, altitude, rho=1.225, efficiency=0.4, capacity_factor=0.35, 
                           hours_per_year=8760):
    """Calculate levelized cost per MWh based on area and altitude."""
    # Adjust air density for altitude
    rho_adjusted = rho * np.exp(-altitude / 8400)
    
    # Wind speed at altitude
    speed = get_wind_speed(altitude)
    
    # Power calculation
    power_theoretical = 0.5 * rho_adjusted * area * speed ** 3
    power_actual = power_theoretical * efficiency
    annual_energy_mwh = power_actual * hours_per_year * capacity_factor / 1e6
    
    # Annual cost with factory economies of scale
    annual_cost = get_floating_cost(area, altitude)
    
    if annual_energy_mwh > 0:
        return annual_cost / annual_energy_mwh
    else:
        return np.inf

# Vectorize the cost calculation function
calculate_cost_per_mwh_vec = np.vectorize(calculate_cost_per_mwh)

# Calculate cost for each pixel
costs = calculate_cost_per_mwh_vec(S, A)

# Handle any infinite or invalid values
costs = np.nan_to_num(costs, nan=costs[np.isfinite(costs)].max(), 
                      posinf=costs[np.isfinite(costs)].max())

# Calculate additional data for hover info
wind_speeds = get_wind_speed(A)
power_output = 0.5 * 1.225 * np.exp(-A / 8400) * S * wind_speeds ** 3 * 0.4 / 1000  # kW
annual_energy = power_output * 8760 * 0.35 / 1000  # MWh
turbine_masses = get_turbine_mass(S)
diameters = 2 * np.sqrt(S / np.pi)

# Apply log scale for visualization
costs_log = np.log10(costs)

# Find optimal altitude for each size
optimal_altitudes = []
optimal_costs_for_size = []

for size_idx, size in enumerate(sizes):
    # Get costs for this size across all altitudes
    costs_for_size = costs[:, size_idx]
    # Find the altitude index with minimum cost
    min_alt_idx = np.argmin(costs_for_size)
    optimal_altitudes.append(altitudes[min_alt_idx])
    optimal_costs_for_size.append(costs_for_size[min_alt_idx])

optimal_altitudes = np.array(optimal_altitudes)
optimal_costs_for_size = np.array(optimal_costs_for_size)

# Find overall optimal point
min_idx = np.unravel_index(np.argmin(costs), costs.shape)
optimal_area = S[min_idx]
optimal_altitude = A[min_idx]
optimal_cost = costs[min_idx]

# Create hover text
hover_text = []
for i in range(resolution):
    hover_row = []
    for j in range(resolution):
        production_type = "Factory" if S[i,j] < 10 else "Custom"
        hover_row.append(
            f"<b>Production:</b> {production_type}<br>" +
            f"<b>Swept Area:</b> {S[i,j]:.2f} m²<br>" +
            f"<b>Diameter:</b> {diameters[i,j]:.2f} m<br>" +
            f"<b>Altitude:</b> {A[i,j]:.0f} m<br>" +
            f"<b>Wind Speed:</b> {wind_speeds[i,j]:.2f} m/s<br>" +
            f"<b>Turbine Mass:</b> {turbine_masses[i,j]:.1f} kg<br>" +
            f"<b>Power Output:</b> {power_output[i,j]:.2f} kW<br>" +
            f"<b>Annual Energy:</b> {annual_energy[i,j]:.2f} MWh<br>" +
            f"<b>Cost per MWh:</b> <b>${costs[i,j]:.2f}</b>"
        )
    hover_text.append(hover_row)

# Create subplots - 2 rows, 2 columns
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Cost Heatmap: Log Area vs Linear Altitude',
        'Optimal Altitude for Each Turbine Size',
        'Minimum Cost per MWh vs Turbine Size',
        'Optimal Configuration Summary'
    ),
    specs=[[{"type": "heatmap"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "table"}]],
    vertical_spacing=0.12,
    horizontal_spacing=0.12,
    row_heights=[0.55, 0.45]
)

# Plot 1: Heatmap with optimal line overlaid
fig.add_trace(go.Heatmap(
    x=sizes,
    y=altitudes,
    z=costs_log,
    colorscale='Viridis_r',
    hovertext=hover_text,
    hoverinfo='text',
    colorbar=dict(
        title='Cost/MWh<br>(USD/MWh)',
        y=0.75,
        len=0.45,
        tickvals=np.linspace(costs_log.min(), costs_log.max(), 6),
        ticktext=[f'${10**val:.0f}' for val in np.linspace(costs_log.min(), costs_log.max(), 6)]
    ),
    name='Cost Map'
), row=1, col=1)

# Add optimal line to heatmap
fig.add_trace(go.Scatter(
    x=sizes,
    y=optimal_altitudes,
    mode='lines',
    line=dict(color='red', width=3, dash='solid'),
    name='Optimal Altitude',
    hovertemplate='<b>Area:</b> %{x:.2f} m²<br><b>Optimal Alt:</b> %{y:.0f} m<extra></extra>'
), row=1, col=1)

# Add overall optimal point
fig.add_trace(go.Scatter(
    x=[optimal_area],
    y=[optimal_altitude],
    mode='markers',
    marker=dict(size=15, color='white', symbol='star', line=dict(width=2, color='red')),
    name=f'Global Optimum',
    hovertext=f'<b>GLOBAL OPTIMUM</b><br>Area: {optimal_area:.2f} m²<br>Alt: {optimal_altitude:.0f} m<br>Cost: ${optimal_cost:.2f}/MWh',
    hoverinfo='text'
), row=1, col=1)

# Add factory threshold line
fig.add_vline(x=10, line_dash="dash", line_color="orange", opacity=0.5, 
              annotation_text="Factory Threshold", row=1, col=1)

# Plot 2: Optimal altitude vs size
fig.add_trace(go.Scatter(
    x=sizes,
    y=optimal_altitudes,
    mode='lines+markers',
    line=dict(color='blue', width=2),
    marker=dict(size=4, color='blue'),
    name='Optimal Altitude',
    hovertemplate='<b>Area:</b> %{x:.2f} m²<br><b>Optimal Alt:</b> %{y:.0f} m<extra></extra>'
), row=1, col=2)

# Add factory threshold line
fig.add_vline(x=10, line_dash="dash", line_color="orange", opacity=0.5,
              annotation_text="Factory", annotation_position="top", row=1, col=2)

# Plot 3: Minimum cost vs size
fig.add_trace(go.Scatter(
    x=sizes,
    y=optimal_costs_for_size,
    mode='lines+markers',
    line=dict(color='green', width=2),
    marker=dict(size=4, color='green'),
    name='Min Cost',
    hovertemplate='<b>Area:</b> %{x:.2f} m²<br><b>Min Cost:</b> $%{y:.2f}/MWh<extra></extra>'
), row=2, col=1)

# Add factory threshold line
fig.add_vline(x=10, line_dash="dash", line_color="orange", opacity=0.5,
              annotation_text="Factory", annotation_position="top", row=2, col=1)

# Plot 4: Summary table
production_type = "Factory-produced" if optimal_area < 10 else "Custom-built"
summary_data = [
    ["Production Type", production_type],
    ["Swept Area", f"{optimal_area:.2f} m²"],
    ["Diameter", f"{2*np.sqrt(optimal_area/np.pi):.2f} m"],
    ["Optimal Altitude", f"{optimal_altitude:.0f} m"],
    ["Wind Speed", f"{get_wind_speed(optimal_altitude):.2f} m/s"],
    ["Turbine Mass", f"{get_turbine_mass(optimal_area):.1f} kg"],
    ["Power Output", f"{0.5 * 1.225 * np.exp(-optimal_altitude / 8400) * optimal_area * get_wind_speed(optimal_altitude) ** 3 * 0.4 / 1000:.2f} kW"],
    ["Annual Energy", f"{0.5 * 1.225 * np.exp(-optimal_altitude / 8400) * optimal_area * get_wind_speed(optimal_altitude) ** 3 * 0.4 * 8760 * 0.35 / 1e9:.2f} MWh"],
    ["Cost per MWh", f"${optimal_cost:.2f}"]
]

fig.add_trace(go.Table(
    header=dict(
        values=['<b>Parameter</b>', '<b>Value</b>'],
        fill_color='paleturquoise',
        align='left',
        font=dict(size=12, color='black')
    ),
    cells=dict(
        values=[[row[0] for row in summary_data], [row[1] for row in summary_data]],
        fill_color='lavender',
        align='left',
        font=dict(size=11)
    )
), row=2, col=2)

# Update axes
fig.update_xaxes(title_text="Swept Area (m²)", type="log", row=1, col=1)
fig.update_yaxes(title_text="Altitude (m)", type="linear", row=1, col=1)

fig.update_xaxes(title_text="Swept Area (m²)", type="log", row=1, col=2)
fig.update_yaxes(title_text="Optimal Altitude (m)", type="linear", row=1, col=2)

fig.update_xaxes(title_text="Swept Area (m²)", type="log", row=2, col=1)
fig.update_yaxes(title_text="Min Cost ($/MWh)", type="log", row=2, col=1)

# Update layout
fig.update_layout(
    title=dict(
        text='Floating Wind Turbine: Comprehensive Analysis from Micro to Macro<br><sub>Red line shows optimal altitude for each size | Hover for details</sub>',
        x=0.5,
        xanchor='center',
        font=dict(size=16)
    ),
    width=1600,
    height=1200,
    hovermode='closest',
    showlegend=True
)

fig.show()

# Print statistics with scale breakdown
print(f"\n{'='*70}")
print(f"FLOATING WIND TURBINE COST ANALYSIS: MICRO TO MACRO")
print(f"{'='*70}")

# Analyze different scales
micro_mask = S < 1  # < 1 m²
small_mask = (S >= 1) & (S < 10)  # 1-10 m² (factory)
medium_mask = (S >= 10) & (S < 100)  # 10-100 m²
large_mask = S >= 100  # > 100 m²

print(f"\nScale Analysis:")
print(f"  Micro (<1 m²):    Min: ${costs[micro_mask].min():.2f}/MWh, Mean: ${costs[micro_mask].mean():.2f}/MWh")
print(f"  Small (1-10 m²):  Min: ${costs[small_mask].min():.2f}/MWh, Mean: ${costs[small_mask].mean():.2f}/MWh")
print(f"  Medium (10-100):  Min: ${costs[medium_mask].min():.2f}/MWh, Mean: ${costs[medium_mask].mean():.2f}/MWh")
print(f"  Large (>100 m²):  Min: ${costs[large_mask].min():.2f}/MWh, Mean: ${costs[large_mask].mean():.2f}/MWh")

print(f"\nOverall Statistics:")
print(f"  Minimum cost per MWh: ${costs.min():.2f}")
print(f"  Maximum cost per MWh: ${costs.max():.2f}")
print(f"  Mean cost per MWh: ${costs.mean():.2f}")

print(f"\nOptimal Configuration ({production_type}):")
print(f"  Swept Area: {optimal_area:.2f} m² (diameter: {2*np.sqrt(optimal_area/np.pi):.2f} m)")
print(f"  Altitude: {optimal_altitude:.0f} m")
print(f"  Wind Speed: {get_wind_speed(optimal_altitude):.2f} m/s")
print(f"  Turbine Mass: {get_turbine_mass(optimal_area):.1f} kg")
print(f"  Power Output: {0.5 * 1.225 * np.exp(-optimal_altitude / 8400) * optimal_area * get_wind_speed(optimal_altitude) ** 3 * 0.4 / 1000:.2f} kW")
print(f"  Cost per MWh: ${optimal_cost:.2f}")
print(f"{'='*70}\n")