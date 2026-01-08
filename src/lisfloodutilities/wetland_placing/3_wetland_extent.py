"""
Wetland Extent Network Extraction and Daily Conversion

This script extracts monthly climatology time series of wetland extent at placed
wetland locations and calculates local (non-upstream) wetland contribution.
Results are exported as both monthly climatologies and daily time series
(linear interpolation and flat broadcast methods).

Workflow:
1. Load placed wetlands (CSV) and upstream accumulated wetland area (NetCDF)
2. Extract monthly time series (2000-2018) at each wetland location
3. Compute monthly climatology (12-month seasonal cycle averaged across years)
4. Build wetland network topology (identify direct upstream wetlands per wetland)
5. Calculate local wetland extent by subtracting direct upstream contributions
6. Convert monthly climatology to daily time series (interpolated and flat methods)
7. Export results as CSV files and generate verification plots

Input:
- placed_wetlands.csv: Wetland locations from wetlands_placing.py
- WAD2M_upstream_inundated_area_km2_allcells.nc: Accumulated wetland area (monthly 2000-2018)
- allupstream_connect_niger.pkl: Network connectivity (all upstream cells per cell)

Output:
- wetland_extent_km2_total.csv: Total upstream wetland extent (12 months × N wetlands)
- wetland_extent_km2_local.csv: Local wetland extent after subtracting upstream (12 months × N wetlands)
- wetland_extent_km2_local_daily_lininterp.csv: Daily local extent via linear interpolation (365 days × N wetlands)
- wetland_extent_km2_local_daily_flat.csv: Daily local extent via flat broadcast (365 days × N wetlands)
- wetland_metadata.csv: Wetland metadata (lat, lon, type, n_direct_upstream_wetlands)
"""

import xarray as xr
import numpy as np
from pathlib import Path
import platform
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import helper_functions as hf
import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import Point

# ========================================
# FUNCTIONS
# ========================================
# Temporal conversion functions moved to helper_functions.py
# Use hf.monthly_to_daily_flat() and hf.monthly_to_daily_lininterp()


def plot_wetland_placement_verification(wad2m_var, wetlands_df, output_path):
    """
    Create verification map showing wetland placement on WAD2M accumulated dataset.
    
    Parameters:
    -----------
    wad2m_var : xarray.DataArray
        WAD2M accumulated wetland area (time × lat × lon)
    wetlands_df : pd.DataFrame
        Wetland locations with columns: wetland_id, lat, lon
    output_path : Path
        Where to save the figure
    """
    import matplotlib.pyplot as plt
    
    # Get time-averaged WAD2M
    wad2m_mean = wad2m_var.mean(dim='time')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Plot WAD2M accumulated wetland area with log scale
    wad2m_mean.plot(
        ax=ax,
        cmap='Blues',
        norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=wad2m_mean.max().values),
        cbar_kwargs={'label': 'Accumulated Wetland Area (km², log scale)'}
    )
    
    # Plot all wetland locations
    ax.scatter(
        wetlands_df['lon'], 
        wetlands_df['lat'],
        c='green',
        s=100,
        edgecolors='black',
        linewidths=1,
        zorder=5,
        label='Wetland locations'
    )
    
    # Add wetland ID labels
    for idx, row in wetlands_df.iterrows():
        ax.text(
            row['lon'] + 0.05,
            row['lat'] + 0.05,
            str(row['wetland_id']),
            fontsize=10,
            color='black',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='none'),
            zorder=6
        )
    
    ax.set_title('Wetland Placement\nGreen dots = wetlands, IDs labeled', fontsize=14)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    #plt.tight_layout()
    #plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #plt.close()
    
    #print(f"   Saved verification map: {output_path}")

def plot_wetland_climatology_dropdown(climatology_local, climatology_wide):
    """
    Create interactive Plotly plot with toggleable wetland traces.

    Parameters:
    -----------
    climatology_local : pd.DataFrame
        Local wetland extent climatology with metadata
    climatology_wide : pd.DataFrame
        Total upstream wetland extent climatology

    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive line plot with legend toggle
    """
    import plotly.graph_objects as go
    
    wetland_ids = sorted(climatology_local['wetland_id'].tolist())
    months = list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    # Create traces for all wetlands
    traces = []
    for wetland_id in wetland_ids:
        local_row = climatology_local[climatology_local['wetland_id'] == wetland_id]
        if len(local_row) == 0:
            continue
        
        local_values = [local_row[f'month_{i}'].values[0] for i in months]
        wtype = local_row['type'].values[0]
        
        traces.append(go.Scatter(
            x=month_names,
            y=local_values,
            mode='lines+markers',
            name=f'Wetland {wetland_id} ({wtype})',
            line=dict(width=2),
            marker=dict(size=8)
        ))
    
    fig = go.Figure(data=traces)
    
    fig.update_layout(
        title='Wetland Local Extent Climatology (Click legend to toggle)',
        xaxis_title='Month',
        yaxis_title='Local Wetland Extent (km²)',
        hovermode='x unified',
        height=600,
        width=1200,
        template='plotly_white'
    )
    
    return fig



# =============================================================================
# 1. CONFIGURATION & SETUP
# =============================================================================
# Set platform-specific base paths and define all input/output file paths.
# Configure whether to include inter-node wetlands or only tributary wetlands.

CASE_NAME = "juba"  # Case study identifier

# --- Paths ---
system = platform.system()
if system == 'Windows':
    BASE = Path(rf'C:\Users\sorger\OneDrive - IIASA\Seed-FD\wetlands\4_case_studies\{CASE_NAME}')
else:
    BASE = Path(rf'/Users/sorger/Library/CloudStorage/OneDrive-IIASA/Seed-FD/wetlands/4_case_studies/{CASE_NAME}')

PATHS = {
    'wetlands_csv': BASE / '2_results' / f'2_placed_wetlands_{CASE_NAME}.csv',
    'wad2m_upstream_nc': BASE / '2_results' / f'1_WAD2M_upstream_area_km2_{CASE_NAME}.nc',
    'connectivity_all': BASE / '1_data' / f'allupstream_connect_{CASE_NAME}.pkl',
    'output_climatology_total': BASE / '2_results' / f'3_wetland_extent_km2_total_{CASE_NAME}.csv',
    'output_climatology_local': BASE / '2_results' / f'3_wetland_extent_km2_local_{CASE_NAME}.csv',
    'output_wl_extent_daily_lininterp': BASE / '2_results' / f'3_wetland_extent_km2_local_daily_lininterp_{CASE_NAME}.csv',
    'output_wl_extent_daily_flat': BASE / '2_results' / f'3_wetland_extent_km2_local_daily_flat_{CASE_NAME}.csv',
    'output_metadata': BASE / '2_results' / f'3_wetland_metadata_{CASE_NAME}.csv',
    'output_wetlands_filtered': BASE / '2_results' / f'3_placed_wetlands_filtered_{CASE_NAME}.csv',
    'output_removed_wetlands': BASE / '2_results' / f'3_removed_wetlands_{CASE_NAME}.csv',
    'shapefile_wetlands_filtered': BASE / '2_results' / f'3_placed_wetlands_filtered_{CASE_NAME}.shp'
}

# --- Parameters ---
MIN_LOCAL_EXTENT_KM2 = 250  # Minimum max local wetland for filtering wetlands with small extents (km²)

# =============================================================================
# 2. DATA LOADING
# =============================================================================
# Load wetland placement results, upstream accumulated wetland area from WAD2M,
# and river network connectivity. Filter wetlands by type if configured.
#
# Functions used:
# - pd.read_csv() - Load placed wetlands CSV
# - xr.open_dataset() - Load NetCDF gridded wetland area
# - pickle.load() - Load network connectivity dictionary

# Load placed wetlands
INCLUDE_INTERNODE_WETLANDS = True  # Set to False to only include tributary wetlands
wetlands_df = pd.read_csv(PATHS['wetlands_csv'])

print(f"""   Loaded {len(wetlands_df)} wetland locations"
      Types: {wetlands_df['type'].value_counts().to_dict()}""")

# Filter by wetland type if configured
if not INCLUDE_INTERNODE_WETLANDS:
    n_total = len(wetlands_df)
    wetlands_df = wetlands_df[wetlands_df['type'] == 'tributary'].reset_index(drop=True)
    n_filtered = len(wetlands_df)
    print(f"""   Loaded {n_total} wetland locations
   Filtered to {n_filtered} tributary wetlands only (excluding {n_total - n_filtered} inter-node wetlands)
   Types: {wetlands_df['type'].value_counts().to_dict()}""")
else:
    print(f"""   Loaded {len(wetlands_df)} wetland locations
   Types: {wetlands_df['type'].value_counts().to_dict()}""")

# Load upstream accumulated wetland area NetCDF
ds_wad2m_ups = xr.open_dataset(PATHS['wad2m_upstream_nc'])
wad2m_var = ds_wad2m_ups["WAD2M_inundated_area_upstream"]  
print(f"""   Loaded NetCDF: {wad2m_var.dims}
Shape: {wad2m_var.shape}")
Time range: {len(wad2m_var.time)} months""")

# Load network connectivity (all upstream cells per cell)
with open(PATHS['connectivity_all'], 'rb') as f:
    allupstream_connect = pickle.load(f)
print(f"   Loaded connectivity: {len(allupstream_connect)} cells")

# =============================================================================
# 3. EXTRACT TIME SERIES AT WETLAND POINTS
# =============================================================================
# Extract monthly WAD2M upstream wetland area at each placed wetland location
# using nearest-neighbor selection. Reshape from wide to long format for climatology.
#
# Functions used:
# - hf.extract_station_data_from_da() - Extract gridded data at point locations
# - pd.melt() - Reshape wide to long format

# Extract time series at wetland locations using helper function
ts_df = hf.extract_station_data_from_da(
    da=wad2m_var,
    stations_df=wetlands_df,
    station_ID='wetland_id',
    lat='lat',
    lon='lon',
    method='nearest'
)

# The result has time as index, wetland_ids as columns
# Reshape to long format for climatology calculation
ts_df = ts_df.reset_index()  # time becomes a column
ts_df = ts_df.melt(id_vars='time', var_name='wetland_id', value_name='upstream_area_km2')

print(f"   Extracted {len(ts_df)} time series records ({len(ts_df['wetland_id'].unique())} wetlands × {len(ts_df['time'].unique())} timesteps)")
print(ts_df.head(10))
# =============================================================================
# 4. COMPUTE MONTHLY CLIMATOLOGY (TOTAL UPSTREAM)
# =============================================================================
# Calculate 12-month seasonal climatology by averaging each month across all years
# (2000-2018). Pivot to wide format and merge with wetland metadata.
#
# Functions used:
# - pd.groupby() - Group by wetland_id and month
# - pd.pivot() - Reshape to wide format (wetland_id × month)
# - pd.merge() - Join with wetland metadata

# Add month-of-year from time coordinate
# Assuming monthly data 2000-2018 (228 months = 19 years)
ts_df['month'] = ts_df['time'].dt.month

# Group by wetland_id and month, calculate mean across years
climatology_df = ts_df.groupby(['wetland_id', 'month'])['upstream_area_km2'].mean().reset_index()
climatology_df.rename(columns={'upstream_area_km2': 'mean_upstream_km2'}, inplace=True)

# Pivot to wide format (wetland_id, month_1, month_2, ..., month_12)
climatology_wide = climatology_df.pivot(index='wetland_id', columns='month', values='mean_upstream_km2')
climatology_wide.columns = [f'month_{i}' for i in range(1, 13)]
climatology_wide = climatology_wide.reset_index()

# Merge back with wetland metadata
climatology_wide = climatology_wide.merge(
    wetlands_df[['wetland_id', 'lat', 'lon', 'type', 'comp_idx']],
    on='wetland_id',
    how='left'
)

print(climatology_df.head(12))
print(climatology_wide.head(12))
# =============================================================================
# 5. BUILD WETLAND NETWORK TOPOLOGY
# =============================================================================
# Identify direct upstream wetlands for each wetland by filtering out wetlands
# that are upstream of other upstream wetlands (avoid double-counting).
#
# Functions used:
# - dict.zip() - Create comp_idx → wetland_id mapping
# - set operations - Filter to direct upstream only


def build_wetland_network_topology(wl_df, allupstream_connect):
    """
    Build wetland network topology identifying direct upstream wetlands.

    Parameters:
        wl_df: DataFrame with wetland locations (must have 'wetland_id', 'comp_idx')
        allupstream_connect: list of lists with all upstream cells per cell

    Returns:
        dict mapping wetland_id -> list of direct upstream wetland_ids
    """
    # Create mapping: comp_idx → wetland_id
    wetland_comp_idx_to_id = dict(zip(wl_df['comp_idx'].astype(int), wl_df['wetland_id']))

    # Build DIRECT upstream wetlands (not all upstream)
    wetland_direct_upstream = {}

    for idx, wetland in wl_df.iterrows():
        wetland_id = wetland['wetland_id']
        comp_idx = int(wetland['comp_idx'])

        # Get ALL upstream cells
        all_upstream_cells = allupstream_connect[comp_idx] if comp_idx < len(allupstream_connect) else []

        # Find which ones are wetlands
        all_upstream_wetlands = [
            wetland_comp_idx_to_id[up_comp]
            for up_comp in all_upstream_cells
            if up_comp in wetland_comp_idx_to_id
        ]

        # Filter to DIRECT upstream only using set difference
        direct = set(all_upstream_wetlands)

        # Remove any that are upstream of other upstream wetlands
        for candidate_id in all_upstream_wetlands:
            candidate_comp = int(wl_df[wl_df['wetland_id'] == candidate_id]['comp_idx'].values[0])
            candidate_upstream_cells = allupstream_connect[candidate_comp] if candidate_comp < len(allupstream_connect) else []

            candidate_upstream_wetlands = [
                wetland_comp_idx_to_id[up_comp]
                for up_comp in candidate_upstream_cells
                if up_comp in wetland_comp_idx_to_id
            ]

            direct = direct - set(candidate_upstream_wetlands)

        wetland_direct_upstream[wetland_id] = list(direct)

    return wetland_direct_upstream


# Build initial network topology
wetland_direct_upstream = build_wetland_network_topology(wetlands_df, allupstream_connect)


# =============================================================================
# 6. CALCULATE LOCAL WETLAND EXTENT (SUBTRACT UPSTREAM)
# =============================================================================
# For each wetland and each month, subtract the sum of direct upstream wetlands'
# climatology to isolate local (non-upstream) wetland contribution.
#
# Functions used:
# - pd.DataFrame.iterrows() - Iterate over wetlands
# - pd.DataFrame.at[] - Update values in place


def calculate_local_extent(clim_wide, wetland_direct_upstream):
    """
    Calculate local wetland extent by subtracting direct upstream contributions.

    Parameters:
        clim_wide: DataFrame with total upstream climatology (wetland_id, month_1...month_12, metadata)
        wetland_direct_upstream: dict mapping wetland_id -> list of direct upstream wetland_ids

    Returns:
        DataFrame with local wetland extent (same structure as clim_wide)
    """
    month_cols = [f'month_{i}' for i in range(1, 13)]
    clim_local = clim_wide.copy()

    for col in month_cols:
        for idx, row in clim_local.iterrows():
            wetland_id = row['wetland_id']
            total_area = row[col]

            direct_upstream_wetlands = wetland_direct_upstream.get(wetland_id, [])

            upstream_sum = sum(
                clim_wide[clim_wide['wetland_id'] == up_id][col].values[0]
                for up_id in direct_upstream_wetlands
                if len(clim_wide[clim_wide['wetland_id'] == up_id]) > 0
            )

            local_extent = total_area - upstream_sum
            clim_local.at[idx, col] = local_extent

    # Add count of direct upstream wetlands
    clim_local['n_direct_upstream_wetlands'] = clim_local['wetland_id'].map(
        lambda wid: len(wetland_direct_upstream.get(wid, []))
    )

    return clim_local


# Calculate initial local extent
climatology_local = calculate_local_extent(climatology_wide, wetland_direct_upstream)

month_cols = [f'month_{i}' for i in range(1, 13)]
n_negative = (climatology_local[month_cols] < 0).sum().sum()

print(f"""   Initial local wetland extent for {len(climatology_local)} wetlands
   Mean local extent: {climatology_local[month_cols].mean().mean():.1f} km²
   Negative values found: {n_negative} {'OK' if n_negative == 0 else '⚠ WARNING: Check direct upstream logic!'}""")


# =============================================================================
# 6.1 ITERATIVE FILTERING OF LOW-EXTENT WETLANDS
# =============================================================================
# Remove wetlands with max local extent < MIN_LOCAL_EXTENT_KM2, then rebuild
# network topology and recalculate local extents. Repeat until no removals needed.

print(f"\n--- Iterative Wetland Filtering (MIN_LOCAL_EXTENT_KM2 = {MIN_LOCAL_EXTENT_KM2}) ---")

removed_wetlands_records = []
iteration = 0

while True:
    # Calculate max local extent across all months for each wetland
    climatology_local['max_local_extent'] = climatology_local[month_cols].max(axis=1)

    # Find wetlands below threshold
    low_extent_wetlands = climatology_local[climatology_local['max_local_extent'] < MIN_LOCAL_EXTENT_KM2]

    if len(low_extent_wetlands) == 0:
        print(f"No more wetlands below {MIN_LOCAL_EXTENT_KM2} km² - filtering complete")
        break

    iteration += 1

    # Find wetland with lowest max extent
    min_idx = low_extent_wetlands['max_local_extent'].idxmin()
    wetland_to_remove = low_extent_wetlands.loc[min_idx]

    removed_wetland_id = wetland_to_remove['wetland_id']
    removed_max_extent = wetland_to_remove['max_local_extent']
    removed_type = wetland_to_remove['type']
    removed_lat = wetland_to_remove['lat']
    removed_lon = wetland_to_remove['lon']

    # Record the removal
    removed_wetlands_records.append({
        'iteration': iteration,
        'wetland_id': removed_wetland_id,
        'max_local_extent_km2': removed_max_extent,
        'type': removed_type,
        'lat': removed_lat,
        'lon': removed_lon
    })

    print(f"  Iteration {iteration}: Removing wetland {removed_wetland_id} "
          f"(max_local_extent={removed_max_extent:.1f} km², type={removed_type})")

    # Remove from wetlands_df
    wetlands_df = wetlands_df[wetlands_df['wetland_id'] != removed_wetland_id].copy()

    # Remove from climatology_wide (needed for recalculation)
    climatology_wide = climatology_wide[climatology_wide['wetland_id'] != removed_wetland_id].copy()

    # Rebuild network topology with updated wetlands
    wetland_direct_upstream = build_wetland_network_topology(wetlands_df, allupstream_connect)

    # Recalculate local extents
    climatology_local = calculate_local_extent(climatology_wide, wetland_direct_upstream)

# Clean up temporary column
if 'max_local_extent' in climatology_local.columns:
    climatology_local = climatology_local.drop(columns=['max_local_extent'])

# Create removed wetlands DataFrame
removed_wetlands_df = pd.DataFrame(removed_wetlands_records)

# Final statistics
n_negative_final = (climatology_local[month_cols] < 0).sum().sum()

print(f"""
Wetland Filtering Summary:
  Initial wetlands:          {len(wetlands_df) + len(removed_wetlands_df)}
  Wetlands removed:          {len(removed_wetlands_df)}
  Final wetlands:            {len(wetlands_df)}
  Iterations:                {iteration}
  Final mean local extent:   {climatology_local[month_cols].mean().mean():.1f} km²
  Negative values remaining: {n_negative_final}""")

if len(removed_wetlands_df) > 0:
    print("\nRemoved wetlands details:")
    print(removed_wetlands_df.to_string(index=False))



# =============================================================================
# 7. EXPORT RESULTS
# =============================================================================
# Export monthly climatologies (total and local), convert to daily time series
# using both linear interpolation and flat broadcast methods, and export metadata.
#
# Functions used:
# - pd.DataFrame.set_index() - Reshape for export
# - monthly_to_daily() - Linear interpolation to daily values
# - monthly_to_daily_flat() - Flat broadcast to daily values
# - pd.DataFrame.to_csv() - Write CSV files

# --- Total upstream climatology ---
# Format: Rows = months (1-12), Columns = wetland_id
month_cols = [f'month_{i}' for i in range(1, 13)]
climatology_export = climatology_wide.set_index('wetland_id')[month_cols].T
climatology_export.index = range(1, 13)  # Rows = months 1-12
climatology_export.index.name = 'month'

climatology_export.to_csv(PATHS['output_climatology_total'])
print(f"   Saved total upstream climatology: {PATHS['output_climatology_total']}")
print(f"   Format: {len(climatology_export)} months × {len(climatology_export.columns)} wetlands")

# --- Local wetland extents monthly km2 ---
# Format: Rows = months (1-12), Columns = wetland_id
local_export = climatology_local.set_index('wetland_id')[month_cols].T
local_export.index = range(1, 13)  # Rows = months 1-12
local_export.index.name = 'month'

local_export.to_csv(PATHS['output_climatology_local'])
print(f"   Saved local climatology: {PATHS['output_climatology_local']}")
print(f"   Format: {len(local_export)} months × {len(local_export.columns)} wetlands")

# ---- Local wetland extents daily km2 ----
# Prepare DataFrame for monthly_to_daily: index=months, columns=wetland_ids
local_wl_extent_monthly = climatology_local.set_index('wetland_id')[month_cols].T
local_wl_extent_monthly.index = range(12)  # 0-11 for months

# Interpolate monthly values to daily values
daily_km2_lininterp = hf.monthly_to_daily_lininterp(local_wl_extent_monthly, 2021)
daily_km2_lininterp.to_csv(PATHS['output_wl_extent_daily_lininterp'], index=False)

# Flatly broadcasting of monthly values to daily values
daily_km2_flat = hf.monthly_to_daily_flat(local_wl_extent_monthly, 2021)
daily_km2_flat.to_csv(PATHS['output_wl_extent_daily_flat'], index=False)

print(f"""
Daily conversion complete:
  Linear interpolation: {len(daily_km2_lininterp)} days × {len(daily_km2_lininterp.columns)-3} wetlands
  Flat broadcast: {len(daily_km2_flat)} days × {len(daily_km2_flat.columns)-3} wetlands
  Files saved:
    - {PATHS['output_wl_extent_daily_lininterp']}
    - {PATHS['output_wl_extent_daily_flat']}""")


# --- Wetland metadata ---
# Format: wetland_id, lat, lon, type, comp_idx, n_upstream_wetlands
metadata_export = climatology_local[['wetland_id', 'lat', 'lon', 'type', 'comp_idx', 'n_direct_upstream_wetlands']]
metadata_export.to_csv(PATHS['output_metadata'], index=False)
print(f"   Saved wetland metadata: {PATHS['output_metadata']}")

# --- Filtered wetlands (updated placed_wetlands) ---
wetlands_df.to_csv(PATHS['output_wetlands_filtered'], index=False)
print(f"   Saved filtered wetlands: {PATHS['output_wetlands_filtered']}")

# --- Removed wetlands (for inspection) ---
if len(removed_wetlands_df) > 0:
    removed_wetlands_df.to_csv(PATHS['output_removed_wetlands'], index=False)
    print(f"   Saved removed wetlands: {PATHS['output_removed_wetlands']}")

# --- Export filtered wetlands as shapefile for QGIS ---
if len(wetlands_df) > 0:
    geometry_wetlands = [Point(xy) for xy in zip(wetlands_df['lon'], wetlands_df['lat'])]
    gdf_wetlands = gpd.GeoDataFrame(wetlands_df, geometry=geometry_wetlands, crs='EPSG:4326')
    gdf_wetlands.to_file(PATHS['shapefile_wetlands_filtered'], driver='ESRI Shapefile')
    print(f"   Saved filtered wetlands shapefile: {PATHS['shapefile_wetlands_filtered']}")

# =============================================================================
# 8. VERIFICATION PLOTS
# =============================================================================
# Generate spatial map of wetland placement, interactive climatology plot,
# and time series visualization of daily wetland extents.
#
# Functions used:
# - plot_wetland_placement_verification() - Spatial verification map
# - plot_wetland_climatology_dropdown() - Interactive Plotly line plot
# - hf.quick_ts_plot() - Static matplotlib time series plot

# Wetland placement overview plot
plot_wetland_placement_verification(
    wad2m_var=wad2m_var,
    wetlands_df=wetlands_df,
    output_path=BASE / '2_results' / 'wetland_placement.png'
)

# Interactive plot of local wetland extents
fig = plot_wetland_climatology_dropdown(climatology_local, climatology_wide)
fig.show()

# Daily interpolated wetland extents
hf.quick_ts_plot(
    daily_km2_flat.set_index(pd.date_range('2020-01-01', periods=len(daily_km2_flat))),
    title='Daily Wetland Inundation Extent',
    ylabel='Area [km²]'
)
