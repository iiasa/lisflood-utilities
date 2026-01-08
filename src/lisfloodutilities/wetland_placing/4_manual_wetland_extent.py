"""
Manual Wetland Extent Extraction

Standalone script for extracting wetland extents from manually provided wetland
locations. Use this when you have your own wetland coordinates and want to skip
the automated placement (Scripts 0-2).

WORKFLOW:
---------
1. Load user's manual wetland CSV (wetland_id, lat, lon)
2. Derive comp_idx from LDD raster (for network topology)
3. Extract monthly time series (2000-2018) at each wetland location
4. Compute monthly climatology (12-month seasonal cycle)
5. Build wetland network topology (identify direct upstream wetlands)
6. Calculate local wetland extent by subtracting upstream contributions
7. Convert to daily time series and export

INPUTS:
-------
- 1_data/manual_wetlands_{case}.csv: User-provided wetland locations
  Required columns: wetland_id, lat, lon
- 1_data/ldd_{case}.map: Flow direction raster (for comp_idx derivation)
- 1_data/allupstream_connect_{case}.pkl: Network connectivity (from Script 0)
- 2_results/1_WAD2M_upstream_area_km2_{case}.nc: Accumulated wetland area (from Script 1)

OUTPUTS:
--------
- 4_wetland_extent_km2_total_{case}.csv: Total upstream extent (12 months × N wetlands)
- 4_wetland_extent_km2_local_{case}.csv: Local extent (12 months × N wetlands)
- 4_wetland_extent_km2_local_daily_lininterp_{case}.csv: Daily (interpolated)
- 4_wetland_extent_km2_local_daily_flat_{case}.csv: Daily (flat broadcast)
- 4_wetland_metadata_{case}.csv: Wetland metadata
- 4_placed_wetlands_filtered_{case}.csv: Final wetlands after filtering
- 4_placed_wetlands_filtered_{case}.shp: Shapefile for QGIS


USAGE:
------
1. Create CSV file: 1_data/manual_wetlands_{case}.csv
2. Set CASE_NAME parameter
3. Run: python 4_manual_wetland_extent.py

Author: FSD
Created: 2025-12
"""

import xarray as xr
import numpy as np
from pathlib import Path
import platform
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import rasterio
import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import Point


# =============================================================================
# COMPRESSION UTILITIES (must match pickle creation script)
# =============================================================================

def compress(input_arr, mask):
    """Compress 2D array to 1D by filtering out masked cells."""
    out = input_arr.ravel()
    return np.ma.compressed(np.ma.masked_array(out, mask))


def decompress(input_arr, mask1, shape, dtype):
    """Decompress 1D array back to 2D using mask."""
    out = mask1.copy().astype(dtype)
    out[~mask1] = input_arr[:]
    out = out.reshape(shape)
    return out


# =============================================================================
# HELPER FUNCTIONS (copied from helper_functions.py for standalone use)
# =============================================================================

def quick_ts_plot(df, title=None, ylabel=None, xlabel='Time', figsize=(14, 6), legend=True, ylim=None):
    """
    Quick lightweight plot of multiple time series from a DataFrame.
    Handles NaN values correctly - lines start/end where data exists.
    """
    df = df.copy().apply(pd.to_numeric, errors='coerce')

    fig, ax = plt.subplots(figsize=figsize)

    for col in df.columns:
        ax.plot(df.index, df[col], label=str(col), linewidth=1)

    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if ylim is not None:
        ax.set_ylim(ylim)

    if legend and len(df.columns) <= 20:
        ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    elif legend:
        ax.legend(ncol=3, frameon=False, fontsize=8)

    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return fig, ax


def extract_station_data_from_da(da, stations_df, station_ID, lat, lon, method="nearest"):
    """
    Build a DataFrame of time series from a gridded DataArray.
    Assumes DA has coords 'lat' and 'lon'. Column names in stations_df
    are provided via station_ID, lat, lon.
    """
    time_series_dict = {}

    for _, row in stations_df.iterrows():
        sid = row[station_ID]
        la = float(row[lat])
        lo = float(row[lon])

        ts = da.sel(lat=la, lon=lo, method=method).to_pandas()
        ts.index = pd.to_datetime(ts.index)
        ts.index.name = "time"

        time_series_dict[sid] = ts
        print(f"{sid} extracted")

    print("Done!")
    df = pd.DataFrame(time_series_dict).sort_index(axis=1)
    df.index = pd.to_datetime(df.index)
    df.index.name = "time"
    return df


def monthly_to_daily_flat(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Convert monthly wetland extent to daily values using flat (step) assignment.

    Each day gets the value of its corresponding month (no interpolation).
    Creates a step function where values change on the 1st of each month.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 12 rows (months 0-11 as index) and data columns
    year : int
        Year for daily time series

    Returns
    -------
    pd.DataFrame
        Daily values with columns: count, day, month, <data_columns>
    """
    daily_idx = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    daily_df = pd.DataFrame(index=daily_idx)

    for col in df.columns:
        daily_df[col] = daily_idx.month.map(lambda m: df.loc[m-1, col])

    out = daily_df.reset_index(drop=True).assign(
        count=lambda x: range(1, len(x) + 1),
        day=lambda x: daily_idx.day,
        month=lambda x: daily_idx.month
    )

    out = out[["count", "day", "month", *df.columns]].round(0).astype(int)
    return out


def monthly_to_daily_lininterp(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """
    Convert monthly wetland extent to daily values using linear interpolation.

    Monthly values are anchored to the 15th of each month with smooth transitions
    between months. Handles year wrap-around by extending with Dec (prev year)
    and Jan (next year).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 12 rows (months 0-11 as index) and data columns
    year : int
        Year for daily time series

    Returns
    -------
    pd.DataFrame
        Daily values with columns: count, day, month, <data_columns>
    """
    anchors = pd.to_datetime([f"{year}-{m:02d}-15" for m in range(1, 13)])
    a = df.copy()
    a.index = anchors

    prev_dec = a.iloc[[11]].set_axis([pd.Timestamp(year-1, 12, 15)])
    next_jan = a.iloc[[0]].set_axis([pd.Timestamp(year+1, 1, 15)])
    anchors_full = pd.concat([prev_dec, a, next_jan]).sort_index()

    daily_idx = pd.date_range(f"{year}-01-01", f"{year+1}-01-01", freq="D")
    interp = anchors_full.reindex(anchors_full.index.union(daily_idx)).interpolate("time")
    daily = interp.reindex(daily_idx)

    out = (
        daily.reset_index(drop=True)
        .assign(count=lambda x: range(1, len(x) + 1),
                day=lambda x: daily.index.day,
                month=lambda x: daily.index.month)
    )

    out = out[["count", "day", "month", *df.columns]].round(0).astype(int)
    return out


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_wetland_placement_verification(wad2m_var, wetlands_df, output_path):
    """Create verification map showing wetland placement on WAD2M."""
    wad2m_mean = wad2m_var.mean(dim='time')

    fig, ax = plt.subplots(figsize=(14, 10))

    wad2m_mean.plot(
        ax=ax,
        cmap='Blues',
        norm=plt.matplotlib.colors.LogNorm(vmin=1, vmax=wad2m_mean.max().values),
        cbar_kwargs={'label': 'Accumulated Wetland Area (km², log scale)'}
    )

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

    ax.set_title('Manual Wetland Placement\nGreen dots = wetlands, IDs labeled', fontsize=14)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


def plot_wetland_climatology_dropdown(climatology_local, climatology_wide):
    """Create interactive Plotly plot with toggleable wetland traces."""
    wetland_ids = sorted(climatology_local['wetland_id'].tolist())
    months = list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

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

CASE_NAME = "juba"  # Case study identifier

# --- Paths ---
system = platform.system()
if system == 'Windows':
    BASE = Path(rf'C:\Users\sorger\OneDrive - IIASA\Seed-FD\wetlands\4_case_studies\{CASE_NAME}')
else:
    BASE = Path(rf'/Users/sorger/Library/CloudStorage/OneDrive-IIASA/Seed-FD/wetlands/4_case_studies/{CASE_NAME}')

PATHS = {
    # Inputs
    'wetlands_csv': BASE / '1_data' / f'manual_wetlands_{CASE_NAME}.csv',
    'ldd': BASE / '1_data' / f'ldd_{CASE_NAME}.map',
    'wad2m_upstream_nc': BASE / '2_results' / f'1_WAD2M_upstream_area_km2_{CASE_NAME}.nc',
    'connectivity_all': BASE / '1_data' / f'allupstream_connect_{CASE_NAME}.pkl',
    # Outputs
    'output_climatology_total': BASE / '2_results' / f'4_wetland_extent_km2_total_{CASE_NAME}.csv',
    'output_climatology_local': BASE / '2_results' / f'4_wetland_extent_km2_local_{CASE_NAME}.csv',
    'output_wl_extent_daily_lininterp': BASE / '2_results' / f'4_wetland_extent_km2_local_daily_lininterp_{CASE_NAME}.csv',
    'output_wl_extent_daily_flat': BASE / '2_results' / f'4_wetland_extent_km2_local_daily_flat_{CASE_NAME}.csv',
    'output_metadata': BASE / '2_results' / f'4_wetland_metadata_{CASE_NAME}.csv',
    'output_wetlands_filtered': BASE / '2_results' / f'4_placed_wetlands_filtered_{CASE_NAME}.csv',
    'output_removed_wetlands': BASE / '2_results' / f'4_removed_wetlands_{CASE_NAME}.csv',
    'shapefile_wetlands_filtered': BASE / '2_results' / f'4_placed_wetlands_filtered_{CASE_NAME}.shp'
}

# --- Parameters ---
MIN_LOCAL_EXTENT_KM2 = 250  # Minimum max local wetland extent to keep [km²]

# =============================================================================
# 2. DATA LOADING
# =============================================================================

print("=" * 70)
print("MANUAL WETLAND EXTENT EXTRACTION")
print("=" * 70)

# Load manual wetlands CSV
print(f"\nLoading manual wetlands from: {PATHS['wetlands_csv']}")
wetlands_df = pd.read_csv(PATHS['wetlands_csv'])

# Validate required columns
required_cols = ['wetland_id', 'lat', 'lon']
missing_cols = [col for col in required_cols if col not in wetlands_df.columns]
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}. CSV must have: {required_cols}")

print(f"  Loaded {len(wetlands_df)} wetland locations")

# =============================================================================
# 2.1 DERIVE COMP_IDX FROM LAT/LON
# =============================================================================
# Convert lat/lon to compressed grid index using LDD raster.
# This is required for network topology and upstream subtraction.

print(f"\nDeriving comp_idx from LDD: {PATHS['ldd']}")

# Load LDD raster
with rasterio.open(PATHS['ldd'], 'r') as src:
    ldd_data = src.read(1)
    ldd_transform = src.transform

ldd = ldd_data.astype(np.int64)
ldd[ldd == 255] = 5  # Make sinks for invalid values

# Create boundary frame with sinks (same as pickle creation)
rows, cols = ldd.shape
for x in range(cols):
    ldd[0, x] = 5
    ldd[rows-1, x] = 5
for y in range(rows):
    ldd[y, 0] = 5
    ldd[y, cols-1] = 5

# Create compression mask (same logic as pickle creation script)
mask = np.invert(np.bool_(ldd.ravel()))
mask1 = np.ma.masked_array(mask, mask)
lddflat = compress(ldd, mask)
lddsize = len(lddflat)
lddorder = np.arange(lddsize)
lddord = decompress(lddorder, mask1, ldd.shape, "int")

# Get LDD coordinate arrays
ldd_lon_coords = np.array([ldd_transform[2] + (i + 0.5) * ldd_transform[0] for i in range(cols)])
ldd_lat_coords = np.array([ldd_transform[5] + (i + 0.5) * ldd_transform[4] for i in range(rows)])

print(f"  LDD shape: {ldd.shape}")
print(f"  Compressed size: {lddsize} cells")

# Assign comp_idx to each wetland
wetlands_df['comp_idx'] = -1  # Initialize

for idx, row in wetlands_df.iterrows():
    lat_idx = np.argmin(np.abs(ldd_lat_coords - row['lat']))
    lon_idx = np.argmin(np.abs(ldd_lon_coords - row['lon']))

    if ldd[lat_idx, lon_idx] != 0:  # Valid LDD cell
        wetlands_df.at[idx, 'comp_idx'] = int(lddord[lat_idx, lon_idx])
    else:
        print(f"  WARNING: Wetland {row['wetland_id']} at ({row['lat']}, {row['lon']}) is outside valid LDD area")

# Add type column
wetlands_df['type'] = 'manual'

# Check for invalid wetlands
invalid_wetlands = wetlands_df[wetlands_df['comp_idx'] == -1]
if len(invalid_wetlands) > 0:
    print(f"  WARNING: {len(invalid_wetlands)} wetlands have invalid comp_idx (outside LDD)")
    wetlands_df = wetlands_df[wetlands_df['comp_idx'] != -1].reset_index(drop=True)
    print(f"  Removed invalid wetlands. Remaining: {len(wetlands_df)}")

print(f"  Assigned comp_idx to {len(wetlands_df)} wetlands")

# =============================================================================
# 2.2 LOAD REMAINING DATA
# =============================================================================

# Load upstream accumulated wetland area NetCDF
print(f"\nLoading WAD2M upstream: {PATHS['wad2m_upstream_nc']}")
ds_wad2m_ups = xr.open_dataset(PATHS['wad2m_upstream_nc'])
wad2m_var = ds_wad2m_ups["WAD2M_inundated_area_upstream"]
print(f"  Shape: {wad2m_var.shape}, Time: {len(wad2m_var.time)} months")

# Load network connectivity
print(f"\nLoading connectivity: {PATHS['connectivity_all']}")
with open(PATHS['connectivity_all'], 'rb') as f:
    allupstream_connect = pickle.load(f)
print(f"  Loaded: {len(allupstream_connect)} cells")

# =============================================================================
# 3. EXTRACT TIME SERIES AT WETLAND POINTS
# =============================================================================

print("\nExtracting time series at wetland locations...")

ts_df = extract_station_data_from_da(
    da=wad2m_var,
    stations_df=wetlands_df,
    station_ID='wetland_id',
    lat='lat',
    lon='lon',
    method='nearest'
)

# Reshape to long format
ts_df = ts_df.reset_index()
ts_df = ts_df.melt(id_vars='time', var_name='wetland_id', value_name='upstream_area_km2')

print(f"  Extracted {len(ts_df)} records ({len(ts_df['wetland_id'].unique())} wetlands × {len(ts_df['time'].unique())} timesteps)")

# =============================================================================
# 4. COMPUTE MONTHLY CLIMATOLOGY (TOTAL UPSTREAM)
# =============================================================================

print("\nComputing monthly climatology...")

ts_df['month'] = ts_df['time'].dt.month

climatology_df = ts_df.groupby(['wetland_id', 'month'])['upstream_area_km2'].mean().reset_index()
climatology_df.rename(columns={'upstream_area_km2': 'mean_upstream_km2'}, inplace=True)

climatology_wide = climatology_df.pivot(index='wetland_id', columns='month', values='mean_upstream_km2')
climatology_wide.columns = [f'month_{i}' for i in range(1, 13)]
climatology_wide = climatology_wide.reset_index()

# Merge with wetland metadata
climatology_wide = climatology_wide.merge(
    wetlands_df[['wetland_id', 'lat', 'lon', 'type', 'comp_idx']],
    on='wetland_id',
    how='left'
)

print(f"  Climatology computed for {len(climatology_wide)} wetlands")

# =============================================================================
# 5. BUILD WETLAND NETWORK TOPOLOGY
# =============================================================================

print("\nBuilding wetland network topology...")


def build_wetland_network_topology(wl_df, allupstream_connect):
    """Build wetland network topology identifying direct upstream wetlands."""
    wetland_comp_idx_to_id = dict(zip(wl_df['comp_idx'].astype(int), wl_df['wetland_id']))
    wetland_direct_upstream = {}

    for idx, wetland in wl_df.iterrows():
        wetland_id = wetland['wetland_id']
        comp_idx = int(wetland['comp_idx'])

        all_upstream_cells = allupstream_connect[comp_idx] if comp_idx < len(allupstream_connect) else []

        all_upstream_wetlands = [
            wetland_comp_idx_to_id[up_comp]
            for up_comp in all_upstream_cells
            if up_comp in wetland_comp_idx_to_id
        ]

        direct = set(all_upstream_wetlands)

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


wetland_direct_upstream = build_wetland_network_topology(wetlands_df, allupstream_connect)

# =============================================================================
# 6. CALCULATE LOCAL WETLAND EXTENT (SUBTRACT UPSTREAM)
# =============================================================================

print("\nCalculating local wetland extent...")


def calculate_local_extent(clim_wide, wetland_direct_upstream):
    """Calculate local wetland extent by subtracting direct upstream contributions."""
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

    clim_local['n_direct_upstream_wetlands'] = clim_local['wetland_id'].map(
        lambda wid: len(wetland_direct_upstream.get(wid, []))
    )

    return clim_local


climatology_local = calculate_local_extent(climatology_wide, wetland_direct_upstream)

month_cols = [f'month_{i}' for i in range(1, 13)]
n_negative = (climatology_local[month_cols] < 0).sum().sum()

print(f"  Mean local extent: {climatology_local[month_cols].mean().mean():.1f} km²")
print(f"  Negative values: {n_negative} {'OK' if n_negative == 0 else '⚠ Check upstream logic!'}")

# =============================================================================
# 6.1 ITERATIVE FILTERING OF LOW-EXTENT WETLANDS
# =============================================================================

print(f"\n--- Iterative Wetland Filtering (MIN_LOCAL_EXTENT_KM2 = {MIN_LOCAL_EXTENT_KM2}) ---")

removed_wetlands_records = []
iteration = 0

while True:
    climatology_local['max_local_extent'] = climatology_local[month_cols].max(axis=1)
    low_extent_wetlands = climatology_local[climatology_local['max_local_extent'] < MIN_LOCAL_EXTENT_KM2]

    if len(low_extent_wetlands) == 0:
        print(f"No more wetlands below {MIN_LOCAL_EXTENT_KM2} km² - filtering complete")
        break

    iteration += 1

    min_idx = low_extent_wetlands['max_local_extent'].idxmin()
    wetland_to_remove = low_extent_wetlands.loc[min_idx]

    removed_wetland_id = wetland_to_remove['wetland_id']
    removed_max_extent = wetland_to_remove['max_local_extent']
    removed_type = wetland_to_remove['type']
    removed_lat = wetland_to_remove['lat']
    removed_lon = wetland_to_remove['lon']

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

    wetlands_df = wetlands_df[wetlands_df['wetland_id'] != removed_wetland_id].copy()
    climatology_wide = climatology_wide[climatology_wide['wetland_id'] != removed_wetland_id].copy()

    wetland_direct_upstream = build_wetland_network_topology(wetlands_df, allupstream_connect)
    climatology_local = calculate_local_extent(climatology_wide, wetland_direct_upstream)

if 'max_local_extent' in climatology_local.columns:
    climatology_local = climatology_local.drop(columns=['max_local_extent'])

removed_wetlands_df = pd.DataFrame(removed_wetlands_records)

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

print("\n" + "=" * 70)
print("EXPORTING RESULTS")
print("=" * 70)

# Ensure output directory exists
PATHS['output_climatology_total'].parent.mkdir(exist_ok=True)

# --- Total upstream climatology ---
month_cols = [f'month_{i}' for i in range(1, 13)]
climatology_export = climatology_wide.set_index('wetland_id')[month_cols].T
climatology_export.index = range(1, 13)
climatology_export.index.name = 'month'

climatology_export.to_csv(PATHS['output_climatology_total'])
print(f"\nSaved total upstream climatology: {PATHS['output_climatology_total']}")

# --- Local wetland extents monthly ---
local_export = climatology_local.set_index('wetland_id')[month_cols].T
local_export.index = range(1, 13)
local_export.index.name = 'month'

local_export.to_csv(PATHS['output_climatology_local'])
print(f"Saved local climatology: {PATHS['output_climatology_local']}")

# --- Daily conversion ---
local_wl_extent_monthly = climatology_local.set_index('wetland_id')[month_cols].T
local_wl_extent_monthly.index = range(12)

daily_km2_lininterp = monthly_to_daily_lininterp(local_wl_extent_monthly, 2021)
daily_km2_lininterp.to_csv(PATHS['output_wl_extent_daily_lininterp'], index=False)

daily_km2_flat = monthly_to_daily_flat(local_wl_extent_monthly, 2021)
daily_km2_flat.to_csv(PATHS['output_wl_extent_daily_flat'], index=False)

print(f"""
Daily conversion complete:
  Linear interpolation: {len(daily_km2_lininterp)} days × {len(daily_km2_lininterp.columns)-3} wetlands
  Flat broadcast: {len(daily_km2_flat)} days × {len(daily_km2_flat.columns)-3} wetlands""")

# --- Wetland metadata ---
metadata_export = climatology_local[['wetland_id', 'lat', 'lon', 'type', 'comp_idx', 'n_direct_upstream_wetlands']]
metadata_export.to_csv(PATHS['output_metadata'], index=False)
print(f"Saved wetland metadata: {PATHS['output_metadata']}")

# --- Filtered wetlands ---
wetlands_df.to_csv(PATHS['output_wetlands_filtered'], index=False)
print(f"Saved filtered wetlands: {PATHS['output_wetlands_filtered']}")

# --- Removed wetlands ---
if len(removed_wetlands_df) > 0:
    removed_wetlands_df.to_csv(PATHS['output_removed_wetlands'], index=False)
    print(f"Saved removed wetlands: {PATHS['output_removed_wetlands']}")

# --- Shapefile ---
if len(wetlands_df) > 0:
    geometry_wetlands = [Point(xy) for xy in zip(wetlands_df['lon'], wetlands_df['lat'])]
    gdf_wetlands = gpd.GeoDataFrame(wetlands_df, geometry=geometry_wetlands, crs='EPSG:4326')
    gdf_wetlands.to_file(PATHS['shapefile_wetlands_filtered'], driver='ESRI Shapefile')
    print(f"Saved shapefile: {PATHS['shapefile_wetlands_filtered']}")

# =============================================================================
# 8. VERIFICATION PLOTS
# =============================================================================

print("\n" + "=" * 70)
print("GENERATING PLOTS")
print("=" * 70)

# Wetland placement overview
plot_wetland_placement_verification(
    wad2m_var=wad2m_var,
    wetlands_df=wetlands_df,
    output_path=BASE / '2_results' / '4_wetland_placement.png'
)

# Interactive climatology plot
fig = plot_wetland_climatology_dropdown(climatology_local, climatology_wide)
fig.show()

# Daily time series plot
quick_ts_plot(
    daily_km2_flat.set_index(pd.date_range('2020-01-01', periods=len(daily_km2_flat))),
    title='Daily Wetland Inundation Extent (Manual Wetlands)',
    ylabel='Area [km²]'
)

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)
