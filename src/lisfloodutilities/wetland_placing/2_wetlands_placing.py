"""
Wetland Placement Algorithm for River Networks

This script identifies optimal locations for placing dynamic wetland cells in a river
network based on:
1. Detecting wetland area jumps along the river network
2. Placing wetlands at tributary confluences
3. Spacing wetlands along inter-node river segments based on distance

The script uses:
- WAD2M wetland inundation dataset
- River network connectivity (upstream links)
- Upstream drainage area
- Accumulated wetland area (flow-accumulated from WAD2M)

Output:
- CSV of placed wetland locations (lat, lon, flat_idx, type)
- CSV of node pairs with flow distances
- Visualizations of placement results
"""

import xarray as xr
import numpy as np
from pathlib import Path
import platform
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import helper_functions as hf
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from collections import deque


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
    'wad2m_regrid': BASE / '1_data' / f'WAD2M_wetlands_2000_2018_{CASE_NAME}.nc',
    'ups': BASE / '1_data' / f'ups_{CASE_NAME}.map',
    'ldd': BASE / '1_data' / f'ldd_{CASE_NAME}.map',
    'connectivity': BASE / '1_data' / f'upstream_connect_{CASE_NAME}.pkl',
    'wad2m_accups': BASE / '2_results' / f'1_WAD2M_upstream_area_km2_{CASE_NAME}.nc',
    'output_wetlands': BASE / '2_results' / f'2_placed_wetlands_{CASE_NAME}.csv',
    'output_nodes': BASE / '2_results' / f'2_node_pairs_{CASE_NAME}.csv',
    'output_removed_nodes': BASE / '2_results' / f'2_removed_nodes_{CASE_NAME}.csv',
    'output_nodes_map': BASE / '2_results' / f'2_wetland_nodes_map_{CASE_NAME}.png',
    'output_placement_map': BASE / '2_results' / f'2_wetland_placement_map_{CASE_NAME}.png',
    'shapefile_nodes': BASE / '2_results' / f'2_wetland_nodes_{CASE_NAME}.shp',
    'shapefile_wetlands': BASE / '2_results' / f'2_placed_wetlands_{CASE_NAME}.shp'
}

# --- Parameters ---

# Wetland jump detection parameters
UPS_THRESHOLD = 15000    # Minimum upstream area to be considered a river (km²)
JUMP_THRESHOLD = 2500    # Minimum wetland area jump to flag a node (km²)

# Inter-node wetland placement parameters
SHORT_DIST_KM = 200      # Distance < 100 km: place 0 wetlands
MEDIUM_DIST_KM = 550     # Distance 100-450 km: place 1 wetland
LONG_DIST_KM = 900       # Distance 450-800 km: place 2 wetlands
VLONG_DIST_KM = 1250     # Distance 800-1150 km: place 3 wetlands
# Distance > 1150 km: place 4 wetlands (extrapolating pattern)

# Node filtering parameters
MIN_NODE_DISTANCE_KM = 50  # Minimum distance between consecutive nodes (km)


# =============================================================================
# 2. DATA LOADING & PREPROCESSING
# =============================================================================
# Functions used:
# - xr.open_dataset() - Load NetCDF datasets
# - pickle.load() - Load network connectivity dictionary

# Load WAD2M wetland dataset (regridded)
ds_wad2m = xr.open_dataset(PATHS['wad2m_regrid'])
print(f"Loaded WAD2M: {ds_wad2m.dims}")

# Compute derived variables
wad2m_var = "Fw"
fw_mean = ds_wad2m[wad2m_var].mean(dim='time')       # Mean wetland fraction
fw_monthly_clim = ds_wad2m[wad2m_var].groupby('time.month').mean('time')  # Monthly climatology
fw_max = ds_wad2m[wad2m_var].max('time')              # Maximum wetland fraction

# Load upstream drainage area (supports both .map and .nc formats)
ups_path = PATHS['ups']

if ups_path.suffix == '.map':
    # Load PCRaster .map file with rasterio
    with rasterio.open(ups_path, 'r') as src:
        ups_data = src.read(1)
        ups_transform = src.transform
        ups_crs = src.crs
        ups_nodata = src.nodata  # Get NoData value from file metadata

    # Convert to xarray DataArray with proper coordinates
    height, width = ups_data.shape
    lon_coords = [ups_transform[2] + (i + 0.5) * ups_transform[0] for i in range(width)]
    lat_coords = [ups_transform[5] + (i + 0.5) * ups_transform[4] for i in range(height)]

    ups_da = xr.DataArray(
        ups_data,
        coords={'lat': lat_coords, 'lon': lon_coords},
        dims=['lat', 'lon']
    )

    # Mask NoData values from PCRaster file
    # PCRaster typically uses -3.40282e+38 or -999 as NoData
    if ups_nodata is not None:
        ups_da = ups_da.where(ups_da != ups_nodata)



elif ups_path.suffix == '.nc':
    # Load NetCDF file directly with xarray
    ds_ups = xr.open_dataset(ups_path)
    # Assume variable name is 'ups' or first variable
    if 'ups' in ds_ups:
        ups_da = ds_ups['ups']
    else:
        ups_da = ds_ups[list(ds_ups.data_vars)[0]]



else:
    raise ValueError(f"Unsupported UPS file format: {ups_path.suffix}. Must be .map or .nc")

# Interpolate upstream area to WAD2M grid
ups = ups_da.interp(
    lat=fw_mean.lat,
    lon=fw_mean.lon,
    method='nearest'
)

print(f"Loaded UPS from {ups_path.suffix} format")
print(f"  UPS range: {ups.min().values:.1f} - {ups.max().values:.1f} km²")
print(f"  Valid UPS cells: {(~ups.isnull()).sum().values}")

# Load flow-accumulated wetland area
ds_wad2m_ups = xr.open_dataset(PATHS['wad2m_accups'])
fw_upsacc = ds_wad2m_ups["WAD2M_inundated_area_upstream"]
fw_upsacc_mean = fw_upsacc.mean(dim="time")
fw_upsacc_max = fw_upsacc.max("time")

# Load network connectivity (upstream links for each cell)
with open(PATHS['connectivity'], 'rb') as f:
    link = pickle.load(f)

print(f"\nLoaded network connectivity: {len(link)} cells")

# =============================================================================
# BUILD COMPRESSED INDEX MAPPING (matches pickle creation logic)
# =============================================================================
# The pickle uses COMPRESSED indices, not simple grid indices.
# We must recreate the same mask and lddord array used when building the pickle.

print("\nBuilding compressed index mapping from LDD...")

# Load LDD raster (same as used to build pickle)
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
lddord = decompress(lddorder, mask1, ldd.shape, "int")  # 2D array: (lat_idx, lon_idx) → compressed_idx

# Build inverse mapping: compressed_idx → (lat_idx, lon_idx)
compressed_to_2d = {}
for lat_idx in range(rows):
    for lon_idx in range(cols):
        if ldd[lat_idx, lon_idx] != 0:  # Valid cell
            comp_idx = lddord[lat_idx, lon_idx]
            compressed_to_2d[comp_idx] = (lat_idx, lon_idx)

print(f"""  LDD shape: {ldd.shape}
  Compressed size: {lddsize} cells
  Link size: {len(link)} cells
  Match: {lddsize == len(link)}""")

# Get LDD coordinate arrays (native grid, before any resampling)
ldd_height, ldd_width = ldd.shape
ldd_lon_coords = np.array([ldd_transform[2] + (i + 0.5) * ldd_transform[0] for i in range(ldd_width)])
ldd_lat_coords = np.array([ldd_transform[5] + (i + 0.5) * ldd_transform[4] for i in range(ldd_height)])


# Helper functions using COMPRESSED indices
def coords_from_compressed(comp_idx):
    """Convert compressed index to (lat, lon) coordinates using LDD grid."""
    if comp_idx in compressed_to_2d:
        lat_idx, lon_idx = compressed_to_2d[comp_idx]
        return ldd_lat_coords[lat_idx], ldd_lon_coords[lon_idx]
    return None, None


def get_compressed_idx(lat_idx, lon_idx):
    """Get compressed index from 2D grid indices (LDD grid)."""
    if 0 <= lat_idx < rows and 0 <= lon_idx < cols:
        if ldd[lat_idx, lon_idx] != 0:
            return lddord[lat_idx, lon_idx]
    return -1


def is_river_cell_compressed(comp_idx, ups_2d_ldd):
    """Check if compressed index corresponds to a river cell."""
    if comp_idx in compressed_to_2d:
        lat_idx, lon_idx = compressed_to_2d[comp_idx]
        if lat_idx < ups_2d_ldd.shape[0] and lon_idx < ups_2d_ldd.shape[1]:
            return ups_2d_ldd[lat_idx, lon_idx] >= UPS_THRESHOLD
    return False


def get_ups_area_compressed(comp_idx, ups_2d_ldd):
    """Get upstream area for a compressed index."""
    if comp_idx in compressed_to_2d:
        lat_idx, lon_idx = compressed_to_2d[comp_idx]
        if lat_idx < ups_2d_ldd.shape[0] and lon_idx < ups_2d_ldd.shape[1]:
            return ups_2d_ldd[lat_idx, lon_idx]
    return 0


# Load UPS on LDD native grid (no resampling - must match pickle grid exactly)
with rasterio.open(PATHS['ups'], 'r') as src:
    ups_ldd = src.read(1)
    ups_nodata_ldd = src.nodata

# Handle nodata
if ups_nodata_ldd is not None:
    ups_ldd = np.where(ups_ldd == ups_nodata_ldd, np.nan, ups_ldd)

# Create river mask on LDD grid
river_mask_ldd = ups_ldd >= UPS_THRESHOLD

print(f"  River cells on LDD grid: {np.nansum(river_mask_ldd)}")

# DIAGNOSTIC: Verify link direction using compressed indices
print("\nDIAGNOSTIC: Verifying link dictionary direction (using compressed indices)...")

river_cells_ldd = np.where(river_mask_ldd)
samples_checked = 0
for i in range(len(river_cells_ldd[0])):
    lat_idx = river_cells_ldd[0][i]
    lon_idx = river_cells_ldd[1][i]

    comp_idx = get_compressed_idx(lat_idx, lon_idx)
    if comp_idx < 0 or comp_idx >= len(link):
        continue

    if len(link[comp_idx]) > 0:
        cell_ups = ups_ldd[lat_idx, lon_idx]
        neighbor_comp_idx = link[comp_idx][0]

        if neighbor_comp_idx in compressed_to_2d:
            n_lat_idx, n_lon_idx = compressed_to_2d[neighbor_comp_idx]
            neighbor_ups = ups_ldd[n_lat_idx, n_lon_idx]

            direction = "UPSTREAM" if neighbor_ups < cell_ups else "DOWNSTREAM"
            print(f"  Cell [{lat_idx},{lon_idx}] UPS: {cell_ups:.1f} → Neighbor [{n_lat_idx},{n_lon_idx}] UPS: {neighbor_ups:.1f} [{direction}]")
            samples_checked += 1
            if samples_checked >= 5:
                break

if samples_checked == 0:
    print("  WARNING: No valid samples found!")

# Create river mask
river_mask = ups >= UPS_THRESHOLD

print(f"\nRiver cells (ups >= {UPS_THRESHOLD}): {river_mask.sum().values}")

# Export river mask as GeoTIFF for QGIS inspection
print("\nExporting river mask for validation...")

# Convert river mask to xarray DataArray (using LDD native grid, not interpolated)
river_mask_da = xr.DataArray(
    river_mask_ldd.astype(np.float32),
    coords={'lat': ldd_lat_coords, 'lon': ldd_lon_coords},
    dims=['lat', 'lon']
)
river_mask_da = river_mask_da.where(river_mask_da == 1, 0)  # Convert bool to 0/1

# Export to GeoTIFF
river_mask_path = PATHS['output_wetlands'].parent / f'2_river_mask_{CASE_NAME}.tif'
river_mask_da.rio.write_crs("EPSG:4326", inplace=True)
river_mask_da.rio.to_raster(river_mask_path, compress='LZW')

print(f"  River mask saved: {river_mask_path}")
print(f"  UPS range: {np.nanmin(ups_ldd):.1f} - {np.nanmax(ups_ldd):.1f} km²")



# =============================================================================
# 3. WETLAND NODE DETECTION & FILTERING
# =============================================================================
# Using COMPRESSED indices to match the pickle format.
# All operations use the LDD native grid (ups_ldd, river_mask_ldd).
#
# Structure:
#   3.1 Detect wetland jump nodes
#   3.2 Build initial node-to-node connectivity
#   3.3 Iterative node filtering (remove nodes too close together)

# --- 3.1 DETECT WETLAND JUMP NODES ---

# Get fw_upsacc arrays (these are on WAD2M grid, may need alignment)
fw_upsacc_mean_2d = fw_upsacc_mean.values
fw_upsacc_max_2d = fw_upsacc_max.values

# For wetland accumulation lookup, we need to map LDD grid coords to WAD2M grid
# Create interpolated versions on LDD grid
fw_upsacc_max_ldd = fw_upsacc_max.interp(
    lat=xr.DataArray(ldd_lat_coords, dims='lat'),
    lon=xr.DataArray(ldd_lon_coords, dims='lon'),
    method='nearest'
).values

fw_upsacc_mean_ldd = fw_upsacc_mean.interp(
    lat=xr.DataArray(ldd_lat_coords, dims='lat'),
    lon=xr.DataArray(ldd_lon_coords, dims='lon'),
    method='nearest'
).values

# Find all river cells on LDD grid
river_cells_lat_idx, river_cells_lon_idx = np.where(river_mask_ldd)
print(f"Analyzing {len(river_cells_lat_idx)} river cells on LDD grid...")

# Loop through all river cells and detect jumps
flagged_nodes = []

for i in range(len(river_cells_lat_idx)):
    lat_idx = river_cells_lat_idx[i]
    lon_idx = river_cells_lon_idx[i]

    # Get COMPRESSED index
    comp_idx = get_compressed_idx(lat_idx, lon_idx)
    if comp_idx < 0 or comp_idx >= len(link):
        continue

    # Get upstream neighbors (these are also compressed indices)
    upstream_neighbors = link[comp_idx]

    if not upstream_neighbors:
        continue  # Headwater cell

    # Filter to river neighbors only
    upstream_river_neighbors = []
    for up_comp_idx in upstream_neighbors:
        if up_comp_idx not in compressed_to_2d:
            continue
        up_lat_idx, up_lon_idx = compressed_to_2d[up_comp_idx]

        if river_mask_ldd[up_lat_idx, up_lon_idx]:
            upstream_river_neighbors.append((up_comp_idx, up_lat_idx, up_lon_idx))

    # Check jump from each upstream river neighbor
    current_wetland_max = fw_upsacc_max_ldd[lat_idx, lon_idx]
    current_ups = ups_ldd[lat_idx, lon_idx]

    for up_comp_idx, up_lat_idx, up_lon_idx in upstream_river_neighbors:
        upstream_wetland_max = fw_upsacc_max_ldd[up_lat_idx, up_lon_idx]
        upstream_ups = ups_ldd[up_lat_idx, up_lon_idx]

        # Calculate jump (downstream - upstream)
        jump_max = current_wetland_max - upstream_wetland_max

        # Flag if exceeds threshold
        if jump_max > JUMP_THRESHOLD:
            lat = ldd_lat_coords[lat_idx]
            lon = ldd_lon_coords[lon_idx]

            flagged_nodes.append({
                'lat': lat,
                'lon': lon,
                'lat_idx': lat_idx,
                'lon_idx': lon_idx,
                'comp_idx': comp_idx,
                'jump_max_km2': jump_max,
                'wetland_acc_max_km2': current_wetland_max,
                'upstream_wetland_max_km2': upstream_wetland_max,
                'ups_area': current_ups,
                'upstream_ups_area': upstream_ups
            })

# Convert to DataFrame, remove duplicates
results_df = pd.DataFrame(flagged_nodes)

if len(results_df) > 0:
    results_df = results_df.sort_values('jump_max_km2', ascending=False)
    results_df = results_df.drop_duplicates(subset=['lat', 'lon'], keep='first')
    results_df = results_df.reset_index(drop=True)
    results_df['node_id'] = range(len(results_df))

print(f"Found {len(results_df)} nodes with wetland jumps > {JUMP_THRESHOLD} km²\n")

if len(results_df) > 0:
    print(f"Nodes where acc weland area jumps above {JUMP_THRESHOLD} km²\n")
    print(results_df.head(10)[['lat', 'lon', 'jump_max_km2', 'wetland_acc_max_km2']])

# Visualization: Wetland jump nodes map
fig, ax = plt.subplots(figsize=(12, 8))

# Background: accumulated wetland area on rivers (use WAD2M grid for plotting)
river_mask_for_plot = ups.interp(
    lat=fw_upsacc_max.lat,
    lon=fw_upsacc_max.lon,
    method='nearest'
) >= UPS_THRESHOLD

fw_upsacc_mean.where(river_mask_for_plot).plot(
    ax=ax,
    cmap='Blues',
    add_colorbar=True,
    cbar_kwargs={'label': 'Accumulated Wetland (km²)'}
)

# Overlay: flagged nodes colored by jump size with node IDs
if len(results_df) > 0:
    scatter = ax.scatter(
        results_df['lon'],
        results_df['lat'],
        c=results_df['jump_max_km2'],
        s=150,
        cmap='YlOrRd',
        edgecolor='black',
        linewidth=1.5,
        vmin=JUMP_THRESHOLD,
        zorder=10,
        alpha=0.8
    )
    plt.colorbar(scatter, ax=ax, label='Wetland Jump (km²)', shrink=0.8)

    # Add node ID labels
    for idx, row in results_df.iterrows():
        ax.text(row['lon'], row['lat'], str(idx), fontsize=8,
                ha='center', va='center', color='white', weight='bold')

ax.set_title(f'Wetland Jump Nodes (Jump > {JUMP_THRESHOLD} km²)\n{len(results_df)} nodes detected',
             fontweight='bold', fontsize=14)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.tight_layout()
plt.savefig(PATHS['output_nodes_map'], dpi=150, bbox_inches='tight')
plt.close()

print(f"Node detection map saved: {PATHS['output_nodes_map']}")


# --- 3.2 BUILD NODE-TO-NODE CONNECTIVITY ---

def build_node_pairs(nodes_df, node_lookup_dict):
    """
    Build node-to-node connectivity using BFS traversal.

    Parameters:
        nodes_df: DataFrame with detected nodes (must have 'comp_idx', 'node_id', 'lat', 'lon')
        node_lookup_dict: dict mapping compressed_idx -> node_id

    Returns:
        list of node pair dictionaries with connectivity info
    """
    pairs = []

    for idx, node in nodes_df.iterrows():
        node_comp_idx = int(node['comp_idx'])

        if node_comp_idx >= len(link):
            continue

        # Get all upstream river neighbors (compressed indices)
        upstream_neighbors = link[node_comp_idx]
        upstream_river_neighbors = [
            up_comp for up_comp in upstream_neighbors
            if is_river_cell_compressed(up_comp, ups_ldd)
        ]

        # For EACH upstream branch, use BFS to find next consecutive node
        for upstream_start_comp_idx in upstream_river_neighbors:
            # Check if immediate neighbor is a node
            if upstream_start_comp_idx in node_lookup_dict:
                upstream_node_id = node_lookup_dict[upstream_start_comp_idx]
                upstream_node = nodes_df[nodes_df['node_id'] == upstream_node_id].iloc[0]

                dist = hf.haversine_distance(node['lat'], node['lon'],
                                            upstream_node['lat'], upstream_node['lon'])

                pairs.append({
                    'downstream_node_id': node['node_id'],
                    'upstream_node_id': upstream_node_id,
                    'downstream_lat': node['lat'],
                    'downstream_lon': node['lon'],
                    'upstream_lat': upstream_node['lat'],
                    'upstream_lon': upstream_node['lon'],
                    'path_cells': [upstream_start_comp_idx],
                    'n_cells': 1,
                    'flow_distance_km': dist
                })
                continue

            # BFS to explore paths from this branch
            visited = set()
            queue = deque([(upstream_start_comp_idx, [upstream_start_comp_idx])])
            found_node_on_this_branch = False

            while queue and not found_node_on_this_branch:
                current_comp, path = queue.popleft()

                if current_comp in visited:
                    continue
                visited.add(current_comp)

                if current_comp in node_lookup_dict:
                    upstream_node_id = node_lookup_dict[current_comp]
                    upstream_node = nodes_df[nodes_df['node_id'] == upstream_node_id].iloc[0]

                    # Calculate flow distance along path
                    total_distance = 0
                    prev_lat, prev_lon = node['lat'], node['lon']

                    for cell_comp_idx in path:
                        curr_lat, curr_lon = coords_from_compressed(cell_comp_idx)
                        if curr_lat is not None:
                            distance = hf.haversine_distance(prev_lat, prev_lon, curr_lat, curr_lon)
                            total_distance += distance
                            prev_lat, prev_lon = curr_lat, curr_lon

                    pairs.append({
                        'downstream_node_id': node['node_id'],
                        'upstream_node_id': upstream_node_id,
                        'downstream_lat': node['lat'],
                        'downstream_lon': node['lon'],
                        'upstream_lat': upstream_node['lat'],
                        'upstream_lon': upstream_node['lon'],
                        'path_cells': path.copy(),
                        'n_cells': len(path),
                        'flow_distance_km': total_distance
                    })

                    found_node_on_this_branch = True
                    break

                if not found_node_on_this_branch:
                    if current_comp < len(link):
                        upstream_of_current = link[current_comp]
                        for neighbor_comp in upstream_of_current:
                            if is_river_cell_compressed(neighbor_comp, ups_ldd):
                                queue.append((neighbor_comp, path + [neighbor_comp]))

    return pairs


# Build initial node lookup and node pairs
node_lookup = {}
for idx, row in results_df.iterrows():
    comp_idx = int(row['comp_idx'])
    node_lookup[comp_idx] = row['node_id']

print(f"\nNode lookup created: {len(node_lookup)} nodes indexed by compressed index")

# Build initial node-to-node connectivity
node_pairs = build_node_pairs(results_df, node_lookup)

print(f"Initial node-to-node connections found: {len(node_pairs)}")

if node_pairs:
    print("\nInitial node connections (downstream → upstream):")
    for pair in node_pairs:
        print(f"  Node {pair['downstream_node_id']} → Node {pair['upstream_node_id']} "
              f"(distance: {pair['flow_distance_km']:.1f} km, cells: {pair['n_cells']})")


# --- 3.3 ITERATIVE NODE FILTERING ---
# Remove nodes that are too close together, keeping the node with larger wetland jump

print(f"\n--- Iterative Node Filtering (MIN_NODE_DISTANCE_KM = {MIN_NODE_DISTANCE_KM}) ---")

removed_nodes_records = []
iteration = 0

while True:
    # Convert current node_pairs to DataFrame for easier filtering
    if not node_pairs:
        print("No node pairs remaining - stopping filtering")
        break

    node_pairs_df_temp = pd.DataFrame(node_pairs)

    # Find pairs below minimum distance threshold
    close_pairs = node_pairs_df_temp[node_pairs_df_temp['flow_distance_km'] < MIN_NODE_DISTANCE_KM]

    if len(close_pairs) == 0:
        print(f"No more node pairs below {MIN_NODE_DISTANCE_KM} km - filtering complete")
        break

    iteration += 1

    # Find the pair with smallest distance
    min_idx = close_pairs['flow_distance_km'].idxmin()
    smallest_pair = close_pairs.loc[min_idx]

    downstream_node_id = smallest_pair['downstream_node_id']
    upstream_node_id = smallest_pair['upstream_node_id']
    pair_distance = smallest_pair['flow_distance_km']

    # Get jump values for both nodes
    downstream_node = results_df[results_df['node_id'] == downstream_node_id].iloc[0]
    upstream_node = results_df[results_df['node_id'] == upstream_node_id].iloc[0]

    downstream_jump = downstream_node['jump_max_km2']
    upstream_jump = upstream_node['jump_max_km2']

    # Remove the node with SMALLER wetland jump (keep the more important one)
    if upstream_jump <= downstream_jump:
        # Remove upstream node
        removed_node_id = upstream_node_id
        kept_node_id = downstream_node_id
        removed_jump = upstream_jump
        kept_jump = downstream_jump
        removal_reason = "upstream_smaller_jump"
    else:
        # Remove downstream node
        removed_node_id = downstream_node_id
        kept_node_id = upstream_node_id
        removed_jump = downstream_jump
        kept_jump = upstream_jump
        removal_reason = "downstream_smaller_jump"

    # Record the removal
    removed_nodes_records.append({
        'iteration': iteration,
        'removed_node_id': removed_node_id,
        'kept_node_id': kept_node_id,
        'removed_jump_km2': removed_jump,
        'kept_jump_km2': kept_jump,
        'distance_km': pair_distance,
        'removal_reason': removal_reason
    })

    print(f"  Iteration {iteration}: Removing node {removed_node_id} (jump={removed_jump:.1f} km²) "
          f"- kept node {kept_node_id} (jump={kept_jump:.1f} km²) - distance was {pair_distance:.1f} km")

    # Remove node from results_df
    results_df = results_df[results_df['node_id'] != removed_node_id].copy()

    # Rebuild node_lookup from updated results_df
    node_lookup = {}
    for idx, row in results_df.iterrows():
        comp_idx = int(row['comp_idx'])
        node_lookup[comp_idx] = row['node_id']

    # Rebuild node_pairs with updated node list
    node_pairs = build_node_pairs(results_df, node_lookup)

# Create removed nodes DataFrame
removed_nodes_df = pd.DataFrame(removed_nodes_records)

print(f"""
Node Filtering Summary:
  Initial nodes:           {len(results_df) + len(removed_nodes_df)}
  Nodes removed:           {len(removed_nodes_df)}
  Final nodes:             {len(results_df)}
  Final node connections:  {len(node_pairs)}
  Iterations:              {iteration}""")

if len(removed_nodes_df) > 0:
    print("\nRemoved nodes details:")
    print(removed_nodes_df.to_string(index=False))


# =============================================================================
# 4. WETLAND PLACEMENT
# =============================================================================
# Using COMPRESSED indices throughout to match the pickle format.
# Uses filtered results_df and node_pairs from Section 3.

# --- STEP 1: IDENTIFY TRIBUTARIES AND PLACE TRIBUTARY WETLANDS ---
tributary_wetlands = []

for idx, node in results_df.iterrows():
    # Get compressed index (already stored, convert to int)
    node_comp_idx = int(node['comp_idx'])

    if node_comp_idx >= len(link):
        continue

    # Get upstream neighbors (compressed indices)
    upstream_neighbors = link[node_comp_idx]

    # Filter to river neighbors only
    upstream_river_neighbors = [
        up_comp for up_comp in upstream_neighbors
        if is_river_cell_compressed(up_comp, ups_ldd)
    ]

    # Place wetland at EACH upstream river branch (directly at the upstream cell)
    for upstream_comp_idx in upstream_river_neighbors:
        # Place wetland directly at this upstream cell
        wetland_lat, wetland_lon = coords_from_compressed(upstream_comp_idx)

        if wetland_lat is not None:
            tributary_wetlands.append({
                'comp_idx': upstream_comp_idx,
                'lat': wetland_lat,
                'lon': wetland_lon,
                'node_id': node['node_id'],
                'type': 'tributary'
            })

print(f"\nPlaced {len(tributary_wetlands)} tributary wetlands")


# --- STEP 2: PLACE INTER-NODE WETLANDS ---

internode_wetlands = []

for pair in node_pairs:
    dist = pair['flow_distance_km']
    path = pair['path_cells']

    # Determine number of wetlands based on distance thresholds
    if dist < SHORT_DIST_KM:
        n_wetlands = 0  # No inter-node wetlands
    elif dist < MEDIUM_DIST_KM:
        n_wetlands = 1
    elif dist < LONG_DIST_KM:
        n_wetlands = 2
    elif dist < VLONG_DIST_KM:
        n_wetlands = 3
    else:
        n_wetlands = 4  # For very long segments > 1150 km

    # Only place wetlands if n_wetlands > 0 and path is long enough
    if n_wetlands > 0 and len(path) > n_wetlands:
        # Divide path into (n_wetlands + 1) segments
        # This creates equal spacing with buffers from both nodes
        segment_length = len(path) / (n_wetlands + 1)
        
        # Place wetlands at segment boundaries
        for i in range(1, n_wetlands + 1):
            # Calculate position for this wetland
            position = int(segment_length * i)
            
            # Ensure position is within valid range
            if position >= len(path):
                position = len(path) - 1
            
            cell_comp_idx = path[position]
            cell_lat, cell_lon = coords_from_compressed(cell_comp_idx)

            if cell_lat is not None:
                internode_wetlands.append({
                    'comp_idx': cell_comp_idx,
                    'lat': cell_lat,
                    'lon': cell_lon,
                    'downstream_node_id': pair['downstream_node_id'],
                    'upstream_node_id': pair['upstream_node_id'],
                    'type': 'internode',
                    'segment_distance_km': dist,
                    'n_wetlands_in_segment': n_wetlands,
                    'position_in_segment': i
                })

print(f"Placed {len(internode_wetlands)} inter-node wetlands")


# --- STEP 5: COMBINE ALL WETLANDS ---

all_wetlands = tributary_wetlands + internode_wetlands
wetlands_df = pd.DataFrame(all_wetlands)

# Remove duplicates (same location might be selected multiple times)
if len(wetlands_df) > 0:
    wetlands_df = wetlands_df.drop_duplicates(subset=['lat', 'lon'], keep='first')
    wetlands_df = wetlands_df.reset_index(drop=True)
    wetlands_df['wetland_id'] = range(len(wetlands_df))

print(f"\nTOTAL WETLANDS PLACED: {len(wetlands_df)}")
print(f"  Tributary wetlands:  {len(wetlands_df[wetlands_df['type'] == 'tributary'])}")
print(f"  Inter-node wetlands: {len(wetlands_df[wetlands_df['type'] == 'internode'])}")

if len(wetlands_df) > 0:
    print("\nSample of placed wetlands:")
    print(wetlands_df.head(10))


# --- STEP 6: VISUALIZE WETLAND PLACEMENT ---

fig, ax = plt.subplots(figsize=(14, 10))

# Background: max accumulated wetland area on rivers
fw_upsacc_max.where(river_mask_for_plot).plot(
    ax=ax,
    cmap='Blues',
    add_colorbar=True,
    cbar_kwargs={'label': 'Max Wetland Upstream (km²)'}
)

# Plot flagged nodes 
ax.scatter(
    results_df['lon'],
    results_df['lat'],
    c='red',
    s=150,
    marker='o',
    edgecolor='black',
    linewidth=2,
    alpha=0.9,
    zorder=12,
    label='Wetland Jump Nodes'
)
for idx, row in results_df.iterrows():
    ax.annotate(
        str(row['node_id']),
        xy=(row['lon'], row['lat']),
        xytext=(5, 5),  # Offset in points
        textcoords='offset points',
        fontsize=10,
        fontweight='bold',
        color='white',
        bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.4),
        zorder=13
    )
# Plot placed wetlands
if len(wetlands_df) > 0:
    # Tributary wetlands 
    trib_wetlands = wetlands_df[wetlands_df['type'] == 'tributary']
    if len(trib_wetlands) > 0:
        ax.scatter(
            trib_wetlands['lon'],
            trib_wetlands['lat'],
            c='green',
            s=80,
            marker='o',
            edgecolor='black',
            linewidth=1,
            alpha=0.8,
            zorder=11,
            label='Tributary Wetlands'
        )

    # Inter-node wetlands
    inter_wetlands = wetlands_df[wetlands_df['type'] == 'internode']
    if len(inter_wetlands) > 0:
        ax.scatter(
            inter_wetlands['lon'],
            inter_wetlands['lat'],
            c='green',
            s=60,
            marker='s',
            edgecolor='black',
            linewidth=1,
            alpha=0.7,
            zorder=10,
            label='Inter-node Wetlands'
        )

ax.set_title('Wetland Placement in River Network', fontsize=14, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.savefig(PATHS['output_placement_map'], dpi=150, bbox_inches='tight')
plt.close()

print(f"Placement map saved: {PATHS['output_placement_map']}")

# =============================================================================
# 5. EXPORT RESULTS
# =============================================================================

# Export wetlands DataFrame
PATHS['output_wetlands'].parent.mkdir(exist_ok=True)
wetlands_df.to_csv(PATHS['output_wetlands'], index=False)

# Export node pairs (for reference)
node_pairs_df = pd.DataFrame(node_pairs)
if len(node_pairs_df) > 0:
    # Remove 'path_cells' column for CSV export (it's a list)
    node_pairs_export = node_pairs_df.drop(columns=['path_cells'])
    node_pairs_export.to_csv(PATHS['output_nodes'], index=False)

# Export removed nodes (for inspection)
if len(removed_nodes_df) > 0:
    removed_nodes_df.to_csv(PATHS['output_removed_nodes'], index=False)
    print(f"Removed nodes CSV saved: {PATHS['output_removed_nodes']}")

# Export shapefiles for QGIS
print("\nExporting shapefiles for QGIS...")

# 1. Export wetland jump nodes as shapefile
if len(results_df) > 0:
    # Create GeoDataFrame from results_df
    geometry_nodes = [Point(xy) for xy in zip(results_df['lon'], results_df['lat'])]
    gdf_nodes = gpd.GeoDataFrame(results_df, geometry=geometry_nodes, crs='EPSG:4326')

    # Export to shapefile
    gdf_nodes.to_file(PATHS['shapefile_nodes'], driver='ESRI Shapefile')
    print(f"  Nodes shapefile saved: {PATHS['shapefile_nodes']}")
else:
    print("  No nodes to export")

# 2. Export placed wetlands as shapefile
if len(wetlands_df) > 0:
    # Create GeoDataFrame from wetlands_df
    geometry_wetlands = [Point(xy) for xy in zip(wetlands_df['lon'], wetlands_df['lat'])]
    gdf_wetlands = gpd.GeoDataFrame(wetlands_df, geometry=geometry_wetlands, crs='EPSG:4326')

    # Export to shapefile
    gdf_wetlands.to_file(PATHS['shapefile_wetlands'], driver='ESRI Shapefile')
    print(f"  Wetlands shapefile saved: {PATHS['shapefile_wetlands']}")
else:
    print("  No wetlands to export")

print(f"""
Completed Script 2: Wetland Placement
  Total wetlands placed:     {len(wetlands_df)}
  Tributary wetlands:        {len(wetlands_df[wetlands_df['type'] == 'tributary']) if len(wetlands_df) > 0 else 0}
  Inter-node wetlands:       {len(wetlands_df[wetlands_df['type'] == 'internode']) if len(wetlands_df) > 0 else 0}

  Node Filtering:
    Initial nodes detected:  {len(results_df) + len(removed_nodes_df)}
    Nodes removed:           {len(removed_nodes_df)}
    Final nodes (filtered):  {len(results_df)}
    Node-to-node connections:{len(node_pairs_df) if len(node_pairs_df) > 0 else 0}

  Files saved:
    CSV Files:
      - {PATHS['output_wetlands']}
      - {PATHS['output_nodes']}
      - {PATHS['output_removed_nodes']}
    Shapefiles:
      - {PATHS['shapefile_nodes']}
      - {PATHS['shapefile_wetlands']}
    Maps:
      - {PATHS['output_nodes_map']}
      - {PATHS['output_placement_map']}""")
