"""
Upstream Accumulation of WAD2M Wetland Inundation Data

This script computes the total upstream inundated wetland area for each river cell
by routing WAD2M monthly inundation data along a pre-built river network topology.

WORKFLOW:
---------
1. Load pre-computed connectivity pickles from Script 0
2. For each month in WAD2M time series (2000-2018):
   a. Read monthly inundation fraction [0-1] from WAD2M NetCDF
   b. Convert to inundated area [km²] by multiplying by cell area
   c. For each river cell (where upstream area >= UPS_THRESHOLD):
      - Sum inundated area from ALL upstream cells
      - Add the cell's own inundated area
      - Result = total upstream wetland area draining to that cell [km²]
   d. Write monthly 2D field to output NetCDF
3. Output preserves temporal resolution (no averaging across time)

INPUTS:
-------
- 1_data/upstream_connect_{case}.pkl: Direct upstream connections (from Script 0)
- 1_data/allupstream_connect_{case}.pkl: Complete upstream catchment (from Script 0)
- 1_data/ldd.map: Flow direction raster (LDD format, values 1-9)
- 1_data/ups.nc: Upstream area raster (number of upstream cells)
- 1_data/cellarea.nc: Cell area raster [m²]
- 1_data/WAD2M_wetlands_2000-2018.nc: Monthly wetland inundation fraction [0-1]

OUTPUTS:
--------
- 2_results/1_WAD2M_upstream_inundated_area_km2.nc: Monthly upstream inundated area [km²]
  Variable: WAD2M_inundated_area_upstream
  Dimensions: (time, lat, lon) - same temporal resolution as input (228 months)

KEY PARAMETERS:
---------------
- UPS_THRESHOLD: Minimum upstream cells to include (default 0 = all cells)
  Set to 0 to include all cells, or higher (e.g., 4000) to mask for major rivers
- CASE_NAME: Case study identifier (e.g., "niger", "amazon")

NOTES:
------
- Script does NOT compute temporal averages - output is a monthly time series
- Cells below UPS_THRESHOLD are set to 0 (e.g., small streams excluded)
- All input rasters must be grid-aligned (same extent, resolution, CRS)
- Upstream accumulation includes the cell itself (cell + all upstream)
- Run Script 0 first to build network connectivity pickles

RESAMPLING WAD2M TO LISFLOOD/CWATM GRID 
-----------------------------------
1. convert cellarea from .map to .nc

import helper_functions as hf 
hf.map_to_nc('cellarea.map', varname='cellarea')

2. remapcon with climate data operators 
ssh -l sorger hpg914.iiasa.ac.at

cdo -L\
    -remapcon,cellarea_juba.nc \
    WAD2M_wetlands_2000-2018_025deg.nc \
    WAD2M_wetlands_2000_2018_juba.nc

Author: Peter Burek (original), refactored by FSD
Created: 2025-12
"""

import numpy as np
from netCDF4 import Dataset
import rasterio
import pickle
from pathlib import Path
import platform
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt
import helper_functions as hf

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
    'connectivity_direct': BASE / '1_data' / f'upstream_connect_{CASE_NAME}.pkl',
    'connectivity_all': BASE / '1_data' / f'allupstream_connect_{CASE_NAME}.pkl',
    'ldd': BASE / '1_data' / f'ldd_{CASE_NAME}.map',
    'ups': BASE / '1_data' / f'ups_{CASE_NAME}.map',
    'cellarea': BASE / '1_data' / f'cellarea_{CASE_NAME}.map',
    'wad2m': BASE / '1_data' / f'WAD2M_wetlands_2000_2018_{CASE_NAME}.nc',
    'output': BASE / '2_results' / f'1_WAD2M_upstream_area_km2_{CASE_NAME}.nc',
    'validation_plot': BASE / '2_results' / f'1_WAD2M_validation_{CASE_NAME}.png',
    'tif_accumulated': BASE / '2_results' / f'1_WAD2M_max_accumulated_{CASE_NAME}.tif',
    'tif_extent': BASE / '2_results' / f'1_WAD2M_max_extent_{CASE_NAME}.tif'
}

# --- Parameters ---
UPS_THRESHOLD = 0  # Minimum upstream cells to include (0 = all cells)

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def compress(input, mask):
    """Compress 2D array to 1D by filtering out masked cells"""
    out = input.ravel()
    out = np.nan_to_num(out, nan=0.0)  # Replace any remaining NaN with 0
    return np.ma.compressed(np.ma.masked_array(out, mask))


def decompress(input, mask1, shape, dtype, fill_value=-9999.0):
    """Decompress 1D array back to 2D using mask"""
    out = np.full(mask1.size, fill_value, dtype=dtype)
    out[~mask1] = input[:]
    out = out.reshape(shape)
    return out


def savenetcdf(outname, nf1, varname, standard, long, unit, ntime, data_var_name):
    """
    Create output NetCDF by cloning structure from reference NetCDF.

    Parameters
    ----------
    outname : str
        Output NetCDF file path
    nf1 : Dataset
        Reference NetCDF to clone structure from
    varname : str
        Name for output variable
    standard : str
        CF standard_name attribute
    long : str
        Long name attribute
    unit : str
        Units attribute
    ntime : int
        Number of time steps
    data_var_name : str
        Name of data variable in reference NetCDF to replace

    Returns
    -------
    Dataset
        Opened output NetCDF ready for writing
    """
    nf3 = Dataset(outname, 'w', format='NETCDF4_CLASSIC')

    # Clone dimensions
    for name in list(nf1.dimensions):
        dimension = nf1.dimensions[name]
        length = len(dimension)
        if name == 'time':
            length = ntime
        nf3.createDimension(name, length if not dimension.isunlimited() else None)

    # Clone variables
    for name in list(nf1.variables):
        variable = nf1.variables[name]
        if name != data_var_name:
            x = nf3.createVariable(name, variable.datatype, variable.dimensions)
            for att in nf1[name].ncattrs():
                nf3[name].setncattr(att, getattr(nf1[name], att))
            if name not in ["time", data_var_name]:
                nf3.variables[name][:] = nf1.variables[name][:]
        else:
            # Use float32 explicitly with appropriate fill value
            x = nf3.createVariable(varname, 'f4', variable.dimensions,
                                 fill_value=-9999.0, zlib=True)
            nf3[varname].setncattr("standard_name", standard)
            nf3[varname].setncattr("long_name", long)
            nf3[varname].setncattr("units", unit)

    # Clone global attributes
    for att in nf1.ncattrs():
        nf3.setncattr(att, getattr(nf1, att))

    return nf3

# =============================================================================
# 3. DATA LOADING
# =============================================================================

print("Loading connectivity pickles...")
with open(PATHS['connectivity_all'], 'rb') as f:
    dirDown2 = pickle.load(f)

print(f"  All upstream connections loaded: {len(dirDown2)} cells")

# Load LDD and UPS to reconstruct masks
print("Loading LDD and upstream area...")
with rasterio.open(PATHS['ldd'], 'r') as src:
    ldd1 = src.read(1)
ldd = ldd1.astype(np.int64)
ldd[ldd == 255] = 5

with rasterio.open(PATHS['ups'], 'r') as src:
    ups = src.read(1)

# Create masks consistent with LDD
mask = np.invert(np.bool8(ldd.ravel()))
mask1 = np.ma.masked_array(mask, mask)
upsflat = compress(ups, mask)

print(f"  Grid shape: {ldd.shape}")
print(f"  Valid cells: {len(upsflat)}")

# Load cell area
print("Loading cell area...")
with rasterio.open(PATHS['cellarea'], 'r') as src:
    cellarea = src.read(1).astype(np.float64) / 1e6  # m² to km²

# Open WAD2M NetCDF
print("Opening WAD2M NetCDF...")
nf1 = Dataset(str(PATHS['wad2m']), 'r')
vars_list = list(nf1.variables.keys())
data_var_name = "WAD2M" if "WAD2M" in vars_list else list(nf1.variables.items())[-1][0]
ntime = nf1.variables[data_var_name].shape[0]

# Check coordinate ordering (NetCDF ascending vs rasterio descending)
nf1_lat = nf1.variables['lat'][:]
NEEDS_FLIP = nf1_lat[0] < nf1_lat[-1]  # True if NetCDF has ascending lat

print(f"""  Data variable: {data_var_name}
  Number of months: {ntime}
  NetCDF lat order: {nf1_lat[0]:.3f} → {nf1_lat[-1]:.3f} ({'ascending' if NEEDS_FLIP else 'descending'})
  Vertical flip needed: {NEEDS_FLIP}""")

# =============================================================================
# 4. UPSTREAM ACCUMULATION
# =============================================================================

print("Creating output NetCDF...")
nf3 = savenetcdf(
    str(PATHS['output']), nf1, "WAD2M_inundated_area_upstream",
    standard="inundated_area_upstream",
    long="monthly upstream-total inundated area",
    unit="km2",
    ntime=ntime,
    data_var_name=data_var_name
)
nf3.variables["time"][:] = nf1.variables["time"][:ntime]

print("Accumulating WAD2M upstream...")
print(f"  UPS_THRESHOLD: {UPS_THRESHOLD} cells")

# Loop over months
for month in range(ntime):
    if month % 12 == 0:  # Print progress every year
        print(f"  Processing month {month}/{ntime} ({month*100//ntime}%)")

    # Read monthly inundation fraction
    frac = nf1.variables[data_var_name][month, :, :].data

    # Replace fill values with 0 (no inundation), not NaN (missing data)
    frac[frac > 1e16] = 0.0    # Large positive fill values → 0 inundation
    frac[frac < -1000] = 0.0   # Negative fill values (e.g., -9999) → 0 inundation

    # Convert fraction to area [km²]
    inun_area = frac * cellarea

    # Compress to 1D
    inun_area2 = compress(inun_area, mask)

    # Upstream sum INCLUDING the cell itself
    result = inun_area2 * 0.0
    for cell in range(len(dirDown2)):
        if upsflat[cell] >= UPS_THRESHOLD:
            s_up = np.nansum(inun_area2[dirDown2[cell]]) if dirDown2[cell] else 0.0
            result[cell] = inun_area2[cell] + s_up
        else:
            result[cell] = 0.0

    # Decompress and write 2D month field
    out2d = decompress(result, mask1, ldd.shape, np.float32, fill_value=-9999.0)

    # Flip vertically if NetCDF has ascending lat (rasterio is always descending)
    if NEEDS_FLIP:
        out2d = np.flipud(out2d)

    nf3.variables["WAD2M_inundated_area_upstream"][month, :, :] = out2d

# =============================================================================
# 5. EXPORT RESULTS
# =============================================================================

nf1.close()
nf3.close()

print(f"""
Completed Script 1: WAD2M Upstream Accumulation
  Months processed: {ntime}
  UPS threshold: {UPS_THRESHOLD} cells
  Output saved:
    - {PATHS['output']}

QC Checklist:
  ☐ Visualize max accumulation: xr.open_dataset('...').max('time').plot()
  ☐ Verify river network visible in accumulated map
  ☐ Check max values reasonable for basin size
  ☐ Confirm NetCDF has correct dimensions (time={ntime}, lat, lon)""")

ds = xr.open_dataset(PATHS["output"], mask_and_scale=True)
da = ds['WAD2M_inundated_area_upstream']

# Mask fill values for statistics (NetCDF fill value is -9999.0)
da_masked = da.where(da != -9999.0)

# Statistics
max_val = float(da_masked.max())
min_val = float(da_masked.min())
mean_val = float(da_masked.mean())
non_zero_mean = float(da_masked.where(da_masked > 0).mean())

print(f"""
{'='*70}
OUTPUT STATISTICS
{'='*70}
Accumulated Upstream Wetland Area:
  Maximum:           {max_val:>10.2f} km²
  Minimum:           {min_val:>10.2f} km²
  Mean (all cells):  {mean_val:>10.2f} km²
  Mean (non-zero):   {non_zero_mean:>10.2f} km²

Latitude order: {ds.lat.values[0]:.3f} → {ds.lat.values[-1]:.3f}
Coordinate order: {'ascending (correct)' if ds.lat.values[0] < ds.lat.values[-1] else 'descending'}
{'='*70}
""")

# Create validation plot
max_time_slice = da_masked.max(dim='time').values

# Flip vertically to match geographic orientation (north at top)
max_time_slice_flipped = np.flipud(max_time_slice)

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(np.log10(max_time_slice_flipped + 1), cmap='Blues', interpolation='nearest')
ax.set_title(f'Maximum Upstream Wetland Accumulation [log10(km²)]\n{CASE_NAME} | Max: {max_val:.2f} km²')
ax.set_xlabel('Longitude index')
ax.set_ylabel('Latitude index')
cbar = plt.colorbar(im, ax=ax, label='log10(Accumulated Area + 1)')

plt.tight_layout()
plt.savefig(PATHS['validation_plot'], dpi=150, bbox_inches='tight')
plt.close()

print(f"  Validation plot saved: {PATHS['validation_plot']}")

# Export GeoTIFFs for QGIS
print("Exporting GeoTIFFs for QGIS...")

# 1. Maximum accumulated upstream wetland area
da_max_accumulated = da_masked.max(dim='time')
da_max_accumulated.rio.write_crs("EPSG:4326", inplace=True)
da_max_accumulated.rio.to_raster(PATHS['tif_accumulated'], compress='LZW')

# 2. Maximum wetland extent (direct inundated area, not accumulated)
# Reopen WAD2M dataset
ds_wad2m = xr.open_dataset(PATHS['wad2m'])
# Get data variable (same logic as earlier in script)
wad2m_var = "WAD2M" if "WAD2M" in ds_wad2m.data_vars else list(ds_wad2m.data_vars.keys())[0]

# Check if cellarea NetCDF exists, if not convert from .map
cellarea_nc = PATHS['cellarea'].with_suffix('.nc')
if not cellarea_nc.exists():
    hf.map_to_nc(str(PATHS['cellarea']), str(cellarea_nc), varname='cellarea')

# Open cellarea as xarray
ds_cellarea = xr.open_dataset(cellarea_nc)
cellarea_da = ds_cellarea['cellarea'] / 1e6  # m² to km²

# Simple: frac × cellarea, then max over time (xarray handles coordinates automatically)
wad2m_frac = ds_wad2m["Fw"].where(ds_wad2m["Fw"] > -1000, 0.0)
wad2m_extent_ts = wad2m_frac * cellarea_da  # Broadcasting handles alignment
wad2m_extent_max = wad2m_extent_ts.max(dim='time')

# Export to GeoTIFF
wad2m_extent_max.rio.write_crs("EPSG:4326", inplace=True)
wad2m_extent_max.rio.to_raster(PATHS['tif_extent'], compress='LZW')

ds_wad2m.close()
ds_cellarea.close()

print(f"""  GeoTIFFs exported:
    - Maximum accumulated: {PATHS['tif_accumulated']}
    - Maximum extent: {PATHS['tif_extent']}""")
