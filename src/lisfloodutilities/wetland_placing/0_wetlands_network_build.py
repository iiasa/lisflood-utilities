"""
Build River Network Connectivity from LDD Raster

This script computes river network topology from a Local Drain Direction (LDD) raster
and saves connectivity pickles for later use in wetland placement workflows.

WORKFLOW:
---------
1. Read LDD (flow direction) raster
2. Create compressed flat indices and masks for valid cells
3. Identify direct upstream neighbors for each cell (8-directional connectivity)
4. Perform post-order tree traversal to compute ALL upstream cells recursively
5. Save two connectivity pickles:
   - upstream_connect_{case}.pkl: Direct parent-child relationships
   - allupstream_connect_{case}.pkl: Complete upstream catchment for each cell
6. Generate network validation plot

INPUTS:
-------
- 1_data/ldd.map: Flow direction raster (PCRaster LDD format, values 1-9 where 5=sink)
- 1_data/ups.nc: Upstream area raster (for validation plot only)

OUTPUTS:
--------
- 1_data/upstream_connect_{case}.pkl: Direct upstream links (list of lists)
- 1_data/allupstream_connect_{case}.pkl: Complete upstream catchment (list of lists)
- 2_results/0_network_validation.png: QC plot showing network structure

KEY PARAMETERS:
---------------
- CASE_NAME: Case study identifier (e.g., "niger")

NOTES:
------
- Run once per case study
- Checks for circular dependencies and reports "wrong" cells
- Post-order traversal supports 8 upstream neighbors per cell
- Recursion limit set to 500,000 to handle large basins

USAGE:
------
# One-time: build connectivity for a case study
python 0_wetlands_network_build.py

Author: Peter Burek (original), refactored by FSD
Created: 2025-12
"""

import numpy as np
import rasterio
import matplotlib.pyplot as plt
import pickle
import sys
from pathlib import Path
import platform

# =============================================================================
# 1. CONFIGURATION & SETUP
# =============================================================================

# Case study identifier
CASE_NAME = "juba"  

# Set recursion limit for large basins
sys.setrecursionlimit(500000)

# --- Paths ---
system = platform.system()
if system == 'Windows':
    BASE = Path(rf'C:\Users\sorger\OneDrive - IIASA\Seed-FD\wetlands\4_case_studies\{CASE_NAME}')
else:
    BASE = Path(rf'/Users/sorger/Library/CloudStorage/OneDrive-IIASA/Seed-FD/wetlands/4_case_studies/{CASE_NAME}')

PATHS = {
    'ldd': BASE / '1_data' / f'ldd_{CASE_NAME}.map',
    'ups': BASE / '1_data' / f'ups_{CASE_NAME}.map',
    'output_direct': BASE / '1_data' / f'upstream_connect_{CASE_NAME}.pkl',
    'output_all': BASE / '1_data' / f'allupstream_connect_{CASE_NAME}.pkl',
    'validation_plot': BASE / '2_results' / '0_network_validation.png'
}

# =============================================================================
# 2. HELPER FUNCTIONS
# =============================================================================

def compress(input, mask):
    """Compress 2D array to 1D by filtering out masked cells"""
    out = input.ravel()
    return np.ma.compressed(np.ma.masked_array(out, mask))


def decompress(input, mask1, shape, dtype):
    """Decompress 1D array back to 2D using mask"""
    out = mask1.copy().astype(dtype)
    out[~mask1] = input[:]
    out = out.reshape(shape)
    return out


def postorder(dirUp, node, wrong, dirDown):
    """
    Perform post-order tree traversal to find all upstream cells.

    Recursively traverses upstream following parent-child relationships until
    reaching headwater cells (no upstream). Supports up to 8 upstream neighbors.

    Parameters
    ----------
    dirUp : list of lists
        Direct upstream connections (link2)
    node : int
        Current node (flat index)
    wrong : list
        Accumulates cells with circular dependencies
    dirDown : list
        Accumulates all upstream cells for current node
    """
    if dirUp[node] != []:
        postorder.counter += 1
        if postorder.counter > 500000:
            wrong.append(node)
            return

        # Recursively process each upstream neighbor
        for i in range(len(dirUp[node])):
            if i < 8:  # Support up to 8 upstream neighbors
                postorder(dirUp, dirUp[node][i], wrong, dirDown)
                dirDown.append(dirUp[node][i])

# =============================================================================
# 3. LOAD LDD RASTER
# =============================================================================

print("Loading LDD raster...")
with rasterio.open(PATHS['ldd'], 'r') as src:
    ldd1 = src.read(1)
    transform = src.transform
    crs = src.crs

rows, cols = ldd1.shape
ldd = ldd1.astype(np.int64)

# LDD format: 1-9 encoding, where 5 = sink (pit)
# Make sinks for invalid values
ldd[ldd == 255] = 5

# Create boundary frame with sinks (avoid edge effects)
for x in range(cols):
    ldd[0, x] = 5
    ldd[rows-1, x] = 5
for y in range(rows):
    ldd[y, 0] = 5
    ldd[y, cols-1] = 5

print(f"  LDD shape: {ldd.shape}")
print(f"  Valid cells: {np.sum(ldd > 0)}")

# =============================================================================
# 4. CREATE COMPRESSED INDICES
# =============================================================================

# Compress: 2D → 1D for valid cells only
mask = np.invert(np.bool8(ldd.ravel()))
mask1 = np.ma.masked_array(mask, mask)
lddflat = compress(ldd, mask)
lddsize = len(lddflat)
lddorder = np.arange(lddsize)
lddord = decompress(lddorder, mask1, ldd.shape, "int")

print(f"  Compressed size: {lddsize} cells")

# =============================================================================
# 5. BUILD DIRECT UPSTREAM CONNECTIONS
# =============================================================================

print("Building direct upstream connections...")

# LDD neighbor offsets (8-directional + sink)
#      0    1   2   3   4  5   6   7    8    9
xd = [999, -1,  0,  1, -1, 0,  1, -1,   0,   1]
yd = [999,  1,  1,  1,  0, 0,  0, -1,  -1,  -1]

link2 = []  # Direct upstream neighbors for each cell

for y in range(rows):
    for x in range(cols):
        d = ldd[y, x]
        upst2 = []

        if d != 0:
            # Search 8 neighboring cells for upstream connections
            for neigh in [1, 2, 3, 4, 6, 7, 8, 9]:  # Skip 5 (self)
                xx = x + xd[neigh]
                yy = y + yd[neigh]

                if (0 <= xx < cols) and (0 <= yy < rows):
                    dd = ldd[yy, xx]
                    if dd != 0:
                        xxx = xx + xd[dd]
                        yyy = yy + yd[dd]

                        # If neighbor flows into current cell, it's upstream
                        if (xxx == x) and (yyy == y):
                            upst2.append(lddord[yy, xx])

            link2.append(upst2)

print(f"  Done scanning LDD")

# =============================================================================
# 6. POST-ORDER TRAVERSAL FOR ALL UPSTREAM CELLS
# =============================================================================

print("Computing all upstream cells (post-order traversal)...")

dirDown2 = []  # All upstream cells for each cell
wrong = []  # Cells with circular dependencies
maxcount = 0

for i in range(lddsize):
    if i % 1000 == 0:
        print(f"  Processing cell {i}/{lddsize}")

    dirDown = []
    if lddflat[i] != 0:
        postorder.counter = 0
        postorder(link2, i, wrong, dirDown)
        if postorder.counter > maxcount:
            maxcount = postorder.counter
        dirDown2.append(dirDown)

print(f"""
Network traversal complete:
  Max recursion depth: {maxcount}
  Cells with circular dependencies: {len(set(wrong))}""")

if len(wrong) > 0:
    print(f"  WARNING: {len(set(wrong))} cells have routing errors (circular dependencies)")

# =============================================================================
# 7. SAVE CONNECTIVITY PICKLES
# =============================================================================

print("Saving connectivity pickles...")

with open(PATHS['output_direct'], 'wb') as f:
    pickle.dump(link2, f)

with open(PATHS['output_all'], 'wb') as f:
    pickle.dump(dirDown2, f)

print(f"""
Pickles saved:
  - {PATHS['output_direct']}
  - {PATHS['output_all']}""")

# =============================================================================
# 8. GENERATE VALIDATION PLOT
# =============================================================================

print("Generating validation plot...")

# Load upstream area for visualization
with rasterio.open(PATHS['ups'], 'r') as src:
    ups = src.read(1)

# Calculate number of upstream cells for each cell
n_upstream = np.array([len(dirDown2[i]) if i < len(dirDown2) else 0
                       for i in range(lddsize)])
n_upstream_2d = decompress(n_upstream, mask1, ldd.shape, "float")

# Create 2-panel plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel 1: Upstream area (log scale)
im1 = axes[0].imshow(np.log10(ups + 1), cmap='Blues')
axes[0].set_title(f'Upstream Area [log10(km²)]\n{CASE_NAME}')
axes[0].set_xlabel('Longitude index')
axes[0].set_ylabel('Latitude index')
plt.colorbar(im1, ax=axes[0], label='log10(Upstream Area + 1)')

# Panel 2: Number of upstream cells
im2 = axes[1].imshow(np.log10(n_upstream_2d + 1), cmap='viridis')
axes[1].set_title(f'Network Connectivity\nlog10(N upstream cells + 1)')
axes[1].set_xlabel('Longitude index')
axes[1].set_ylabel('Latitude index')
plt.colorbar(im2, ax=axes[1], label='log10(N cells + 1)')

plt.tight_layout()
plt.savefig(PATHS['validation_plot'], dpi=150, bbox_inches='tight')
plt.close()

print(f"""
Validation plot saved:
  - {PATHS['validation_plot']}

Network building complete!
  Total cells: {lddsize}
  Mean upstream cells: {n_upstream.mean():.1f}
  Max upstream cells: {n_upstream.max()}

QC Checklist:
  ☐ Inspect validation plot - river network should be visible
  ☐ Check max recursion < 500,000
  ☐ Verify no circular dependencies (wrong cells = 0)
  ☐ Confirm pickles created in 1_data/""")
