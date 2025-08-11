#!/usr/bin/env python3
"""
Compute melt-pool width (Δx) and depth (Δy) on the z=0.001 slice across time,
and save both a CSV and a plot.

Assumptions:
- Data files: postProcessing/T_slice/<time>/planeZ001.vtk (legacy VTK PolyData)
- Melt pool exists iff any points have T >= 1580 and it touches y=0 when present.
- Width  = max_x - min_x
- Depth  = max_y - max(min_y, 0)

Outputs:
- meltpool_metrics_planeZ001.csv  (time,width_dx,depth_dy,min_x,max_x,min_y,max_y)
- meltpool_metrics_planeZ001.png  (width & depth vs time)
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

# ---- config ----
BASE = Path("postProcessing/T_slice")
SLICE_FILE = "planeZ001.vtk"
T_THRESHOLD = 1580.0
CSV_OUT = "meltpool_metrics_planeZ001.csv"
PNG_OUT = "meltpool_metrics_planeZ001.png"
# ---------------

# VTK (legacy .vtk PolyData)
from vtkmodules.vtkIOLegacy import vtkPolyDataReader
from vtkmodules.vtkFiltersCore import vtkCellDataToPointData
from vtkmodules.util.numpy_support import vtk_to_numpy

def is_time_dir(p: Path) -> bool:
    try:
        return p.is_dir() and float(p.name) is not None
    except ValueError:
        return False

def load_points_and_T(vtk_path: Path):
    """Read legacy .vtk PolyData and return (points Nx3, T_per_point N)."""
    r = vtkPolyDataReader()
    r.SetFileName(str(vtk_path))
    r.Update()
    poly = r.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise RuntimeError(f"Empty/invalid polydata: {vtk_path}")

    def get_T_from_pointdata(p):
        pd = p.GetPointData()
        arr = pd.GetArray("T") or pd.GetScalars()
        return vtk_to_numpy(arr) if arr is not None else None

    T = get_T_from_pointdata(poly)
    if T is None:
        c2p = vtkCellDataToPointData()
        c2p.SetInputData(poly)
        c2p.Update()
        poly = c2p.GetOutput()
        T = get_T_from_pointdata(poly)
        if T is None and poly.GetPointData().GetNumberOfArrays() == 1:
            T = vtk_to_numpy(poly.GetPointData().GetArray(0))
    if T is None:
        raise RuntimeError(f"No temperature array found in {vtk_path}")

    pts = vtk_to_numpy(poly.GetPoints().GetData())
    return pts, T

def main():
    time_dirs = sorted([p for p in BASE.iterdir() if is_time_dir(p)],
                       key=lambda p: float(p.name))
    rows = []
    t_list, w_list, d_list = [], [], []

    for tdir in time_dirs:
        vtk_path = tdir / SLICE_FILE
        if not vtk_path.exists():
            continue
        try:
            pts, T = load_points_and_T(vtk_path)
        except Exception as e:
            print(f"[skip] {vtk_path}: {e}")
            continue

        mask = T >= T_THRESHOLD
        tval = float(tdir.name)
        if not np.any(mask):
            # no melt pool at this time
            rows.append([tval, "", "", "", "", "", ""])
            continue

        p = pts[mask]
        min_x, max_x = float(p[:,0].min()), float(p[:,0].max())
        min_y, max_y = float(p[:,1].min()), float(p[:,1].max())

        width = max_x - min_x
        depth = max_y - max(min_y, 0.0)  # clamp lower edge to y=0

        rows.append([tval, width, depth, min_x, max_x, min_y, max_y])
        t_list.append(tval); w_list.append(width); d_list.append(depth)

    # write CSV
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "width_dx", "depth_dy", "min_x", "max_x", "min_y", "max_y"])
        w.writerows(rows)
    print(f"Wrote {CSV_OUT} with {len(rows)} rows.")

    # plot (skip times with no melt pool)
    if t_list:
        plt.figure(figsize=(8, 4.5))
        plt.plot(t_list, w_list, label="Width Δx")
        plt.plot(t_list, d_list, label="Depth Δy")
        plt.xlabel("Time")
        plt.ylabel("Length")
        plt.title("Melt pool width & depth vs time (z = 0.001, T ≥ 1580)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(PNG_OUT, dpi=150)
        print(f"Wrote {PNG_OUT}.")

if __name__ == "__main__":
    main()

