#!/usr/bin/env python3
"""
Compute melt-pool width (Δx) and depth (Δy) on the z=0.001 slice across time,
and save both a CSV and a plot.

Assumptions:
- Data files: postProcessing/T_slice/<time>/planeZ0005.vtk (legacy VTK PolyData)
- Melt pool exists iff any points have T >= 1571.15 and it touches y=0 when present.
- Width  = max_x - min_x
- Depth  = max_y - max(min_y, 0)

Outputs:
- meltpool_metrics_planeZ0005.csv  (time,width_dx,depth_dy,min_x,max_x,min_y,max_y)
- meltpool_metrics_planeZ0005.png  (width & depth vs time)
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ---- config ----
BASE = Path("postProcessing/T_slice")
SLICE_FILE = "planeZ0005.vtk"
T_THRESHOLD = 1571.15
CSV_OUT = "meltpool_metrics_planeZ0005.csv"
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
    t_list = []
    min_x_list, max_x_list, min_y_list = [], [], []
    agg_min_x, agg_max_x, agg_min_y = None, None, None
    all_boundary_points = []  # Collect boundary points for all time-steps

    for tdir in time_dirs:
        vtk_path = tdir / SLICE_FILE
        if not vtk_path.exists():
            continue
        try:
            pts, T = load_points_and_T(vtk_path)
        except Exception as e:
            print(f"[skip] {vtk_path}: {e}")
            continue

        tval = float(tdir.name)
        # Find melt boundary using linear interpolation between neighbors crossing threshold
        # Build a KDTree for neighbor search
        from scipy.spatial import cKDTree
        tree = cKDTree(pts[:, :2])
        boundary_points = []
        neighbor_radius = 25e-6  # 25 microns, slightly larger than grid spacing
        for i, (pt, temp) in enumerate(zip(pts, T)):
            for j in tree.query_ball_point(pt[:2], r=neighbor_radius):
                if j == i:
                    continue
                t1, t2 = temp, T[j]
                if (t1 >= T_THRESHOLD and t2 < T_THRESHOLD) or (t1 < T_THRESHOLD and t2 >= T_THRESHOLD):
                    # Linear interpolation for boundary location
                    frac = (T_THRESHOLD - t1) / (t2 - t1) if t2 != t1 else 0.5
                    edge_pt = pt + frac * (pts[j] - pt)
                    boundary_points.append(edge_pt)

        if not boundary_points:
            # no melt pool at this time
            rows.append([tval, "", "", "", "", "", ""])
            continue

        boundary_points = np.array(boundary_points)
        all_boundary_points.append(boundary_points)  # Collect for plotting
        min_x, max_x = float(boundary_points[:,0].min()), float(boundary_points[:,0].max())
        min_y, max_y = float(boundary_points[:,1].min()), float(boundary_points[:,1].max())

        # Update aggregate bounding box
        if agg_min_x is None or min_x < agg_min_x:
            agg_min_x = min_x
        if agg_max_x is None or max_x > agg_max_x:
            agg_max_x = max_x
        if agg_min_y is None or min_y < agg_min_y:
            agg_min_y = min_y

        rows.append([tval, max_x - min_x, max_y - max(min_y, 0.0), min_x, max_x, min_y, max_y])
        t_list.append(tval)
        min_x_list.append(min_x)
        max_x_list.append(max_x)
        min_y_list.append(min_y)
    # Plot all melt boundaries together, with convex hulls, colored by time
    if all_boundary_points:
        from scipy.spatial import ConvexHull
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # Define custom colormap: balanced blue -> purple -> red
        custom_cmap = mcolors.LinearSegmentedColormap.from_list(
            "balanced_blue_purple_red", ["#4F81BD", "#9E5CB2", "#E15759"])
        fig, ax = plt.subplots(figsize=(8, 8))
        # Normalize time for colormap
        norm = mpl.colors.Normalize(vmin=min(t_list), vmax=max(t_list))
        cmap = custom_cmap
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        for idx, (bp, tval) in enumerate(zip(all_boundary_points, t_list)):
            color = cmap(norm(tval))
            ax.plot(bp[:, 0], -bp[:, 1], '.', color=color, alpha=0.7, markersize=2)
            if len(bp) >= 3:
                hull = ConvexHull(bp[:, :2])
                hull_pts = bp[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                ax.plot(hull_pts[:, 0], -hull_pts[:, 1], '-', color=color, alpha=0.7, linewidth=1.0)
        ax.set_xlabel("x")
        ax.set_ylabel("-y")
        ax.set_title(f"All melt boundary points and outlines across time (z = 0.001, T ≥ {T_THRESHOLD})")
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')  # Ensure equal aspect ratio
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(sm, cax=cax, label="Time")
        fig.tight_layout()
        fig.savefig("boundaries.png", dpi=150)
        print("Wrote boundaries.png.")

    # Write aggregate bounding box info
    if agg_min_x is not None:
        print(f"Aggregate bounding box (all time): min_x={agg_min_x:.6f}, max_x={agg_max_x:.6f}, min_y={agg_min_y:.6f}, max_y=0.0")
        with open("meltpool_aggregate_bbox.txt", "w") as fagg:
            fagg.write(f"min_x,max_x,min_y,max_y\n{agg_min_x},{agg_max_x},{agg_min_y},0.0\n")


    # write CSV
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "width_dx", "depth_dy", "min_x", "max_x", "min_y", "max_y"])
        w.writerows(rows)
    print(f"Wrote {CSV_OUT} with {len(rows)} rows.")

    # Plot width and depth of the melted area vs time, with max value lines and annotations
    if t_list:
        width_list = [mx - mn for mn, mx in zip(min_x_list, max_x_list)]
        # For depth, use max_y_list (not max_x_list!)
        max_y_list = [float(bp[:,1].max()) if len(bp) else 0.0 for bp in all_boundary_points]
        depth_list = [max_y - max(mn_y, 0.0) for mn_y, max_y in zip(min_y_list, max_y_list)]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(t_list, width_list, label="Width (max_x - min_x)", color="purple")
        ax.plot(t_list, depth_list, label="Depth (max_y - max(min_y, 0))", color="orange")

        # Annotate max width
        max_width = max(width_list)
        max_width_idx = width_list.index(max_width)
        max_width_time = t_list[max_width_idx]
        ax.axhline(max_width, xmax=(max_width_time-ax.get_xlim()[0])/float(ax.get_xlim()[1]-ax.get_xlim()[0]), color="purple", linestyle="--", alpha=0.7)
        # Move label a bit to the right of y-axis
        x_offset = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.03
        ax.text(ax.get_xlim()[0] + x_offset, max_width, f"{max_width:.4g}", color="purple", va="bottom", ha="left", fontsize=10, fontweight='bold', bbox=dict(facecolor='white', edgecolor='purple', boxstyle='round,pad=0.2'))

        # Annotate max depth
        max_depth = max(depth_list)
        max_depth_idx = depth_list.index(max_depth)
        max_depth_time = t_list[max_depth_idx]
        ax.axhline(max_depth, xmax=(max_depth_time-ax.get_xlim()[0])/float(ax.get_xlim()[1]-ax.get_xlim()[0]), color="orange", linestyle="--", alpha=0.7)
        ax.text(ax.get_xlim()[0] + x_offset, max_depth, f"{max_depth:.4g}", color="orange", va="bottom", ha="left", fontsize=10, fontweight='bold', bbox=dict(facecolor='white', edgecolor='orange', boxstyle='round,pad=0.2'))

        ax.set_xlabel("Time")
        ax.set_ylabel("Length")
        ax.set_title(f"Melted area width and depth vs time (z = 0.001, T ≥ {T_THRESHOLD})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig("metrics.png", dpi=150)
        print(f"Wrote metrics.png.")

if __name__ == "__main__":
    main()

