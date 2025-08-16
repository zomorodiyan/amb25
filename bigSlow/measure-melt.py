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
import argparse


# ---- config ----
BASE = Path("postProcessing/T_slice")
T_THRESHOLD = 1571.15
DEFAULT_ZLIST = [5, 10, 15, 20]
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
    parser = argparse.ArgumentParser(description="Compute melt-pool width/depth for multiple Z-slices.")
    parser.add_argument('--zlist', nargs='+', type=int, default=DEFAULT_ZLIST,
                        help="List of Z-slice indices (e.g. 5 10 15 20 25 30 35 for planeZ0005, planeZ0010, ...)")
    parser.add_argument('--boundaries-z', type=int, default=10,
                        help="Z-slice index to use for boundaries.png coloring by time (default: 10 for planeZ0010)")
    args = parser.parse_args()
    zlist = args.zlist
    boundaries_z = args.boundaries_z

    time_dirs = sorted([p for p in BASE.iterdir() if is_time_dir(p)],
                       key=lambda p: float(p.name))

    rows = []
    all_boundary_points = []  # List of (z, tval, boundary_points)
    agg_min_x, agg_max_x, agg_min_y = None, None, None

    for z in zlist:
        zstr = f"{z:04d}"
        SLICE_FILE = f"planeZ{zstr}.vtk"
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
            from scipy.spatial import cKDTree
            tree = cKDTree(pts[:, :2])
            boundary_points = []
            neighbor_radius = 25e-6
            for i, (pt, temp) in enumerate(zip(pts, T)):
                for j in tree.query_ball_point(pt[:2], r=neighbor_radius):
                    if j == i:
                        continue
                    t1, t2 = temp, T[j]
                    if (t1 >= T_THRESHOLD and t2 < T_THRESHOLD) or (t1 < T_THRESHOLD and t2 >= T_THRESHOLD):
                        frac = (T_THRESHOLD - t1) / (t2 - t1) if t2 != t1 else 0.5
                        edge_pt = pt + frac * (pts[j] - pt)
                        boundary_points.append(edge_pt)

            if not boundary_points:
                rows.append([z, tval, "", "", "", "", "", ""])
                continue

            boundary_points = np.array(boundary_points)
            all_boundary_points.append((z, tval, boundary_points))
            min_x, max_x = float(boundary_points[:,0].min()), float(boundary_points[:,0].max())
            min_y, max_y = float(boundary_points[:,1].min()), float(boundary_points[:,1].max())

            if agg_min_x is None or min_x < agg_min_x:
                agg_min_x = min_x
            if agg_max_x is None or max_x > agg_max_x:
                agg_max_x = max_x
            if agg_min_y is None or min_y < agg_min_y:
                agg_min_y = min_y

            rows.append([z, tval, max_x - min_x, max_y - max(min_y, 0.0), min_x, max_x, min_y, max_y])

    # Plot boundaries for a single z-slice (colored by time, with colorbar)
    if all_boundary_points:
        from scipy.spatial import ConvexHull
        import matplotlib as mpl
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        # Filter to requested z slice and sort by time
        z_points = sorted([(t, bp) for zz, t, bp in all_boundary_points if zz == boundaries_z], key=lambda x: x[0])
        if z_points:
            t_vals = [t for t, _ in z_points]
            # Custom colormap: black -> purple -> yellow, normalized by time
            custom_cmap = mcolors.LinearSegmentedColormap.from_list(
                "black_purple_yellow", ["#000000", "#9E5CB2", "#FFFF00"])
            norm = mpl.colors.Normalize(vmin=min(t_vals), vmax=max(t_vals))
            cmap = custom_cmap
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            fig, ax = plt.subplots(figsize=(8, 8))
            for tval, bp in z_points:
                color = cmap(norm(tval))
                ax.plot(bp[:, 0], -bp[:, 1], '.', color=color, alpha=0.7, markersize=2)
                if len(bp) >= 3:
                    hull = ConvexHull(bp[:, :2])
                    hull_pts = bp[hull.vertices]
                    hull_pts = np.vstack([hull_pts, hull_pts[0]])
                    ax.plot(hull_pts[:, 0], -hull_pts[:, 1], '-', color=color, alpha=0.7, linewidth=1.0)
            ax.set_xlabel("x")
            ax.set_ylabel("-y")
            ax.set_title(f"Melt boundary points and outlines across time (z = {boundaries_z/10000:.4f}, T ≥ {T_THRESHOLD})")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(sm, cax=cax, label="Time")
            fig.tight_layout()
            fig.savefig("boundaries.png", dpi=150)
            print(f"Wrote boundaries.png (z={boundaries_z}/10000).")
        else:
            print(f"No boundary data found for z={boundaries_z}. Skipping boundaries.png.")

    # Write aggregate bounding box info
    if agg_min_x is not None:
        print(f"Aggregate bounding box (all z, all time): min_x={agg_min_x:.6f}, max_x={agg_max_x:.6f}, min_y={agg_min_y:.6f}, max_y=0.0")
        with open("meltpool_aggregate_bbox.txt", "w") as fagg:
            fagg.write(f"min_x,max_x,min_y,max_y\n{agg_min_x},{agg_max_x},{agg_min_y},0.0\n")

    # Write CSV with z column
    CSV_OUT = "meltpool_metrics_planes.csv"
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["z", "time", "width_dx", "depth_dy", "min_x", "max_x", "min_y", "max_y"])
        w.writerows(rows)
    print(f"Wrote {CSV_OUT} with {len(rows)} rows.")

    # Plot width and depth vs time for all z-slices
    if all_boundary_points:
        zvals = sorted(set(z for z, _, _ in all_boundary_points))
        # Assign a distinct color per z-slice
        z_cmap = plt.get_cmap('tab10') if len(zvals) <= 10 else plt.get_cmap('tab20')
        z_to_color = {z: z_cmap(i % z_cmap.N) for i, z in enumerate(zvals)}
        fig, ax = plt.subplots(figsize=(10, 6))
        # Keep lists for z=10 to annotate max later
        z10_t, z10_widths, z10_depths = None, None, None
        for z in zvals:
            z_points = sorted([(t, bp) for zz, t, bp in all_boundary_points if zz == z], key=lambda x: x[0])
            t_list = [t for t, _ in z_points]
            min_x_list = [float(bp[:,0].min()) for _, bp in z_points]
            max_x_list = [float(bp[:,0].max()) for _, bp in z_points]
            min_y_list = [float(bp[:,1].min()) for _, bp in z_points]
            max_y_list = [float(bp[:,1].max()) for _, bp in z_points]
            width_list = [mx - mn for mn, mx in zip(min_x_list, max_x_list)]
            depth_list = [max_y - max(mn_y, 0.0) for mn_y, max_y in zip(min_y_list, max_y_list)]
            color = z_to_color[z]
            ax.plot(t_list, width_list, label=f"Width z={z/10000:.4f}", color=color, linestyle='-')
            ax.plot(t_list, depth_list, label=f"Depth z={z/10000:.4f}", color=color, linestyle='--')
            if z == 10:
                z10_t, z10_widths, z10_depths = t_list, width_list, depth_list
        # Add dashed max lines and labels for z=10 if present
        if z10_t and z10_widths:
            # Ensure axis limits exist
            ax.relim(); ax.autoscale()
            x_min, x_max = ax.get_xlim()
            # Width max
            max_w = max(z10_widths)
            max_w_idx = z10_widths.index(max_w)
            max_w_time = z10_t[max_w_idx]
            frac_w = (max_w_time - x_min) / float(x_max - x_min) if x_max > x_min else 1.0
            ax.axhline(max_w, xmax=frac_w, color=z_to_color.get(10, 'k'), linestyle='--', alpha=0.8)
            x_offset = (x_max - x_min) * 0.03
            ax.text(x_min + x_offset, max_w, f"{max_w:.4g}", color=z_to_color.get(10, 'k'), va="bottom", ha="left",
                    fontsize=10, fontweight='bold', bbox=dict(facecolor='white', edgecolor=z_to_color.get(10, 'k'), boxstyle='round,pad=0.2'))
            # Depth max
            max_d = max(z10_depths)
            max_d_idx = z10_depths.index(max_d)
            max_d_time = z10_t[max_d_idx]
            frac_d = (max_d_time - x_min) / float(x_max - x_min) if x_max > x_min else 1.0
            ax.axhline(max_d, xmax=frac_d, color=z_to_color.get(10, 'k'), linestyle='--', alpha=0.8)
            ax.text(x_min + x_offset, max_d, f"{max_d:.4g}", color=z_to_color.get(10, 'k'), va="bottom", ha="left",
                    fontsize=10, fontweight='bold', bbox=dict(facecolor='white', edgecolor=z_to_color.get(10, 'k'), boxstyle='round,pad=0.2'))
        ax.set_xlabel("Time")
        ax.set_ylabel("Length")
        ax.set_title(f"Melted area width and depth vs time (multiple z, T ≥ {T_THRESHOLD})")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, loc='lower right')
        fig.tight_layout()
        fig.savefig("metrics.png", dpi=150)
        print(f"Wrote metrics.png.")

if __name__ == "__main__":
    main()

