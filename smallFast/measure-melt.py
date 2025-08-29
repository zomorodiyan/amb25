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
    parser.add_argument('--boundaries-z', type=int, default=5,
                        help="Z-slice index to use for boundaries.png coloring by time (default: 10 for planeZ0010)")
    args = parser.parse_args()
    zlist = args.zlist
    boundaries_z = args.boundaries_z

    time_dirs = sorted([p for p in BASE.iterdir() if is_time_dir(p)],
                       key=lambda p: float(p.name))

    # Limit to first 800 time directories for faster processing while getting more peaks
    time_dirs = time_dirs[:800]

    print(f"Found {len(time_dirs)} time directories to process (limited to first 800)")
    print(f"Processing Z-slices: {zlist}")
    print("Starting boundary detection...")

    rows = []
    all_boundary_points = []  # List of (z, tval, boundary_points)
    agg_min_x, agg_max_x, agg_min_y = None, None, None
    # Track peak values across all z-slices - moved to top
    all_width_peaks = {}  # z -> [(peak_width, time_at_peak), ...]
    all_depth_peaks = {}  # z -> [(peak_depth, time_at_peak), ...]

    total_files = len(time_dirs) * len(zlist)
    processed_files = 0
    import time
    start_time = time.time()

    for z in zlist:
        zstr = f"{z:04d}"
        SLICE_FILE = f"planeZ{zstr}.vtk"
        print(f"\nProcessing Z-slice {z} (z={z/10000:.4f})...")
        slice_processed = 0
        
        for tdir in time_dirs:
            vtk_path = tdir / SLICE_FILE
            if not vtk_path.exists():
                processed_files += 1
                continue
            
            # Progress indicator
            slice_processed += 1
            processed_files += 1
            
            # Show progress every 50 files or at the beginning
            if processed_files % 50 == 0 or processed_files == 1:
                elapsed_time = time.time() - start_time
                progress_percent = 100 * processed_files / total_files
                if processed_files > 1:
                    estimated_total_time = elapsed_time * total_files / processed_files
                    remaining_time = estimated_total_time - elapsed_time
                    print(f"  Progress: {processed_files}/{total_files} ({progress_percent:.1f}%) - "
                          f"Elapsed: {elapsed_time:.1f}s, Est. remaining: {remaining_time:.1f}s")
                else:
                    print(f"  Progress: {processed_files}/{total_files} ({progress_percent:.1f}%) - Starting...")
            
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

    # Final progress report
    total_elapsed = time.time() - start_time
    print(f"\nCompleted boundary detection for {len(zlist)} Z-slices and {len(time_dirs)} time steps")
    print(f"Found boundary data for {len(all_boundary_points)} time/z combinations")
    print(f"Total processing time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")

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
        print("\nGenerating plots...")
        print("=" * 50)
        zvals = sorted(set(z for z, _, _ in all_boundary_points))
        print(f"Creating metrics plot for {len(zvals)} Z-slices...")
        print(f"Starting peak detection...")
        # Assign a distinct color per z-slice
        z_cmap = plt.get_cmap('tab10') if len(zvals) <= 10 else plt.get_cmap('tab20')
        z_to_color = {z: z_cmap(i % z_cmap.N) for i, z in enumerate(zvals)}
        fig, ax = plt.subplots(figsize=(10, 6))
        # Keep lists for z=10 to annotate max later
        z10_t, z10_widths, z10_depths = None, None, None
        
        def find_peaks(values, times, min_prominence=None, min_time_separation=0.001):
            """Find local maxima (peaks) in the data with optional minimum prominence and time separation."""
            if len(values) < 3:
                return []
            
            # Convert to numpy arrays for easier processing
            vals = np.array(values)
            ts = np.array(times)
            
            # If no prominence specified, use 5% of the data range as minimum prominence
            if min_prominence is None:
                data_range = vals.max() - vals.min()
                min_prominence = 0.05 * data_range
            
            peaks = []
            for i in range(1, len(vals) - 1):
                # Check if this point is a local maximum
                if vals[i] > vals[i-1] and vals[i] > vals[i+1]:
                    # Check prominence (how much higher than surrounding minima)
                    left_min = vals[:i+1].min()
                    right_min = vals[i:].min()
                    prominence = vals[i] - max(left_min, right_min)
                    
                    if prominence >= min_prominence:
                        peaks.append((vals[i], ts[i]))
            
            # Filter peaks that are too close in time, keeping the larger one
            if not peaks:
                return []
                
            # Sort peaks by time
            peaks.sort(key=lambda x: x[1])
            
            filtered_peaks = []
            for value, time in peaks:
                # Check if this peak is too close to any already accepted peak
                too_close = False
                for i, (prev_value, prev_time) in enumerate(filtered_peaks):
                    if abs(time - prev_time) < min_time_separation:
                        too_close = True
                        # If current peak is larger, replace the previous one
                        if value > prev_value:
                            filtered_peaks[i] = (value, time)
                        break
                
                # If not too close to any existing peak, add it
                if not too_close:
                    filtered_peaks.append((value, time))
            
            # Sort by time again and return
            filtered_peaks.sort(key=lambda x: x[1])
            return filtered_peaks
        
        for z in zvals:
            print(f"  Processing peaks for z={z/10000:.4f}...")
            z_points = sorted([(t, bp) for zz, t, bp in all_boundary_points if zz == z], key=lambda x: x[0])
            t_list = [t for t, _ in z_points]
            min_x_list = [float(bp[:,0].min()) for _, bp in z_points]
            max_x_list = [float(bp[:,0].max()) for _, bp in z_points]
            min_y_list = [float(bp[:,1].min()) for _, bp in z_points]
            max_y_list = [float(bp[:,1].max()) for _, bp in z_points]
            width_list = [mx - mn for mn, mx in zip(min_x_list, max_x_list)]
            depth_list = [max_y - max(mn_y, 0.0) for mn_y, max_y in zip(min_y_list, max_y_list)]
            
            # Find peaks in width and depth
            if width_list and len(width_list) >= 3:
                width_peaks = find_peaks(width_list, t_list)
                all_width_peaks[z] = width_peaks
                print(f"    Found {len(width_peaks)} width peaks")
            
            if depth_list and len(depth_list) >= 3:
                depth_peaks = find_peaks(depth_list, t_list)
                all_depth_peaks[z] = depth_peaks
                print(f"    Found {len(depth_peaks)} depth peaks")
            
            color = z_to_color[z]
            ax.plot(t_list, width_list, label=f"Width z={z/10000:.4f}", color=color, linestyle='-')
            ax.plot(t_list, depth_list, label=f"Depth z={z/10000:.4f}", color=color, linestyle='--')
            
            # Mark width peaks with 'x' markers
            if z in all_width_peaks and all_width_peaks[z]:
                for peak_val, peak_time in all_width_peaks[z]:
                    ax.plot(peak_time, peak_val, 'x', color=color, markersize=8, markeredgewidth=2)
            
            # Mark depth peaks with 'x' markers
            if z in all_depth_peaks and all_depth_peaks[z]:
                for peak_val, peak_time in all_depth_peaks[z]:
                    ax.plot(peak_time, peak_val, 'x', color=color, markersize=8, markeredgewidth=2)
            
            if z == 10:
                z10_t, z10_widths, z10_depths = t_list, width_list, depth_list
        
        # Calculate overlap depth for each z-slice
        def calculate_overlap_depth(z_slice, depth_peak_times, all_boundary_data):
            """Calculate overlap depth by finding geometric intersections between consecutive melted area boundaries"""
            from scipy.spatial.distance import cdist
            
            overlap_depths = []
            
            # Get boundary points for this z-slice at depth peak times
            z_boundary_data = [(t, bp) for zz, t, bp in all_boundary_data if zz == z_slice]
            z_boundary_dict = {t: bp for t, bp in z_boundary_data}
            
            # Find boundary data closest to each peak time
            peak_boundaries = []
            for peak_val, peak_time in depth_peak_times:
                # Find the closest time in boundary data
                closest_time = min(z_boundary_dict.keys(), key=lambda t: abs(t - peak_time))
                if abs(closest_time - peak_time) < 0.0001:  # Within reasonable tolerance
                    boundary_points = z_boundary_dict[closest_time]
                    # Filter out points very close to surface (y=0), but be less aggressive
                    filtered_bp = boundary_points[boundary_points[:,1] < -5e-7]  # y < -0.5 micrometers
                    if len(filtered_bp) > 3:  # Need at least a few points for meaningful analysis
                        peak_boundaries.append((peak_time, peak_val, filtered_bp))
                    else:
                        # If filtering removes too many points, use original boundary
                        print(f"    Warning: filtering removed too many points, using full boundary")
                        peak_boundaries.append((peak_time, peak_val, boundary_points))
            
            # Find intersections between consecutive boundaries
            for i in range(1, len(peak_boundaries)):
                prev_time, prev_peak, prev_bp = peak_boundaries[i-1]
                curr_time, curr_peak, curr_bp = peak_boundaries[i]
                
                # Find intersection points between the two boundary curves (not point clouds)
                intersection_depths = []
                
                print(f"    Analyzing intersection between boundary curves at t={prev_time:.6f} and t={curr_time:.6f}")
                print(f"    Previous boundary: {len(prev_bp)} points, Current boundary: {len(curr_bp)} points")
                
                # Create boundary curves using convex hulls and find their intersection
                if len(prev_bp) >= 3 and len(curr_bp) >= 3:
                    try:
                        from scipy.spatial import ConvexHull
                        
                        # Create convex hulls for both boundaries
                        prev_hull = ConvexHull(prev_bp[:, :2])
                        curr_hull = ConvexHull(curr_bp[:, :2])
                        
                        prev_hull_points = prev_bp[prev_hull.vertices]
                        curr_hull_points = curr_bp[curr_hull.vertices]
                        
                        # Create ordered boundary lines from hull vertices
                        prev_boundary_lines = []
                        curr_boundary_lines = []
                        
                        # Create line segments for previous boundary
                        for i in range(len(prev_hull_points)):
                            p1 = prev_hull_points[i]
                            p2 = prev_hull_points[(i + 1) % len(prev_hull_points)]
                            prev_boundary_lines.append((p1[:2], p2[:2]))
                        
                        # Create line segments for current boundary  
                        for i in range(len(curr_hull_points)):
                            p1 = curr_hull_points[i]
                            p2 = curr_hull_points[(i + 1) % len(curr_hull_points)]
                            curr_boundary_lines.append((p1[:2], p2[:2]))
                        
                        # Find intersections between line segments
                        def line_intersection(line1, line2):
                            """Find intersection point between two line segments"""
                            (x1, y1), (x2, y2) = line1
                            (x3, y3), (x4, y4) = line2
                            
                            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                            if abs(denom) < 1e-10:  # Lines are parallel
                                return None
                            
                            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                            u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
                            
                            # Check if intersection is within both line segments
                            if 0 <= t <= 1 and 0 <= u <= 1:
                                x = x1 + t * (x2 - x1)
                                y = y1 + t * (y2 - y1)
                                return (x, y)
                            return None
                        
                        # Find all line-to-line intersections
                        intersections = []
                        for prev_line in prev_boundary_lines:
                            for curr_line in curr_boundary_lines:
                                intersection = line_intersection(prev_line, curr_line)
                                if intersection is not None:
                                    x, y = intersection
                                    # Convert back to depth (use absolute y value)
                                    depth = abs(y)
                                    intersections.append((x, y, depth))
                                    print(f"    Line intersection found at x={x:.6f}, y={y:.6f}, depth={depth:.6f}")
                        
                        # If we found intersections, use them
                        if intersections:
                            # Sort by depth and take the deepest intersection as the main one
                            intersections.sort(key=lambda x: x[2], reverse=True)
                            deepest_intersection = intersections[0]
                            intersection_depths.append(deepest_intersection[2])
                            print(f"    Using deepest intersection at depth: {deepest_intersection[2]:.6f}")
                        else:
                            print(f"    No line intersections found between boundary curves")
                    
                    except Exception as e:
                        print(f"    Error in boundary curve intersection: {e}")
                
                # Fallback: If no line intersections found, try closest approach method
                if len(intersection_depths) == 0:
                    print("    Trying closest approach method as fallback...")
                    
                    # Find the single closest pair of points between boundaries
                    if len(prev_bp) > 0 and len(curr_bp) > 0:
                        from scipy.spatial.distance import cdist
                        distances = cdist(curr_bp[:, :2], prev_bp[:, :2])
                        
                        # Find the single closest pair
                        min_dist_idx = np.unravel_index(distances.argmin(), distances.shape)
                        curr_idx, prev_idx = min_dist_idx
                        
                        min_distance = distances[curr_idx, prev_idx]
                        if min_distance < 50e-6:  # Within 50 micrometers
                            curr_point = curr_bp[curr_idx]
                            prev_point = prev_bp[prev_idx]
                            
                            intersection_depth = (abs(curr_point[1]) + abs(prev_point[1])) / 2
                            intersection_depths.append(intersection_depth)
                            print(f"    Closest approach intersection at depth: {intersection_depth:.6f} (distance: {min_distance*1e6:.1f} μm)")
                
                print(f"    Total intersection depths found: {len(intersection_depths)}")
                
                # Alternative method: Find actual line intersections using convex hulls
                if len(intersection_depths) == 0:
                    try:
                        from scipy.spatial import ConvexHull
                        # Skip shapely-based methods to avoid import errors
                        pass
                    except Exception:
                        # Any error with geometric intersection, continue with distance-based method
                        pass
                
                # If we found intersection depths, record them
                if intersection_depths:
                    # Use the maximum intersection depth (deepest overlap)
                    max_overlap_depth = max(intersection_depths)
                    avg_overlap_depth = np.mean(intersection_depths)
                    
                    overlap_depths.append({
                        'prev_time': prev_time,
                        'curr_time': curr_time,
                        'overlap_depth': max_overlap_depth,
                        'avg_overlap_depth': avg_overlap_depth,
                        'intersection_points': len(intersection_depths),
                        'prev_depth_range': (float(prev_bp[:,1].min()), float(prev_bp[:,1].max())),
                        'curr_depth_range': (float(curr_bp[:,1].min()), float(curr_bp[:,1].max()))
                    })
            
            return overlap_depths
        
        # Calculate overlap depths for all z-slices
        all_overlap_depths = {}
        print("\n" + "=" * 50)
        print("Calculating overlap depths...")
        for i, z in enumerate(zvals, 1):
            if z in all_depth_peaks and all_depth_peaks[z]:
                print(f"  [{i}/{len(zvals)}] Processing overlap for z={z/10000:.4f} ({len(all_depth_peaks[z])} depth peaks)")
                overlap_data = calculate_overlap_depth(z, all_depth_peaks[z], all_boundary_points)
                if overlap_data:
                    all_overlap_depths[z] = overlap_data
                    print(f"    → Found {len(overlap_data)} overlaps")
                else:
                    print(f"    → No overlaps found")
            else:
                print(f"  [{i}/{len(zvals)}] No depth peaks for z={z/10000:.4f} - skipping overlap calculation")
        print("Overlap calculation complete.")
        
        # Now generate the boundaries plot with the detected peaks and overlaps
        print("\n" + "=" * 50)
        print(f"Creating boundaries plot for z={boundaries_z/10000:.4f}")
        
        # Check if we have depth peaks for this z-slice
        if boundaries_z in all_depth_peaks and all_depth_peaks[boundaries_z]:
            from scipy.spatial import ConvexHull
            
            # Filter to requested z slice and sort by time
            z_points = sorted([(t, bp) for zz, t, bp in all_boundary_points if zz == boundaries_z], key=lambda x: x[0])
            
            peak_times = [peak_time for _, peak_time in all_depth_peaks[boundaries_z]]
            peak_boundary_data = []
            
            print(f"Depth peak times: {[f'{t:.6f}' for t in peak_times]}")
            
            # Find boundary data closest to each peak time
            for peak_time in peak_times:
                closest_data = min(z_points, key=lambda x: abs(x[0] - peak_time))
                if abs(closest_data[0] - peak_time) < 0.0001:  # Within tolerance
                    peak_boundary_data.append((peak_time, closest_data[1]))
                    print(f"  Found boundary data for peak time {peak_time:.6f} (closest: {closest_data[0]:.6f})")
            
            if peak_boundary_data:
                # Create the plot
                fig2, ax2 = plt.subplots(figsize=(12, 10))
                
                # Use different colors for each peak
                colors = plt.cm.tab10(np.linspace(0, 1, len(peak_boundary_data)))
                
                # Plot each melted area at peak times
                for i, (peak_time, bp) in enumerate(peak_boundary_data):
                    color = colors[i % len(colors)]
                    
                    # Plot boundary points
                    ax2.plot(bp[:, 0], -bp[:, 1], '.', color=color, alpha=0.7, markersize=4, 
                           label=f'Peak {i+1} (t={peak_time:.6f})')
                    
                    # Plot convex hull outline
                    if len(bp) >= 3:
                        try:
                            hull = ConvexHull(bp[:, :2])
                            hull_pts = bp[hull.vertices]
                            hull_pts = np.vstack([hull_pts, hull_pts[0]])
                            ax2.plot(hull_pts[:, 0], -hull_pts[:, 1], '-', color=color, alpha=0.8, linewidth=2.5)
                        except:
                            pass
                
                # Add intersection visualization if we have overlap data
                if boundaries_z in all_overlap_depths and all_overlap_depths[boundaries_z]:
                    overlaps = all_overlap_depths[boundaries_z]
                    print(f"Adding intersection visualization for {len(overlaps)} intersections...")
                    
                    for i, overlap_info in enumerate(overlaps):
                        prev_time = overlap_info['prev_time']
                        curr_time = overlap_info['curr_time']
                        
                        # Find the corresponding boundary data
                        prev_bp = None
                        curr_bp = None
                        for peak_time, bp in peak_boundary_data:
                            if abs(peak_time - prev_time) < 0.0001:
                                prev_bp = bp
                            if abs(peak_time - curr_time) < 0.0001:
                                curr_bp = bp
                        
                        if prev_bp is not None and curr_bp is not None:
                            # Recalculate single intersection point for visualization
                            intersection_point = None
                            
                            # Use the same boundary curve intersection method
                            if len(prev_bp) >= 3 and len(curr_bp) >= 3:
                                try:
                                    from scipy.spatial import ConvexHull
                                    
                                    # Create convex hulls for both boundaries
                                    prev_hull = ConvexHull(prev_bp[:, :2])
                                    curr_hull = ConvexHull(curr_bp[:, :2])
                                    
                                    prev_hull_points = prev_bp[prev_hull.vertices]
                                    curr_hull_points = curr_bp[curr_hull.vertices]
                                    
                                    # Create boundary lines
                                    prev_boundary_lines = []
                                    curr_boundary_lines = []
                                    
                                    for j in range(len(prev_hull_points)):
                                        p1 = prev_hull_points[j]
                                        p2 = prev_hull_points[(j + 1) % len(prev_hull_points)]
                                        prev_boundary_lines.append((p1[:2], p2[:2]))
                                    
                                    for j in range(len(curr_hull_points)):
                                        p1 = curr_hull_points[j]
                                        p2 = curr_hull_points[(j + 1) % len(curr_hull_points)]
                                        curr_boundary_lines.append((p1[:2], p2[:2]))
                                    
                                    # Line intersection function
                                    def line_intersection(line1, line2):
                                        (x1, y1), (x2, y2) = line1
                                        (x3, y3), (x4, y4) = line2
                                        
                                        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                                        if abs(denom) < 1e-10:
                                            return None
                                        
                                        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                                        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
                                        
                                        if 0 <= t <= 1 and 0 <= u <= 1:
                                            x = x1 + t * (x2 - x1)
                                            y = y1 + t * (y2 - y1)
                                            return (x, y)
                                        return None
                                    
                                    # Find the deepest intersection
                                    intersections = []
                                    for prev_line in prev_boundary_lines:
                                        for curr_line in curr_boundary_lines:
                                            intersection = line_intersection(prev_line, curr_line)
                                            if intersection is not None:
                                                x, y = intersection
                                                intersections.append((x, y, abs(y)))
                                    
                                    if intersections:
                                        # Use the deepest intersection
                                        intersections.sort(key=lambda x: x[2], reverse=True)
                                        intersection_point = intersections[0][:2]  # (x, y)
                                
                                except Exception:
                                    pass
                            
                            # Fallback to closest approach if no line intersection found
                            if intersection_point is None:
                                from scipy.spatial.distance import cdist
                                distances = cdist(curr_bp[:, :2], prev_bp[:, :2])
                                min_dist_idx = np.unravel_index(distances.argmin(), distances.shape)
                                curr_idx, prev_idx = min_dist_idx
                                
                                if distances[curr_idx, prev_idx] < 50e-6:
                                    curr_point = curr_bp[curr_idx]
                                    prev_point = prev_bp[prev_idx]
                                    intersection_x = (curr_point[0] + prev_point[0]) / 2
                                    intersection_y = (curr_point[1] + prev_point[1]) / 2
                                    intersection_point = (intersection_x, intersection_y)
                            
                            # Plot single intersection point
                            if intersection_point is not None:
                                x, y = intersection_point
                                ax2.plot(x, -y, 'x', color='red', markersize=15, markeredgewidth=4,
                                        label=f'Intersection {i+1}' if i == 0 else "",
                                        alpha=0.9)
                                
                                # Add text annotation using the actual intersection depth
                                actual_intersection_depth = abs(y)  # Use the actual y-coordinate of the intersection
                                ax2.annotate(f'Depth: {actual_intersection_depth:.6f}', 
                                           xy=(x, -y), 
                                           xytext=(15, 15), textcoords='offset points',
                                           fontsize=10, fontweight='bold', color='red',
                                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                                                   edgecolor='red', alpha=0.9),
                                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
                
                ax2.set_xlabel("x (m)")
                ax2.set_ylabel("-y (m)")
                ax2.set_title(f"Melted areas at depth peak times with overlaps (z = {boundaries_z/10000:.4f} m, T ≥ {T_THRESHOLD} K)")
                ax2.grid(True, alpha=0.3)
                ax2.set_aspect('equal', adjustable='box')
                
                # Create legend
                handles, labels = ax2.get_legend_handles_labels()
                if len(handles) > 8:  # Too many for inline legend
                    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                else:
                    ax2.legend(loc='best', fontsize=8)
                
                fig2.tight_layout()
                fig2.savefig("boundaries.png", dpi=150, bbox_inches='tight')
                print(f"Wrote boundaries.png showing {len(peak_boundary_data)} melted areas at depth peak times with overlap visualization.")
            else:
                print(f"No boundary data found matching depth peak times for z={boundaries_z}.")
        else:
            print(f"No depth peaks found for z={boundaries_z}. Cannot create boundaries plot.")
            # Create empty plot with message
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            ax2.text(0.5, 0.5, f'No depth peaks found for z={boundaries_z/10000:.4f}', 
                   ha='center', va='center', transform=ax2.transAxes, fontsize=14)
            ax2.set_title(f"No melted areas at depth peaks (z = {boundaries_z/10000:.4f})")
            fig2.savefig("boundaries.png", dpi=150)
            print("Wrote empty boundaries.png.")
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
        
        # Print maximum values for all z-slices
        print("\n=== PEAK VALUES SUMMARY ===")
        print("Width Peaks:")
        for z in sorted(all_width_peaks.keys()):
            if all_width_peaks[z]:
                print(f"  z={z/10000:.4f}:")
                for i, (peak_width, time_at_peak) in enumerate(all_width_peaks[z], 1):
                    print(f"    Peak {i}: Width = {peak_width:.6f} at time = {time_at_peak:.6f}")
            else:
                print(f"  z={z/10000:.4f}: No peaks detected")
        
        print("\nDepth Peaks:")
        for z in sorted(all_depth_peaks.keys()):
            if all_depth_peaks[z]:
                print(f"  z={z/10000:.4f}:")
                for i, (peak_depth, time_at_peak) in enumerate(all_depth_peaks[z], 1):
                    print(f"    Peak {i}: Depth = {peak_depth:.6f} at time = {time_at_peak:.6f}")
            else:
                print(f"  z={z/10000:.4f}: No peaks detected")
        
        print("\nOverlap Depths (Geometric Intersections):")
        if all_overlap_depths:
            total_overlaps = sum(len(overlaps) for overlaps in all_overlap_depths.values())
            print(f"Found {total_overlaps} total intersections across all z-slices:")
            for z in sorted(all_overlap_depths.keys()):
                overlaps = all_overlap_depths[z]
                print(f"  z={z/10000:.4f} ({len(overlaps)} intersections):")
                for i, overlap_info in enumerate(overlaps, 1):
                    print(f"    Intersection {i}: Depth = {overlap_info['overlap_depth']:.6f}")
                    print(f"      Between peak times {overlap_info['prev_time']:.6f} and {overlap_info['curr_time']:.6f}")
                    if 'avg_overlap_depth' in overlap_info:
                        print(f"      Average intersection depth = {overlap_info['avg_overlap_depth']:.6f}")
                    if 'intersection_points' in overlap_info:
                        print(f"      Number of intersection points = {overlap_info['intersection_points']}")
                    print(f"      Previous melt depth range: ({overlap_info['prev_depth_range'][0]:.6f}, {overlap_info['prev_depth_range'][1]:.6f})")
                    print(f"      Current melt depth range:  ({overlap_info['curr_depth_range'][0]:.6f}, {overlap_info['curr_depth_range'][1]:.6f})")
        else:
            print("  No intersection depths calculated")
        
        # Calculate and display averages
        print("\n=== AVERAGE VALUES SUMMARY ===")
        
        # Calculate averages for each z-slice
        for z in sorted(all_width_peaks.keys()):
            if all_width_peaks[z] and all_depth_peaks[z]:
                # Extract peak values
                width_values = [peak[0] for peak in all_width_peaks[z]]  # peak[0] is the width value
                depth_values = [peak[0] for peak in all_depth_peaks[z]]  # peak[0] is the depth value
                
                # Calculate averages
                avg_width = np.mean(width_values) * 1e6  # Convert to micrometers
                avg_depth = np.mean(depth_values) * 1e6  # Convert to micrometers
                
                print(f"z={z/10000:.4f}:")
                print(f"  Avg. Width (μm): {avg_width:.3f}")
                print(f"  Avg. Depth (μm): {avg_depth:.3f}")
                
                # Calculate average overlap depth if available
                if z in all_overlap_depths and all_overlap_depths[z]:
                    overlap_values = [overlap['overlap_depth'] for overlap in all_overlap_depths[z]]
                    avg_overlap_depth = np.mean(overlap_values) * 1e6  # Convert to micrometers
                    print(f"  Avg. Overlap Depth (μm): {avg_overlap_depth:.3f}")
                    print(f"  Number of peaks: Width={len(width_values)}, Depth={len(depth_values)}, Overlaps={len(overlap_values)}")
                else:
                    print(f"  Avg. Overlap Depth (μm): No overlaps detected")
                    print(f"  Number of peaks: Width={len(width_values)}, Depth={len(depth_values)}, Overlaps=0")
                print()
        
        print("==============================\n")

if __name__ == "__main__":
    main()

