#!/usr/bin/env python3
"""
Temperature Analysis and Melting Surface Calculation and Representation (TAMSCR)

Generates:
  1) A three-subplot figure:
       - scatter of final temperature with selected pixels highlighted
       - temperature vs time for those pixels with threshold line
       - table of crossings & TAM for those pixels
  2) A TAM (Time Above Melt) contour over the x–z plane (0 for no TAM shown as -1e-6)
  3) An SCR (Solidus Cooling Rate) contour over the x–z plane (undefined SCR shown as -1e-6)
  4) A single CSV listing ALL pixels that HAVE a TAM (>0), with columns:
       [pixel_index, x, y, z, TAM_seconds, TAM_end_time_s, SCR_K_per_s]
     If SCR is unavailable for a pixel, it is written as NaN.

Assumptions:
  - Your data directory contains subdirectories named by time (e.g., "0.0001", "0.0025", ...)
  - Each time directory contains a VTK file (default "planeY0.vtk") with POINTS and temperature data.
  - Temperatures are in Kelvin.

Edit the CONFIG section below to match your paths and parameters.

Date: August 27, 2025
"""

import os
import csv
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ----------------------- CONFIG -----------------------
BASE_DIR = "/media/data2/August/amb25/smallFast/postProcessing/T_slice"  # folder containing time-step directories
VTK_FILENAME = "planeY0.vtk"                                             # VTK file name inside each time folder
THRESHOLD_TEMP = 1533.15                                                 # K (solidus threshold used throughout)
TARGET_COOL_TEMP = 1423.15                                               # K (SCR target)
UNDEFINED_FILL = np.nan                                                  # value to use in contours when undefine0lh
N_PIXELS_FOR_TIMESERIES = 5                                              # how many pixels to plot in the time-series
MAX_TIME_STEPS = 100000                                                     # limit for loading (evenly sampled across range)
RESULTS_DIR = "/media/data2/August/amb25/results"                        # outputs folder
# ------------------------------------------------------


class VTKTemperatureAnalyzer:
    def create_tam_point_plot(self, tam: np.ndarray, figsize: Tuple[int, int] = (10, 4)):
        """Scatter plot of TAM over x–z; undefined TAM (<=0 or NaN) -> UNDEFINED_FILL. Uses square markers."""
        vals = tam.copy()
        vals[~(vals > 0)] = UNDEFINED_FILL
        x = self.coordinates[:, 0]
        z = self.coordinates[:, 2]
        fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(x, z, c=vals, cmap='viridis', s=20, marker='s', edgecolors='none')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("TAM (s)")
        ax.set_aspect('equal')
        ax.set_xlabel('(m)')
        ax.set_ylabel('(m)')
        ax.set_title('TAM Point Plot (pixel values, squares)')
        ax.grid(True, alpha=0.15)
        return fig

    def create_scr_point_plot(self, scr: np.ndarray, max_scr: float = 10.0, figsize: Tuple[int, int] = (10, 4)):
        """Scatter plot of SCR over x–z; undefined (NaN, <=0) or >max_scr -> UNDEFINED_FILL. Uses square markers."""
        vals = scr.copy()
        # Mask invalid (NaN, <=0) or too large values
        mask_invalid = (np.isnan(vals)) | (vals <= 0) | (vals > max_scr)
        plot_vals = np.full_like(vals, UNDEFINED_FILL, dtype=float)
        plot_vals[~mask_invalid] = vals[~mask_invalid]
        x = self.coordinates[:, 0]
        z = self.coordinates[:, 2]
        fig, ax = plt.subplots(figsize=figsize)
        sc = ax.scatter(x, z, c=plot_vals, cmap='viridis', s=20, marker='s', edgecolors='none')
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Solidus Cooling Rate (K/s)")
        ax.set_aspect('equal')
        ax.set_xlabel('(m)')
        ax.set_ylabel('(m)')
        ax.set_title(f'SCR Point Plot (pixel values, squares)\n(cooling from {THRESHOLD_TEMP} K to {TARGET_COOL_TEMP} K, >{max_scr} hidden)')
        ax.grid(True, alpha=0.15)
        return fig
    """Analyzes temperature data from VTK files across time steps."""

    def __init__(self, data_directory: str, vtk_filename: str = "planeY0.vtk"):
        """
        Initialize the analyzer.
        """
        self.data_directory = data_directory
        self.vtk_filename = vtk_filename
        self.time_steps: List[float] = []
        self.time_step_paths: Dict[float, str] = {}
        self.coordinates: Optional[np.ndarray] = None  # shape (N,3)
        self.temperature_data: Dict[float, np.ndarray] = {}  # time -> temps (N,)

    # ---------- Data IO ----------
    def scan_time_steps(self) -> List[float]:
        """Scan directory for available time steps with the target VTK file."""
        time_dirs = []
        for item in os.listdir(self.data_directory):
            item_path = os.path.join(self.data_directory, item)
            if os.path.isdir(item_path):
                try:
                    tval = float(item)
                    vtk_path = os.path.join(item_path, self.vtk_filename)
                    if os.path.exists(vtk_path):
                        time_dirs.append((tval, item_path))
                except ValueError:
                    continue
        if not time_dirs:
            raise FileNotFoundError("No time step directories with the target VTK file were found.")
        time_dirs.sort(key=lambda x: x[0])
        self.time_steps = [t for t, _ in time_dirs]
        self.time_step_paths = {t: p for t, p in time_dirs}
        print(f"Found {len(self.time_steps)} time steps ranging from {min(self.time_steps):.6f} to {max(self.time_steps):.6f}")
        return self.time_steps

    def parse_vtk_file(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse a VTK file to extract coordinates and temperature data.
        Returns (coordinates[N,3], temperatures[N,])
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # POINTS section
        n_points = None
        coord_start_line = None
        for i, line in enumerate(lines):
            if line.startswith('POINTS'):
                parts = line.split()
                n_points = int(parts[1])
                coord_start_line = i + 1
                break
        if n_points is None:
            raise ValueError(f"Could not find POINTS in {file_path}")

        # Coordinates
        coords_flat = []
        j = coord_start_line
        while len(coords_flat) < n_points * 3 and j < len(lines):
            s = lines[j].strip()
            if s:
                coords_flat.extend([float(x) for x in s.split()])
            j += 1
        coordinates = np.array(coords_flat[:n_points * 3]).reshape(-1, 3)

        # Temperatures: scan from end; fall back to broader scan
        temps: List[float] = []
        for k in range(len(lines) - 1, -1, -1):
            s = lines[k].strip()
            if not s:
                continue
            try:
                vals = [float(x) for x in s.split()]
                if any(200 < v < 5000 for v in vals):
                    temps = vals + temps
                    if len(temps) >= n_points:
                        break
            except ValueError:
                continue
        if len(temps) < n_points:
            temps = []
            started = False
            for s in reversed([ln.strip() for ln in lines]):
                if not s:
                    continue
                try:
                    vals = [float(x) for x in s.split()]
                    if any(200 < v < 5000 for v in vals):
                        temps = vals + temps
                        started = True
                        if len(temps) >= n_points:
                            break
                    elif started:
                        break
                except ValueError:
                    if started:
                        break
                    continue

        temperatures = np.array(temps[:n_points], dtype=float)
        if len(temperatures) != n_points:
            raise ValueError(f"Temperature data size mismatch: expected {n_points}, got {len(temperatures)}")
        return coordinates, temperatures

    def load_all_data(self, max_time_steps: int = None):
        """Load temperature data for all time steps (or an evenly sampled subset)."""
        if not self.time_steps:
            self.scan_time_steps()
        if max_time_steps is not None and len(self.time_steps) > max_time_steps:
            idx = np.linspace(0, len(self.time_steps) - 1, max_time_steps, dtype=int)
            self.time_steps = [self.time_steps[i] for i in idx]
            print(f"Limited to {max_time_steps} evenly distributed time steps")

        print("Loading temperature data...")
        loaded = 0
        for i, t in enumerate(self.time_steps):
            vtk_path = os.path.join(self.time_step_paths[t], self.vtk_filename)
            try:
                coords, temps = self.parse_vtk_file(vtk_path)
                if self.coordinates is None:
                    self.coordinates = coords
                    print(f"Grid size: {len(coords)} points")
                    print(f"X range: {coords[:,0].min():.6f} .. {coords[:,0].max():.6f}")
                    print(f"Z range: {coords[:,2].min():.6f} .. {coords[:,2].max():.6f}")
                self.temperature_data[t] = temps
                loaded += 1
                if loaded % 10 == 0:
                    print(f"Loaded {loaded}/{len(self.time_steps)}")
            except Exception as e:
                print(f"Error loading {vtk_path}: {e}")
        print(f"Successfully loaded {loaded} time steps")

    # ---------- Utilities ----------
    def get_pixel_temperature_evolution(self, pixel_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        times = np.array(sorted(self.temperature_data.keys()), dtype=float)
        temps = np.array([self.temperature_data[t][pixel_idx] for t in times], dtype=float)
        return times, temps

    def find_threshold_crossing_times(self, pixel_idx: int, threshold: float) -> List[float]:
        """All (interpolated) times the pixel crosses the threshold (either direction)."""
        times, temps = self.get_pixel_temperature_evolution(pixel_idx)
        cross = []
        for i in range(len(times) - 1):
            t1, t2 = times[i], times[i+1]
            y1, y2 = temps[i], temps[i+1]
            if (y1 - threshold) * (y2 - threshold) <= 0 and y1 != y2:
                ct = t1 + (threshold - y1) * (t2 - t1) / (y2 - y1)
                cross.append(float(ct))
            elif y1 == threshold:
                cross.append(float(t1))
        return cross

    # ---------- TAM (longest continuous >= threshold) ----------
    def calculate_tam_and_end(self, pixel_idx: int, threshold: float) -> Tuple[float, Optional[float]]:
        """
        Returns the longest continuous duration with temp >= threshold
        and the end time of that longest segment (when it cools back down to threshold,
        or the last available time if still above at the end).
        """
        times, temps = self.get_pixel_temperature_evolution(pixel_idx)
        if len(times) < 2:
            return 0.0, None

        max_dur = 0.0
        end_of_max: Optional[float] = None
        in_seg = False
        entry_time: Optional[float] = None

        if temps[0] >= threshold:
            in_seg = True
            entry_time = float(times[0])

        for i in range(len(times) - 1):
            t1, t2 = float(times[i]), float(times[i+1])
            y1, y2 = float(temps[i]), float(temps[i+1])

            # upward crossing
            if (y1 < threshold) and (y2 >= threshold):
                cross_up = t1 + (threshold - y1) * (t2 - t1) / (y2 - y1) if y2 != y1 else t1
                in_seg = True
                entry_time = float(cross_up)

            # downward crossing
            if in_seg and (y1 >= threshold) and (y2 < threshold):
                cross_down = t1 + (threshold - y1) * (t2 - t1) / (y2 - y1) if y2 != y1 else t2
                dur = float(cross_down) - float(entry_time if entry_time is not None else t1)
                if dur > max_dur:
                    max_dur = dur
                    end_of_max = float(cross_down)
                in_seg = False
                entry_time = None

            # exact-threshold edge cases
            if not in_seg and (y1 == threshold and y2 > threshold):
                in_seg = True
                entry_time = t1
            if in_seg and (y1 > threshold and y2 == threshold):
                cross_down = t2
                dur = float(cross_down) - float(entry_time if entry_time is not None else t1)
                if dur > max_dur:
                    max_dur = dur
                    end_of_max = float(cross_down)
                in_seg = False
                entry_time = None

        # still above at the end
        if in_seg and entry_time is not None:
            cross_down = float(times[-1])
            dur = cross_down - float(entry_time)
            if dur > max_dur:
                max_dur = dur
                end_of_max = cross_down

        return float(max_dur), (None if max_dur == 0.0 else float(end_of_max))

    def compute_tam_for_all_pixels(self, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            tam (N,), tam_end_time (N,)
            - tam == 0.0 means undefined (never above threshold)
            - tam_end_time = NaN if tam undefined
        """
        if self.coordinates is None:
            raise RuntimeError("No coordinates loaded.")
        n = self.coordinates.shape[0]
        tam = np.zeros(n, dtype=float)
        tam_end = np.full(n, np.nan, dtype=float)
        for idx in range(n):
            dur, end_t = self.calculate_tam_and_end(idx, threshold)
            tam[idx] = dur
            if end_t is not None:
                tam_end[idx] = end_t
        return tam, tam_end

    # ---------- SCR (cooling from threshold to TARGET_COOL_TEMP) ----------
    def calculate_scr_for_pixel(self, pixel_idx: int, threshold: float, target: float = TARGET_COOL_TEMP
                               ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Returns (tam_end_time, time_to_target, scr_K_per_s) for one pixel.
        - tam_end_time: None if no TAM (never reached threshold)
        - time_to_target: None if it never cools to 'target' after tam_end
        - scr: (threshold - target) / time_to_target (K/s), None if not computable
        """
        tam_dur, tam_end = self.calculate_tam_and_end(pixel_idx, threshold)
        if tam_end is None:
            return None, None, None

        times, temps = self.get_pixel_temperature_evolution(pixel_idx)
        times = times.astype(float)
        temps = temps.astype(float)

        if tam_end < times[0] or tam_end > times[-1]:
            return float(tam_end), None, None

        i = np.searchsorted(times, tam_end) - 1
        i = max(0, min(i, len(times) - 2))
        t1, t2 = times[i], times[i+1]
        y1, y2 = temps[i], temps[i+1]
        y_at_end = y1 + (tam_end - t1) * (y2 - y1) / (t2 - t1) if t2 != t1 else y1

        def segment_cross_time(t1, y1, t2, y2, y_target):
            if (y1 - y_target) * (y2 - y_target) <= 0 and y1 != y2:
                return t1 + (y_target - y1) * (t2 - t1) / (y2 - y1)
            elif y1 == y_target:
                return t1
            elif y2 == y_target:
                return t2
            return None

        cross = None
        if t2 > tam_end:
            cross = segment_cross_time(tam_end, y_at_end, t2, y2, target)
        j = i + 1
        while cross is None and j < len(times) - 1:
            cross = segment_cross_time(times[j], temps[j], times[j+1], temps[j+1], target)
            j += 1

        if cross is None:
            return float(tam_end), None, None

        dt = float(cross) - float(tam_end)
        if dt <= 0:
            return float(tam_end), None, None

        scr = (float(threshold) - float(target)) / dt  # K/s
        return float(tam_end), float(dt), float(scr)

    def compute_scr_for_all_pixels(self, threshold: float, target: float = TARGET_COOL_TEMP
                                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns arrays (N,):
            tam_end (NaN where no TAM),
            time_to_target (NaN if not reached),
            scr (NaN if not defined)
        """
        n = self.coordinates.shape[0]
        tam_end_arr = np.full(n, np.nan, dtype=float)
        dt_arr = np.full(n, np.nan, dtype=float)
        scr_arr = np.full(n, np.nan, dtype=float)
        for idx in range(n):
            end_t, dt, scr_val = self.calculate_scr_for_pixel(idx, threshold, target)
            if end_t is not None:
                tam_end_arr[idx] = end_t
            if dt is not None:
                dt_arr[idx] = dt
            if scr_val is not None:
                scr_arr[idx] = scr_val
        return tam_end_arr, dt_arr, scr_arr

    # ---------- Pixel selection for time-series viz ----------
    def select_pixels_for_analysis(self, n_pixels: int = 5, method: str = 'diverse',
                                   x_range: Tuple[float, float] = None) -> List[int]:
        if method == 'x_range' and x_range is not None:
            x_coords = self.coordinates[:, 0]
            z_coords = self.coordinates[:, 2]
            x_min, x_max = x_range
            mask = (x_coords >= x_min) & (x_coords <= x_max)
            valid_indices = np.where(mask)[0]
            if len(valid_indices) == 0:
                print(f"Warning: No pixels found in x range ({x_min:.6f}, {x_max:.6f})")
                return []
            print(f"Found {len(valid_indices)} pixels in x range ({x_min:.6f}, {x_max:.6f})")
            if len(valid_indices) <= n_pixels:
                selected_pixels = valid_indices.tolist()
            else:
                valid_z = z_coords[valid_indices]
                z_min, z_max = valid_z.min(), valid_z.max()
                selected_pixels = []
                for i in range(n_pixels):
                    z_target = z_min + (i + 0.5) * (z_max - z_min) / n_pixels
                    distances = np.abs(valid_z - z_target)
                    closest_local_idx = np.argmin(distances)
                    closest_global_idx = valid_indices[closest_local_idx]
                    selected_pixels.append(closest_global_idx)
        elif method == 'diverse':
            x_coords = self.coordinates[:, 0]
            z_coords = self.coordinates[:, 2]
            x_min, x_max = x_coords.min(), x_coords.max()
            z_min, z_max = z_coords.min(), z_coords.max()
            selected_pixels = []
            grid_size = int(np.ceil(np.sqrt(n_pixels)))
            for i in range(min(n_pixels, grid_size * grid_size)):
                row = i // grid_size
                col = i % grid_size
                x_target = x_min + (col + 0.5) * (x_max - x_min) / grid_size
                z_target = z_min + (row + 0.5) * (z_max - z_min) / grid_size
                distances = np.sqrt((x_coords - x_target)**2 + (z_coords - z_target)**2)
                closest_idx = np.argmin(distances)
                selected_pixels.append(closest_idx)
                if len(selected_pixels) >= n_pixels:
                    break
        elif method == 'hottest':
            max_temps = []
            for i in range(len(self.coordinates)):
                temps = [self.temperature_data[t][i] for t in self.temperature_data.keys()]
                max_temps.append(max(temps))
            hottest_indices = np.argsort(max_temps)[-n_pixels:]
            selected_pixels = hottest_indices.tolist()
        elif method == 'random':
            selected_pixels = np.random.choice(len(self.coordinates), n_pixels, replace=False).tolist()
        return selected_pixels

    # ---------- Visualization ----------
    def create_visualization(self, pixel_indices: List[int], threshold: float,
                             figsize: Tuple[int, int] = (12, 14)):
        """
        Three vertically stacked subplots:
          (1) scatter of final temperature with selected pixels highlighted,
          (2) temperature evolution curves + threshold,
          (3) table summarizing crossing times & TAM for selected pixels.
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.05)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0])
        ax_table = fig.add_subplot(gs[2, 0])

        # Top: scatter of final temperature
        x_coords = self.coordinates[:, 0]
        z_coords = self.coordinates[:, 2]
        final_time = max(self.temperature_data.keys())
        final_temps = self.temperature_data[final_time]
        scatter = ax1.scatter(x_coords, z_coords, c=final_temps, cmap='hot', s=1, alpha=0.6)
        colors = plt.cm.Set1(np.linspace(0, 1, max(1, len(pixel_indices))))
        for i, pixel_idx in enumerate(pixel_indices):
            x, z = self.coordinates[pixel_idx, 0], self.coordinates[pixel_idx, 2]
            ax1.scatter(x, z, color=colors[i % len(colors)], s=50, marker='o',
                        edgecolors='black', linewidth=2, label=f'Pixel {i+1} ({pixel_idx})')
        ax1.set_xlabel('(m)')
        ax1.set_title('Surface Temperature (x–z plane), Selected Pixels')
        ax1.legend(loc='upper right', fontsize=9, framealpha=0.8)
        ax1.set_aspect('equal')
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="3%", pad=0.10)
        cbar = plt.colorbar(scatter, cax=cax)
        cbar.set_label('Temperature (K)')

        # Middle: temperature evolution with threshold & crossings
        all_crossing_times = []
        pixel_data = []
        for i, pixel_idx in enumerate(pixel_indices):
            times, temperatures = self.get_pixel_temperature_evolution(pixel_idx)
            ax2.plot(times, temperatures, color=colors[i % len(colors)], linewidth=2,
                     label=f'Pixel {i+1} ({pixel_idx})')
            crossing_times = self.find_threshold_crossing_times(pixel_idx, threshold)
            if crossing_times:
                all_crossing_times.extend(crossing_times)
                for ct in crossing_times:
                    ax2.scatter(ct, threshold, color=colors[i % len(colors)],
                                marker='x', s=80, linewidth=2, zorder=5)
            tam_val, _ = self.calculate_tam_and_end(pixel_idx, threshold)
            crossing_times_str = ', '.join([f'{t:.6f}' for t in crossing_times]) if crossing_times else 'None'
            tam_str = f'{tam_val:.6f}' if tam_val > 0 else 'None'
            pixel_data.append([f'Pixel {i+1} ({pixel_idx})', crossing_times_str, tam_str])

        time_range = [min(self.time_steps), max(self.time_steps)]
        ax2.plot(time_range, [threshold, threshold], 'k--', linewidth=2, label=f'Threshold ({threshold} K)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Temperature (K)')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Bottom: summary table
        ax_table.axis('off')
        table_data = [['Pixel', 'Threshold Crossing Times (s)', 'TAM (s)']] + pixel_data
        table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                               cellLoc='center', loc='center', colWidths=[0.25, 0.50, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.6)
        for r in range(len(table_data)):
            for c in range(len(table_data[0])):
                cell = table[(r, c)]
                if r == 0:
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f1f1f2' if r % 2 == 0 else 'white')
        return fig, all_crossing_times

    def create_tam_contour(self, tam: np.ndarray, figsize: Tuple[int, int] = (10, 4)):
        """tricontourf of TAM over x–z; undefined TAM (<=0 or NaN) -> UNDEFINED_FILL"""
        vals = tam.copy()
        vals[~(vals > 0)] = UNDEFINED_FILL
        x = self.coordinates[:, 0]
        z = self.coordinates[:, 2]
        fig, ax = plt.subplots(figsize=figsize)
        cs = ax.tricontourf(x, z, vals, levels=20)
        cbar = fig.colorbar(cs, ax=ax); cbar.set_label("TAM (s)")
        ax.set_aspect('equal'); ax.set_xlabel('(m)'); ax.set_ylabel('(m)')
        ax.set_title('Time Above Melt (TAM) Contour'); ax.grid(True, alpha=0.15)
        return fig

    def create_scr_contour(self, scr: np.ndarray, figsize: Tuple[int, int] = (10, 4)):
        """tricontourf of SCR; undefined (NaN, <=0) -> UNDEFINED_FILL"""
        vals = scr.copy()
        vals[np.isnan(vals) | (vals <= 0)] = UNDEFINED_FILL
        x = self.coordinates[:, 0]
        z = self.coordinates[:, 2]
        fig, ax = plt.subplots(figsize=figsize)
        cs = ax.tricontourf(x, z, vals, levels=20)
        cbar = fig.colorbar(cs, ax=ax)
        cbar.set_label("Solidus Cooling Rate (K/s)")
        ax.set_aspect('equal')
        ax.set_xlabel('(m)')
        ax.set_ylabel('(m)')
        ax.set_title(f'SCR Contour (cooling from {THRESHOLD_TEMP} K to {TARGET_COOL_TEMP} K)')
        ax.grid(True, alpha=0.15)
        return fig


def main():
    # --- Setup paths ---
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ANALYSIS_PNG     = os.path.join(RESULTS_DIR, "temperature_analysis.png")
    TAM_CONTOUR_PNG  = os.path.join(RESULTS_DIR, "tam_contour.png")
    SCR_CONTOUR_PNG  = os.path.join(RESULTS_DIR, "scr_contour.png")
    COMBINED_CSV     = os.path.join(RESULTS_DIR, "tam_scr_combined.csv")  # single CSV

    print("Starting TAMSCR...")
    print("=" * 80)

    # --- Load data ---
    analyzer = VTKTemperatureAnalyzer(BASE_DIR, VTK_FILENAME)
    analyzer.scan_time_steps()
    analyzer.load_all_data(max_time_steps=MAX_TIME_STEPS)

    # --- Three-subplot figure for selected pixels ---
    print(f"\nSelecting {N_PIXELS_FOR_TIMESERIES} pixels in x range (0.00004, 0.00012) for time-series...")
    # Adjust x_range as needed for your domain; or change method='diverse'
    pixel_indices = analyzer.select_pixels_for_analysis(
        n_pixels=N_PIXELS_FOR_TIMESERIES, method='x_range', x_range=(0.00004, 0.00012)
    )
    print("Selected pixels:")
    for i, idx in enumerate(pixel_indices):
        xyz = analyzer.coordinates[idx]
        print(f"  Pixel {i+1} -> index {idx} at ({xyz[0]:.6f}, {xyz[1]:.6f}, {xyz[2]:.6f})")

    print(f"\nCreating three-subplot visualization with threshold {THRESHOLD_TEMP} K...")
    fig, _ = analyzer.create_visualization(pixel_indices, THRESHOLD_TEMP)
    plt.savefig(ANALYSIS_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {ANALYSIS_PNG}")

    # --- Compute per-pixel metrics ---
    print("\nComputing TAM and TAM end times for all pixels...")
    tam, tam_end = analyzer.compute_tam_for_all_pixels(THRESHOLD_TEMP)

    print("Computing SCR for all pixels...")
    tam_end_arr, dt_to_target, scr = analyzer.compute_scr_for_all_pixels(THRESHOLD_TEMP, TARGET_COOL_TEMP)

    # --- Save single CSV: include ALL pixels (TAM and SCR for every pixel; NaN for undefined) ---
    print("Writing complete TAM, SCR CSV (all pixels; NaN for undefined values)...")
    with open(COMBINED_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pixel_index", "x", "y", "z", "TAM_seconds", "TAM_end_time_s", "SCR_K_per_s"])
        for i, (xyz, tam_i, end_i, scr_i) in enumerate(zip(analyzer.coordinates, tam, tam_end, scr)):
            w.writerow([
                i,
                float(xyz[0]),
                float(xyz[1]),
                float(xyz[2]),
                float(tam_i) if np.isfinite(tam_i) else np.nan,
                float(end_i) if np.isfinite(end_i) else np.nan,
                float(scr_i) if np.isfinite(scr_i) else np.nan
            ])
    print(f"Saved: {COMBINED_CSV}")

    # --- Save sorted CSV by TAM_end_time (only valid, only [tam_end_time, tam, scr]) ---
    SORTED_CSV = os.path.join(RESULTS_DIR, "tam_scr_sorted_by_endtime.csv")
    print("Writing sorted TAM, SCR CSV (only valid rows, columns: tam_end_time, tam, scr)...")
    rows = []
    for tam_i, end_i, scr_i in zip(tam, tam_end, scr):
        if np.isfinite(tam_i) and np.isfinite(end_i) and np.isfinite(scr_i):
            rows.append([float(end_i), float(tam_i), float(scr_i)])
    rows.sort(key=lambda r: r[0])  # sort by tam_end_time
    with open(SORTED_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tam_end_time", "tam", "scr"])
        w.writerows(rows)
    print(f"Saved: {SORTED_CSV}")

    # --- Print summary statistics for TAM and SCR ---
    # Only consider valid (TAM > 0) and valid SCR (finite, >0, <= max_scr)
    valid_tam = tam[tam > 0]
    valid_scr = scr[(~np.isnan(scr)) & (scr > 0) & (scr <= 3e6)]
    print("\nSummary statistics:")
    print(f"{'N-datapoints':>12}\t{'TAM Mean':>10}\t{'TAM Median':>10}\t{'TAM Std. Dev.':>12}\t{'N-datapoints':>12}\t{'SCR Mean':>10}\t{'SCR Median':>10}\t{'SCR Std. Dev.':>12}")
    print(f"{len(valid_tam):12d}\t{np.mean(valid_tam):10.4f}\t{np.median(valid_tam):10.4f}\t{np.std(valid_tam):12.4f}\t{len(valid_scr):12d}\t{np.mean(valid_scr):10.4f}\t{np.median(valid_scr):10.4f}\t{np.std(valid_scr):12.4f}")

    # --- Contours with UNDEFINED_FILL ---
    print("Creating TAM point plot (undefined -> -1e-6, squares)...")
    fig_tam = analyzer.create_tam_point_plot(tam)
    plt.savefig(TAM_CONTOUR_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig_tam)
    print(f"Saved: {TAM_CONTOUR_PNG}")

    print("Creating SCR point plot (undefined -> -1e-6, squares, values > max_scr hidden)...")
    fig_scr = analyzer.create_scr_point_plot(scr, max_scr=3e6)
    plt.savefig(SCR_CONTOUR_PNG, dpi=300, bbox_inches='tight')
    plt.close(fig_scr)
    print(f"Saved: {SCR_CONTOUR_PNG}")

    print("\nDone.")

if __name__ == "__main__":
    main()

