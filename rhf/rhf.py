import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
import os

# =========================================================
# Fixed geometry in X (45 tracks; 44 equal gaps of 0.00011 m)
# =========================================================
x_min, x_max = 0.00008, 0.00492
y_min        = 0.00008
n_tracks     = 45
pitch        = (x_max - x_min) / (n_tracks - 1)  # 0.00011 m
step         = 50e-6                              # ~50 µm along-track spacing
v            = 0.96                               # scan speed [m/s]

TURN1 = 0.00075   # seconds (0.75 ms)  ← original case
TURN2 = 0.00500   # seconds (5   ms)   ← new case

# R,T pairs for the 2x2 subplots
pairs = [
    (0.20e-3, 2.0e-3),
    (0.80e-3, 2.0e-3),
    (0.20e-3, 8.0e-3),
    (0.80e-3, 8.0e-3),
]

# Output directory = same folder as this script (fallback: CWD if __file__ unavailable)
outdir = Path(__file__).resolve().parent if "__file__" in globals() else Path(os.getcwd())


# =========================================================
# Helpers
# =========================================================
def sample_vertical_track(x, y_start, y_end, step):
    L = abs(y_end - y_start)
    n = max(2, int(np.floor(L / step)) + 1)
    ys = np.linspace(y_start, y_end, n)
    xs = np.full_like(ys, x)
    return np.column_stack([xs, ys]), L

def build_serpentine(y_max, turn_over_time):
    """Build serpentine path (points, times) for given Y extent and turn-over pause."""
    all_track_pts, all_track_times, track_lengths = [], [], []
    t_now = 0.0
    for k in range(n_tracks):
        x = x_min + k * pitch
        if k % 2 == 0:  # even -> up
            p, L = sample_vertical_track(x, y_min, y_max, step)
        else:           # odd  -> down
            p, L = sample_vertical_track(x, y_max, y_min, step)

        local_t = np.linspace(0.0, L / v, len(p))   # ON-time along this track
        all_track_pts.append(p)
        all_track_times.append(t_now + local_t)
        track_lengths.append(len(p))

        t_now += (L / v)
        if k < n_tracks - 1:
            t_now += turn_over_time               # OFF pause before next track

    pts   = np.vstack(all_track_pts)
    times = np.concatenate(all_track_times)
    n1    = track_lengths[0]
    return pts, times, all_track_pts, track_lengths, n1

def compute_rhf(pts, times, n1, R, T):
    """Paper-like RHF; normalize so mid of first track == 1 (for this R,T)."""
    def kernel(dist, dt):
        return ((R - dist) / R) ** 2 * ((T - dt) / T)

    N = len(pts)
    rhf_raw = np.zeros(N)
    j0 = 0
    for i in range(N):
        # keep only prior points within T seconds
        while j0 < i and (times[i] - times[j0]) > T:
            j0 += 1
        if j0 >= i:
            continue

        dt   = times[i] - times[j0:i]
        dxy  = pts[i] - pts[j0:i]
        dist = np.hypot(dxy[:, 0], dxy[:, 1])

        m = (dt > 0.0) & (dt <= T) & (dist > 0.0) & (dist <= R)
        if not np.any(m):
            continue

        rhf_raw[i] = kernel(dist[m], dt[m]).sum()

    # Normalize to the middle of the FIRST track
    mid_idx  = n1 // 2
    baseline = rhf_raw[mid_idx]
    if baseline <= 0:
        i0, i1 = int(0.4 * n1), int(0.6 * n1)
        baseline = np.median(rhf_raw[i0:i1]) if np.any(rhf_raw[i0:i1] > 0) else 1.0
    return (rhf_raw / baseline) if baseline > 0 else rhf_raw

def draw_grid_figure(y_max, turn_over_time, filename):
    """Make a 2x2 grid for given y_max and turn_over_time; save to PNG."""
    pts, times, all_track_pts, track_lengths, n1 = build_serpentine(y_max, turn_over_time)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.ravel()
    cmap = plt.cm.viridis

    for ax, (R, T) in zip(axes, pairs):
        rhf = compute_rhf(pts, times, n1, R, T)  # per-panel normalization inside

        # Per-panel color normalization
        vmin, vmax = float(np.min(rhf)), float(np.max(rhf))
        if vmax <= vmin:
            vmax = vmin + 1e-12
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        # Solid lines with linearly varying color between nodes
        offset = 0
        for k, p in enumerate(all_track_pts):
            n = track_lengths[k]
            r = rhf[offset:offset + n]
            segs = np.stack([p[:-1], p[1:]], axis=1)      # (n-1, 2, 2)
            seg_vals = 0.5 * (r[:-1] + r[1:])            # linear interp color
            lc = LineCollection(segs, array=seg_vals, cmap=cmap, norm=norm, linewidths=2.0, zorder=2)
            ax.add_collection(lc)
            offset += n

        # Keep the points as well
        ax.scatter(pts[:, 0], pts[:, 1], c=rhf, cmap=cmap, norm=norm,
                   s=10, marker='s', edgecolors='none', zorder=3)

        # Per-panel colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)
        ax.set_title(f"R = {R*1e3:.2f} mm,  T = {T*1e3:.2f} ms")

    plt.tight_layout()
    outpath = outdir / filename
    fig.savefig(outpath, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")

def export_rhf_table(y_max, turn_over_time, R, T, csv_name=None):
    """
    Build serpentine (y_max, turn_over_time), compute RHF with (R, T),
    and write a CSV with columns: t, x, y, rhf.
    R [m], T [s], y_max [m], turn_over_time [s].
    """
    pts, times, _, _, n1 = build_serpentine(y_max, turn_over_time)
    rhf = compute_rhf(pts, times, n1, R, T)

    # Minimal filename if none provided
    if csv_name is None:
        # Naming: 5x5 = large domain, 5x1 = small domain; fast/slow = turn time
        if y_max > 0.003 and turn_over_time < 0.002:
            case_ = '5x5_fast'
        elif y_max > 0.003 and turn_over_time >= 0.002:
            case_ = '5x5_slow'
        elif y_max <= 0.003 and turn_over_time < 0.002:
            case_ = '5x1_fast'
        else:
            case_ = '5x1_slow'
        csv_name = f"constant/rhfTable_{case_}.csv"

    csv_path = outdir / csv_name

    arr = np.column_stack([times, pts[:, 0], pts[:, 1], rhf])
    np.savetxt(csv_path, arr, delimiter=",", header="t,x,y,rhf", comments="")
    print(f"Saved RHF table: {csv_path}")
    return csv_path


def calculate_simulation_duration(y_max, turn_over_time):
    """
    Calculate timing details for a simulation.
    Returns: total_time, single_track_time, total_track_time, total_turn_around_time
    """
    # Calculate single track length and time
    single_track_length = abs(y_max - y_min)
    single_track_time = single_track_length / v

    # Total time for all tracks (scanning)
    total_track_time = n_tracks * single_track_time

    # Total turn-around time (n_tracks - 1 pauses between tracks)
    total_turn_around_time = (n_tracks - 1) * turn_over_time

    # Total simulation time
    total_time = total_track_time + total_turn_around_time

    return total_time, single_track_time, total_track_time, total_turn_around_time

def print_simulation_durations():
    """Print duration details for all 4 simulation cases."""
    print("\n" + "="*80)
    print("SIMULATION DURATION ANALYSIS")
    print("="*80)

    cases = [
        ("Case A: Large domain, fast turnaround", 0.00492, TURN1),
        ("Case B: Small domain, fast turnaround", 0.00092, TURN1),
        ("Case C: Large domain, slow turnaround", 0.00492, TURN2),
        ("Case D: Small domain, slow turnaround", 0.00092, TURN2),
    ]

    for case_name, y_max, turn_time in cases:
        total_time, single_track, total_track, total_turn = calculate_simulation_duration(y_max, turn_time)

        print(f"\n{case_name}:")
        print(f"  Domain size (Y): {y_max*1e3:.2f} mm")
        print(f"  Turnaround time: {turn_time*1e3:.4f} ms")
        print(f"  Single track time: {single_track*1e3:.4f} ms")
        print(f"  Total track time: {total_track*1e3:.4f} ms")
        print(f"  Total turnaround time: {total_turn*1e3:.4f} ms")
        print(f"  TOTAL SIMULATION TIME: {total_time*1e3:.4f} ms ({total_time:.4f} s)")
        print(f"  Track time fraction: {(total_track/total_time)*100:.1f}%")
        print(f"  Turnaround time fraction: {(total_turn/total_time)*100:.1f}%")

def export_all_four_tables(R=8e-6, T=3e-3):
    """
    Write four CSVs for the 4 pad cases:
      A) y_max=0.00492, turn=TURN1
      B) y_max=0.00092, turn=TURN1
      C) y_max=0.00492, turn=TURN2
      D) y_max=0.00092, turn=TURN2
    """
    cases = [
        (0.00492, TURN1),
        (0.00092, TURN1),
        (0.00492, TURN2),
        (0.00092, TURN2),
    ]
    paths = []
    for y_max, turn in cases:
        paths.append(export_rhf_table(y_max, turn, R, T))
    return paths

def export_time_position_power(
    y_max,
    turn_over_time,
    power=142.5,
    power_func=None,          # kept for compatibility; ignored
    map_2d_to_3d="x0z",
    pos_filename="constant/timeVsLaserPosition",
    pow_filename="constant/timeVsLaserPower",
    float_fmt_time="{:.6f}",
    float_fmt_coord=("{:.6f}", "{:.6f}", "{:.6f}"),
    float_fmt_power="{:.6f}",
    include_final_power_zero=True,
):
    """
    Export two OpenFOAM-style files with MINIMAL entries:
      Position: exactly 2 lines per vertical laser track (start & end).
      Power:    constant 'power' while scanning a track, 0 during turn-around pauses.

    Position file pattern (example):
      (
          (t_start_track0 (x y z))
          (t_end_track0   (x y z))
          (t_start_track1 (x y z))
          (t_end_track1   (x y z))
          ...
      )

    Power file pattern (events where value changes):
      (
          (t_start_track0 power)
          (t_end_track0   0)
          (t_start_track1 power)
          (t_end_track1   0)
          ...
          (t_end_last     0)   # optional (controlled by include_final_power_zero)
      )

    Notes
    -----
    - power_func retained but ignored (model is piecewise constant ON/OFF).
    - Tracks follow serpentine: even index goes y_min -> y_max; odd goes y_max -> y_min.
    - Turn-around pause duration = turn_over_time (laser OFF, position not recorded separately).
    - Mapping options:
         "xy0": (x, y, 0)
         "x0z": (x, 0, y)   (sample provided earlier)
         "0xy": (0, x, y)
    """
    # Basic geometric/time quantities
    track_length = abs(y_max - y_min)
    track_time   = track_length / v

    # Prepare directory
    constant_dir = outdir / "constant"
    constant_dir.mkdir(exist_ok=True)
    pos_path = outdir / pos_filename
    pow_path = outdir / pow_filename

    fx_fmt, fy_fmt, fz_fmt = float_fmt_coord
    p_fmt = float_fmt_power  # power formatter

    # Write position file (2 lines per track)
    with open(pos_path, "w") as fpos:
        fpos.write("(\n")
        for k in range(n_tracks):
            x = x_min + k * pitch
            # Serpentine direction
            if k % 2 == 0:  # up
                y_start, y_end = y_min, y_max
            else:           # down
                y_start, y_end = y_max, y_min

            t_start = k * (track_time + turn_over_time)
            t_end   = t_start + track_time

            # Map to 3D
            def map_point(xp, yp):
                if map_2d_to_3d == "xy0":
                    return xp, yp, 0.0
                elif map_2d_to_3d == "x0z":
                    return xp, 0.0, yp
                elif map_2d_to_3d == "0xy":
                    return 0.0, xp, yp
                else:
                    raise ValueError("Unsupported map_2d_to_3d option.")
            x_s, y_s, z_s = map_point(x, y_start)
            x_e, y_e, z_e = map_point(x, y_end)

            fpos.write(
                f"    ({float_fmt_time.format(t_start)}   "
                f"({fx_fmt.format(x_s)} {fy_fmt.format(y_s)} {fz_fmt.format(z_s)}))\n"
            )
            fpos.write(
                f"    ({float_fmt_time.format(t_end)}   "
                f"({fx_fmt.format(x_e)} {fy_fmt.format(y_e)} {fz_fmt.format(z_e)}))\n"
            )
        fpos.write(")\n")



    # Write power file (events at changes, with 1us offset to avoid repeated times)
    dt = 1e-6  # 1 microsecond offset
    with open(pow_path, "w") as fpow:
        fpow.write("(\n")
        for k in range(n_tracks):
            t_start = k * (track_time + turn_over_time)
            t_end   = t_start + track_time

            # Power ON at t_start
            fpow.write(f"    ({float_fmt_time.format(t_start)}   {p_fmt.format(power)})\n")
            # Power ON just before t_end
            fpow.write(f"    ({float_fmt_time.format(t_end - dt)}   {p_fmt.format(power)})\n")
            # Power OFF at t_end
            fpow.write(f"    ({float_fmt_time.format(t_end)}   {p_fmt.format(0.0)})\n")
            # Power OFF just before next t_start (if not last track)
            if k < n_tracks - 1:
                t_next_start = (k + 1) * (track_time + turn_over_time)
                fpow.write(f"    ({float_fmt_time.format(t_next_start - dt)}   {p_fmt.format(0.0)})\n")
        fpow.write(")\n")

    print(f"Saved: {pos_path}")
    print(f"Saved: {pow_path}")
    return pos_path, pow_path


# New function: single figure with 4 subplots for fixed R, T
def plot_all_cases_subplots(R=0.80e-3, T=2.0e-3, filename="rhf_all_cases.png"):
    """
    Plot a 2x2 grid of subplots for all 4 (y_max, turn_over_time) cases, using fixed R and T.
    """
    cases = [
        (0.00492, TURN1, "A: Large domain, fast turn"),
        (0.00092, TURN1, "B: Small domain, fast turn"),
        (0.00492, TURN2, "C: Large domain, slow turn"),
        (0.00092, TURN2, "D: Small domain, slow turn"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    cmap = plt.cm.viridis

    for ax, (y_max, turn_over_time, label) in zip(axes, cases):
        pts, times, all_track_pts, track_lengths, n1 = build_serpentine(y_max, turn_over_time)
        rhf = compute_rhf(pts, times, n1, R, T)

        vmin, vmax = float(np.min(rhf)), float(np.max(rhf))
        if vmax <= vmin:
            vmax = vmin + 1e-12
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        offset = 0
        for k, p in enumerate(all_track_pts):
            n = track_lengths[k]
            r = rhf[offset:offset + n]
            segs = np.stack([p[:-1], p[1:]], axis=1)
            seg_vals = 0.5 * (r[:-1] + r[1:])
            lc = LineCollection(segs, array=seg_vals, cmap=cmap, norm=norm, linewidths=2.0, zorder=2)
            ax.add_collection(lc)
            offset += n

        ax.scatter(pts[:, 0], pts[:, 1], c=rhf, cmap=cmap, norm=norm,
                   s=10, marker='s', edgecolors='none', zorder=3)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.2)
        ax.set_title(f"{label}\n$y_{{max}}$={y_max*1e3:.2f} mm, Turn={turn_over_time*1e3:.2f} ms")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    fig.suptitle(f"Residual Heat Factor, R = {R*1e3:.2f} mm, T = {T*1e3:.2f} ms", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    outpath = outdir / filename
    fig.savefig(outpath, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {outpath}")

# Call the new function to generate the single figure with 4 subplots
plot_all_cases_subplots(R=0.80e-3, T=2.0e-3, filename="rhf_all_cases.png")

# Print simulation duration analysis
print_simulation_durations()

# Generate the four RHF tables for R=800 µm, T=3 ms:
export_all_four_tables(R=0.80e-3, T=2.0e-3)

# Example usage (uncomment to generate minimal OpenFOAM-style inputs):
export_time_position_power(
    y_max=0.00492,
    turn_over_time=TURN1,
    power=285,
    map_2d_to_3d="x0z"
)
