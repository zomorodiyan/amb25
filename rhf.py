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


# =========================================================
# Generate FOUR figures:
#   A) y_max = 0.00492, turn_over_time = 0.00075 s
#   B) y_max = 0.00092, turn_over_time = 0.00075 s
#   C) y_max = 0.00492, turn_over_time = 0.00500 s
#   D) y_max = 0.00092, turn_over_time = 0.00500 s
# =========================================================
draw_grid_figure(y_max=0.00492, turn_over_time=TURN1, filename="RHF_5x5_turn0p75.png")
draw_grid_figure(y_max=0.00092, turn_over_time=TURN1, filename="RHF_5x1_turn0p75.png")
draw_grid_figure(y_max=0.00492, turn_over_time=TURN2, filename="RHF_5x5_turn5p0.png")
draw_grid_figure(y_max=0.00092, turn_over_time=TURN2, filename="RHF_5x1_turn5p0.png")

