#!/usr/bin/env python3
"""
Calculate 3D integral for a tilted cylindrical object with tapered "pencil" shape.

The cylinder is tilted 5° to the y=0 surface, cut off by it, and tapers to a point.
Uses a radial Gaussian function with sigma = laserRadius/2.

Parameters:
- effectiveRadius: maximum radius
- laserHeight: total length along tilted axis
- taperLength: length of tapered section
- laserRadius: Gaussian parameter (sigma = laserRadius/2)
"""

import numpy as np
import matplotlib.pyplot as plt

TILT_ANGLE = 5.0  # degrees

def gaussian_function(r, sigma):
    """Gaussian function that diminishes with radial distance."""
    return np.exp(-r**2 / (2 * sigma**2))

def get_radius_at_position(s, effectiveRadius, laserHeight, taperLength):
    """Calculate radius at position s along the tilted axis."""
    if s < 0 or s > laserHeight:
        return 0.0

    constant_length = laserHeight - taperLength
    if s <= constant_length:
        return effectiveRadius
    else:
        # Linear taper to zero
        taper_progress = (s - constant_length) / taperLength
        return effectiveRadius * (1 - taper_progress)

def to_cartesian(r, theta, s):
    """Convert tilted cylindrical coordinates to Cartesian."""
    tilt_rad = np.radians(TILT_ANGLE)
    x = r * np.cos(theta)
    y = s * np.cos(tilt_rad) - r * np.sin(theta) * np.sin(tilt_rad)
    z = s * np.sin(tilt_rad) + r * np.sin(theta) * np.cos(tilt_rad)
    return x, y, z

def integrand(r, theta, s, effectiveRadius, laserHeight, taperLength, laserRadius):
    """Calculate integrand value at (r, theta, s) in tilted cylindrical coordinates."""
    # Check if point is within object bounds
    max_radius = get_radius_at_position(s, effectiveRadius, laserHeight, taperLength)
    if r > max_radius:
        return 0.0

    # Check if point is above y=0 surface (cut-off constraint)
    x, y, z = to_cartesian(r, theta, s)
    if y < 0:
        return 0.0

    # Calculate Gaussian value
    sigma = laserRadius / 2.0
    gaussian_val = gaussian_function(r, sigma)

    # Return with cylindrical Jacobian
    return gaussian_val * r

def calculate_integral_loops(effectiveRadius, laserHeight, taperLength, laserRadius, n_s=100, n_r=50, n_theta=100):
    """Calculate integral using simple nested loops."""
    sigma = laserRadius / 2.0
    integral_sum = 0.0

    # Create integration grids
    s_vals = np.linspace(-laserRadius, laserHeight, n_s)
    theta_vals = np.linspace(0, 2*np.pi, n_theta)

    # Grid spacing for integration
    ds = laserHeight / n_s
    dtheta = 2*np.pi / n_theta

    for s in s_vals:
        max_r = get_radius_at_position(s, effectiveRadius, laserHeight, taperLength)
        if max_r > 0:
            # Create r grid for this s
            r_vals = np.linspace(0, max_r, n_r)
            dr = max_r / n_r

            for r in r_vals:
                for theta in theta_vals:
                    # Check if point is above y=0 surface
                    x, y, z = to_cartesian(r, theta, s)
                    if y >= 0:
                        # Calculate integrand: Gaussian * r (cylindrical Jacobian)
                        gaussian_val = gaussian_function(r, sigma)
                        integrand_val = gaussian_val * r

                        # Add to integral with volume element
                        integral_sum += integrand_val * dr * dtheta * ds

    return integral_sum

def visualize_object(effectiveRadius, laserHeight, taperLength, laserRadius):
    """Visualize the tilted object and Gaussian function."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Radius profile
    s_vals = np.linspace(0, laserHeight, 100)
    radii = [get_radius_at_position(s, effectiveRadius, laserHeight, taperLength) for s in s_vals]
    ax1.plot(s_vals, radii, 'b-', linewidth=2, label='Radius profile')
    ax1.plot(s_vals, [-r for r in radii], 'b-', linewidth=2)
    ax1.set_xlabel('Position along tilted axis')
    ax1.set_ylabel('Radius')
    ax1.set_title('Object Cross-section')
    ax1.grid(True)
    ax1.legend()

    # Plot 2: Gaussian function
    sigma = laserRadius / 2.0
    r_vals = np.linspace(0, effectiveRadius * 1.5, 100)
    gaussian_vals = [gaussian_function(r, sigma) for r in r_vals]
    ax2.plot(r_vals, gaussian_vals, 'r-', linewidth=2, label=f'Gaussian (σ={sigma:.3f})')
    ax2.axvline(x=sigma, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Radial distance')
    ax2.set_ylabel('Gaussian value')
    ax2.set_title('Gaussian Function')
    ax2.grid(True)
    ax2.legend()

    # Plot 3: 3D object
    ax3 = plt.subplot(133, projection='3d')
    theta = np.linspace(0, 2*np.pi, 20)
    s = np.linspace(0, laserHeight, 50)
    S, Theta = np.meshgrid(s, theta)

    # Calculate radius and convert to Cartesian
    R = np.array([[get_radius_at_position(s_val, effectiveRadius, laserHeight, taperLength)
                   for s_val in s] for _ in theta])

    X, Y, Z = [], [], []
    for i, s_val in enumerate(s):
        for j, theta_val in enumerate(theta):
            r_val = R[j, i]
            x, y, z = to_cartesian(r_val, theta_val, s_val)
            if y >= 0:  # Only plot above y=0
                X.append(x)
                Y.append(y)
                Z.append(z)

    ax3.scatter(X, Y, Z, alpha=0.6, s=1)
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title(f'{TILT_ANGLE}° Tilted Object')

    plt.tight_layout()
    plt.show()

def check_geometry(effectiveRadius, laserHeight, taperLength):
    """Check if the tilted geometry makes physical sense."""
    tilt_rad = np.radians(TILT_ANGLE)

    # At s=0 (start), what's the y-coordinate for points at max radius?
    s = 0
    r = effectiveRadius

    # Check the most extreme points (theta where sin(theta) = ±1)
    y_min_at_start = s * np.cos(tilt_rad) - r * 1.0 * np.sin(tilt_rad)  # sin(theta) = 1
    y_max_at_start = s * np.cos(tilt_rad) - r * (-1.0) * np.sin(tilt_rad)  # sin(theta) = -1

    # At s=laserHeight (end), what's the y-coordinate?
    s = laserHeight
    r = get_radius_at_position(s, effectiveRadius, laserHeight, taperLength)
    y_at_end = s * np.cos(tilt_rad)  # r=0 at the tip

    print("Geometry Check:")
    print(f"  At start (s=0): y ranges from {y_min_at_start:.6e} to {y_max_at_start:.6e}")
    print(f"  At end (s={laserHeight:.6e}): y = {y_at_end:.6e}")
    print(f"  Tilt creates y-offset of {effectiveRadius * np.sin(tilt_rad):.6e} at max radius")

    # Check if any volume exists above y=0
    if y_max_at_start <= 0:
        print("  WARNING: Entire cylinder may be below y=0 plane!")
        return False
    elif y_min_at_start < 0:
        print("  Some volume is cut off by y=0 plane (this is expected)")
        return True
    else:
        print("  All volume is above y=0 plane")
        return True

def main():
    """Calculate the integral with example parameters."""
    # Parameters
    effectiveRadius = 0.000050
    laserHeight = 0.000140
    taperLength = 0.000100
    laserRadius = 0.000036
    # effectiveRadius = 0.000100
    # laserHeight = 0.000280
    # taperLength = 0.000200
    # laserRadius = 0.000072

    print("3D Integral for Tilted Tapered Cylinder")
    print("=" * 40)
    print(f"Effective Radius: {effectiveRadius}")
    print(f"Laser Height: {laserHeight}")
    print(f"Taper Length: {taperLength}")
    print(f"Laser Radius: {laserRadius}")
    print(f"Gaussian σ: {laserRadius/2}")
    print(f"Tilt Angle: {TILT_ANGLE}°")
    print()

    # Check geometry first
    check_geometry(effectiveRadius, laserHeight, taperLength)
    print()

    # Calculate integral using loops
    print("Calculating integral using nested loops...")
    result = calculate_integral_loops(effectiveRadius, laserHeight, taperLength, laserRadius)
    print(f"Result: {result:.6e}")

    # Visualization
    print("\nGenerating plots...")
    visualize_object(effectiveRadius, laserHeight, taperLength, laserRadius)

if __name__ == "__main__":
    main()
