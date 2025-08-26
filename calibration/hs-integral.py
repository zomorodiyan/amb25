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
from mpl_toolkits.mplot3d import Axes3D

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

def calculate_integral_monte_carlo(effectiveRadius, laserHeight, taperLength, laserRadius, n_samples=1000000):
    """Calculate integral using Monte Carlo method."""
    s_samples = np.random.uniform(0, laserHeight, n_samples)
    integral_sum = 0.0
    valid_samples = 0

    for s in s_samples:
        max_r = get_radius_at_position(s, effectiveRadius, laserHeight, taperLength)
        if max_r > 0:
            r = np.random.uniform(0, max_r)
            theta = np.random.uniform(0, 2*np.pi)

            # Check if point is above y=0 surface
            x, y, z = to_cartesian(r, theta, s)
            if y >= 0:
                sigma = laserRadius / 2.0
                gaussian_val = gaussian_function(r, sigma)
                volume_element = max_r**2 * laserHeight * 2*np.pi
                integral_sum += gaussian_val * r * volume_element
                valid_samples += 1

    return integral_sum / valid_samples if valid_samples > 0 else 0.0

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

def main():
    """Calculate the integral with example parameters."""
    # Parameters
    effectiveRadius = 0.000050
    laserHeight = 0.000140
    taperLength = 0.000100
    laserRadius = 0.000036

    print("3D Integral for Tilted Tapered Cylinder")
    print("=" * 40)
    print(f"Effective Radius: {effectiveRadius}")
    print(f"Laser Height: {laserHeight}")
    print(f"Taper Length: {taperLength}")
    print(f"Laser Radius: {laserRadius}")
    print(f"Gaussian σ: {laserRadius/2}")
    print(f"Tilt Angle: {TILT_ANGLE}°")
    print()

    # Monte Carlo integration
    print("Calculating integral using Monte Carlo method...")
    result = calculate_integral_monte_carlo(effectiveRadius, laserHeight, taperLength, laserRadius)
    print(f"Result: {result:.6f}")

    # Visualization
    print("\nGenerating plots...")
    visualize_object(effectiveRadius, laserHeight, taperLength, laserRadius)

if __name__ == "__main__":
    main()
