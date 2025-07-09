import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_spline(points):
    """
    Minimal 3D visualization of spline fitting through 8 points.
    
    Args:
        points: List of (x, y, z) coordinates of the 8 points
    """
    # Unpack points
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    z = [p[2] for p in points]
    
    # Create parameter for interpolation (arc length approximation)
    t = np.linspace(0, 1, len(x))
    t_smooth = np.linspace(0, 1, 300)
    
    # Create cubic spline interpolation for each dimension
    cs_x = CubicSpline(t, x, bc_type='natural')
    cs_y = CubicSpline(t, y, bc_type='natural')
    cs_z = CubicSpline(t, z, bc_type='natural')
    
    # Generate smooth curve
    x_smooth = cs_x(t_smooth)
    y_smooth = cs_y(t_smooth)
    z_smooth = cs_z(t_smooth)
    
    # Create minimap-style 3D plot
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot spline curve
    ax.plot(x_smooth, y_smooth, z_smooth, 'b-', linewidth=2, label='Spline Fit')
    
    # Plot original points
    ax.scatter(x, y, z, c='r', s=50, label='Control Points')
    
    # Minimap styling
    ax.set_xticks(np.linspace(min(x), max(x), 5))
    ax.set_yticks(np.linspace(min(y), max(y), 5))
    ax.set_zticks(np.linspace(min(z), max(z), 5))
    ax.tick_params(labelsize=8)
    ax.set_title('3D Minimap: Spline Fitting through 8 Points', fontsize=10)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.show()

# Example usage with 8 random 3D points
if __name__ == "__main__":
    np.random.seed(42)
    points = np.column_stack((
        np.sort(np.random.uniform(0, 10, 8)),
        np.random.uniform(0, 10, 8),
        np.random.uniform(0, 10, 8)
    ))
    
    visualize_3d_spline(points)