import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D

def visualize_3d_spline(points_set, down_vec, text, filename):
    """
    Minimal 3D visualization of spline fitting through 8 points.
    
    Args:
        points: List of (x, y, z) coordinates of the 8 points
    """
        
    # Create minimap-style 3D plot
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')

    spline_colors = ['b-', 'y-']
    
    # Unpack points
    for i, points in enumerate(points_set):
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
        
        # Plot spline curve
        ax.plot(x_smooth, y_smooth, z_smooth, spline_colors[i], linewidth=2, label='Spline Fit')
    
        # Plot original points
        ax.scatter(x, y, z, c='r', s=50, label='Control Points')

    # Origin point
    origin = [0, 0, 0]

    # Unit vectors for x, y, z axes
    vectors = np.array([[1, 0, 0],  # x-axis (red)
                        [0, 1, 0],  # y-axis (green)
                        [0, 0, 1],  # z-axis (blue)
                        down_vec])  # gravity (back)
    vectors *= 100

    # RGB colors corresponding to each vector
    colors = ['red', 'green', 'blue', 'black']
    labels = ['X', 'Y', 'Z', 'G']

    # Plot each vector
    for vec, color, label in zip(vectors, colors, labels):
        ax.quiver(*origin, *vec, color=color, arrow_length_ratio=0.1, linewidth=2, label=label + '-axis')
    
    # Minimap styling
    ax.set_xticks(np.linspace(-500, 500, 5))
    ax.set_yticks(np.linspace(-500, 500, 5))
    ax.set_zticks(np.linspace(-500, 500, 5))
    ax.tick_params(labelsize=8)
    ax.set_title('Spline Fitting ' + text, fontsize=10)
    ax.legend(fontsize=8)
    
    # Save to file
    # plt.tight_layout()
    # plt.savefig(filename, dpi=150, bbox_inches='tight')
    # plt.close()  # Close the figure to free memory
    
    # Show
    plt.tight_layout()
    plt.show()


def plot_velocity_comparison(estimated_vel, ground_truth_vel, time_axis=None, title="Velocity Comparison", 
                             xlabel="Time", ylabel="Velocity", legend_loc="best", figsize=(10, 6)):
    """
    Plot estimated velocity against ground truth velocity.
    
    Parameters:
    - estimated_vel: numpy array of estimated velocities
    - ground_truth_vel: numpy array of ground truth velocities
    - time_axis: optional time axis values (if None, will use array indices)
    - title: plot title
    - xlabel: label for x-axis
    - ylabel: label for y-axis
    - legend_loc: location of legend
    - figsize: figure size
    """
    
    # Create time axis if not provided
    if time_axis is None:
        time_axis = np.arange(len(estimated_vel)) * 0.05 # 20Hz
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    # Plot both velocity curves
    plt.plot(time_axis, estimated_vel, 'b-', label='Estimated Velocity', alpha=0.7)
    plt.plot(time_axis, ground_truth_vel, 'r--', label='Ground Truth Velocity', alpha=0.7)
    
    # Add plot elements
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc=legend_loc)
    
    # Show the plot
    plt.tight_layout()
    plt.show()