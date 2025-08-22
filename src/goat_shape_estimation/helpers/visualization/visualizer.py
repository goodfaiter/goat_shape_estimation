import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D

# matplotlib.use('qtagg')

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
    vectors *= 0.1

    # RGB colors corresponding to each vector
    colors = ['red', 'green', 'blue', 'black']
    labels = ['X', 'Y', 'Z', 'G']

    # Plot each vector
    for vec, color, label in zip(vectors, colors, labels):
        ax.quiver(*origin, *vec, color=color, arrow_length_ratio=0.1, linewidth=2, label=label + '-axis')
    
    # Minimap styling
    ax.set_xticks(np.linspace(-0.5, 0.5, 5))
    ax.set_yticks(np.linspace(-0.5, 0.5, 5))
    ax.set_zticks(np.linspace(-0.5, 0.5, 5))
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

def visualize_3d_spline_minimal(points_set, down_vec):
    """
    Minimal 3D visualization of spline fitting through points.
    No legends, background, or axis ticks.
    """
    fig = plt.figure(figsize=(6, 6), facecolor='none')
    ax = fig.add_subplot(111, projection='3d')
    
    # Remove background and axis panes
    ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.set_facecolor('none')
    
    # Make axis lines invisible
    ax.xaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.yaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    ax.zaxis.line.set_color((0.0, 0.0, 0.0, 0.0))
    
    spline_colors = ['b-', 'y-']
    
    # Plot splines and points
    for i, points in enumerate(points_set):
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        z = [p[2] for p in points]
        
        # Create spline interpolation
        t = np.linspace(0, 1, len(x))
        t_smooth = np.linspace(0, 1, 300)
        
        cs_x = CubicSpline(t, x, bc_type='natural')
        cs_y = CubicSpline(t, y, bc_type='natural')
        cs_z = CubicSpline(t, z, bc_type='natural')
        
        # Generate smooth curve
        x_smooth = cs_x(t_smooth)
        y_smooth = cs_y(t_smooth)
        z_smooth = cs_z(t_smooth)
        
        # Plot spline curve and points
        ax.plot(x_smooth, y_smooth, z_smooth, spline_colors[i], linewidth=2)
        ax.scatter(x, y, z, c='r', s=50)
    
    # Plot coordinate vectors
    origin = [0, 0, 0]
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], down_vec]) * 0.1
    colors = ['red', 'green', 'blue', 'black']
    
    for vec, color in zip(vectors, colors):
        ax.quiver(*origin, *vec, color=color, arrow_length_ratio=0.1, linewidth=2)
    
    # Remove axis ticks and grid
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    plt.axis('equal')
    
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

def plot_trajectories(data, labels=None, title="2D Trajectories", xlabel="Travel Distance [m]", ylabel="Deviation from Straight Line [m]", ylim = None):
    """
    Plot multiple 2D trajectories on a single graph.
    
    Parameters:
    -----------
    M : numpy array of shape [M, T, 2]
        Array containing M trajectories, each with T time steps and 2 dimensions (x, y)
    labels : list of str, optional
        Labels for each trajectory (length M)
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    """
    
    M, _, _ = data.shape
    
    plt.figure(figsize=(10, 8))
    
    # Create a colormap for different trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, M))
    
    for i in range(M):
        trajectory = data[i]
        x_coords = trajectory[:, 0]
        y_coords = trajectory[:, 1]
        
        # Plot the trajectory
        if labels:
            plt.plot(x_coords, y_coords, color=colors[i], label=labels[i], linewidth=2, alpha=0.8)
        else:
            plt.plot(x_coords, y_coords, color=colors[i], linewidth=2, alpha=0.8)
        
        # Mark the start and end points
        plt.scatter(x_coords[0], y_coords[0], color=colors[i], s=100, marker='o', edgecolors='black')
        plt.scatter(x_coords[-1], y_coords[-1], color=colors[i], s=100, marker='s', edgecolors='black')
    
    plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True, alpha=0.3)

    # Larger tick labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    
    if labels:
        plt.legend(loc='best', fontsize=16)
    
    # Equal aspect ratio for proper spatial representation
    # plt.axis('equal')
    plt.tight_layout()
    
    # Show the plot
    plt.show()

def plot_time_series(data, labels=None, title="Yaw Rate Tracking", xlabel="Time [s]", ylabel="Yaw Rate [rad/s]", ylim=None):
    """
    Plot multiple 2D trajectories on a single graph.
    
    Parameters:
    -----------
    M : numpy array of shape [M, T, 2]
        Array containing M trajectories, each with T time steps and 2 dimensions (x, y)
    labels : list of str, optional
        Labels for each trajectory (length M)
    title : str, optional
        Plot title
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    """
    
    M = len(data)
    T = len(data[0])

    time_axis = np.arange(T) * 0.05 # 20Hz
    
    plt.figure(figsize=(10, 8))
    
    # Create a colormap for different trajectories
    colors = plt.cm.viridis(np.linspace(0, 1, M))
    
    for i in range(M):
        trajectory = data[i]
        x_coords = time_axis
        y_coords = trajectory[:]
        
        # Plot the trajectory
        if labels:
            plt.plot(x_coords, y_coords, color=colors[i], label=labels[i], linewidth=2, alpha=0.8)
        else:
            plt.plot(x_coords, y_coords, color=colors[i], linewidth=2, alpha=0.8)
        
        # Mark the start and end points
        plt.scatter(x_coords[0], y_coords[0], color=colors[i], s=100, marker='o', edgecolors='black')
        plt.scatter(x_coords[-1], y_coords[-1], color=colors[i], s=100, marker='s', edgecolors='black')
    
    if title:
        plt.title(title, fontsize=20)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.grid(True, alpha=0.3)

    # Larger tick labels
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    
    if labels:
        plt.legend(loc='best', fontsize=16)
    
    # Equal aspect ratio for proper spatial representation
    # plt.axis('equal')
    plt.tight_layout()
    
    # Show the plot
    plt.show()