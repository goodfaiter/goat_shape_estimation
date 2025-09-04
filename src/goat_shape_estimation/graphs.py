import torch
import pandas as pd
import numpy as np
from helpers.data_processer import DataProcessorGoat, ema_2d_optimized
from helpers.visualization.visualizer import visualize_3d_spline, plot_velocity_comparison, plot_trajectories, plot_time_series, visualize_3d_spline_minimal, plot_time_series_two_axis
import roma

NUM_POINTS = 12
NUM_POINTS_PER_CIRCLE = 9
INDEX_POINTS = 0
INDEX_GRAVITY = INDEX_POINTS + NUM_POINTS * 3
INDEX_VELOCITY = INDEX_GRAVITY + 3
NUM_OUTPUT = INDEX_VELOCITY + 6
INDICES_RING_ONE = [0, 1, 2, 3, 4, 5, 6, 7, 0]
INDICES_RING_TWO = [0, 8, 2, 9, 4, 10, 6, 11, 0]
DEVICE = torch.device("cpu")
# MODEL_PATH = "/workspace/data/output/2025_08_22_11_19_04/2025_08_22_11_19_04_best_lstm_model.pt"
# MODEL_PATH = "/workspace/data/output/2025_08_26_15_22_18/2025_08_26_15_22_18_100.pt"
MODEL_PATH = "/workspace/data/output/2025_08_26_16_08_42/2025_08_26_16_08_42_100.pt" # current best on overleaf
# MODEL_PATH = "/workspace/data/output/2025_08_29_11_00_37/2025_08_29_11_00_37_100.pt" # new best
# MODEL_PATH = "/workspace/data/output/latest_lstm_model.pt"

np.set_printoptions(precision=1)

def simulate_data_with_model(model_path, data):
    device = device = torch.device("cpu")
    num_data = data["/imu/data/orientation_w"].size
    traced_model = torch.jit.load(model_path, map_location="cpu")
    data_processor_goat = DataProcessorGoat(device)
    inputs = data_processor_goat.process_input_data(data)
    outputs = np.zeros([num_data, NUM_OUTPUT])
    for i in range(num_data):
        outputs[i, :] = traced_model(inputs[i, :].unsqueeze(0).unsqueeze(0))[0, -1, :].numpy()

    for j in range(NUM_POINTS * 3):
        data[f"/frame_points/data_{j}"] = outputs[:, INDEX_POINTS + j]
    data[f"/gravity_vector/data_{0}"] = outputs[:, INDEX_GRAVITY + 0]
    data[f"/gravity_vector/data_{1}"] = outputs[:, INDEX_GRAVITY + 1]
    data[f"/gravity_vector/data_{2}"] = outputs[:, INDEX_GRAVITY + 2]
    data[f"/estimated_twist/linear_x"] = outputs[:, INDEX_VELOCITY + 0]
    data[f"/estimated_twist/linear_y"] = outputs[:, INDEX_VELOCITY + 1]
    data[f"/estimated_twist/linear_z"] = outputs[:, INDEX_VELOCITY + 2]
    data[f"/estimated_twist/angular_x"] = outputs[:, INDEX_VELOCITY + 3]
    data[f"/estimated_twist/angular_y"] = outputs[:, INDEX_VELOCITY + 4]
    data[f"/estimated_twist/angular_z"] = outputs[:, INDEX_VELOCITY + 5]

    frame_points = outputs[:, INDEX_POINTS:INDEX_GRAVITY].reshape(-1, NUM_POINTS, 3)
    avg_distance = np.mean(frame_points[:, [1, 3, 8, 9], :], axis = 1) - np.mean(frame_points[:,[5, 7, 10, 11], :], axis=1)
    frame_width = np.linalg.norm(avg_distance, axis = 1) - 0.1 # 0.1 comes from the 5 [cm] x 2 marker to wheel offset
    data['/estimated_width/data'] = frame_width

def load_model_outputs(data):
    num_data = data["/imu/data/orientation_w"].size
    output = torch.zeros([NUM_OUTPUT], dtype=torch.float, device=DEVICE)
    outputs = torch.zeros([num_data, NUM_OUTPUT], dtype=torch.float, device=DEVICE)
    for i in range(num_data):
        for j in range(NUM_POINTS * 3):
            output[INDEX_POINTS + j] = data[f"/frame_points/data_{j}"][i]
        output[INDEX_GRAVITY + 0] = data[f"/gravity_vector/data_{0}"][i]
        output[INDEX_GRAVITY + 1] = data[f"/gravity_vector/data_{1}"][i]
        output[INDEX_GRAVITY + 2] = data[f"/gravity_vector/data_{2}"][i]
        output[INDEX_VELOCITY + 0] = data[f"/estimated_twist/linear_x"][i]
        output[INDEX_VELOCITY + 1] = data[f"/estimated_twist/linear_y"][i]
        output[INDEX_VELOCITY + 2] = data[f"/estimated_twist/linear_z"][i]
        output[INDEX_VELOCITY + 3] = data[f"/estimated_twist/angular_x"][i]
        output[INDEX_VELOCITY + 4] = data[f"/estimated_twist/angular_y"][i]
        output[INDEX_VELOCITY + 5] = data[f"/estimated_twist/angular_z"][i]
        outputs[i, :] = output
    return outputs

def test_graphs(data):
    data_processor_goat = DataProcessorGoat(DEVICE)
    inputs = data_processor_goat.process_input_data(data)
    targets = data_processor_goat.process_output_data(data)

    num_data = inputs.shape[0]
    estimated_points = torch.zeros([NUM_POINTS, 3], dtype=torch.float, device=DEVICE)
    estimated_points_to_visualize = torch.zeros([2, NUM_POINTS_PER_CIRCLE, 3], dtype=torch.float, device=DEVICE)
    estimated_gravity = torch.zeros([3], dtype=torch.float, device=DEVICE)
    estimated_velocities = torch.zeros([inputs.shape[0], 6], dtype=torch.float, device=DEVICE)
    output = torch.zeros([NUM_OUTPUT], dtype=torch.float, device=DEVICE)

    outputs = load_model_outputs(data)

    for i in range(num_data):
        output = outputs[i]

        # Points and Gravity
        if i % 100 == 0:
            estimated_points = output[INDEX_POINTS:INDEX_GRAVITY].view(NUM_POINTS, 3)
            estimated_gravity = output[INDEX_GRAVITY:INDEX_VELOCITY]
            estimated_points_to_visualize[0, :, :] = estimated_points[INDICES_RING_ONE, :]
            estimated_points_to_visualize[1, :, :] = estimated_points[INDICES_RING_TWO, :]

            visualize_3d_spline(
                estimated_points_to_visualize.detach().numpy(),
                estimated_gravity.detach().numpy(),
                text=f" with Tendon length {inputs[i, -2]:.2f}[m] and {inputs[i, -1]:.2f}[m], timestep {i*0.05:.2f}[s]",
                filename="data/output/" + f"spline_{i}",
            )

        estimated_velocities[i, :] = output[INDEX_VELOCITY:]

    ### Plot velocities
    plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 0], targets.detach().numpy()[:, INDEX_VELOCITY + 0], title="Linear X")
    # plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 1], targets.detach().numpy()[:, INDEX_VELOCITY + 1], title="Linear Y")
    # plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 2], targets.detach().numpy()[:, INDEX_VELOCITY + 2], title="Linear Z")
    # plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 3], targets.detach().numpy()[:, INDEX_VELOCITY + 3], title="Angular X")
    # plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 4], targets.detach().numpy()[:, INDEX_VELOCITY + 4], title="Angular Y")
    plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 5], targets.detach().numpy()[:, INDEX_VELOCITY + 5], title="Angular Z")

def top_down_view(datas):
    """ Plot mocap trajecotires side by side to compare open and closed loop"""
    min_input = np.inf
    max_input = 0
    for data in datas:
        data_processor = DataProcessorGoat(DEVICE)
        inputs = data_processor.process_input_data(data)
        min_input = min(min_input, inputs.shape[0])
        max_input = max(max_input, inputs.shape[0])

    drive_pos_in_world = np.zeros([len(datas), max_input, 2])

    for i, data in enumerate(datas):
        data_processor = DataProcessorGoat(DEVICE)
        inputs = data_processor.process_input_data(data)
        num_data = inputs.shape[0]
        targets = data_processor.process_output_data(data)
        drive_pos_in_world[i, :inputs.shape[0], :] = data_processor.drive_pos_in_world[:, :2].detach().numpy()
        drive_pos_in_world[i, :, :] -= drive_pos_in_world[i, 0, :]
        drive_rotmat_drive_to_world = data_processor.drive_rotmat_drive_to_world[0, :2, :2].detach().numpy()
        drive_rot_mat_world_to_drive = drive_rotmat_drive_to_world.transpose()
        for t in range(num_data):
            drive_pos_in_world[i, t, :] = np.matmul(drive_rot_mat_world_to_drive, drive_pos_in_world[i, t, :])
    plot_trajectories(drive_pos_in_world[:, :min_input, :], title=None, 
                      labels=['Open Loop 1', 'Open Loop 2', 'Open Loop 3', 'Closed Loop 1', 'Closed Loop 2', 'Closed Loop 3'], 
                      linestyle=[None, None, None, "--", "--", "--"],
                      ylim=[-1.25, 0.5], filename="top_down_view")

def exponential_moving_average(data, alpha=0.3):
    """
    Apply exponential moving average filter to a [T, D] array.
    
    Parameters:
    -----------
    data : numpy.ndarray
        Input array of shape [T, D] where T is time steps, D is dimensions
    alpha : float, optional
        Smoothing factor (0 < alpha <= 1). Higher alpha = less smoothing
    
    Returns:
    --------
    smoothed : numpy.ndarray
        Smoothed array of same shape as input
    """
    if len(data.shape) == 1:
        T = data.shape[0]
        D = 1
    else:
        T, D = data.shape
    smoothed = np.zeros_like(data)
    
    # Initialize with first value
    smoothed[0] = data[0]
    
    for t in range(1, T):
        # Standard EMA formula
        smoothed[t] = smoothed[t-1] * (1 - alpha) + data[t] * alpha
    
    return smoothed

def yaw_tracking(datas, dur, labels = None, filename = ""):
    series = []
    for data, to_plot_dict in datas:
        data_processor = DataProcessorGoat(DEVICE)
        data_processor.process_input_data(data)
        data_processor.process_output_data(data)
        # Do this or ignore to use rosbag data.
        # simulate_data_with_model(MODEL_PATH, data)
        for name, start in to_plot_dict.items():
            if name == '/imu/data/angular_velocity_z':
                data[name] *= -1.0
                smooth_imu = exponential_moving_average(data[name], alpha=0.3)
                series.append(smooth_imu[start:start + dur])
            else:
                series.append(data[name][start:start + dur])
    plot_time_series(series, labels=labels, xlabel="Time [s]", ylabel="Yaw Rate [rad/s]", ylim=[-0.1, 1.5], filename=filename)


def x_tracking(datas, dur, labels):
    rates = []
    for data, to_plot_dict in datas:
        data_processor = DataProcessorGoat(DEVICE)
        data_processor.process_input_data(data)
        data_processor.process_output_data(data)
        # Do this or ignore to use rosbag data.
        # simulate_data_with_model(MODEL_PATH, data)
        for name, start in to_plot_dict.items():
            rates.append(data[name][start:start + dur])
    plot_time_series(rates, labels=labels, xlabel="Time [s]", ylabel="Velocity [m/s]", ylim=[-0.1, 0.5], filename="x_velocity_estimation")

def plot_minimal_shape(data):
    data_processor = DataProcessorGoat(DEVICE)
    inputs = data_processor.process_input_data(data)
    targets = data_processor.process_output_data(data)
    # Do this or ignore to use rosbag data.
    # simulate_data_with_model(MODEL_PATH, data)
    num_data = inputs.shape[0]
    ground_truth_points = np.zeros([NUM_POINTS, 3])
    estimated_points = np.zeros([NUM_POINTS, 3])
    estimated_points_to_visualize = np.zeros([2, NUM_POINTS_PER_CIRCLE, 3])
    estimated_gravity = np.zeros([3])
    for t in range(num_data):
        for i in range(NUM_POINTS):
            for j in range(3):
                estimated_points[i][j] = data[f'/frame_points/data_{i*3 + j}'][t]
                ground_truth_points = targets[t, :INDEX_GRAVITY].view(NUM_POINTS, 3).detach().numpy()
                estimated_gravity[j] = data[f'/gravity_vector/data_{j}'][t]
        
        # if t > 600 and t % 10 == 0:
        if t % 50 == 0:
            text=f"Tendon length {inputs[t, -2]:.2f}[m] and {inputs[t, -1]:.2f}[m], index {t}, timestep {t*0.05:.2f}[s]"
            print(text)
            estimated_points_to_visualize[0, :, :] = estimated_points[INDICES_RING_ONE, :]
            estimated_points_to_visualize[1, :, :] = estimated_points[INDICES_RING_TWO, :]
            # estimated_points_to_visualize[0, :, :] = ground_truth_points[INDICES_RING_ONE, :]
            # estimated_points_to_visualize[1, :, :] = ground_truth_points[INDICES_RING_TWO, :]

            visualize_3d_spline_minimal(
                estimated_points_to_visualize,
                estimated_gravity,
            )

def calculate_rmse_reconstruction(data, start=0, end = None):
    data_processor = DataProcessorGoat(DEVICE)
    inputs = data_processor.process_input_data(data)
    targets = data_processor.process_output_data(data)
    num_data = inputs.shape[0]
    if not end:
        end = num_data
    # Do this or ignore to use rosbag data.
    # simulate_data_with_model(MODEL_PATH, data)
    ground_truth_points = np.zeros([end - start, NUM_POINTS, 3])
    estimated_points = np.zeros([end - start, NUM_POINTS, 3])
    for t in range(start, end):
        for i in range(NUM_POINTS):
            for j in range(3):
                estimated_points[t - start][i][j] = data[f'/frame_points/data_{i*3 + j}'].values[t]
                ground_truth_points[t-start] = targets[t, :INDEX_GRAVITY].view(NUM_POINTS, 3).detach().numpy()
    dist = np.linalg.norm(estimated_points - ground_truth_points, axis=2)

    # this
    # rmse_per_timestep = np.sqrt(np.mean((dist)**2, axis=1))
    # rmse = np.mean(rmse_per_timestep, axis=0)
    # or
    rmse = np.sqrt(np.mean((dist)**2))
    return rmse

def calculate_rmse_twist_reconstruction(data, start = 0, end = None):
    data_processor = DataProcessorGoat(DEVICE)
    inputs = data_processor.process_input_data(data)
    targets = data_processor.process_output_data(data)
    num_data = inputs.shape[0]
    if end is None:
        end = num_data
    # Do this or ignore to use rosbag data.
    # simulate_data_with_model(MODEL_PATH, data)
    estimated_twist = np.zeros([end - start, 6])
    ground_truth_twist = np.zeros([end - start, 6])
    estimated_twist[:, 0] = data['/estimated_twist/linear_x'].values[start:end] * 1000
    estimated_twist[:, 1] = data['/estimated_twist/linear_y'].values[start:end] * 1000
    estimated_twist[:, 2] = data['/estimated_twist/linear_z'].values[start:end] * 1000
    estimated_twist[:, 3] = data['/estimated_twist/angular_x'].values[start:end] * 180 / np.pi
    estimated_twist[:, 4] = data['/estimated_twist/angular_y'].values[start:end] * 180 / np.pi
    estimated_twist[:, 5] = data['/estimated_twist/angular_z'].values[start:end] * 180 / np.pi
    ground_truth_twist[:, 0] = data['/ground_truth/twist_linear_x'].values[start:end] * 1000
    ground_truth_twist[:, 1] = data['/ground_truth/twist_linear_y'].values[start:end] * 1000
    ground_truth_twist[:, 2] = data['/ground_truth/twist_linear_z'].values[start:end] * 1000
    ground_truth_twist[:, 3] = data['/ground_truth/twist_angular_x'].values[start:end] * 180 / np.pi
    ground_truth_twist[:, 4] = data['/ground_truth/twist_angular_y'].values[start:end] * 180 / np.pi
    ground_truth_twist[:, 5] = data['/ground_truth/twist_angular_z'].values[start:end] * 180 / np.pi
    return np.sqrt(np.mean((estimated_twist-ground_truth_twist)**2, axis=0))


def plot_morphing(data, start, end = None, filename = ""):
    data_processor = DataProcessorGoat(DEVICE)
    inputs = data_processor.process_input_data(data)
    targets = data_processor.process_output_data(data)
    num_data = inputs.shape[0]
    if end is None:
        end = num_data
    # Do this or ignore to use rosbag data.
    # simulate_data_with_model(MODEL_PATH, data)
    ground_truth_points = np.zeros([end - start, NUM_POINTS, 3])
    estimated_points = np.zeros([end - start, NUM_POINTS, 3])
    for t in range(start, end):
        for i in range(NUM_POINTS):
            for j in range(3):
                estimated_points[t - start][i][j] = data[f'/frame_points/data_{i*3 + j}'].values[t]
        ground_truth_points[t - start] = targets[t, :INDEX_GRAVITY].view(NUM_POINTS, 3).detach().numpy()
    dist = np.linalg.norm(estimated_points - ground_truth_points, axis=2)
    rmse = np.sqrt(np.mean((dist)**2, axis=1))

    series_left = np.zeros([3, end-start])
    labels_left = []
    series_left[0, :] = data['/estimated_width/data'].values[start:end]
    labels_left.append("Estimated Width (Left)")
    series_left[1, :] = data['/tendon_length_node_1/tendon_length/data'].values[start:end]
    labels_left.append("Tendon Length 1 (Left)")
    series_left[2, :] = data['/tendon_length_node_2/tendon_length/data'].values[start:end]
    labels_left.append("Tendon Length 2 (Left)")
    series_right = np.zeros([1, end-start])
    labels_right = []
    series_right[0, :] = rmse * 1000
    labels_right.append("Reconstruction RMSE (Right)")

    plot_time_series_two_axis(data_left=series_left, 
                              data_right=series_right,
                              labels_left=labels_left, 
                              labels_right= labels_right, 
                              xlabel="Time [s]", ylabel_left="Tendon Lengths and Width [m]", ylabel_right="Reconstruction RMSE [mm]",
                              ylim_left=[0.1, 2.75], ylim_right=[30, 150], filename=filename)


# Alternative function using dot product (more direct)
def angle_between_vectors_dot(u, v):
    """
    Alternative method using dot product (works for any vectors, not just unit vectors)
    """
    u = np.array(u)
    v = np.array(v)
    
    # Normalize if not already unit vectors
    u_norm = u / np.linalg.norm(u)
    v_norm = v / np.linalg.norm(v)
    
    # Calculate dot product and clip to avoid numerical issues
    dot_product = np.dot(u_norm, v_norm)
    dot_product = np.clip(dot_product, -1.0, 1.0)  # Ensure valid range for arccos
    
    angle = np.arccos(dot_product)
    return angle


def calculate_gravity_error(data, start = 0, end = None):
    data_processor = DataProcessorGoat(DEVICE)
    inputs = data_processor.process_input_data(data)
    targets = data_processor.process_output_data(data)
    num_data = inputs.shape[0]
    if end is None:
        end = num_data
    # Do this or ignore to use rosbag data.
    # simulate_data_with_model(MODEL_PATH, data)
    estimated_gravity = torch.zeros([end - start, 3])
    ground_truth_gravity = torch.zeros([end - start, 3])
    estimated_gravity[:, 0] = torch.tensor(data['/gravity_vector/data_0'].values[start:end])
    estimated_gravity[:, 1] =  torch.tensor(data['/gravity_vector/data_1'].values[start:end])
    estimated_gravity[:, 2] =  torch.tensor(data['/gravity_vector/data_2'].values[start:end])
    ground_truth_gravity[:, 0] =  torch.tensor(data['/ground_truth/gravity_x'].values[start:end])
    ground_truth_gravity[:, 1] =  torch.tensor(data['/ground_truth/gravity_y'].values[start:end])
    ground_truth_gravity[:, 2] =  torch.tensor(data['/ground_truth/gravity_z'].values[start:end])
    rad_angle_errors = roma.rotvec_geodesic_distance(estimated_gravity, ground_truth_gravity)
    return np.sqrt(np.mean((rad_angle_errors.detach().numpy())**2)) * 180 / np.pi


# scenarios
static_circle = "/workspace/data/2025_08_20/rosbag2_2025_08_20-17_30_24_goat_training.parquet"
static_rover = "/workspace/data/2025_08_20/rosbag2_2025_08_20-17_35_07_goat_training.parquet"
static_sphere = "/workspace/data/2025_08_20/rosbag2_2025_08_20-17_37_32_goat_training.parquet"
front_w_tendon = "/workspace/data/2025_08_20/rosbag2_2025_08_20-17_39_14_goat_training.parquet"
front_w_o_tendon = "/workspace/data/2025_08_20/rosbag2_2025_08_20-17_41_31_goat_training.parquet"
forward_pid = "/workspace/data/2025_08_13/rosbag2_2025_08_13-14_32_37_goat_training.parquet"
rot_w_frame = "/workspace/data/2025_08_13/rosbag2_2025_08_13-14_22_42_goat_training.parquet"
rot_w_o_frame = "/workspace/data/2025_08_13/rosbag2_2025_08_13-14_26_13_goat_training.parquet"
forward_pid_broken_wel = "/workspace/data/2025_08_13/rosbag2_2025_08_13-17_22_15_goat_training.parquet"

# Top Down View
# datas = [
#     pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-16_49_04_goat_training.parquet"),
#     pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-16_49_43_goat_training.parquet"),
#     pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-16_51_23_goat_training.parquet"),
#     pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-17_22_15_goat_training.parquet"),
#     pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-17_23_05_goat_training.parquet"),
#     pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-17_23_48_goat_training.parquet"),
# ]
# top_down_view(datas)

# Yaw Rate w/o, w, w + PID Tracking
# datas = [
#     (pd.read_parquet(rot_w_frame), {'/desired_twist/angular_z': 1183}), # w/o frame, open
#     (pd.read_parquet(rot_w_o_frame), {'/ground_truth/twist_angular_z': 394}), # w/o frame, open
#     (pd.read_parquet(rot_w_frame), {'/ground_truth/twist_angular_z': 254}), # with frame, open
#     (pd.read_parquet(rot_w_frame), {'/ground_truth/twist_angular_z': 1183}), # with frame, closed
# ]
# yaw_tracking(datas, 80, labels=['Desired', 'Open Loop w/o Frame Estimation', 'Open Loop w/ Frame Estimation', 'Closed Loop w/ Frame Estimation'], filename="desired_yaw_tracking")

# Linear Velocity Estimation
# datas = [
#     (pd.read_parquet(forward_pid), {'/desired_twist/linear_x': 70}),
#     (pd.read_parquet(forward_pid), {'/estimated_twist/linear_x': 69}),
#     (pd.read_parquet(forward_pid), {'/ground_truth/twist_linear_x': 70}),
# ]
# x_tracking(datas, 60, labels=['Desired', 'Estimated Linear Forward Velocity', 'Ground Truth Linear Forward Velocity'])

# Yaw Rate Estimation
# datas = [
#     (pd.read_parquet(rot_w_frame), {'/desired_twist/angular_z': 1183}),
#     (pd.read_parquet(rot_w_frame), {'/estimated_twist/angular_z': 1183}),
#     (pd.read_parquet(rot_w_frame), {'/ground_truth/twist_angular_z': 1183}),
#     (pd.read_parquet(rot_w_frame), {'/imu/data/angular_velocity_z': 1183}),
# ]
# yaw_tracking(datas, 80, labels=['Desired', 'Estimated Yaw Rate', 'Ground Truth Yaw Rate', 'Smoothed IMU Yaw Rate'], filename="yaw_rate_estimation")

# Plot Shape
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-15_09_51_goat_training.parquet")  # front fold w/o tendon but was trained on it
# data = pd.read_parquet(static_circle)  # circle
# data = pd.read_parquet(static_rover)  # rover
# data = pd.read_parquet(static_sphere)  # front fold with tendon
# data = pd.read_parquet(front_w_o_tendon)  # front fold w/o tendon
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_36_03_goat_training.parquet")  # rover to ball, bad data
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_37_32_goat_training.parquet")  # ball to circle, bad data
# plot_minimal_shape(data)

recons = {"(A) Static Circle": np.array([]),
          "(B) Static Rover" : np.array([]),
          "(C) Static Sphere": np.array([]),
          "(D) Static Intermediate w/ Tendon": np.array([]),
          "(E) Static Intermediate w/o Tendon": np.array([]),
          "Driving Forwards": np.array([]),
          "Yawing in Place": np.array([]),
          "Driving Forwards w/ dist.": np.array([]),}
data = pd.read_parquet(static_circle) # (A) Static Circle
rmse = calculate_rmse_reconstruction(start= 50, data=data) * 1000
recons["(A) Static Circle"] = np.append(recons["(A) Static Circle"], rmse)

data = pd.read_parquet(static_rover) # (B) Static Rover
rmse = calculate_rmse_reconstruction(start= 50, data=data) * 1000
recons["(B) Static Rover"] = np.append(recons["(B) Static Rover"], rmse)

data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_37_32_goat_training.parquet") # static ball
rmse = calculate_rmse_reconstruction(start= 10, end= 40, data=data) * 1000
recons["(C) Static Sphere"] = np.append(recons["(C) Static Sphere"], rmse)

data = pd.read_parquet(static_sphere) # front fold w/ tendon
rmse = calculate_rmse_reconstruction(start= 50, data=data) * 1000
recons["(D) Static Intermediate w/ Tendon"] = np.append(recons["(D) Static Intermediate w/ Tendon"], rmse)

data = pd.read_parquet(front_w_o_tendon) # front fold w/o tendon
rmse = calculate_rmse_reconstruction(start= 50, end=400, data=data) * 1000
recons["(E) Static Intermediate w/o Tendon"] = np.append(recons["(E) Static Intermediate w/o Tendon"], rmse)

data = pd.read_parquet(forward_pid) # forward PID
rmse = calculate_rmse_reconstruction(start= 50, data=data) * 1000
recons["Driving Forwards"] = np.append(recons["Driving Forwards"], rmse)

data = pd.read_parquet(rot_w_frame) # rot with and without PID with frame
rmse = calculate_rmse_reconstruction(start= 50, data=data) * 1000
recons["Yawing in Place"] = np.append(recons["Yawing in Place"], rmse)

data = pd.read_parquet(forward_pid_broken_wel) # forward with yaw PID broken drive
rmse = calculate_rmse_reconstruction(start= 50, data=data) * 1000
recons["Driving Forwards w/ dist."] = np.append(recons["Driving Forwards w/ dist."], rmse)

# Gravity
data = pd.read_parquet(static_circle) # (A) Static Circle
recons["(A) Static Circle"] = np.append(recons["(A) Static Circle"], calculate_gravity_error(data))
data = pd.read_parquet(static_rover) # (B) Static Rover
recons["(B) Static Rover"] = np.append(recons["(B) Static Rover"], calculate_gravity_error(data))
data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_37_32_goat_training.parquet") # static ball
recons["(C) Static Sphere"] = np.append(recons["(C) Static Sphere"], calculate_gravity_error(data))
data = pd.read_parquet(static_sphere) # front fold w/ tendon
recons["(D) Static Intermediate w/ Tendon"] = np.append(recons["(D) Static Intermediate w/ Tendon"], calculate_gravity_error(data))
data = pd.read_parquet(front_w_o_tendon) # front fold w/o tendon
recons["(E) Static Intermediate w/o Tendon"] = np.append(recons["(E) Static Intermediate w/o Tendon"], calculate_gravity_error(data))
data = pd.read_parquet(forward_pid) # forward PID
recons["Driving Forwards"] = np.append(recons["Driving Forwards"], calculate_gravity_error(data))
data = pd.read_parquet(rot_w_frame) # rot with and without PID with frame
recons["Yawing in Place"] = np.append(recons["Yawing in Place"], calculate_gravity_error(data))
data = pd.read_parquet(forward_pid_broken_wel) # forward with yaw PID broken drive
recons["Driving Forwards w/ dist."] = np.append(recons["Driving Forwards w/ dist."], calculate_gravity_error(data))

data = pd.read_parquet(static_circle) # (A) Static Circle
rmse = calculate_rmse_twist_reconstruction(data)
recons["(A) Static Circle"] = np.append(recons["(A) Static Circle"], rmse)

data = pd.read_parquet(static_rover) # (B) Static Rover
rmse = calculate_rmse_twist_reconstruction(data)
recons["(B) Static Rover"] = np.append(recons["(B) Static Rover"], rmse)

data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_37_32_goat_training.parquet") # static ball
rmse = calculate_rmse_twist_reconstruction(data, start= 10, end= 40)
recons["(C) Static Sphere"] = np.append(recons["(C) Static Sphere"], rmse)

data = pd.read_parquet(static_sphere) # front fold w/ tendon
rmse = calculate_rmse_twist_reconstruction(data)
recons["(D) Static Intermediate w/ Tendon"] = np.append(recons["(D) Static Intermediate w/ Tendon"], rmse)

data = pd.read_parquet(front_w_o_tendon) # front fold w/o tendon
rmse = calculate_rmse_twist_reconstruction(data)
recons["(E) Static Intermediate w/o Tendon"] = np.append(recons["(E) Static Intermediate w/o Tendon"], rmse)

data = pd.read_parquet(forward_pid)  # forward PID
rmse = calculate_rmse_twist_reconstruction(data)
recons["Driving Forwards"] = np.append(recons["Driving Forwards"], rmse)

data = pd.read_parquet(rot_w_frame)  # rot with and without PID with frame
rmse = calculate_rmse_twist_reconstruction(data)
recons["Yawing in Place"] = np.append(recons["Yawing in Place"], rmse)

data = pd.read_parquet(forward_pid_broken_wel) # forward with yaw PID broken drive
rmse = calculate_rmse_twist_reconstruction(data)
recons["Driving Forwards w/ dist."] = np.append(recons["Driving Forwards w/ dist."], rmse)

recons_mat = []
for key, value in recons.items():
    recons_mat.append(value)
    line = ""
    line += key
    i = 0
    for v in value:
        line += " & "
        line += f"{v:.3}"
        if i == 4:
            line += " & "
            line += f"{value[2:5].mean():.3}"
        if i == 7:
            line += " & "
            line += f"{value[5:8].mean():.3}"
        i += 1
    line += " "
    line += chr(92)
    line += chr(92)
    print(line)

recons_mat = np.array(recons_mat)

print(chr(92) + "midrule")

line = chr(92) + "textbf{Mean}"
for i in range(recons_mat.shape[1]):
    line += " & "
    line += f"{recons_mat[:, i].mean():.3}"
    if i == 4:
        line += " & "
    if i == 7:
        line += " & "
line += " "
line += chr(92)
line += chr(92)
print(line)

line = chr(92) + "textbf{Std Dev.}"
for i in range(recons_mat.shape[1]):
    line += " & "
    line += f"{recons_mat[:, i].std():.3}"
    if i == 4:
        line += " & "
    if i == 7:
        line += " & "
line += " "
line += chr(92)
line += chr(92)
print(line)

# data = pd.read_parquet(static_circle)  # circle
# plot_morphing(data, 50)
# data = pd.read_parquet(static_rover)  # rover
# plot_morphing(data, 50)
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_36_03_goat_training.parquet") # rover to ball, shitty mocap
# plot_morphing(data, 50, 650)
data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_37_32_goat_training.parquet") # ball to circle, shitty mocap
plot_morphing(data, 20, filename="morphing_timeseries_sphere_to_circle")
# data = pd.read_parquet("/workspace/data/2025_09_03/rosbag2_2025_09_03-16_12_23_goat_training.parquet") # circle to ball to circle
# plot_morphing(data, 0, filename="morphing_timeseries_sphere_to_circle")

