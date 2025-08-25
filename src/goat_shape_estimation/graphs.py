import torch
import pandas as pd
import numpy as np
from helpers.data_processer import DataProcessorGoat, ema_2d_optimized
from helpers.visualization.visualizer import visualize_3d_spline, plot_velocity_comparison, plot_trajectories, plot_time_series, visualize_3d_spline_minimal

NUM_POINTS = 12
NUM_POINTS_PER_CIRCLE = 9
INDEX_POINTS = 0
INDEX_GRAVITY = INDEX_POINTS + NUM_POINTS * 3
INDEX_VELOCITY = INDEX_GRAVITY + 3
NUM_OUTPUT = INDEX_VELOCITY + 6
INDICES_RING_ONE = [0, 1, 2, 3, 4, 5, 6, 7, 0]
INDICES_RING_TWO = [0, 8, 2, 9, 4, 10, 6, 11, 0]
DEVICE = torch.device("cpu")
MODEL_PATH = "/workspace/data/output/2025_08_22_11_19_04/2025_08_22_11_19_04_best_lstm_model.pt"

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

    # frame_points = outputs[:, INDEX_POINTS:INDEX_GRAVITY].reshape(-1, NUM_POINTS, 3)
    # avg_distance = np.mean(frame_points[:, [1, 3, 8, 9], :], axis = 1) - np.mean(frame_points[:,[5, 7, 10, 11], :], axis=1)
    # frame_width = np.linalg.norm(avg_distance, axis = 1) - 0.1 # 0.1 comes from the 5 [cm] x 2 marker to wheel offset
    # data['/estimated_width/data'] = frame_width

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
    plot_trajectories(drive_pos_in_world[:, :min_input, :], labels=['Open Loop 1', 'Open Loop 2', 'Open Loop 3', 'Closed Loop 1', 'Closed Loop 2', 'Closed Loop 3'], ylim=[-1.25, 0.5])


def yaw_tracking(datas, dur, title="Yaw Rate Tracking", labels = None):
    yaw_rates = []
    for data, to_plot_dict in datas:
        data_processor = DataProcessorGoat(DEVICE)
        data_processor.process_input_data(data)
        data_processor.process_output_data(data)
        for name, start in to_plot_dict.items():
            if name == '/imu/data/angular_velocity_z':
                data[name] *= -1.0
            yaw_rates.append(data[name][start:start + dur])
            # test = ema_2d_optimized(data[name][start:start + dur])
            # yaw_rates[-1] = ema_2d_optimized(torch.tensor([yaw_rates[-1]])).detach().numpy()
    plot_time_series(yaw_rates, labels=labels, title=title, xlabel="Time [s]", ylabel="Yaw Rate [rad/s]", ylim=[-0.1, 1.25])


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
    plot_time_series(rates, labels=labels, title="Linear Velocity Tracking", xlabel="Time [s]", ylabel="Velocity [m/s]", ylim=[-0.1, 0.5])

def plot_minimal_shape(data):
    data_processor = DataProcessorGoat(DEVICE)
    inputs = data_processor.process_input_data(data)
    targets = data_processor.process_output_data(data)
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
        if t % 100 == 0:
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

def calculate_rmse_reconstruction(data):
    seq_length = 50
    data_processor = DataProcessorGoat(DEVICE)
    inputs = data_processor.process_input_data(data)
    targets = data_processor.process_output_data(data)
    num_data = inputs.shape[0]
    # Do this or ignore to use rosbag data.
    # simulate_data_with_model(MODEL_PATH, data)
    ground_truth_points = np.zeros([num_data - seq_length, NUM_POINTS, 3])
    estimated_points = np.zeros([num_data - seq_length, NUM_POINTS, 3])
    for t in range(seq_length, num_data):
        for i in range(NUM_POINTS):
            for j in range(3):
                estimated_points[t - seq_length][i][j] = data[f'/frame_points/data_{i*3 + j}'].values[t]
                ground_truth_points[t-seq_length] = targets[t, :INDEX_GRAVITY].view(NUM_POINTS, 3).detach().numpy()
    dist = np.linalg.norm(estimated_points - ground_truth_points, axis=2)

    # this
    # rmse_per_timestep = np.sqrt(np.mean((dist)**2, axis=1))
    # rmse = np.sqrt(np.mean((rmse_per_timestep)**2))
    # or
    rmse = np.sqrt(np.mean((dist)**2))
    return rmse

def calculate_rmse_twist_reconstruction(data):
    seq_length = 50
    data_processor = DataProcessorGoat(DEVICE)
    inputs = data_processor.process_input_data(data)
    targets = data_processor.process_output_data(data)
    num_data = inputs.shape[0]
    # Do this or ignore to use rosbag data.
    # simulate_data_with_model(MODEL_PATH, data)
    estimated_twist = np.zeros([num_data - seq_length, 6])
    ground_truth_twist = np.zeros([num_data - seq_length, 6])
    estimated_twist[:, 0] = data['/estimated_twist/linear_x'].values[seq_length:] * 1000
    estimated_twist[:, 1] = data['/estimated_twist/linear_y'].values[seq_length:] * 1000
    estimated_twist[:, 2] = data['/estimated_twist/linear_z'].values[seq_length:] * 1000
    estimated_twist[:, 3] = data['/estimated_twist/angular_x'].values[seq_length:]
    estimated_twist[:, 4] = data['/estimated_twist/angular_y'].values[seq_length:]
    estimated_twist[:, 5] = data['/estimated_twist/angular_z'].values[seq_length:]
    ground_truth_twist[:, 0] = data['/ground_truth/twist_linear_x'].values[seq_length:] * 1000
    ground_truth_twist[:, 1] = data['/ground_truth/twist_linear_y'].values[seq_length:] * 1000
    ground_truth_twist[:, 2] = data['/ground_truth/twist_linear_z'].values[seq_length:] * 1000
    ground_truth_twist[:, 3] = data['/ground_truth/twist_angular_x'].values[seq_length:]
    ground_truth_twist[:, 4] = data['/ground_truth/twist_angular_y'].values[seq_length:]
    ground_truth_twist[:, 5] = data['/ground_truth/twist_angular_z'].values[seq_length:]
    return np.sqrt(np.mean((estimated_twist-ground_truth_twist)**2, axis=0))


def plot_morphing(data):
    seq_length = 50
    data_processor = DataProcessorGoat(DEVICE)
    inputs = data_processor.process_input_data(data)
    targets = data_processor.process_output_data(data)
    num_data = inputs.shape[0]
    # Do this or ignore to use rosbag data.
    simulate_data_with_model(MODEL_PATH, data)
    ground_truth_points = np.zeros([num_data - seq_length, NUM_POINTS, 3])
    estimated_points = np.zeros([num_data - seq_length, NUM_POINTS, 3])
    for t in range(seq_length, num_data):
        for i in range(NUM_POINTS):
            for j in range(3):
                estimated_points[t - seq_length][i][j] = data[f'/frame_points/data_{i*3 + j}'].values[t]
        ground_truth_points[t - seq_length] = targets[t, :INDEX_GRAVITY].view(NUM_POINTS, 3).detach().numpy()
    dist = np.linalg.norm(estimated_points - ground_truth_points, axis=2) * 1000
    rmse = np.sqrt(np.mean((dist)**2, axis=1))

    series = []
    series.append(data['/estimated_width/data'].values)
    series.append(data['/tendon_length_node_1/tendon_length/data'].values)
    series.append(data['/tendon_length_node_2/tendon_length/data'].values)
    plot_time_series(series, labels=["Estimated Width", "Tendon Length 1", "Tendon Length 2"], title=None, xlabel="Time [s]", ylabel="[m]", ylim=None)

# test_graphs(datas[0])

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
#     (pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-14_22_42_goat_training.parquet"), {'/desired_twist/angular_z': 1183}), # w/o frame, open
#     (pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-14_26_13_goat_training.parquet"), {'/ground_truth/twist_angular_z': 394}), # w/o frame, open
#     (pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-14_22_42_goat_training.parquet"), {'/ground_truth/twist_angular_z': 254}), # with frame, open
#     (pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-14_22_42_goat_training.parquet"), {'/ground_truth/twist_angular_z': 1183}), # with frame, closed
# ]
# yaw_tracking(datas, 80, labels=['Desired', 'Open Loop w/o Frame Estimation', 'Open Loop w/ Frame Estimation', 'Closed Loop w/ Frame Estimation'])

# Linear Velocity Estimation
# datas = [
#     (pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-14_32_37_goat_training.parquet"), {'/desired_twist/linear_x': 70}),
#     (pd.read_parquet("/workspace/data/2025_08_13/roerror - errorsbag2_2025_08_13-14_32_37_goat_training.parquet"), {'/estimated_twist/linear_x': 69}),
#     (pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-14_32_37_goat_training.parquet"), {'/ground_truth/twist_linear_x': 70}),
# ]
# x_tracking(datas, 60, labels=['Desired', 'Estimated Linear Forward Velocity', 'Ground Truth Linear Forward Velocity'])

# Yaw Rate Estimation
# datas = [
#     (pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-14_22_42_goat_training.parquet"), {'/desired_twist/angular_z': 1183}),
#     (pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-14_22_42_goat_training.parquet"), {'/estimated_twist/angular_z': 1183}),
#     (pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-14_22_42_goat_training.parquet"), {'/ground_truth/twist_angular_z': 1183}),
#     (pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-14_22_42_goat_training.parquet"), {'/imu/data/angular_velocity_z': 1183}),
# ]
# yaw_tracking(datas, 80, title="Yaw Rate Estimation", labels=['Desired', 'Estimated Yaw Rate', 'Ground Truth Yaw Rate', 'IMU Yaw Rate'])

# Plot Shape
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-15_09_51_goat_training.parquet")  # front fold w/o tendon but was trained on it
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_30_24_goat_training.parquet")  # circle
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_35_07_goat_training.parquet")  # rover
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_39_14_goat_training.parquet")  # front fold with tendon
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_41_31_goat_training.parquet")  # front fold w/o tendon
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_36_03_goat_training.parquet")  # rover to ball
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_37_32_goat_training.parquet")  # ball to circle
# plot_minimal_shape(data)

# rmse = calculate_rmse_reconstruction(pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_30_24_goat_training.parquet")) * 1000
# print(f" RMSE: Circle: {rmse:.3}")
# rmse = calculate_rmse_reconstruction(pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_35_07_goat_training.parquet")) * 1000
# print(f" RMSE: Rover: {rmse:.3}")
# rmse = calculate_rmse_reconstruction(pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_37_32_goat_training.parquet")) * 1000
# print(f" RMSE: Ball: {rmse:.3}")
# rmse = calculate_rmse_reconstruction(pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_39_14_goat_training.parquet")) * 1000
# print(f" RMSE: Front Fold w/ Tendon: {rmse:.3}")
# rmse = calculate_rmse_reconstruction(pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_41_31_goat_training.parquet")) * 1000
# print(f" RMSE: Front Fold w/o Tendon: {rmse:.3}")

# rmse = calculate_rmse_twist_reconstruction(pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-14_22_42_goat_training.parquet"))  # rot with and without PID with frame
# print(f" RMSE: X Rate Estimation: {rmse[0]:.3}")
# print(f" RMSE: Y Rate Estimation: {rmse[1]:.3}")
# print(f" RMSE: Z Rate Estimation: {rmse[2]:.3}")
# print(f" RMSE: Roll Rate Estimation: {rmse[3]:.3}")
# print(f" RMSE: Pitch Rate Estimation: {rmse[4]:.3}")
# print(f" RMSE: Yaw Rate Estimation: {rmse[5]:.3}")

# rmse = calculate_rmse_twist_reconstruction(pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-14_32_37_goat_training.parquet"))  # forward PID
# print(f" RMSE: X Rate Estimation: {rmse[0]:.3}")
# print(f" RMSE: Y Rate Estimation: {rmse[1]:.3}")
# print(f" RMSE: Z Rate Estimation: {rmse[2]:.3}")
# print(f" RMSE: Roll Rate Estimation: {rmse[3]:.3}")
# print(f" RMSE: Pitch Rate Estimation: {rmse[4]:.3}")
# print(f" RMSE: Yaw Rate Estimation: {rmse[5]:.3}")

# rmse = calculate_rmse_twist_reconstruction(pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-17_22_15_goat_training.parquet"))  # forward with yaw PID broken drive
# print(f" RMSE: X Rate Estimation: {rmse[0]:.3}")
# print(f" RMSE: Y Rate Estimation: {rmse[1]:.3}")
# print(f" RMSE: Z Rate Estimation: {rmse[2]:.3}")
# print(f" RMSE: Roll Rate Estimation: {rmse[3]:.3}")
# print(f" RMSE: Pitch Rate Estimation: {rmse[4]:.3}")
# print(f" RMSE: Yaw Rate Estimation: {rmse[5]:.3}")

# rec = np.array([143, 74, 199, 151, 287])
# print(rec.mean(), rec.std())

# table = np.array([[14.7, 11.6, 4.69], [29.7, 25.4, 4.54], [16.9, 26.6, 6.04]])
# print('top down means', table.mean(axis=1))
# print('left to right means', table.mean(axis=0))
# print('left to right std', table.std(axis=0))

# table = np.array([[0.02, 0.02, 0.10], [0.02, 0.02, 0.06], [0.03, 0.02, 0.05]])
# print('top down means', table.mean(axis=1))
# print('left to right means', table.mean(axis=0))
# print('left to right std', table.std(axis=0))

data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_37_32_goat_training.parquet")
plot_morphing(data)