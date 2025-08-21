import torch
import pandas as pd
import numpy as np
from helpers.data_processer import DataProcessorGoat
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


def yaw_tracking(datas, dur, labels):
    yaw_rates = []
    for data, to_plot_dict in datas:
        data_processor = DataProcessorGoat(DEVICE)
        data_processor.process_input_data(data)
        data_processor.process_output_data(data)
        for name, start in to_plot_dict.items():
            if name is '/imu/data/angular_velocity_z':
                data[name] *= -1.0
            yaw_rates.append(data[name][start:start + dur])
    plot_time_series(yaw_rates, labels=labels, title="Yaw Rate Tracking", xlabel="Time [s]", ylabel="Yaw Rate [rad/s]", ylim=[-0.1, 1.25])


def x_tracking(datas, dur, labels):
    rates = []
    for data, to_plot_dict in datas:
        data_processor = DataProcessorGoat(DEVICE)
        data_processor.process_input_data(data)
        data_processor.process_output_data(data)
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
#     (pd.read_parquet("/workspace/data/2025_08_13/rosbag2_2025_08_13-14_32_37_goat_training.parquet"), {'/estimated_twist/linear_x': 69}),
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
# yaw_tracking(datas, 80, labels=['Desired', 'Estimated Yaw Rate', 'Ground Truth Yaw Rate', 'IMU Yaw Rate'])

# Plot Shape
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-15_09_51_goat_training.parquet")  # front fold w/o tendon but was trained on it
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_30_24_goat_training.parquet")  # circle
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_35_07_goat_training.parquet")  # rover
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_39_14_goat_training.parquet")  # front fold with tendon
data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_41_31_goat_training.parquet")  # front fold w/o tendon
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_36_03_goat_training.parquet")  # rover to ball
# data = pd.read_parquet("/workspace/data/2025_08_20/rosbag2_2025_08_20-17_37_32_goat_training.parquet")  # ball to circle
plot_minimal_shape(data)