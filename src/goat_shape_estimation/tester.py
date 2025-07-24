import torch
import pandas as pd
import numpy as np
from helpers.data_processer import DataProcessorGoat
from helpers.visualizer import visualize_3d_spline, plot_velocity_comparison

# Load the traced model
traced_model = torch.jit.load("/workspace/data/output/latest_lstm_model.pt", map_location="cpu")

data = pd.read_parquet("/workspace/data/2025_07_21/rosbag2_2025_07_22-10_54_41_goat_training.parquet")  # still
# data = pd.read_parquet("/workspace/data/2025_07_21/rosbag2_2025_07_21-14_16_19_goat_training.parquet")  # yaw in circle mode
# data = pd.read_parquet("/workspace/data/2025_07_21/rosbag2_2025_07_22-09_41_22_goat_training.parquet")  # circle -> s but with broken front tendon
# data = pd.read_parquet("/workspace/data/2025_07_21/rosbag2_2025_07_22-10_56_30_goat_training.parquet")  # circle -> s 

device = device = torch.device("cpu")

data_processor_goat = DataProcessorGoat(device)
inputs = data_processor_goat.process_input_data(data)
targets = data_processor_goat.process_output_data(data)

num_data = inputs.shape[0]
num_points = 12
num_points_per_circle = 9
points_index = 0
estimated_points = torch.zeros([num_points, 3], dtype=torch.float, device=torch.device("cpu"))
estimated_points_to_visualize = torch.zeros([2, num_points_per_circle, 3], dtype=torch.float, device=torch.device("cpu"))
estimated_gravity = torch.zeros([3], dtype=torch.float, device=torch.device("cpu"))
grav_index = points_index + num_points * 3
estimated_velocities = torch.zeros([inputs.shape[0], 6], dtype=torch.float, device=torch.device("cpu"))
vel_index = grav_index + 3

for i in range(num_data):
    sample_input = inputs[i, :].unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = traced_model(sample_input).squeeze().squeeze()

    ## Points and Gravity
    # if i % 100 == 0:
    #     estimated_points = output[:grav_index].view(num_points, 3)
    #     estimated_gravity = output[grav_index:vel_index]
    #     estimated_points_to_visualize[0, :, :] = estimated_points[[0, 1, 2, 3, 4, 5, 6, 7, 0], :]
    #     estimated_points_to_visualize[1, :, :] = estimated_points[[0, 8, 2, 9, 4, 10, 6, 11, 0], :]

    #     visualize_3d_spline(
    #         estimated_points_to_visualize.detach().numpy(),
    #         estimated_gravity.detach().numpy(),
    #         text=f" with Tendon length {inputs[i, -2]:.2f}[m] and {inputs[i, -1]:.2f}[m], timestep {i*0.05:.2f}[s]",
    #         filename="data/output/" + f"spline_{i}",
    #     )

    estimated_velocities[i, :] = output[vel_index:]

### Plot velocities
plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 0], targets.detach().numpy()[:, vel_index], title="Linear X")
plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 1], targets.detach().numpy()[:, vel_index + 1], title="Linear Y")
plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 2], targets.detach().numpy()[:, vel_index + 2], title="Linear Z")
plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 3], targets.detach().numpy()[:, vel_index + 3], title="Angular X")
plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 4], targets.detach().numpy()[:, vel_index + 4], title="Angular Y")
plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 5], targets.detach().numpy()[:, vel_index + 5], title="Angular Z")
