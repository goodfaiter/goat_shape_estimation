import torch
import pandas as pd
import numpy as np
from helpers.data_processer import DataProcessorGoat
from helpers.visualizer import visualize_3d_spline, plot_velocity_comparison

# Load the traced model
traced_model = torch.jit.load("/workspace/data/output/latest_lstm_model.pt", map_location="cpu")
# traced_model.reset()
# data = pd.read_parquet("/workspace/data/2025_06_04/2025_06_04_16_24_22_goat_training.parquet") # c -> s -> c
# data = pd.read_parquet("/workspace/data/2025_06_04/2025_06_04_15_52_28_goat_training.parquet") # yaw in circle mode
data = pd.read_parquet("/workspace/data/2025_06_04/2025_06_04_15_49_09_goat_training.parquet") # c -> rover + drive
# data = pd.read_parquet("/workspace/data/2025_06_04/2025_06_04_15_50_40_goat_training.parquet") # rov -> c

data_processor_goat = DataProcessorGoat()
inputs = data_processor_goat.process_input_data(data)
targets = data_processor_goat.process_output_data(data)

estimated_velocities = torch.zeros([inputs.shape[0], 6], dtype=torch.float, device=torch.device('cpu'))

for i in range(inputs.shape[0]):
    sample_input = inputs[i, :].unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output = traced_model(sample_input)

    ## Points and Gravity
    # if i % 100 == 0:
    #     print(output.squeeze().squeeze().numpy())
    #     print(targets[i, :].numpy())
    #     # Get single new observation
    #     new_data_point = sample_input.squeeze().squeeze().numpy()
    #     vis_points = output[0, 0:24].detach().reshape(8, 3).numpy()
    #     vis_points = np.append(vis_points, vis_points[0, :]) # Append first point to last to close the spline loop
    #     down_vec = output[0, 24:27].detach().numpy()
    #     # target_data_point = targets[i, 0:24].detach().reshape(8, 3).numpy()
    #     # down_vec = targets[i, 24:27].detach().numpy()
    #     # vis_points = target_data_point
    #     # vis_points = np.append(vis_points, vis_points[0, :]) # Append first point to last to close the spline loop
        
    #     visualize_3d_spline(
    #         vis_points.reshape(9, 3),
    #         down_vec,
    #         text=f" with Tendon length {new_data_point[-2]} and {new_data_point[-1]}, iteration {i}",
    #         filename="data/output/" + f"spline_{i}",
    #     )
    #     print(new_data_point[20:22])

    estimated_velocities[i, :] = output[0, 27:]

### Plot velocities
plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 0], targets.detach().numpy()[:, 0])
plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 1], targets.detach().numpy()[:, 1])
plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 2], targets.detach().numpy()[:, 2])
plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 3], targets.detach().numpy()[:, 3])
plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 4], targets.detach().numpy()[:, 4])
plot_velocity_comparison(estimated_velocities.detach().numpy()[:, 5], targets.detach().numpy()[:, 5])
