import torch
from torch.utils.data import DataLoader
from helpers.dataset import GoatDataset
from helpers.model import RNNModel, LSTMModel, SelfAttentionModel, SelfAttentionRNNModel, BeliefEncoderRNNModel, MLP, train_model
from helpers.data_processer import DataProcessorGoat, create_sequences
from helpers.noise import NoiseClean, NoiseGaussian, NoiseOffset, NoiseSinusoidal
import torch.nn as nn
import pandas as pd
from datetime import datetime
import os


def concat(x, y):
    if x is None:
        return y
    else:
        return torch.cat((x, y), dim=0)


def main():
    # Configuration
    paths = [
        "/workspace/data/2025_07_21/rosbag2_2025_07_22-10_58_28",
        "/workspace/data/2025_07_21/rosbag2_2025_07_22-10_56_30",
        "/workspace/data/2025_07_21/rosbag2_2025_07_22-10_54_41",
        "/workspace/data/2025_07_21/rosbag2_2025_07_22-09_43_08",
        "/workspace/data/2025_07_21/rosbag2_2025_07_22-09_41_22",
        "/workspace/data/2025_07_21/rosbag2_2025_07_22-09_36_32",
        "/workspace/data/2025_07_21/rosbag2_2025_07_22-08_53_52",
        "/workspace/data/2025_07_21/rosbag2_2025_07_21-14_37_36",
        "/workspace/data/2025_08_08/rosbag2_2025_08_08-08_06_10",
        "/workspace/data/2025_08_08/rosbag2_2025_08_08-08_07_55",
        "/workspace/data/2025_08_08/rosbag2_2025_08_08-08_09_46",
        "/workspace/data/2025_08_08/rosbag2_2025_08_08-08_11_23",
        "/workspace/data/2025_08_08/rosbag2_2025_08_08-10_51_28",
        "/workspace/data/2025_08_08/rosbag2_2025_08_08-10_56_21",
        "/workspace/data/2025_08_12/rosbag2_2025_08_12-16_04_26",
        "/workspace/data/2025_08_12/rosbag2_2025_08_12-16_07_25",
        "/workspace/data/2025_08_12/rosbag2_2025_08_12-17_14_46",
        "/workspace/data/2025_08_12/rosbag2_2025_08_12-17_16_12",
        "/workspace/data/2025_08_12/rosbag2_2025_08_12-17_17_39",
        "/workspace/data/2025_08_12/rosbag2_2025_08_12-17_19_01",
        "/workspace/data/2025_08_12/rosbag2_2025_08_12-17_43_41",
        "/workspace/data/2025_08_13/rosbag2_2025_08_13-12_11_45", # rotations both with and without PID
        "/workspace/data/2025_08_13/rosbag2_2025_08_13-14_22_42", # rot with and without PID with frame
        "/workspace/data/2025_08_13/rosbag2_2025_08_13-14_26_13", # rot with and without PID without frame
        "/workspace/data/2025_08_13/rosbag2_2025_08_13-14_29_52", # forward no PID
        "/workspace/data/2025_08_13/rosbag2_2025_08_13-14_32_37", # forward PID
        "/workspace/data/2025_08_13/rosbag2_2025_08_13-14_33_19", # backwards no PID
        "/workspace/data/2025_08_13/rosbag2_2025_08_13-16_49_04", # forward no PID broken wheel
        "/workspace/data/2025_08_13/rosbag2_2025_08_13-16_49_43", # forward no PID broken wheel
        "/workspace/data/2025_08_13/rosbag2_2025_08_13-16_51_23", # forward no PID broken wheel
        "/workspace/data/2025_08_13/rosbag2_2025_08_13-17_23_05", # forward with PID broken wheel
        "/workspace/data/2025_08_13/rosbag2_2025_08_13-17_23_48", # forward with PID broken wheel
        "/workspace/data/2025_08_20/rosbag2_2025_08_20-14_27_38", # circle to sphere to circle 
        "/workspace/data/2025_08_20/rosbag2_2025_08_20-14_30_44", # rover to circle
        "/workspace/data/2025_08_20/rosbag2_2025_08_20-15_09_51", # front fold w/o tendon
        "/workspace/data/2025_08_20/rosbag2_2025_08_20-15_16_50", # random transitions
        "/workspace/data/2025_08_20/rosbag2_2025_08_20-15_17_47", # to rover
        "/workspace/data/2025_08_20/rosbag2_2025_08_20-15_18_50", # to sphere
        "/workspace/data/2025_08_20/rosbag2_2025_08_20-17_30_24", # circle
        "/workspace/data/2025_08_20/rosbag2_2025_08_20-17_35_07", # rover
        "/workspace/data/2025_08_20/rosbag2_2025_08_20-17_50_14", # driving forwards in rover
        "/workspace/data/2025_09_03/rosbag2_2025_09_03-14_32_04", # circle
        "/workspace/data/2025_09_03/rosbag2_2025_09_03-14_37_20", # rover
        "/workspace/data/2025_09_03/rosbag2_2025_09_03-16_12_23", # circle to sphere to circle, poor data cause tendons are offset
        "/workspace/data/2025_09_04/rosbag2_2025_09_04-17_41_33", # circle to rover
        "/workspace/data/2025_09_04/rosbag2_2025_09_04-17_45_31", # sphere to circle
        "/workspace/data/2025_09_15/rosbag2_2025_09_15-12_33_43", # upside down circle
        "/workspace/data/2025_09_15/rosbag2_2025_09_15-12_35_40", # upside down circle -> sphere -> circle
        "/workspace/data/2025_09_15/rosbag2_2025_09_15-12_38_29", # upside down rover
    ]
    test_paths = [
        # "/workspace/data/2025_08_13/rosbag2_2025_08_13-17_22_15", # forward with PID broken wheel
        # "/workspace/data/2025_08_20/rosbag2_2025_08_20-17_39_14", # front fold with tendon
        # "/workspace/data/2025_08_20/rosbag2_2025_08_20-17_41_31", # front fold w/o tendon
        # "/workspace/data/2025_09_03/rosbag2_2025_09_03-16_06_38", # yaw w/ width, w/o and w/ PID
        # "/workspace/data/2025_09_03/rosbag2_2025_09_03-16_10_07", # yaw w/o width (0.5), w/o and w/ PID
        # "/workspace/data/2025_09_03/rosbag2_2025_09_03-16_21_15", # forward w/ width, w/ PID
        # "/workspace/data/2025_09_04/rosbag2_2025_09_04-17_40_29", # circle
        # "/workspace/data/2025_09_04/rosbag2_2025_09_04-17_42_33", # rover
        # "/workspace/data/2025_09_04/rosbag2_2025_09_04-17_44_23", # rover to sphere
    ]
    sequence_length = 50
    target_length = 1
    test_size=0.01
    batch_size = 64
    epochs = 101
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create filepaths
    now = datetime.now()
    os.mkdir("data/output/" + now.strftime("%Y_%m_%d_%H_%M_%S"))
    file_prefix = "data/output/" + now.strftime("%Y_%m_%d_%H_%M_%S") + "/" + now.strftime("%Y_%m_%d_%H_%M_%S_")

    # Setup data noise
    noises = [NoiseClean()]
    noises.append(NoiseGaussian(index = [i for i in range(21)],mean=0.0, std=0.1))
    # noises.append(NoiseOffset(index = [19, 20], offset=0.1))
    # noises.append(NoiseOffset(index = [19, 20], offset=0.3))
    noises.append(NoiseOffset(index = [19, 20], offset=1.0))
    noises.append(NoiseSinusoidal(index = [i for i in range(21)], amplitude=0.1, frequency=1))

    # Prep data
    x_train, x_test, y_train, y_test = None, None, None, None

    # inputs = torch.zeros([0, 45], device=device)
    # for path in paths:
    #     # Load data from bags/csvs
    #     data = pd.read_parquet(path + '_goat_training.parquet')

    #     # Apply point transformations, velocity calcuations
    #     data_processor_goat = DataProcessorGoat(device)
    #     input = data_processor_goat.process_input_data(data)
    #     output = data_processor_goat.process_output_data(data)
    #     inputs = torch.concat((inputs, output), dim=0)

    # input_mean = torch.mean(inputs, dim=0)
    # input_std = torch.std(inputs, dim=0)

    for path in paths:
        # Load data from bags/csvs
        data = pd.read_parquet(path + '_goat_training.parquet')

        # Apply point transformations, velocity calcuations
        data_processor_goat = DataProcessorGoat(device)
        inputs = data_processor_goat.process_input_data(data)
        inputs = data_processor_goat.scale_input_data_tensor(inputs)
        targets = data_processor_goat.process_output_data(data)
        targets = data_processor_goat.scale_output_data_tensor(targets)

        for noise in noises:
            noisy_inputs = noise(inputs)

            # Create sequences
            x_train_seq, x_test_seq, y_train_seq, y_test_seq = create_sequences(
                input=noisy_inputs, target=targets, sequence_length=sequence_length, target_length=target_length, test_size=test_size
            )

            # Concat to the rest of the training data
            x_train = concat(x_train, x_train_seq)
            x_test = concat(x_test, x_test_seq)
            y_train = concat(y_train, y_train_seq)
            y_test = concat(y_test, y_test_seq)

    for path in test_paths:
        # Load data from bags/csvs
        data = pd.read_parquet(path + '_goat_training.parquet')

        # Apply point transformations, velocity calcuations
        data_processor_goat = DataProcessorGoat(device)
        inputs = data_processor_goat.process_input_data(data)
        inputs = data_processor_goat.scale_input_data_tensor(inputs)
        targets = data_processor_goat.process_output_data(data)
        targets = data_processor_goat.scale_output_data_tensor(targets)

        # Create sequences
        x_train_seq, x_test_seq, y_train_seq, y_test_seq = create_sequences(
            input=inputs, target=targets, sequence_length=sequence_length, target_length=target_length, test_size=test_size
        )

        # Concat to the rest of the training data
        x_test = concat(x_test, x_test_seq)
        y_test = concat(y_test, y_test_seq)


    # Create PyTorch datasets and dataloaders
    train_dataset = GoatDataset(x_train, y_train)
    val_dataset = GoatDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    input_size = x_train.shape[2]
    output_size = y_train.shape[2]

    # model = RNNModel(input_size=input_size, hidden_size=64, num_layers=1, output_size=output_size, device=device)
    # model = SelfAttentionModel(input_size=input_size, embed_dim=64, num_heads=8, output_size=output_size)
    # model = SelfAttentionRNNModel(input_size=input_size, embed_dim=64, num_heads=8, hidden_size=128, num_layers=2, output_size=output_size)
    model = BeliefEncoderRNNModel(input_size=input_size, latent_size=32, hidden_size=32, num_layers=1, output_size=output_size, device=device, noisy_index=[19, 20])
    # model = MLP(input_dim=input_size, history_dim=sequence_length, hidden_sizes=[256, 128, 64], output_dim=output_size)

    # Train model
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs,
        file_prefix,
        data_processor_goat.input_mean,
        data_processor_goat.input_std,
        data_processor_goat.output_mean,
        data_processor_goat.output_std,
    )


if __name__ == "__main__":
    main()
