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
from random import random


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
        "/workspace/data/2025_07_21/rosbag2_2025_07_22-08_51_55",
        "/workspace/data/2025_07_21/rosbag2_2025_07_22-08_50_04",
        "/workspace/data/2025_07_21/rosbag2_2025_07_21-14_37_36",
        "/workspace/data/2025_07_21/rosbag2_2025_07_21-14_16_19",
    ]
    sequence_length = 50
    target_length = 1
    batch_size = 64
    epochs = 250
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create filepaths
    now = datetime.now()
    os.mkdir("data/output/" + now.strftime("%Y_%m_%d_%H_%M_%S"))
    file_prefix = "data/output/" + now.strftime("%Y_%m_%d_%H_%M_%S") + "/" + now.strftime("%Y_%m_%d_%H_%M_%S_")

    # Setup data noise
    noises = [NoiseClean()]
    for i in range(6):
        noises.append(NoiseGaussian(mean=0.0, std=0.1 + 0.2 * i))
        noises.append(NoiseOffset(offset=0.4 * i - 1.0))
        noises.append(NoiseSinusoidal(amplitude=0.2 + 0.2 * i, frequency=3.0 - 0.5 * i))

    # Prep data
    x_train, x_test, y_train, y_test = None, None, None, None
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
                input=noisy_inputs, target=targets, sequence_length=sequence_length, target_length=target_length
            )

            # Concat to the rest of the training data
            x_train = concat(x_train, x_train_seq)
            x_test = concat(x_test, x_test_seq)
            y_train = concat(y_train, y_train_seq)
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
    model = BeliefEncoderRNNModel(input_size=input_size, latent_size=64, hidden_size=64, num_layers=1, output_size=output_size, device=device)
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
