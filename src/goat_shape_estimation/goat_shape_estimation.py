import torch
from torch.utils.data import DataLoader
from helpers.data_loader import ROSBagDataLoader
from helpers.dataset import ROSBagDataset
from helpers.model import LSTMModel, train_model

def main():
    # Configuration
    bag_paths = [
        ('path/to/ros1.bag', ['/topic1', '/topic2']),  # ROS1 bag with topics
        ('path/to/ros2.bag', ['/topic3', '/topic4'])   # ROS2 bag with topics
    ]
    sequence_length = 50
    target_length = 1
    batch_size = 32
    epochs = 100
    learning_rate = 0.001
    
    # Load data from bags
    loader = ROSBagDataLoader()
    for bag_path, topics in bag_paths:
        loader.load_bag(bag_path, topics)
    
    # Create sequences
    X_train, X_test, y_train, y_test = loader.create_sequences(
        sequence_length=sequence_length,
        target_length=target_length
    )
    
    # Create PyTorch datasets and dataloaders
    train_dataset = ROSBagDataset(X_train, y_train)
    val_dataset = ROSBagDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_size = X_train.shape[2]  # Number of features
    hidden_size = 64
    num_layers = 2
    output_size = target_length
    
    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        epochs=epochs
    )
    
    print("Training completed. Best model saved to 'best_lstm_model.pth'")

if __name__ == "__main__":
    main()