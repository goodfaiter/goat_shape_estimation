import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

class ROSBagDataset(Dataset):
    """PyTorch Dataset for ROS bag data"""
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.targets[idx])

class LSTMModel(nn.Module):
    """LSTM Model with PyTorch"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class ROSBagDataLoader:
    def __init__(self):
        self.data = {}
        self.timestamps = []
        
    def load_bag(self, bag_path, topics):
        """Load data from a ROS1 or ROS2 bag file"""
        with AnyReader([Path(bag_path)]) as reader:
            for connection in reader.connections:
                if connection.topic in topics:
                    for _, timestamp, rawdata in reader.messages(connections=[connection]):
                        msg = reader.deserialize(rawdata, connection.msgtype)
                        self._process_message(connection.topic, timestamp, msg)
    
    def _process_message(self, topic, timestamp, msg):
        """Process individual ROS messages and store data"""
        # Get timestamp (works for both ROS1 and ROS2)
        ts = getattr(msg, 'header', msg).stamp if hasattr(msg, 'header') else timestamp
        self.timestamps.append(ts)
        
        # Extract data (customize based on your message types)
        if not hasattr(msg, '__slots__'):
            return
            
        for field in msg.__slots__:
            val = getattr(msg, field)
            if isinstance(val, (int, float)):
                if f"{topic}/{field}" not in self.data:
                    self.data[f"{topic}/{field}"] = []
                self.data[f"{topic}/{field}"].append(float(val))
    
    def create_sequences(self, sequence_length=50, target_length=1, test_size=0.2):
        """Convert the data into sequences for LSTM training"""
        # Create DataFrame from collected data
        df = pd.DataFrame(self.data)
        df = df.interpolate().fillna(method='bfill')  # Handle missing values
        
        # Normalize data
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(df.values)
        
        # Create sequences and targets
        sequences = []
        targets = []
        
        for i in range(len(scaled_data) - sequence_length - target_length):
            sequences.append(scaled_data[i:i+sequence_length])
            targets.append(scaled_data[i+sequence_length:i+sequence_length+target_length, 0])  # Predicting first feature
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(sequences), np.array(targets), test_size=test_size, shuffle=False
        )
        
        return X_train, X_test, y_train, y_test

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=100):
    """Train the LSTM model"""
    model.to(device)
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        # Calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_lstm_model.pth')
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    
    return model

def main():
    # Configuration
    bag_paths = [
        ('/workspace/data/first_time_goat_rosbag2_2025_05_29-13_48_18', ['/topic3', '/topic4']),
        ('/workspace/data/2025-05-29-16-42-22.bag', ['/topic1', '/topic2'])
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