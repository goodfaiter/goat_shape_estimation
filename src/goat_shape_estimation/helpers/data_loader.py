import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class ROSBagDataLoader:
    def __init__(self):
        self.data = {}
        self.timestamps = []
        self.scaler = None
        
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