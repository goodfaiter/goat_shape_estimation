from torch.utils.data import Dataset

class GoatDataset(Dataset):
    """PyTorch Dataset for ROS bag data"""
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]