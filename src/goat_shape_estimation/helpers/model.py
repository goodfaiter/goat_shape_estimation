import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from helpers.wrapper import ScaledModelWrapper
from typing import Optional


class RNNModel(nn.Module):
    """RNN Model with PyTorch"""

    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, device=device)
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size, device=device)

    def forward(self, x, h0: Optional[torch.Tensor] = None):
        if h0 is None: # For the training model case
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        
        return out.unsqueeze(1), h0


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
        return out.unsqueeze(1)


class SelfAttentionModel(nn.Module):
    """Attention-based Model with PyTorch"""

    def __init__(self, input_size, embed_dim, num_heads, output_size):
        super(SelfAttentionModel, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True  # Use batch-first format
        )
        # A simple feedforward layer
        self.fc_encoder = nn.Linear(input_size, embed_dim)  # Predict a single value per sequence element
        self.fc_decoder = nn.Linear(embed_dim, output_size)  # Predict a single value per sequence element

    def forward(self, x):
        x = self.fc_encoder(x)
        attn_output, _ = self.attention(x, x, x)  
        output = self.fc_decoder(attn_output)
        return output


class SelfAttentionRNNModel(nn.Module):
    """Attention-based Model with PyTorch"""

    def __init__(self, input_size, embed_dim, num_heads, hidden_size, num_layers, output_size):
        super(SelfAttentionRNNModel, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True  # Use batch-first format
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True)

        # A simple feedforward layer
        self.fc_encoder = nn.Linear(input_size, embed_dim)  # Predict a single value per sequence element
        self.fc_decoder = nn.Linear(hidden_size, output_size)  # Predict a single value per sequence element

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        x = self.fc_encoder(x)
        attn_output, _ = self.attention(x, x, x)

        out, _ = self.rnn(attn_output, h0)
        out = self.fc_decoder(out[:, -1, :])
        return out.unsqueeze(1)


class BeliefEncoderRNNModel(nn.Module):
    """Attention-based Model with PyTorch"""

    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, device=device)

        # A simple feedforward layer
        self.ga = nn.Linear(hidden_size, input_size, device=device)
        self.gb = nn.Linear(hidden_size, input_size, device=device)
        self.fc_decoder = nn.Linear(input_size, output_size, device=device)

    def forward(self, x, h0: Optional[torch.Tensor] = None):
        if h0 is None: # For the training model case
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        latent_b, h0 = self.rnn(x, h0)

        a = self.ga(latent_b)
        a = torch.nn.functional.tanh(a)
        a = x * a # elementwise multiplication

        b = self.gb(latent_b)
        
        out = a + b
        out = self.fc_decoder(out)

        return out[:, -1, :].unsqueeze(1), h0


def train_model(
    model, train_loader, val_loader, criterion, optimizer, device, epochs, file_prefix, input_mean, input_std, output_mean, output_std
):
    """Train the LSTM model"""
    model.to(device)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
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
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        # Calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Save last model
        if epoch == epochs - 1:
            wrapper = ScaledModelWrapper(model, input_mean, input_std, output_mean, output_std)
            wrapper.to('cpu')
            wrapper.freeze()
            wrapper.trace_and_save(file_prefix + "best_lstm_model.pt")
            wrapper.trace_and_save("data/output/latest_lstm_model.pt")
            wrapper.unfreeze()
            wrapper.to(device)

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(file_prefix + "training_history.png")
    plt.close()

    return model
