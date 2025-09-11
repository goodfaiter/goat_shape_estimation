import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from helpers.wrapper import ScaledModelWrapper
from typing import Optional
import torch.optim as optim
from torch.optim import lr_scheduler


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

    def __init__(self, input_size, latent_size, hidden_size, num_layers, output_size, device, noisy_index):
        super().__init__()
        self.noisy_index = noisy_index
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers

        self.fc = nn.Linear(input_size, self.latent_size, device=device)

        self.rnn = nn.RNN(self.latent_size, hidden_size, num_layers, batch_first=True, device=device)

        # A simple feedforward layer
        self.h_to_t = nn.Linear(hidden_size, len(self.noisy_index), device=device)
        self.t_to_l = nn.Linear(len(self.noisy_index), self.latent_size, device=device)
        self.h_to_l = nn.Linear(hidden_size, self.latent_size, device=device)
        self.fc_decoder = nn.Linear(self.latent_size, output_size, device=device)

    def forward(self, x, h0: Optional[torch.Tensor] = None):
        if h0 is None: # For the training model case
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        x = self.fc(x)
        x = torch.nn.functional.tanh(x)

        latent_b, h0 = self.rnn(x, h0)

        a = self.h_to_t(latent_b)
        a = torch.nn.functional.tanh(a)
        a = x[:, :, self.noisy_index] * a # elementwise multiplication
        a = self.t_to_l(a)

        b = self.h_to_l(latent_b)
        
        out = torch.nn.functional.tanh(a + b)
        out = self.fc_decoder(out)

        return out[:, -1, :].unsqueeze(1), h0


class MLP(nn.Module):
    def __init__(self, input_dim, history_dim, hidden_sizes, output_dim):
        """
        Args:
            input_dim (int): Dimension of input features (DIM if flattening [B, H, DIM]).
            hidden_sizes (list): List of hidden layer sizes (e.g., [64, 32]).
            output_dim (int): Output dimension.
            flatten_input (bool): If True, reshapes [B, H, DIM] → [B, H * DIM].
        """
        super(MLP, self).__init__()
        
        # Build layers dynamically
        layers = []
        prev_size = input_dim * history_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.Tanh())
            prev_size = hidden_size
        
        # Final layer (no activation)
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x.flatten(start_dim=1))


def train_model(
    model, train_loader, val_loader, device, epochs, file_prefix, input_mean, input_std, output_mean, output_std
):
    """Train the LSTM model"""
    model.to(device)
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            # loss = criterion(outputs, targets)
            loss = torch.pow(outputs - targets, 2) / (2.0 + targets.abs())
            loss = loss.mean()
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

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = min(best_val_loss, val_loss)
            wrapper = ScaledModelWrapper(model, input_mean, input_std, output_mean, output_std)
            wrapper.to('cpu')
            wrapper.freeze()
            path = file_prefix + "best_lstm_model.pt"
            wrapper.trace_and_save(file_prefix + "best_lstm_model.pt")
            wrapper.trace_and_save("data/output/best_lstm_model.pt")
            print(f"Saving best polcies to /workspace/{path}")
            wrapper.unfreeze()
            wrapper.to(device)

        if epoch % 10 == 0:
            best_val_loss = min(best_val_loss, val_loss)
            wrapper = ScaledModelWrapper(model, input_mean, input_std, output_mean, output_std)
            wrapper.to('cpu')
            wrapper.freeze()
            path = file_prefix + f"{epoch}.pt"
            wrapper.trace_and_save(path)
            wrapper.trace_and_save("data/output/latest_lstm_model.pt")
            print(f"Saving latest polcies to /workspace/{path}")
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
