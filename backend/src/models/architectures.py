"""Neural Network Architectures for DeviceGuard AI Failure Prediction.

Contains LSTM and GRU sequential models designed for temporal SMART telemetry.
"""

import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    """LSTM-based binary classifier for hard-drive failure early warning."""

    def __init__(self, input_size: int = 11, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Logits tensor of shape (batch_size, 1)
        """
        # lstm_out: (batch_size, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)
        # Take the output from the final time step
        last_step = lstm_out[:, -1, :]
        out = self.dropout(last_step)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits


class GRUClassifier(nn.Module):
    """GRU-based binary classifier for hard-drive failure early warning."""

    def __init__(self, input_size: int = 11, hidden_size: int = 32, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            Logits tensor of shape (batch_size, 1)
        """
        gru_out, _ = self.gru(x)
        last_step = gru_out[:, -1, :]
        out = self.dropout(last_step)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits
