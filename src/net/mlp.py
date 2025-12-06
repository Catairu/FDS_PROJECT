import torch
import torch.nn as nn


class MLPBlockLazy(nn.Module):
    """
    MLP modulare con LazyLinear e LazyBatchNorm1d.
    Adatto a collegare CNN+LSTM senza conoscere in_features.
    """

    def __init__(self, hidden_sizes=[128, 64], output_size=6, dropout=0.5):
        super().__init__()

        self.fc1 = nn.LazyLinear(hidden_sizes[0])
        self.bn1 = nn.LazyBatchNorm1d()
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden_sizes[1], output_size)

    def forward(self, x):

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.drop1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.drop2(x)

        x = self.fc_out(x)
        return x
