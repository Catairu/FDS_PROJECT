import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# ============================================================
#  TEMPORAL BLOCK (residual, weight norm, causal padding)
# ============================================================

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()

        # Padding necessario per la causalità
        padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            )
        )
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            )
        )
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        # Residual connection se cambia il numero di canali
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else None

    def forward(self, x):
        out = self.conv1(x)
        out = out[..., :x.size(2)]   # trim right padding
        out = self.relu1(out)
        out = self.drop1(out)

        out2 = self.conv2(out)
        out2 = out2[..., :x.size(2)]  # trim
        out2 = self.relu2(out2)
        out2 = self.drop2(out2)

        res = x if self.downsample is None else self.downsample(x)

        return nn.ReLU()(out2 + res)


# ============================================================
#  FULL TCN (6 layers, exponential dilation)
# ============================================================

class TCN(nn.Module):
    def __init__(self,
                 input_channels=64,
                 channels=[64, 64, 128, 128, 128, 128],
                 kernel_size=5,
                 dropout=0.3):
        super().__init__()

        layers = []
        for i in range(len(channels)):
            dilation = 2 ** i  # 1, 2, 4, 8, 16, 32

            in_ch = input_channels if i == 0 else channels[i - 1]
            out_ch = channels[i]

            layers.append(
                TemporalBlock(
                    in_ch,
                    out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# ============================================================
#  CLASSIFIER (6 classes)
# ============================================================

class TCNClassifier(nn.Module):
    def __init__(self, num_classes=6, cnn_output_channels=64):
        super().__init__()

        self.tcn = TCN(
            input_channels=cnn_output_channels, 
            channels=[64, 64, 128, 128],        
            kernel_size=3,                     
            dropout=0.3
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.tcn(x)       
        out = out.mean(dim=-1) 
        return self.fc(out)