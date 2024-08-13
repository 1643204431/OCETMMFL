import torch
import torch.nn as nn
import torch.nn.functional as F

batch_size = 16

class modality_model(nn.Module):
    def __init__(self, in_channels=1, num_classes=6, hidden_dim=84, conv_kernel_size=5, pool_kernel_size=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 3, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(3, 3, kernel_size=conv_kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=2)
        )


        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 6),
            # nn.ReLU(),
            # nn.Linear(32, num_classes),
        )


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out


class weight(nn.Module):
    def __init__(self):
        super(weight, self).__init__()
        self.parm = torch.nn.Parameter(torch.FloatTensor([0.5]), requires_grad=True)
    def forward(self, x):
        return self.parm * x

class FusionNet(nn.Module):
    def __init__(self, n_modalities=9):
        super().__init__()
        self.n_modalities = n_modalities
        self.weights = nn.ModuleList()
        for m in range(n_modalities):
            self.weights.append(weight())

    def forward(self, deep_features):
        outs = []
        for m in range(self.n_modalities):
            outs.append(self.weights[m](deep_features[m]))
        return sum(outs)
