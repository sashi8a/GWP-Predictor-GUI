import torch
import torch.nn as nn

class CustomNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CustomNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),  
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),  
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)
