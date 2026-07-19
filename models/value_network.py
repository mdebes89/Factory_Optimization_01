"""Value Network"""
import torch
import torch.nn as nn

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_size=128):
        super(ValueNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.network(x).squeeze(-1)
