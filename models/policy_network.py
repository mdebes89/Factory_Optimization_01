"""Policy Network"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(PolicyNetwork, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        action_logits = self.policy_head(features)
        state_value = self.value_head(features).squeeze(-1)
        return action_logits, state_value
    
    def get_action(self, state, deterministic=False):
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0)
        action_logits, _ = self.forward(state)
        dist = torch.distributions.Categorical(logits=action_logits)
        if deterministic:
            action = torch.argmax(action_logits, dim=-1).item()
        else:
            action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action))
        return action, log_prob.item()
