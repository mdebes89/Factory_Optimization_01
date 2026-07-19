"""PPO Algorithm"""
import torch
import torch.optim as optim
import torch.nn.functional as F

class PPO:
    def __init__(self, policy_network, config=None):
        self.policy_network = policy_network
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network.to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.config.get('lr', 3e-4))
        self.gamma = self.config.get('gamma', 0.99)
        self.clip_epsilon = self.config.get('clip_epsilon', 0.2)
        self.ppo_epochs = self.config.get('ppo_epochs', 4)
        self.value_loss_coef = self.config.get('value_loss_coef', 0.5)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
    
    def get_action(self, state, deterministic=False):
        return self.policy_network.get_action(state, deterministic)
    
    def save(self, path):
        torch.save(self.policy_network.state_dict(), path)
    
    def load(self, path):
        self.policy_network.load_state_dict(torch.load(path, map_location=self.device))
