"""PPO Algorithm"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class PPO:
    def __init__(self, policy_network, config=None):
        self.policy_network = policy_network
        self.config = config or {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_network.to(self.device)
        
        # Hyperparameters with defaults
        self.lr = self.config.get('lr', 3e-4)
        self.gamma = self.config.get('gamma', 0.99)
        self.clip_epsilon = self.config.get('clip_epsilon', 0.2)
        self.ppo_epochs = self.config.get('ppo_epochs', 4)
        self.batch_size = self.config.get('batch_size', 64)
        self.value_loss_coef = self.config.get('value_loss_coef', 0.5)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.max_grad_norm = self.config.get('max_grad_norm', 0.5)
        self.gae_lambda = self.config.get('gae_lambda', 0.95)
        self.num_mini_batches = self.config.get('num_mini_batches', 4)
        
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        
        # Storage for training data
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
    
    def get_action(self, state, deterministic=False):
        return self.policy_network.get_action(state, deterministic)
    
    def save(self, path):
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'config' in checkpoint:
            self.config.update(checkpoint['config'])
    
    def store_transition(self, state, action, log_prob, reward, done, value):
        """Store a transition for later training"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
    
    def compute_returns_and_advantages(self):
        """Compute returns and advantages using GAE (Generalized Advantage Estimation)"""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        # Compute returns
        returns = np.zeros_like(rewards)
        discounted_sum = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1] * (1 - dones[t + 1])
            
            discounted_sum = rewards[t] + self.gamma * next_value
            returns[t] = discounted_sum
        
        # Compute advantages using GAE
        advantages = np.zeros_like(rewards)
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_gae = 0
            else:
                next_value = values[t + 1] * (1 - dones[t + 1])
                next_gae = advantages[t + 1]
            
            delta = rewards[t] + self.gamma * next_value - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_gae
            advantages[t] = gae
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        return torch.FloatTensor(returns).to(self.device), torch.FloatTensor(advantages).to(self.device)
    
    def update(self):
        """Perform PPO update using stored transitions"""
        if len(self.states) == 0:
            return
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        old_actions = torch.LongTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        old_values = torch.FloatTensor(np.array(self.values)).to(self.device)
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages()
        
        # Normalize states
        # states = (states - states.mean()) / (states.std() + 1e-8)
        
        # Optimize policy for K epochs
        for _ in range(self.ppo_epochs):
            # Get action logits and state values
            action_logits, state_values = self.policy_network(states)
            
            # Calculate new log probabilities
            dist = torch.distributions.Categorical(logits=action_logits)
            new_log_probs = dist.log_prob(old_actions)
            
            # Calculate ratio (pi_theta / pi_theta_old)
            ratio = (new_log_probs - old_log_probs).exp()
            
            # Calculate surrogate losses
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(state_values.squeeze(), returns)
            
            # Entropy bonus
            entropy = dist.entropy().mean()
            
            # Total loss
            loss = (policy_loss + 
                   self.value_loss_coef * value_loss - 
                   self.entropy_coef * entropy)
            
            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
        
        # Clear stored transitions
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        return loss.item()
    
    def clear_memory(self):
        """Clear stored transitions"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
