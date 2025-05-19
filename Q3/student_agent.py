import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=256, scale=1.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu = nn.Linear(hidden, output_dim)
        self.log_std = nn.Linear(hidden, output_dim)
        self.output_scale = scale

    def forward(self, obs):
        x = F.relu(self.fc1(obs))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), min=-20, max=2)
        return mu, log_std.exp()

    def predict_action(self, obs_tensor):
        mu, _ = self.forward(obs_tensor)
        return torch.tanh(mu) * self.output_scale

class Agent(object):
    def __init__(self):
        self.state_dim = 67  
        self.action_dim = 21
        self.max_action = 1.0
        self.hidden_dim = 256

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(self.state_dim, self.action_dim, self.hidden_dim, self.max_action)
        checkpoint = torch.load("upload.pth", map_location=self.device)
        self.policy.load_state_dict(checkpoint["actor"])
        self.policy.to(self.device)
        self.policy.eval()

    def act(self, observation):
        obs_tensor = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.policy.predict_action(obs_tensor)
        return action.cpu().numpy().flatten() 
