import torch
from torch import nn
from torch.nn import functional as F

class network(nn.Module):
    def __init__(self, num_states, num_actions):
        super(network, self).__init__()

        self.Value_net = Value_net(num_states)
        self.Policy_net = Policy_net(num_states, num_actions)

    def forward(self, x):
        state_value = self.Value_net(x)
        pi = self.Policy_net(x)
        return state_value, pi

class Value_net(nn.Module):
    def __init__(self, num_states):
        super(Value_net, self).__init__()
        self.fc1 = nn.Linear(num_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        value = self.value(x)
        return value

class Policy_net(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Policy_net, self).__init__()
        self.fc1 = nn.Linear(num_states, 64)
        self.fc2 = nn.Linear(64, 64)
        self.action_mean = nn.Linear(64, num_actions)
        self.sigma_log = nn.Parameter(torch.zeros(1, num_actions))

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        mean = self.action_mean(x)
        sigma_log = self.sigma_log.expand_as(mean)
        sigma = torch.exp(sigma_log)
        pi = (mean, sigma)
        
        return pi
