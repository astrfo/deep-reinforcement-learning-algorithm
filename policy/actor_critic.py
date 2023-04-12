import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class ActorCritic:
    def __init__(self):
        self.alpha_p = 0.0002
        self.alpha_v = 0.0005
        self.gamma = 0.98
        self.model_p = PolicyNet(input_size=4, hidden_size=128, output_size=2)
        self.model_v = ValueNet(input_size=4, hidden_size=128, output_size=1)
        self.optimizer_p = optim.Adam(self.model_p.parameters(), lr=self.alpha_p)
        self.optimizer_v = optim.Adam(self.model_v.parameters(), lr=self.alpha_v)
        self.criterion = nn.MSELoss()

    def reset(self):
        self.model_p = PolicyNet(input_size=4, hidden_size=128, output_size=2)
        self.model_v = ValueNet(input_size=4, hidden_size=128, output_size=1)
        self.optimizer_p = optim.Adam(self.model_p.parameters(), lr=self.alpha_p)
        self.optimizer_v = optim.Adam(self.model_v.parameters(), lr=self.alpha_v)

    def action(self, state):
        s = torch.tensor(state[np.newaxis, :])
        prob = self.model_p(s)
        prob = prob[0]
        action = Categorical(prob).sample().item()
        return action, prob[action]

    def update(self, state, action_prob, reward, next_state, done):
        s = torch.tensor(state[np.newaxis, :])
        ns = torch.tensor(next_state[np.newaxis, :])

        target = reward + self.gamma * self.model_v(ns) * (1 - done)
        target.detach()

        v = self.model_v(s)
        loss_v = self.criterion(v, target)

        delta = target - v
        loss_p = -torch.log(action_prob) * delta.item()

        self.optimizer_v.zero_grad()
        self.optimizer_p.zero_grad()
        loss_v.backward()
        loss_p.backward()
        self.optimizer_v.step()
        self.optimizer_p.step()


class PolicyNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(x, dim=1)
        return x


class ValueNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x