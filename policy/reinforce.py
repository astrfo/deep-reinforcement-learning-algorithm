import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class REINFORCE:
    def __init__(self):
        self.alpha = 0.0002
        self.gamma = 0.98
        self.device = torch.device('cpu')
        self.model = PolicyNet(input_size=4, hidden_size=128, output_size=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.memory = []

    def reset(self):
        self.model = PolicyNet(input_size=4, hidden_size=128, output_size=2)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    def action(self, state):
        s = torch.tensor(state[np.newaxis, :])
        prob = self.model(s)
        prob = prob[0]
        action = Categorical(prob).sample().item()
        return action, prob[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        Gt, loss = 0, 0
        for r, p in reversed(self.memory):
            Gt = r + self.gamma * Gt
            loss += -torch.log(p) * Gt

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.memory = []


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