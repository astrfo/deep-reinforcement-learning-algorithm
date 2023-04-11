import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyGradient:
    def __init__(self):
        self.alpha = 0.0002
        self.gamma = 0.98
        self.device = torch.device('cpu')
        self.model = PolicyNet(input_size=4, hidden_size=128, output_size=2)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.memory = []

    def reset(self):
        self.model = PolicyNet(input_size=4, hidden_size=128, output_size=2)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    def action(self, state):
        s = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            prob = self.model(s).squeeze().to('cpu').detach().numpy().copy()
        action = np.random.choice(2, p=prob)
        return action, prob[action]

    def add(self, reward, prob):
        data = (reward, prob)
        self.memory.append(data)

    def update(self):
        G, loss = 0, 0
        for r, p in reversed(self.memory):
            G = r + self.gamma * G
        
        for r, p in self.memory:
            loss += -np.log(p) * G
        loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True).to(self.device)

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
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.softmax(x)
        return x