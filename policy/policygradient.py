import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PolicyGradient:
    def __init__(self):
        self.alpha = 0.01
        self.gamma = 0.98
        self.device = torch.device('cpu')
        self.model = PolicyNet(input_size=4, hidden_size=128, output_size=2)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    def reset(self):
        self.model = PolicyNet(input_size=4, hidden_size=128, output_size=2)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    def episode_reset(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def action(self, state):
        s = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            pi = self.model(s)
        action = torch.multinomial(pi, num_samples=1).item()
        return action

    def update(self, state, action, reward, next_state, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

        if not done:
            return
        
        states = torch.tensor(np.array(self.states), dtype=torch.float32).to(self.device)
        pi = self.model(states)
        selected_log_probs = torch.log(pi.gather(1, torch.tensor(self.actions).view(-1, 1).to(self.device)))

        discounted_rewards = []
        for t in range(len(self.rewards)):
            Gt = 0
            for r, step_reward in enumerate(self.rewards[t:]):
                Gt += step_reward * (self.gamma ** r)
            discounted_rewards.append(Gt)

        discounted_rewards = torch.tensor(discounted_rewards).to(self.device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        loss = []
        for log_prob, Gt in zip(selected_log_probs, discounted_rewards):
            loss.append(-log_prob * Gt)

        self.optimizer.zero_grad()
        loss = torch.cat(loss).sum()
        loss.backward()
        self.optimizer.step()


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