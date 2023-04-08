import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DDQN:
    def __init__(self):
        self.alpha = 0.01
        self.gamma = 0.98
        self.epsilon = 0.1
        self.tau = 0.1
        self.buffer_size = 10**4
        self.batch_size = 32
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.device = torch.device('cpu')
        self.model = QNet(input_size=4, hidden_size=128, output_size=2)
        self.model.to(self.device)
        self.model_target = QNet(input_size=4, hidden_size=128, output_size=2)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)
        self.criterion = nn.MSELoss()

    def reset(self):
        self.replay_buffer.reset()
        self.model = QNet(input_size=4, hidden_size=128, output_size=2)
        self.model.to(self.device)
        self.model_target = QNet(input_size=4, hidden_size=128, output_size=2)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    def q_value(self, state):
        s = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(s).squeeze().to('cpu').detach().numpy().copy()

    def action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(2)
        else:
            q_values = self.q_value(state)
            action = np.random.choice(np.where(q_values == max(q_values))[0])
        return action

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        if len(self.replay_buffer.memory) < self.batch_size:
            return

        s, a, r, ns, d = self.replay_buffer.encode()
        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        ns = torch.tensor(ns, dtype=torch.float32).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        d = torch.tensor(d, dtype=torch.float32).to(self.device)

        q = self.model(s)
        qa = q[np.arange(self.batch_size), a]
        next_q = self.model(ns)
        next_qa = torch.argmax(next_q, dim=1, keepdim=True)
        next_q_target = self.model_target(ns)
        next_qa_target = next_q_target.gather(1, next_qa).squeeze()
        target = r + self.gamma * next_qa_target * (1 - d)

        self.optimizer.zero_grad()
        loss = self.criterion(qa, target)
        loss.backward()
        self.optimizer.step()
        self.sync_model()

    def sync_model(self):
        # self.model_target.load_state_dict(self.model.state_dict())
        with torch.no_grad():
            for target_param, local_param in zip(self.model_target.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau*local_param.data + (np.float64(1.0)-self.tau)*target_param.data)


class QNet(nn.Module):
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


class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory_size = buffer_size
        self.batch_size = batch_size
        self.memory = deque(maxlen=self.memory_size)

    def reset(self):
        self.memory = deque(maxlen=self.memory_size)

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def encode(self):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        indices = np.random.randint(0, len(self.memory), self.batch_size)
        for index in indices:
            s, a, r, ns, d = self.memory[index]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)