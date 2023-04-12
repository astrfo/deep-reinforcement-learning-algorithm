import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CategoricalDQN:
    def __init__(self):
        self.alpha = 0.01
        self.gamma = 0.98
        self.epsilon = 0.1
        self.tau = 0.1
        self.v_min = -10
        self.v_max = 10
        self.n_atoms = 51
        self.buffer_size = 10**4
        self.batch_size = 32
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.batch_size)
        self.device = torch.device('cpu')
        self.model = QNet(input_size=4, hidden_size=128, output_size=2*self.n_atoms)
        self.model.to(self.device)
        self.model_target = QNet(input_size=4, hidden_size=128, output_size=2*self.n_atoms)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    def reset(self):
        self.replay_buffer.reset()
        self.model = QNet(input_size=4, hidden_size=128, output_size=2*self.n_atoms)
        self.model.to(self.device)
        self.model_target = QNet(input_size=4, hidden_size=128, output_size=2*self.n_atoms)
        self.model_target.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)

    def q_value(self, state):
        s = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(s).view(-1, 2, self.n_atoms).to('cpu').detach().numpy().copy()

    def action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(2)
        else:
            q_values = np.dot(self.q_value(state), np.linspace(self.v_min, self.v_max, self.n_atoms))
            action = np.argmax(q_values, axis=-1)[0]
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

        z = self.model(s).view(-1, 2, self.n_atoms)
        z = F.softmax(z, dim=2)
        
        z_prime = self.model_target(ns).view(-1, 2, self.n_atoms)
        z_prime = F.softmax(z_prime, dim=2)
        q_values = z * torch.linspace(self.v_min, self.v_max, self.n_atoms).to(self.device)
        na = q_values.sum(2).argmax(1).unsqueeze(1).unsqueeze(1).expand(-1, -1, self.n_atoms).long()
        z_prime = z_prime.gather(1, na).squeeze(1)

        target_z = r.unsqueeze(1) + self.gamma * z_prime * (1 - d.unsqueeze(1))
        target_z = target_z.clamp(self.v_min, self.v_max)

        b = (target_z - self.v_min) / (self.v_max - self.v_min) * (self.n_atoms - 1)
        lower_bound = b.floor().long()
        upper_bound = b.ceil().long()
        lower_residue = (upper_bound.float() - b)
        upper_residue = (b - lower_bound.float())

        target_z_projected = torch.zeros(self.batch_size, self.n_atoms).to(self.device)
        target_z_projected.scatter_add_(1, lower_bound, lower_residue)
        target_z_projected.scatter_add_(1, upper_bound, upper_residue)

        z = z.gather(2, torch.tensor(a).unsqueeze(1).unsqueeze(1).expand(-1, -1, self.n_atoms).long()).squeeze(1)
        loss = -(target_z_projected * z.log()).sum(1).mean()

        self.optimizer.zero_grad()
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