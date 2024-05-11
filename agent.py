import random
from collections import namedtuple, deque

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    def __init__(self,action_size,state_size,cfg):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(cfg.replay_memory_size)
        self.device = cfg.device
        self.gamma = cfg.gamma  # Discount rate
        self.epsilon = cfg.epsilon  # Exploration rate
        self.epsilon_min = cfg.epsilon_min
        self.epsilon_decay =  cfg.epsilon_decay
        self.learning_rate =  cfg.learning_rate
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        self.criterion = nn.MSELoss()
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
        return np.argmax(action_values.gpu().data.numpy())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            target = reward
            if not done:
                next_state = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
                target = reward + self.gamma * torch.max(self.model(next_state).detach())
            target_f = self.model(torch.from_numpy(state).float().unsqueeze(0).to(self.device))
            target_f[0][action] = target
            targets.append(target_f)
        states = torch.from_numpy(np.vstack(states)).float().to(self.device)
        targets = torch.vstack(targets).to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(states)
        loss = self.criterion(outputs, targets)
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay