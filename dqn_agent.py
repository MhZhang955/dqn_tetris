# dqn_agent.py（优化版）
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class DQNAgent:
    def __init__(self, state_size, n_neurons=[64, 64], lr=3e-4,
                 mem_size=20000, discount=0.90, epsilon=1.0,
                 epsilon_min=0.05, epsilon_stop_episode=20000,  # 提高最小探索率
                 replay_start_size=5000, grad_clip=1.0):  # 新增梯度裁剪

        self.state_size = state_size
        self.memory = deque(maxlen=mem_size)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (epsilon - epsilon_min) / epsilon_stop_episode
        self.replay_start_size = replay_start_size
        self.grad_clip = grad_clip  # 梯度裁剪阈值

        # 网络结构保持不变但初始化更稳定
        layers = []
        input_dim = state_size
        for neurons in n_neurons:
            layers.append(nn.Linear(input_dim, neurons))
            layers.append(nn.LayerNorm(neurons))  # 新增层归一化
            layers.append(nn.ReLU())
            input_dim = neurons
        layers.append(nn.Linear(input_dim, 1))

        self.model = nn.Sequential(*layers)
        # 初始化权重
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                nn.init.constant_(layer.bias, 0.0)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)  # 新增L2正则
        self.loss_fn = nn.SmoothL1Loss()  # 改用Huber Loss更稳定

    def best_state(self, states):
        if random.random() < self.epsilon:
            return random.randint(0, len(states) - 1)
        with torch.no_grad():
            states = torch.stack(states)
            q_values = self.model(states)
            return torch.argmax(q_values).item()

    def add_to_memory(self, state, next_state, reward, done):
        self.memory.append((state, next_state, reward, done))

    def train(self, batch_size):
        if len(self.memory) < self.replay_start_size:
            return 0.0

        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states = torch.stack([x[0] for x in batch])
        next_states = torch.stack([x[1] for x in batch])
        rewards = torch.FloatTensor([x[2] for x in batch])
        dones = torch.BoolTensor([x[3] for x in batch])

        with torch.no_grad():
            next_q = self.model(next_states).squeeze()

        targets = rewards + (~dones).float() * self.discount * next_q
        current_q = self.model(states).squeeze()

        loss = self.loss_fn(current_q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # 新增梯度裁剪（关键改进）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)
        return loss.item()