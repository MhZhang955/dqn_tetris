import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from torch.utils.tensorboard import SummaryWriter


class DQNAgent:
    def __init__(self, state_size, n_neurons=[128, 128], lr=3e-5,
                 mem_size=20000, discount=0.95, epsilon=1.0,
                 epsilon_min=0.1, epsilon_stop_episode=2000,
                 replay_start_size=5000, grad_clip=0.5):

        # Environment parameters
        self.state_size = state_size
        self.discount = discount

        # Exploration parameters
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = (epsilon - epsilon_min) / epsilon_stop_episode

        # Memory buffer
        self.memory = deque(maxlen=mem_size)
        self.replay_start_size = replay_start_size

        # Network architecture (modified BatchNorm handling)
        layers = []
        input_dim = state_size
        for neurons in n_neurons:
            layers.extend([
                nn.Linear(input_dim, neurons),
                nn.BatchNorm1d(neurons),
                nn.ReLU()
            ])
            input_dim = neurons
        layers.append(nn.Linear(input_dim, 1))

        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            eps=1e-7,
            weight_decay=1e-5
        )
        self.loss_fn = nn.MSELoss()
        self.grad_clip = grad_clip

        # Tracking
        self.total_steps = 0
        self.writer = SummaryWriter()
        self.train_mode = True

    def set_train_mode(self, mode):
        """Control BatchNorm and Dropout behavior"""
        self.train_mode = mode
        if mode:
            self.model.train()
        else:
            self.model.eval()

    def best_state(self, states):
        """Select best state with proper BatchNorm handling"""
        if random.random() < self.epsilon:
            return random.randint(0, len(states) - 1)

        self.set_train_mode(False)  # Switch to eval mode
        with torch.no_grad():
            states = torch.stack(states)
            # Handle single sample case
            if len(states) == 1:
                states = torch.cat([states, states.clone()])  # Duplicate
                q_values = self.model(states)[0:1]  # Take first prediction
            else:
                q_values = self.model(states)
            best_idx = torch.argmax(q_values).item()
        self.set_train_mode(self.train_mode)  # Restore original mode
        return best_idx

    def add_to_memory(self, state, next_state, reward, done):
        """Store transition with automatic tensor conversion"""
        self.memory.append((
            torch.FloatTensor(state),
            torch.FloatTensor(next_state),
            float(reward),
            bool(done)
        ))

    def train(self, batch_size):
        """Training step with gradient clipping"""
        if len(self.memory) < self.replay_start_size:
            return 0.0

        self.set_train_mode(True)

        # Sample batch
        batch = random.sample(self.memory, min(batch_size, len(self.memory)))
        states = torch.stack([x[0] for x in batch])
        next_states = torch.stack([x[1] for x in batch])
        rewards = torch.FloatTensor([x[2] for x in batch])
        dones = torch.BoolTensor([x[3] for x in batch])

        # Compute targets
        with torch.no_grad():
            next_q = self.model(next_states).squeeze()
        targets = rewards + (~dones).float() * self.discount * next_q

        # Compute loss
        current_q = self.model(states).squeeze()
        loss = self.loss_fn(current_q, targets)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        # Logging
        self.total_steps += 1
        self.writer.add_scalar("Loss/train", loss.item(), self.total_steps)
        self.writer.add_scalar("Params/epsilon", self.epsilon, self.total_steps)

        return loss.item()

    def save_model(self, path):
        """Save complete model state"""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'total_steps': self.total_steps,
            'config': {
                'state_size': self.state_size,
                'n_neurons': [layer.out_features
                              for layer in self.model
                              if isinstance(layer, nn.Linear)][:-1]
            }
        }, path)

    def load_model(self, path):
        """Load complete model state"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.epsilon = checkpoint['epsilon']
        self.total_steps = checkpoint['total_steps']