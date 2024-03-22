import gymnasium as gym
import torch
import random

from torch.optim import Adam
from torch import nn
from replay_buffers import ReplayBuffer
from DQN import DQN


class BaseAgent:

    def __init__(self, _env: gym.envs, use_conv=True, _lr=3e-4, _gamma=0.99,
                 _buffer_size=10000, _batch_size=64, _model_path='best_model.pth', eval_mode=False):
        self.env = _env
        self.lr = _lr
        self.gamma = _gamma
        self.epsilon = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.replay_buffer = ReplayBuffer(_buffer_size)
        self.model_path = _model_path
        self.eval_mode = eval_mode
        self.batch_size = _batch_size
        self.frame = 0
        self.learn_start = self.batch_size
        self.update_freq = 1

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.use_conv = use_conv
        self.net = DQN(_env.observation_space.shape[0], _env.action_space.n, use_conv=use_conv).to(self.device)
        self.optimizer = Adam(self.net.parameters())
        self.MSE_loss = nn.MSELoss()

    def get_action(self, _state: torch.tensor):  # what is the type of state?
        _state = _state.unsqueeze(0).to(self.device)
        qvals = self.net(_state)
        _action = torch.argmax(qvals).item()
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

        return self.env.action_space.sample() if random.random() < self.epsilon and not self.eval_mode else _action

    def compute_loss(self, _batch):
        states, actions, rewards, next_states, dones = zip(*_batch)
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions).to(self.device)
        rewards = torch.tensor(rewards).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones_tensors = [torch.tensor([done], dtype=torch.float32, device=self.device) for done in dones]
        dones = torch.stack(dones_tensors).to(self.device)

        Q_expected = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)  # (B)
        Q_target_next = self.net(next_states).detach().max(1)[0]  # (B, A)

        Q_target = rewards + self.gamma * Q_target_next * (1 - dones)

        loss = self.MSE_loss(Q_expected, Q_target).float()
        return loss

    def append_to_replay(self, s, a, r, s_, te, tr):
        """
        Virtual Function:
        Used and implemented in the child class that has multistep learning"""
        pass

    def update(self, *args):
        self.frame += 1
        self.append_to_replay(*args)
        if self.frame < self.learn_start or self.frame % self.update_freq != 0:
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load(self, model_path=None):
        if model_path is None:
            model_path = self.model_path
        self.net.load_state_dict(torch.load(model_path))
