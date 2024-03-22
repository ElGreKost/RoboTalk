import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch import from_numpy, no_grad, save, load, tensor, clamp
from torch import float as torch_float
from torch import long as torch_long
from torch import min as torch_min
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import numpy as np
from torch import manual_seed
from collections import namedtuple


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from collections import namedtuple, defaultdict
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])

class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.network(x)


class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x)


class PPOAgent:
    def __init__(self, input_size, output_size, clip_param=0.2, ppo_epochs=4, mini_batch_size=64, gamma=0.99, lambda_gae=0.95, actor_lr=3e-4, critic_lr=3e-4):
        self.actor = Actor(input_size, output_size)
        self.critic = Critic(input_size)
        self.clip_param = clip_param
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.buffer = []

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state):
        state = torch.from_numpy(np.array(state)).float().unsqueeze(0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def store_transition(self, transition):
        self.buffer.append(transition)

    def calculate_advantages(self, rewards, values, next_values):
        deltas = [rewards[i] + self.gamma * next_values[i] - values[i] for i in range(len(rewards))]
        advantages = []
        advantage = 0.0
        for delta in reversed(deltas):
            advantage = delta + self.gamma * self.lambda_gae * advantage
            advantages.insert(0, advantage)
        return advantages

    def train(self):
        transitions = Transition(*zip(*self.buffer))
        states = torch.tensor(transitions.state, dtype=torch.float32)
        actions = torch.tensor(transitions.action, dtype=torch.int64).view(-1, 1)
        rewards = torch.tensor(transitions.reward, dtype=torch.float32)
        old_log_probs = torch.tensor(transitions.a_log_prob, dtype=torch.float32).view(-1, 1)

        # Calculate state values and next state values
        values = self.critic(states)
        next_values = torch.cat((values[1:], torch.tensor([[0.0]])), dim=0)

        # Calculate advantages and returns
        advantages = self.calculate_advantages(rewards, values.detach(), next_values.detach())
        advantages = torch.tensor(advantages, dtype=torch.float32).detach()
        returns = advantages + values.detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        for _ in range(self.ppo_epochs):
            sampler = BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.mini_batch_size, drop_last=False)
            for indices in sampler:
                sampled_states = states[indices]
                sampled_actions = actions[indices]
                sampled_old_log_probs = old_log_probs[indices]
                sampled_returns = returns[indices]
                sampled_advantages = advantages[indices]

                # Get current policy outputs for sampled states
                new_log_probs = self.actor(sampled_states).gather(1, sampled_actions)
                entropy = Categorical(self.actor(sampled_states)).entropy().mean()
                ratio = (new_log_probs - sampled_old_log_probs).exp()

                # Clipped surrogate function
                surr1 = ratio * sampled_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * sampled_advantages
                actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy

                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_optimizer.step()

                # Update critic
                critic_loss = F.mse_loss(self.critic(sampled_states).view(-1), sampled_returns.view(-1))
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.critic_optimizer.step()

        # Clear the buffer
        del self.buffer[:]
