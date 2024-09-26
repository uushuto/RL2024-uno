import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])

class PPOAgent(object):
    def __init__(self, state_shape, action_shape, device=None,
                 rollout_length=2048):
        
        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.total_t = 0
        self.learning_steps = 0

        self.actor = PPOActor(
            state_shape=state_shape,
            action_shape=action_shape,
        )

        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=self.device
        )
    
    def is_update(self, total_step):
        return total_step % self.rollout_length == 0

    def feed(self, ts):
        (state, action, reward, next_state, done) = tuple(ts)
        _, log_pi = self.explore(state)
        self.buffer.append(state, action, reward, done, log_pi, next_state)
        self.total_t += 1
        if self.is_update(self.total_t):
            self.train()

    def explore(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()
    
    def step(self, state):
        action, _ = self.explore(state)
        return action

class PPOActor(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_shape[0]),
        )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)


class RolloutBuffer:
    def __init__(self, buffer_size, state_shape, action_shape, device=torch.device('cuda')):
        self.states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)
        self.actions = torch.empty((buffer_size, *action_shape), dtype=torch.float, device=device)
        self.rewards = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.dones = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.log_pis = torch.empty((buffer_size, 1), dtype=torch.float, device=device)
        self.next_states = torch.empty((buffer_size, *state_shape), dtype=torch.float, device=device)

        self._p = 0
        self.buffer_size = buffer_size

    def append(self, state, action, reward, done, log_pi, next_state):
        self.states[self._p].copy_(torch.from_numpy(state))
        self.actions[self._p].copy_(torch.from_numpy(action))
        self.rewards[self._p] = float(reward)
        self.dones[self._p] = float(done)
        self.log_pis[self._p] = float(log_pi)
        self.next_states[self._p].copy_(torch.from_numpy(next_state))
        self._p = (self._p + 1) % self.buffer_size

    def get(self):
        assert self._p == 0, 'Buffer needs to be full before training.'
        return self.states, self.actions, self.rewards, self.dones, self.log_pis, self.next_states


def atanh(x):
    return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))

def evaluate_lop_pi(means, log_stds, actions):
    noises = (atanh(actions) - means) / (log_stds.exp() + 1e-8)
    return calculate_log_pi(log_stds, noises, actions)

def calculate_log_pi(log_stds, noises, actions):
    gaussian_log_probs = \
        (-0.5 * noises.pow(2) - log_stds).sum(dim=-1, keepdim=True) - 0.5 * math.log(2 * math.pi) * log_stds.size(-1)
    log_pis = gaussian_log_probs - torch.log(1 - actions.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
    return log_pis

def reparameterize(means, log_stds):
        stds = log_stds.exp()
        noises = torch.randn_like(means)
        us = means + noises * stds
        actions = torch.tanh(us)
        log_pis = calculate_log_pi(log_stds, noises, actions)
        return actions, log_pis

# MEMO TODO
# - PPO, def feed, tuple->bufferにおける変数構造の調整
# - PPO, def step, log_piをtsのstate指定（tranjectories[0][-1][0]のstate['obs']）で計算
