import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])

class PPOAgent(object):
    def __init__(self,
                 state_shape,
                 action_shape,
                 device=None):
        
        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.actor = PPOActor(
            state_shape=state_shape,
            action_shape=action_shape,
        )

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
# - PPO, def step, Buffer.appendをdef feedで実装
# - PPO, def step, log_piをtsのstate指定（tranjectories[0][-1][0]のstate['obs']）で計算
