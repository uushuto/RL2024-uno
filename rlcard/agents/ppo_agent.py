import os
import random
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])

class PPOAgent(object):
    def __init__(self, 
                 device=None):
        
        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def explore(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()
    
    def step(self, state):
        action, _ = self.explore(state)
        return action

# MEMO TODO
# - PPO, def step, Buffer.appendをdef feedで実装
# - PPO, def step, log_piをtsのstate指定（tranjectories[0][-1][0]のstate['obs']）で計算
