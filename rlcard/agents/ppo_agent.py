import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
from collections import namedtuple, deque

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done', 'legal_actions'])

class PPOAgent(object):
    def __init__(self, state_shape, action_shape, device=None,
                 batch_size=32, gamma=0.995, rollout_length=1024,
                 num_updates=5, lr_actor=3e-4, lr_critic=3e-4, num_actions=2,
                 epsilon_start=1.0, epsilon_end=0.1, epsilon_decay_steps=1024,
                 clip_eps=0.2, coef_ent=0.0, lambd=0.97, max_grad_norm=0.5):
        
        # Torch device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.use_raw = False
        self.total_t = 0
        self.learning_steps = 0
        self.gamma = gamma
        self.lambd = lambd
        self.num_updates = num_updates
        self.batch_size = batch_size
        self.rollout_length = rollout_length
        self.max_grad_norm = max_grad_norm
        self.clip_eps = clip_eps
        self.coef_ent = coef_ent
        
        self.num_actions = num_actions
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
        self.queue_actions = deque()
        self.queue_log_pis = deque()

        self.actor = PPOActor(
            state_shape=state_shape,
            action_shape=action_shape,
            num_actions = num_actions
        ).to(self.device)

        self.critic = PPOCritic(
            state_shape=state_shape,
        ).to(self.device)

        self.buffer = RolloutBuffer(
            buffer_size=rollout_length,
            state_shape=state_shape,
            action_shape=action_shape,
            device=self.device
        )

        self.optim_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optim_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic) 

    def is_update(self, total_step):
        return total_step % self.rollout_length == 0

    def feed(self, ts):
        (state, _, reward, next_state, done) = tuple(ts)
        # _, log_pi = self.explore(state['obs'])
        action = self.queue_actions.popleft()
        log_pi = self.queue_log_pis.popleft()
        self.buffer.append(state['obs'], action, reward, done, log_pi, next_state['obs'])
        self.total_t += 1
        if self.is_update(self.total_t):
            self.train()

    def explore(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_pi = self.actor.sample(state)
        return action.cpu().numpy()[0], log_pi.item()
    
    def step(self, state):
        action, log_pi = self.explore(state['obs'])
        
        self.queue_actions.append(action)
        self.queue_log_pis.append(log_pi)
        
        masked_action = -np.inf * np.ones(self.num_actions, dtype=float)
        legal_actions = list(state['legal_actions'].keys())
        masked_action[legal_actions] = action[legal_actions]
        
        epsilon = self.epsilons[min(self.total_t, self.epsilon_decay_steps-1)]
        # legal_actions = list(state['legal_actions'].keys())
        probs = np.ones(len(legal_actions), dtype=float) * epsilon / len(legal_actions)
        best_action_idx = legal_actions.index(np.argmax(masked_action))
        probs[best_action_idx] += (1.0 - epsilon)
        action_idx = np.random.choice(np.arange(len(probs)), p=probs)

        return legal_actions[action_idx]
        # return action

    def exploit(self, state):
        state = torch.tensor(state, dtype=torch.float, device=self.device).unsqueeze_(0)
        with torch.no_grad():
            action = self.actor(state)
        return action.cpu().numpy()[0]

    def eval_step(self, state):
        action = self.exploit(state['obs'])
        
        masked_action = -np.inf * np.ones(self.num_actions, dtype=float)
        legal_actions = list(state['legal_actions'].keys())
        masked_action[legal_actions] = action[legal_actions]
        
        best_action_idx = np.argmax(masked_action[legal_actions])
        return legal_actions[best_action_idx], None

    def train(self):
        self.learning_steps += 1
        states, actions, rewards, dones, log_pis, next_states = self.buffer.get()
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
        targets, advantages = calculate_advantage(values, rewards, dones, next_values, self.gamma, self.lambd)

        for _ in range(self.num_updates):
            indices = np.arange(self.rollout_length)
            np.random.shuffle(indices)
            
            for start in range(0, self.rollout_length, self.batch_size):
                idxes = indices[start:start+self.batch_size]
                self.update_critic(states[idxes], targets[idxes])
                self.update_actor(states[idxes], actions[idxes], log_pis[idxes], advantages[idxes])

    def update_critic(self, states, targets):
        loss_critic = (self.critic(states) - targets).pow_(2).mean()

        self.optim_critic.zero_grad()
        loss_critic.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()
    
    def update_actor(self, states, actions, log_pis_old, advantages):
        log_pis = self.actor.evaluate_log_pi(states, actions)
        mean_entropy = -log_pis.mean()

        ratios = (log_pis - log_pis_old).exp_()
        loss_actor1 = -ratios * advantages
        loss_actor2 = -torch.clamp(
            ratios,
            1.0 - self.clip_eps,
            1.0 + self.clip_eps
        ) * advantages
        loss_actor = torch.max(loss_actor1, loss_actor2).mean() - self.coef_ent * mean_entropy

        self.optim_actor.zero_grad()
        loss_actor.backward(retain_graph=False)
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.optim_actor.step()

class PPOActor(nn.Module):
    def __init__(self, state_shape, action_shape, num_actions):
        super().__init__()

        self.num_actions = num_actions
        self.state_shape = state_shape
        
        layer_dims = [np.prod(self.state_shape)]
        fc = [nn.Flatten()]
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            fc.append(nn.Tanh())
        fc.append(nn.Linear(layer_dims[-1], self.num_actions))
        self.net = nn.Sequential(*fc)
        
        # self.net = nn.Sequential(
        #     nn.Linear(4 * 4* 15, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, 64),
        #     nn.Tanh(),
        #     nn.Linear(64, action_shape[0]),
        # )
        self.log_stds = nn.Parameter(torch.zeros(1, action_shape[0]))

    def forward(self, states):
        return torch.tanh(self.net(states))

    def sample(self, states):
        # states = states.view(-1, 4 * 4 * 15)
        return reparameterize(self.net(states), self.log_stds)

    def evaluate_log_pi(self, states, actions):
        return evaluate_lop_pi(self.net(states), self.log_stds, actions)

# class PPOActorNetwork(nn.Module):
    


class PPOCritic(nn.Module):
    def __init__(self, state_shape):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(4 * 4 * 15, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, states):
        states = states.view(-1, 4 * 4 * 15)
        return self.net(states)


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

def calculate_advantage(values, rewards, dones, next_values, gamma=0.995, lambd=0.997):
    deltas = rewards + gamma * next_values * (1 - dones) - values
    advantages = torch.empty_like(rewards)
    advantages[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        advantages[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * advantages[t+1]
    targets = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return targets, advantages

# MEMO TODO
# DONE Actor, Criticの入出力次元をdqn_agent.pyと同じように書き換え
# DONE sampleのviewは不要となる？

# DONE def step, actionをndarray -> ある1つに決定

# def stepでactionとlog_piのndarrayを保存 => def feed, buffer保存時にaction, log_piはstep保存を仕様
