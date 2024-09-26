import torch
import torch.nn as nn
import torch.optim as optim
from A2C_network import ActorCriticNetwork  # ActorCriticNetworkをインポート

class A2CAgent:
    def __init__(self, state_shape, num_actions, lr=0.001, gamma=0.99):
        self.gamma = gamma
        self.network = ActorCriticNetwork(state_shape, num_actions)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
    
    def select_action(self, state):
    # 状態が辞書型であれば、その中の 'obs' 部分を取り出す
        obs = state['obs']  # state は辞書型で、'obs' キーに観測データが格納されていることを想定
        obs = torch.FloatTensor(obs).unsqueeze(0)  # Add batch dimension
    
    # アクション確率を予測
        action_probs, _ = self.network(obs)
    
    # アクションをサンプリング
        action = torch.multinomial(action_probs, 1).item()
        return action


    
    def train(self, trajectory):
        # Unpack trajectory
        states, actions, rewards, next_states, dones = zip(*trajectory)
        
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Get current state values and action probabilities
        action_probs, state_values = self.network(states)
        _, next_state_values = self.network(next_states)
        
        # Critic loss (value loss)
        td_target = rewards + self.gamma * next_state_values.squeeze() * (1 - dones)
        critic_loss = (td_target - state_values.squeeze()).pow(2).mean()
        
        # Actor loss (policy loss)
        log_probs = torch.log(action_probs.gather(1, actions))
        advantages = (td_target - state_values.squeeze()).detach()
        actor_loss = -(log_probs.squeeze() * advantages).mean()
        
        # Total loss
        loss = actor_loss + critic_loss
        
        # Update networks
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
