import torch
import torch.nn as nn

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_shape, num_actions):
        super(ActorCriticNetwork, self).__init__()
        
        # Actor network (for action selection)
        self.actor = nn.Sequential(
            nn.Linear(state_shape[0] * state_shape[1] * state_shape[2], 128),  # 状態をフラット化
            nn.ReLU(),
            nn.Linear(128, num_actions),
            nn.Softmax(dim=-1)  # Action probabilities
        )
        
        # Critic network (for value estimation)
        self.critic = nn.Sequential(
            nn.Linear(state_shape[0] * state_shape[1] * state_shape[2], 128),  # 状態をフラット化
            nn.ReLU(),
            nn.Linear(128, 1)  # State value
        )
    
    def forward(self, state):
        # 状態をフラット化
        state = state.view(state.size(0), -1)
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value
