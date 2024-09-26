# A2C_training.pyの修正例

import rlcard
from A2C_agent import A2CAgent  # A2Cエージェント
from A2C_network import ActorCriticNetwork  # ネットワーク定義

# 環境を初期化 (Gym ラッパーを使わず直接RLCard)
env = rlcard.make('uno')

# A2C エージェントの初期化
state_shape = env.state_shape[0]
num_actions = env.num_actions
a2c_agent = A2CAgent(state_shape, num_actions)

# 学習ループ
def train_a2c_agent(env, agent, num_episodes, trajectory_length):
    for episode in range(num_episodes):
        trajectory = []
        state = env.reset()  # 環境をリセット
        for t in range(trajectory_length):
            action = agent.select_action(state['obs'])  # 行動を選択
            next_state, reward, done, _ = env.step(action)  # ステップを実行

            # ここで学習
            trajectory.append((state['obs'], action, reward, next_state['obs'], done))
            state = next_state

            if done:
                break
        
        agent.train(trajectory)

# 学習の実行
train_a2c_agent(env, a2c_agent, num_episodes=5000, trajectory_length=100)