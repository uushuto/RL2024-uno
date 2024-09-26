# A2C_training.py

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
        state = env.reset()
        trajectory = []

        for t in range(trajectory_length):
            # 状態の観測を取得
            obs = state['obs']  # stateから'obs'キーを取得

            # 行動を選択
            action = agent.select_action(obs)  # obs のみを使って行動を選択

            # ステップを実行
            next_state, reward, done, _ = env.step(action)

            # 次の状態の観測を取得
            next_obs = next_state['obs']

            # 訓練用のトラジェクトリを保存
            trajectory.append((obs, action, reward, next_obs, done))

            # 状態を更新
            state = next_state

            if done:
                break

        # エージェントの学習
        agent.train(trajectory)


# 学習の実行
train_a2c_agent(env, a2c_agent, num_episodes=5000, trajectory_length=100)
