def train_a2c_agent(env, agent, num_episodes=5000, trajectory_length=100):
    for episode in range(num_episodes):
        state, _ = env.reset()
        trajectory = []
        episode_reward = 0
        
        for t in range(trajectory_length):
            action = agent.select_action(state['obs'])  # 'obs'で状態を取得
            next_state, reward, done, _ = env.step(action)
            trajectory.append((state['obs'], action, reward, next_state['obs'], done))
            episode_reward += reward
            
            if len(trajectory) >= trajectory_length:
                agent.train(trajectory)
                trajectory = []
            
            if done:
                break
            state = next_state

        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")
