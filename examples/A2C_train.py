def train_a2c_agent(env, agent, num_episodes, trajectory_length):
    for episode in range(num_episodes):
        state = env.reset()
        trajectory = []
        episode_reward = 0
        
        for _ in range(trajectory_length):
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            trajectory.append((state, action, reward, next_state, done))
            episode_reward += reward
            
            state = next_state
            if done:
                break
        
        # Update the agent
        agent.train(trajectory)
        
        print(f"Episode {episode}, Total Reward: {episode_reward}")