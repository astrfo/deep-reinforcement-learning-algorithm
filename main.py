import gym


if __name__ == '__main__':
    env = gym.make('CartPole-v1', render_mode='human').unwrapped

    for i in range(10):
        observation = env.reset()[0]
        terminated, truncated = False, False
        while not (terminated or truncated):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
    env.close()
