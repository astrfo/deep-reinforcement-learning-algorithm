def simulation(sims, epis, env):
    for sim in range(sims):
        for epi in range(epis):
            state = env.reset()[0]
            terminated, truncated = False, False
            while not (terminated or truncated):
                action = env.action_space.sample()
                next_state, reward, terminated, truncated, info = env.step(action)
    env.close()