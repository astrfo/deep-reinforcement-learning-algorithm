from policy.dqn import DQN
from policy.ddqn import DDQN


def simulation(sims, epis, env):
    policy = DDQN()
    for sim in range(sims):
        policy.reset()
        for epi in range(epis):
            state = env.reset()[0]
            terminated, truncated = False, False
            while not (terminated or truncated):
                action = policy.action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                policy.update(state, action, reward, next_state, (terminated or truncated))
                state = next_state
    env.close()