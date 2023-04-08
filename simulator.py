from policy.dqn import DQN
from policy.ddqn import DDQN


class Simulation:
    def __init__(self, sim, epi, env):
        self.sim = sim
        self.epi = epi
        self.env = env

    def run(self):
        policy = DDQN()
        for s in range(self.sim):
            policy.reset()
            for e in range(self.epi):
                state = self.env.reset()[0]
                terminated, truncated = False, False
                while not (terminated or truncated):
                    action = policy.action(state)
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    policy.update(state, action, reward, next_state, (terminated or truncated))
                    state = next_state
        self.env.close()