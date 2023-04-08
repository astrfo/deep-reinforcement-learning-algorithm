from tqdm import tqdm
from policy.dqn import DQN
from policy.doubledqn import DoubleDQN
from policy.duelingdqn import DuelingDQN
from collector import Collector


class Simulation:
    def __init__(self, sim, epi, env):
        self.sim = sim
        self.epi = epi
        self.env = env
        self.collector = Collector(sim, epi)
        self.policy = DuelingDQN()

    def run(self):
        for s in range(self.sim):
            self.collector.reset()
            self.policy.reset()
            for e in tqdm(range(self.epi)):
                state = self.env.reset()[0]
                terminated, truncated, total_reward = False, False, 0
                while not (terminated or truncated) and (total_reward < 500):
                    action = self.policy.action(state)
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                    self.policy.update(state, action, reward, next_state, (terminated or truncated))
                    state = next_state
                    total_reward += reward
                self.collector.collect_episode_data(total_reward)
            self.collector.save_episode_data()
        self.collector.save_simulation_data()
        self.env.close()