import numpy as np
import matplotlib.pyplot as plt


class Collector:
    def __init__(self, sim, epi, is_save_img=True):
        self.sim = sim
        self.epi = epi
        self.is_save_img = is_save_img
        self.reward_sim = np.zeros(self.epi)

    def reset(self):
        self.reward_epi = []

    def collect_episode_data(self, reward):
        self.reward_epi.append(reward)

    def save_episode_data(self):
        self.reward_sim += self.reward_epi

    def save_simulation_data(self):
        self.reward_sim /= self.sim
        self.save_img()

    def save_img(self):
        figure = plt.figure(figsize=(12, 8))
        plt.plot(self.reward_sim, alpha=0.6)
        plt.xlabel('episode')
        plt.savefig('reward.png')
        plt.close()