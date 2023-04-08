import gym
from simulator import simulation


if __name__ == '__main__':
    sim = 1
    epi = 10
    env = gym.make('CartPole-v1', render_mode='human').unwrapped

    simulation(sim, epi, env)