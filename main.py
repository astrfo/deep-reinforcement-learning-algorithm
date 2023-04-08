import gym
from simulator import Simulation


if __name__ == '__main__':
    sim = 1
    epi = 10
    env = gym.make('CartPole-v1', render_mode='human').unwrapped

    simulation = Simulation(sim, epi, env)
    simulation.run()