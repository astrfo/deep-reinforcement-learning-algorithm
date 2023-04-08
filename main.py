import gym
from simulator import Simulation


if __name__ == '__main__':
    sim = 1
    epi = 500
    env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped

    simulation = Simulation(sim, epi, env)
    simulation.run()