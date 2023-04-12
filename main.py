import gym
from simulator import Simulation, PGSimulation, ACSimulation


if __name__ == '__main__':
    sim = 1
    epi = 5000
    env = gym.make('CartPole-v1', render_mode='rgb_array').unwrapped

    # simulation = Simulation(sim, epi, env)
    # simulation = PGSimulation(sim, epi, env)
    simulation = ACSimulation(sim, epi, env)
    simulation.run()