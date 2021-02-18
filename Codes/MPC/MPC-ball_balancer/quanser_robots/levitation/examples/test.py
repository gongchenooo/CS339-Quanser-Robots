"""
Test environment
"""

import numpy as np
import gym
import quanser_robots


if __name__ == "__main__":

    env = gym.make('Levitation-v1')

    obs = env.reset()

    done = False
    while not done:
        act = np.random.uniform(-24.0, 24.0, size=((1, )))
        obs, r, done, _ = env.step(act)
        print(obs, act, r)

    env.close()
