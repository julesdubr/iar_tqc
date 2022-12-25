import gym
from gym import spaces

import numpy as np


class MDP(gym.Env):
    def __init__(self):
        super().__init__()
        # Define action and observation spaces
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1,), dtype=np.float32
        )
        # Set MDP parameters
        self.A0 = 0.3
        self.A1 = 0.9
        self.nu = 5
        self.sigma = 0.25

    def f(self, action):
        # Calculate mean reward function
        return (self.A0 + (self.A1 - self.A0) / 2 * (action + 1)) * np.cos(
            self.nu * action
        )

    def step(self, action):
        f = self.f(action)
        # Generate reward with Gaussian noise
        reward = np.random.normal(f, self.sigma)
        reward = reward.item()

        return np.zeros(1), reward, False, {}

    def reset(self):
        # Reset state and done flag
        return np.zeros(1)

    def render(self, mode="human"):
        # Print current state
        pass
