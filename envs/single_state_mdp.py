import gym
from gym import spaces

import numpy as np


class SingleStateMDP(gym.Env):
    def __init__(self):
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Discrete(1)

        self.state = 0

        self.A0 = 0.3
        self.A1 = 0.9
        self.nu = 5
        self.sigma = 0.25

    def _mean_reward(self, action):
        # Calculate mean reward function
        A = self.A0 + (self.A1 - self.A0) / 2 * (action + 1)
        return A * np.cos(self.nu * action)

    def step(self, action):
        # Generate reward with Gaussian noise
        reward = self._mean_reward(action) + np.random.normal(0, self.sigma)
        reward = reward.item()
        next_state = 0
        return next_state, reward, False, {}

    def reset(self):
        # Reset state and done flag
        self.state = 0
        return self.state

    def render(self, mode="human"):
        # Print current state
        pass
