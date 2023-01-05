import numpy as np
import gym
from gym import spaces


class SingleStateMDP(gym.Env):
    def __init__(self, A0=0.3, A1=0.9, nu=5, sigma=0.25, seed=None):
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Discrete(1)

        self.state = 0

        self.A0 = A0
        self.A1 = A1
        self.nu = nu
        self.sigma = sigma

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

    def _mean_reward(self, action):
        A = self.A0 + (self.A1 - self.A0) * (action + 1) * 0.5
        return A * np.sin(self.nu * action)

    def step(self, action):
        reward = self._mean_reward(action) + self.rng.normal(0, self.sigma)
        reward = reward.item()
        next_state = 0
        return next_state, reward, True, {}

    def reset(self):
        self.state = 0
        return self.state

    def render(self, mode="human"):
        pass
