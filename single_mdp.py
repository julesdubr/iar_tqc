import gym
import numpy as np


class MDP(gym.Env):
    def __init__(self):
        # Define action and observation spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))
        # Set MDP parameters
        self.A0 = 0.3
        self.A1 = 0.9
        self.nu = 5
        self.sigma = 0.25

    def mean_reward(self, action):
        # Calculate mean reward
        return (self.A0 + (self.A1 - self.A0) / 2 * (action + 1)) * np.cos(
            self.nu * action
        )

    def step(self, action):
        mean_reward = self.mean_reward(action)
        # Generate reward with Gaussian noise
        reward = np.random.normal(mean_reward, self.sigma)
        # Return observation, reward, and done flag
        return 0, reward, False, {}

    def reset(self):
        # Reset state and done flag
        return 0

    def render(self):
        # Print current state
        print(self.state)
