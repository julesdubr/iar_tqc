import gym
import numpy as np
from gym import spaces

import seaborn as sns
import matplotlib.pyplot as plt
from utils.utils import custom_params


class SingleStateMDP(gym.Env):
    def __init__(self, A0=0.3, A1=0.9, nu=5, sigma=0.25):
        self.action_space = spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Discrete(1)

        self.state = 0

        self.A0 = A0
        self.A1 = A1
        self.nu = nu
        self.sigma = sigma

    def _mean_reward(self, action):
        A = self.A0 + (self.A1 - self.A0) / 2 * (action + 1)
        return A * np.cos(self.nu * action)

    def step(self, action):
        reward = self._mean_reward(action) + np.random.normal(0, self.sigma)
        reward = reward.item()
        next_state = 0
        return next_state, reward, True, {}

    def reset(self):
        self.state = 0
        return self.state

    def render(self, mode="human"):
        pass


def plot_mdp_reward(n, env, savefig=False):
    actions = np.linspace(-1, 1, n)
    rewards = np.zeros(n)
    targets = np.zeros(n)

    # Build the reward function
    env.reset()
    for i, action in enumerate(actions):
        a = action.item()
        _, reward, _, _ = env.step(a)
        targets[i] = env._mean_reward(a)
        rewards[i] = reward

    # Plot the reward function
    sns.set_theme(style="ticks", palette="tab10", rc=custom_params)

    plt.plot(actions, targets, color="r", linewidth=3, label="$f(a)$")
    plt.scatter(actions, rewards, label="samples from $R(a)$")

    plt.ylim(ymin=-2)
    plt.legend()

    if savefig:
        plt.savefig("plots/reward_function")

    plt.show()
