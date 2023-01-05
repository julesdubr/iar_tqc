import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gym
from gym import spaces

from tqc.functions import quantile_huber_loss_f

DEVICE = "cpu"


class RescaleAction(gym.ActionWrapper):
    def __init__(self, env, a, b):
        assert isinstance(
            env.action_space, spaces.Box
        ), "expected Box action space, got {}".format(type(env.action_space))
        assert np.less_equal(a, b).all(), (a, b)
        super(RescaleAction, self).__init__(env)
        self.a = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + a
        self.b = np.zeros(env.action_space.shape, dtype=env.action_space.dtype) + b
        self.action_space = spaces.Box(
            low=a, high=b, shape=env.action_space.shape, dtype=env.action_space.dtype
        )

    def action(self, action):
        assert np.all(np.greater_equal(action, self.a)), (action, self.a)
        assert np.all(np.less_equal(action, self.b)), (action, self.b)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (high - low) * ((action - self.a) / (self.b - self.a))
        action = np.clip(action, low, high)
        return action


class ReplayBuffer:
    def __init__(self, env, action_grid, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.transition_names = ("state", "action", "reward")
        sizes = (1, 1, 1)
        for name, size in zip(self.transition_names, sizes):
            setattr(self, name, np.empty((max_size, size)))

        state = env.reset()
        for action in action_grid:
            reward = env.step(action)[1]
            self.add(state, action, reward)

    def add(self, state, action, reward):
        values = (state, action, reward)
        for name, value in zip(self.transition_names, values):
            getattr(self, name)[self.ptr] = value

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size, mode="random"):
        if mode == "random":
            ind = np.random.randint(0, self.size, size=batch_size)
        elif mode == "order":
            ind = np.arange(0, min(self.size, batch_size))
        names = self.transition_names
        return (
            torch.FloatTensor(getattr(self, name)[ind]).to(DEVICE) for name in names
        )


class Mlp(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.fcs = []
        in_size = input_size
        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            self.add_module(f"fc{i}", fc)
            self.fcs.append(fc)
            in_size = next_size
        self.last_fc = nn.Linear(in_size, output_size)

    def forward(self, input):
        h = input
        for fc in self.fcs:
            h = F.relu(fc(h))
        output = self.last_fc(h)
        return output


class Critic(nn.Module):
    def __init__(self, n_nets, n_quantiles=1):
        super().__init__()
        self.nets = []
        self.n_quantiles = n_quantiles
        self.n_nets = n_nets

        # Define a list of Q-networks with the given architecture
        for i in range(n_nets):
            net = Mlp(2, [50, 50], n_quantiles)
            self.add_module(f"qf{i}", net)
            self.nets.append(net)

    def forward(self, state, action):
        sa = torch.cat((state, action), dim=1)
        quantiles = torch.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles


class Trainer:
    def __init__(
        self, critic, gamma=0.99, quantiles_to_drop=0, bias_correction_method="TQC"
    ):
        self.critic = critic
        self.gamma = gamma
        self.quantiles_to_drop = quantiles_to_drop * self.critic.n_nets
        self.bias_correction_method = bias_correction_method

        self.optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.quantiles_total = critic.n_quantiles * critic.n_nets

    def train(self, replay_buffer, num_iterations=3000, batch_size=50, verbose=0):
        state = 0
        for i in range(num_iterations):
            # Sample a batch of transitions from the replay buffer
            state, action, reward = replay_buffer.sample(50)

            if self.bias_correction_method == "AVG":
                # Approximate the value for each action of the replay buffer
                # by taking the average of the N Q-networks
                cur_q = self.critic(state, action)

                with torch.no_grad():
                    q_values = cur_q.mean(dim=1)

                    next_q = torch.ones(cur_q.shape) * q_values.max()

                    # Compute the critic target using the the max Q-value in respect
                    # to the greedy policy objective
                    rewards_repeated = reward.unsqueeze(1).repeat(1, cur_q.shape[1], 1)
                    target = rewards_repeated + self.gamma * next_q

                loss = F.mse_loss(cur_q, target)

            elif self.bias_correction_method == "MIN":
                # Approximate the value for each action of the replay buffer
                # by taking the average of the N Q-networks
                cur_q = self.critic(state, action)

                with torch.no_grad():
                    q_values = cur_q.min(dim=1)[0]

                    next_q = torch.ones(cur_q.shape) * q_values.max()

                    # Compute the critic target using the the max Q-value in respect
                    # to the greedy policy objective
                    rewards_repeated = reward.unsqueeze(1).repeat(1, cur_q.shape[1], 1)
                    target = rewards_repeated + self.gamma * next_q

                loss = F.mse_loss(cur_q, target)

            elif self.bias_correction_method == "TQC":
                # Calculate the loss using the TQC bias correction method
                cur_z = self.critic(state, action)

                with torch.no_grad():
                    z_values = cur_z.mean(dim=(1, 2))
                    next_action = torch.ones(action.shape) * action[z_values.argmax()]

                    next_z = self.critic(state, next_action)
                    sorted_z, _ = torch.sort(next_z.reshape(batch_size, -1))
                    sorted_z_part = sorted_z[
                        :, : self.quantiles_total - self.quantiles_to_drop
                    ]

                    target = reward + self.gamma * sorted_z_part

                loss = quantile_huber_loss_f(cur_z, target)

            if verbose:
                print(f"[{i}] loss: {loss}")

            # Backpropagate the loss and update the weights
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def evaluate(self, env, replay_buffer, verbose=0):
        state, action, reward = replay_buffer.sample(2000, mode="order")
        q_values = self.critic(state, action)

        with torch.no_grad():
            approx = q_values.mean(dim=(1, 2)).view(-1, 1)

        discrepancies = approx - reward
        mean = discrepancies.mean().item()
        var = discrepancies.var().item()

        argmax = action[approx.argmax()].item()

        mean_reward = np.apply_along_axis(env._mean_reward, 0, action)
        a_star = action[mean_reward.argmax()].item()

        if verbose:
            print(f"argmax: {argmax} - a*: {a_star}")

        distance = abs(a_star - argmax)

        return mean, var, distance
