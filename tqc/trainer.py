import numpy as np

import torch
import torch.nn.functional as F

from tqc.functions import quantile_huber_loss_f

DEVICE = "cpu"


class Trainer:
    def __init__(
        self,
        critic,
        gamma=0.99,
        quantiles_to_drop=0,
        bias_correction_method="TQC",
    ):
        self.critic = critic
        self.gamma = gamma

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.quantiles_to_drop = quantiles_to_drop * self.critic.n_nets
        self.quantiles_total = critic.n_quantiles * critic.n_nets

        self.bias_correction_method = bias_correction_method

    def train(self, replay_buffer, num_iterations=3000, batch_size=50, verbose=0):
        state, action, reward = replay_buffer.sample(50)
        alpha = torch.exp(self.log_alpha)

        for i in range(num_iterations):
            # Sample a batch of transitions from the replay buffer

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

                critic_loss = F.mse_loss(cur_q, target)

            elif self.bias_correction_method == "MIN":
                # Approximate the value for each action of the replay buffer
                # by taking the minimum of the N Q-networks
                cur_q = self.critic(state, action)

                with torch.no_grad():
                    q_values, _ = cur_q.min(dim=1)

                    next_q = torch.ones(cur_q.shape) * q_values.max()

                    # Compute the critic target using the the max Q-value in respect
                    # to the greedy policy objective
                    rewards_repeated = reward.unsqueeze(1).repeat(1, cur_q.shape[1], 1)
                    target = rewards_repeated + self.gamma * next_q

                critic_loss = F.mse_loss(cur_q, target)

            elif self.bias_correction_method == "TQC":
                # Approximate the value for each action of the replay buffer
                # by using the TQC bias correction method
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

                critic_loss = quantile_huber_loss_f(cur_z, target)

            if verbose:
                print(f"[{i}] loss: {critic_loss}")

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

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
