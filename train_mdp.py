"""
The task is simplistic infinite horizon MDP (S, A,P, R, p0) with only one state S = {s0} and 1-dimensional action space A = [-1, 1]. Since there is only one state, the state transition function P and initial state distribution p0 are delta functions.

On each step agent get stochastic reward r(a) ~ f(a) + N (0, σ), where σ = 0.25. Mean reward function is the cosine with
slowly increasing amplitude (Figure 8):

f(a) = [ A0 + (A1 - A0) / 2 * (a + 1) ] cos νa,
    where A0 = 0.3; A1 = 0.9; ν = 5.

The discount factor is γ = 0.99.

In the following, Q'π(a) and Z'π(a) are the approximations of the true Q-value and Z-value Qπ(a) and Zπ(a) of the action a. We describe a sum from i = 0 to kN as Σi=1->(kN).

In the toy experiment we evaluate bias correction techniques (Table 4) in this MDP. We train Q-networks (or Z-networks, depending on the method) with two hidden layers of size 50 from scratch on the replay buffer of size 50 for 3000 iterations. We populate the buffer by sampling a reward once for each action from a uniform action grid of size 50. At each step of temporal difference learning, we use a policy, which is greedy with respect to the objective in Table 4.

We define ∆(a) := Q'π(a) - Qπ(a) as a signed discrepancy between the approximate and the true Q-value. For TQC Q'π(a) = EZ'π(a) = 1/kN Σi=1->(kN) z(i)(a). We vary the parameters controlling the overestimation for each method and report the robust average (10% of each tail is truncated) over 100 seeds of Ea~U(-1,1) [ ∆(a) ] and Vara~U(-1,1) [∆(a)]. Expectation and variance estimated over dense uniform grid of actions of size 2000 and then averaged over seeds.

For AVG and MIN we vary the number of networks N from [3, 5, 10, 20, 50] and [2, 3, 4, 6, 8, 10] correspondingly. For TQC — number of dropped quantiles per network d = M - k from [0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 16] out of 25. We present the results in Figure 4 with bubbles of diameter, inversely proportional to the averaged over the seeds absolute distance between the optimal a* and the arg max of the policy objective.

To prevent interference of policy optimization subtleties into conclusions about Q-function approximation quality, we use implicit deterministic policy induced by value networks: the argmax of the approximation. To find the maximum, we evaluated the approximation over the dense uniform grid in the range [-1, 1] with a step ∆a = 0.001.

Each dataset consists of uniform grid of actions and sampled corresponding rewards. For each method we average results over several datasets and evaluate on different dataset sizes. In this way current policy defined implicitly as greedy one with respect to value function. This policy doesn’t interact with the environment instead actions predefined to be uniform.
"""

import gym
from gym import spaces

import numpy as np
import torch


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


class QNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        return x


def avg_bias_correction(q_networks):
    def bias_correction(state):
        q_values = [q_network(state).flatten() for q_network in q_networks]
        return torch.mean(q_values, dim=0)

    return bias_correction


def min_bias_correction(q_networks):
    def bias_correction(state):
        q_values = [q_network(state).flatten() for q_network in q_networks]
        return torch.min(q_values, dim=0)

    return bias_correction


def tqc_bias_correction(q_networks, num_dropped_quantiles):
    def bias_correction(state):
        q_values = [q_network(state).flatten() for q_network in q_networks]
        z_values = [torch.sort(q_value)[0] for q_value in q_values]
        z_values = z_values[:, num_dropped_quantiles:-num_dropped_quantiles]
        return torch.mean(z_values, dim=0)

    return bias_correction


def train_q_network(
    replay_buffer, bias_correction_fn, num_networks=1, num_dropped_quantiles=0
):
    # define the Q-networks, optimizers and loss functions
    q_networks = [QNetwork(1, 50, 1) for _ in range(num_networks)]
    optimizers = [torch.optim.Adam(q_network.parameters()) for q_network in q_networks]
    loss_fns = [torch.nn.MSELoss() for _ in range(num_networks)]

    for loss_fn, optimizer in zip(loss_fns, optimizers):
        for state, reward, action in replay_buffer:
            # convert state, action and reward to tensors
            state = torch.tensor(state)
            action = torch.tensor(action)
            reward = torch.tensor(reward)

            # apply the bias correction function to the Q-network's output
            if bias_correction_fn == tqc_bias_correction:
                q_values = bias_correction_fn(q_networks, num_dropped_quantiles)(state)
            else:
                q_values = bias_correction_fn(q_networks)(state)

            # get the estimated q_value and compute the loss
            q_value = q_values[0][action]
            loss = loss_fn(q_value, reward)

            # backpropagate the error and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def evaluate(env, bias_correction_fn, q_networks, num_dropped_quantiles=1):
    # reset the environment
    state = env.reset()

    # apply the bias correction function to the Q-network's output
    if bias_correction_fn == tqc_bias_correction:
        q_values = bias_correction_fn(q_networks, num_dropped_quantiles)(state)
    else:
        q_values = bias_correction_fn(q_networks)(state)

    # choose the action with the highest Q-value, a*
    action = q_values.argmax().item()

    # take a step in the environment
    _, reward, _, _ = env.step(action)
    return reward


def run_experiment(env, bias_correction_fn, parameters):
    # create an empty replay buffer
    replay_buffer = []

    # sample rewards for each action from a uniform action grid
    for action in np.linspace(-1, 1, 50):
        state = 0
        reward = env.step(action)[1]
        replay_buffer.append((state, reward, action))

    # initialize lists to store the results
    mean_discrepancy = []
    var_discrepancy = []

    # iterate over the number of networks or dropped quantiles values
    for param in parameters:

        if bias_correction_fn == tqc_bias_correction:
            # train the Q-network
            q_networks = train_q_network(
                replay_buffer,
                bias_correction_fn,
                2,
                param,
            )
        else:
            # train the Q-network
            q_networks = train_q_network(
                replay_buffer,
                bias_correction_fn,
                param,
                0,
            )

        # evaluate the Q-network
        discrepancies = []
        d = param if bias_correction_fn == tqc_bias_correction else 0

        for action in np.linspace(-1, 1, 2000):
            reward = env.step(action)[1]
            predicted_reward = evaluate(
                env,
                bias_correction_fn,
                q_networks,
                d,
            )
            discrepancy = predicted_reward - reward
            discrepancies.append(discrepancy)

        # calculate the mean and variance of the discrepancies
        mean_discrepancy.append(np.mean(discrepancies))
        var_discrepancy.append(np.var(discrepancies))

    return mean_discrepancy, var_discrepancy


# create the environment
env = SingleStateMDP()

# define the bias correction function
bias_correction_fn = min_bias_correction

# define the number of networks values
parameters = [3, 5, 10, 20, 50]  # for avg and min
# parameters = [0, 1, 2, 3, 4, 5, 6, 7, 10, 13, 16]  # for tqc

# run the experiment
mean_discrepancy, var_discrepancy = run_experiment(env, bias_correction_fn, parameters)

# print the results
print(f"Mean discrepancy: {mean_discrepancy}")
print(f"Variance discrepancy: {var_discrepancy}")
