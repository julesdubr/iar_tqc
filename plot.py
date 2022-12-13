import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

with np.load("logs/Hopper-v3/SAC_2/evaluations.npz") as data:
    sac_rewards = data["results"]
    timesteps = data["timesteps"]

with np.load("logs/Hopper-v3/TQC_3/evaluations.npz") as data:
    tqc_rewards = data["results"]
    assert data["timesteps"].shape == timesteps.shape

tqc_mean = tqc_rewards.mean(axis=1)
tqc_std = tqc_rewards.std(axis=1)

sac_mean = sac_rewards.mean(axis=1)
sac_std = sac_rewards.std(axis=1)

sns.set_context("paper")
sns.set_theme("darkgrid")
fig, ax = plt.subplots()

ax.plot(timesteps, tqc_mean, label="TQC")
ax.fill_between(timesteps, tqc_mean + tqc_std, tqc_mean - tqc_std, alpha=0.2)

ax.plot(timesteps, sac_mean, label="SAC")
ax.fill_between(timesteps, sac_mean + sac_std, sac_mean - sac_std, alpha=0.2)

ax.set_title("Hopper")
ax.set_xlabel("Frames")
ax.set_ylabel("Evaluation returns")
ax.legend()

fig.show()
fig.savefig("plots/tqc_sac_s65423.png", facecolor="white", transparent=False)
