import gym

import yaml
import glob
import os

from stable_baselines3 import SAC, PPO
from sb3_contrib import TQC
from stable_baselines3.common.monitor import Monitor

from utils.wrappers import RescaleAction


ALGOS = {"sac": SAC, "tqc": TQC}


def get_run_id(log_path: str, algo: str) -> int:
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, algo + "_[0-9]*")):
        file_name = os.path.basename(path)
        ext = file_name.split("_")[-1]
        if (
            algo == "_".join(file_name.split("_")[:-1])
            and ext.isdigit()
            and int(ext) > max_run_id
        ):
            max_run_id = int(ext)
    return max_run_id + 1


def read_hyperparameters(algo, env):
    with open("hyperparams.yml", "r") as f:
        hyperparams_dict = yaml.safe_load(f)
        hyperparams = hyperparams_dict["common"]
        hyperparams.update(hyperparams_dict[algo])
        if algo == "tqc":
            hyperparams.update(hyperparams_dict[env])

    return hyperparams


def make_env(env_id, seed, a=-1.0, b=1.0, eval_env=False):
    env = gym.make(env_id)
    env.seed(seed)
    env.action_space.seed(seed)

    if eval_env:
        env = Monitor(env)

    return RescaleAction(env, a, b)