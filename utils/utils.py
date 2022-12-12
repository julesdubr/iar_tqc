import glob
import os

from stable_baselines3 import SAC, PPO
from sb3_contrib import TQC

ALGOS = {
    "ppo": PPO,
    "sac": SAC,
    "tqc": TQC,
}


def get_latest_run_id(log_path: str, algo: str) -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: path to log folder
    :param env_id:
    :return: latest run number
    """
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
    return max_run_id
