import numpy as np

import argparse
import os

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

from tqc.utils import ALGOS, get_run_id, make_env


def train(env_name, algo, seed, policy_kwargs, tensorboard_log, save_path, verbose):
    # Setup environments
    env = make_env(env_name, seed)
    eval_env = make_env(env_name, seed, eval_env=True)

    model = ALGOS[algo](
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-3,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        seed=seed,
        verbose=verbose,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        n_eval_episodes=10,
        log_path=save_path,
        eval_freq=int(1e3),
        deterministic=True,
        verbose=verbose,
    )

    # Start learning
    model.learn(total_timesteps=int(2e5), callback=eval_callback)

    # Save the model
    print(f"Saving to {save_path}")
    model.save(f"{save_path}/{env_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--log-dir", default="logs", type=str)
    parser.add_argument("-tb", "--tensorboard-log", default="logs", type=str)
    parser.add_argument("--verbose", default=0, type=int)
    args = parser.parse_args()

    env_name = "Pendulum-v0"

    policy_kwargs = {
        "net_arch": {
            "pi": [256, 256],
            "qf": [256, 256],
        }
    }

    # Seed
    seeds = np.random.randint(0, 10000, 4, dtype="int64")

    # SAC Training
    algo = "sac"
    for n_critics in [1, 2, 3, 4]:

        policy_kwargs.update({"n_critics": n_critics})

        for seed in seeds:
            set_random_seed(seed)

            # Logging
            tensorboard_log = os.path.join(args.tensorboard_log, env_name)

            log_dir = f"{args.log_dir}/{env_name}"
            algo = algo.upper()
            save_path = os.path.join(log_dir, f"{algo}_{get_run_id(log_dir, algo)}")

            print(f"Logging to {save_path}")

            train(
                env_name,
                algo,
                seed,
                policy_kwargs,
                tensorboard_log,
                save_path,
                args.verbose,
            )

            new_path = os.path.join(log_dir, f"{algo}_{n_critics}_{seed}")
            os.rename(save_path, new_path)

    # TQC Training
    algo = "tqc"
    for n_quantiles in [1, 2, 5, 10, 25]:

        policy_kwargs.update({"n_critics": 5, "n_quantiles": n_quantiles})

        for seed in seeds:
            set_random_seed(seed)

            # Logging
            tensorboard_log = os.path.join(args.tensorboard_log, env_name)

            log_dir = f"{args.log_dir}/{env_name}"
            algo = algo.upper()
            save_path = os.path.join(log_dir, f"{algo}_{get_run_id(log_dir, algo)}")

            print(f"Logging to {save_path}")

            train(
                env_name,
                algo,
                seed,
                policy_kwargs,
                tensorboard_log,
                save_path,
                args.verbose,
            )

            new_path = os.path.join(log_dir, f"{algo}_{n_quantiles}_{seed}")
            os.rename(save_path, new_path)
