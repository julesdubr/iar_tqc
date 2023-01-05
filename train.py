import numpy as np

import argparse
import os

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed

from tqc import DEVICE
from tqc.utils import ALGOS, get_run_id, read_hyperparameters, make_env


def train(args, tensorboard_log, save_path, verbose):
    # Setup environments
    env = make_env(args.env, args.seed)
    eval_env = make_env(args.env, args.seed, eval_env=True)

    # Setup model
    hyperparams = read_hyperparameters(args.algo, args.env, args.hyperparams)

    model = ALGOS[args.algo](
        env=env,
        tensorboard_log=tensorboard_log,
        seed=args.seed,
        verbose=verbose,
        **hyperparams,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        n_eval_episodes=10,
        log_path=save_path,
        eval_freq=int(args.eval_freq),
        deterministic=True,
        verbose=verbose,
    )

    # Start learning
    model.learn(total_timesteps=int(args.n_timesteps), callback=eval_callback)

    # Save the model
    print(f"Saving to {save_path}")
    model.save(f"{save_path}/{args.env}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="tqc", choices=list(ALGOS.keys()))
    parser.add_argument("--env", default="Hopper-v3", type=str)
    parser.add_argument("--eval-freq", default=1e3, type=int)
    parser.add_argument("-n", "--n-timesteps", default=1e5, type=int)
    parser.add_argument("-f", "--log-dir", default="logs", type=str)
    parser.add_argument("-tb", "--tensorboard-log", default="logs", type=str)
    parser.add_argument("--seed", default=-1, type=int)
    parser.add_argument("--hyperparams", default="mujoco", type=str)
    parser.add_argument("--verbose", default=0, type=int)
    args = parser.parse_args()

    # Seed
    if args.seed < 0:
        args.seed = np.random.randint(2**32 - 1, dtype="int64").item()

    set_random_seed(args.seed)
    print(f"Starting training on seed <{args.seed}> using {DEVICE}...")

    # Logging
    tensorboard_log = os.path.join(args.tensorboard_log, args.env)

    log_dir = f"{args.log_dir}/{args.env}"
    algo = args.algo.upper()
    save_path = os.path.join(log_dir, f"{algo}_{get_run_id(log_dir, algo)}")

    print(f"Logging to {save_path}")

    train(args, tensorboard_log, save_path, args.verbose)
