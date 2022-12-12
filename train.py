import torch
import gym

import os
import argparse

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from wrappers import RescaleAction
from utils import ALGOS, N_QUANTILES_TO_DROP, get_latest_run_id


def main(args, tensorboard_log, save_path):
    # Setup environments
    env = gym.make(args.env)
    eval_env = Monitor(gym.make(args.env))

    env = RescaleAction(env, -1.0, 1.0)
    eval_env = RescaleAction(eval_env, -1.0, 1.0)

    # Setup model
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 256], qf=[512, 512, 512]),
        n_critics=5,
        n_quantiles=25,
    )

    model = ALGOS[args.algo](
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=int(1e6),
        policy_kwargs=policy_kwargs,
        batch_size=256,
        tau=0.005,
        top_quantiles_to_drop_per_net=N_QUANTILES_TO_DROP[args.env],
        tensorboard_log=tensorboard_log,
        verbose=args.verbose,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        n_eval_episodes=10,
        log_path=save_path,
        eval_freq=int(args.eval_freq),
        deterministic=True,
        verbose=args.verbose,
    )

    # Start learning
    model.learn(total_timesteps=int(args.n_timesteps), callback=eval_callback)

    # Save the model
    print(f"Saving to {save_path}")
    model.save(f"{save_path}/{args.env}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="tqc", choices=list(ALGOS.keys()))
    parser.add_argument("--env", default="Hopper-v3")
    parser.add_argument("--eval-freq", default=1e3, type=int)
    parser.add_argument("-n", "--n-timesteps", default=1e5, type=int)
    parser.add_argument("-f", "--log-dir", default="logs")
    parser.add_argument("-tb", "--tensorboard-log", default="logs")
    parser.add_argument("--verbose", default=0, type=int)
    args = parser.parse_args()

    tensorboard_log = os.path.join(args.tensorboard_log, args.env)

    log_dir = f"{args.log_dir}/{args.env}"
    algo = args.algo.upper()
    save_path = os.path.join(log_dir, f"{algo}_{get_latest_run_id(log_dir, algo) + 1}")

    main(args, tensorboard_log, save_path)
