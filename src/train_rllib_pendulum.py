#!/usr/bin/env python
import argparse
import logging
import os

from packaging import version
import ray
from ray import tune


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ray_include_webui", type=str2bool, nargs="?", const=True, default=False, help="Start Ray Web UI [False]"
    )
    parser.add_argument(
        "--ray_local_mode",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Start Ray in local mode (a.k.a. disable Ray's multiprocessing) [False]",
    )
    parser.add_argument("--num_gpus", type=int, default=0, help="Configure Ray to manage N GPUs [0]")
    args = parser.parse_args()

    ray_is_085_or_above = version.parse(ray.__version__) >= version.parse("0.8.5")
    activation_key = "fcnet_activation" if ray_is_085_or_above else "hidden_activation"
    layer_sizes_key = "fcnet_hiddens" if ray_is_085_or_above else "hidden_layer_sizes"
    training_cfg = {
        "pendulum_sac": {
            "config": {
                "Q_model": {activation_key: "relu", layer_sizes_key: [256, 256]},
                "clip_actions": False,
                "horizon": 200,
                "learning_starts": 256,
                "metrics_smoothing_episodes": 5,
                "n_step": 1,
                "no_done_at_end": True,
                "normalize_actions": True,
                "num_gpus": 0,
                "num_workers": 0,
                "optimization": {
                    "actor_learning_rate": 0.0003,
                    "critic_learning_rate": 0.0003,
                    "entropy_learning_rate": 0.0003,
                },
                "policy_model": {activation_key: "relu", layer_sizes_key: [256, 256]},
                "prioritized_replay": False,
                "rollout_fragment_length": 1,
                "soft_horizon": False,
                "target_entropy": "auto",
                "target_network_update_freq": 1,
                "tau": 0.005,
                "timesteps_per_iteration": 1000,
                "train_batch_size": 256,
            },
            "env": "Pendulum-v0",
            "run": "SAC",
            "stop": {"episode_reward_mean": -150},
        }
    }

    # Start Ray cluster and run Tune experiment
    ray.init(
        include_webui=args.ray_include_webui,
        logging_level=logging.WARNING,
        local_mode=args.ray_local_mode,
        num_gpus=args.num_gpus,
    )
    tune.run_experiments(training_cfg, verbose=1)  # set verbose=1/2 to see Tune trial status


if __name__ == "__main__":
    main()

