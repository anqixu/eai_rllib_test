#!/usr/bin/env python
import logging
import os

import ray
import tensorflow as tf
from ray import tune


class TestOptFnTrainable(tune.Trainable):
    def _setup(self, config):
        print(f"@ TestOptFnTrainable._setup() - tf.test.is_gpu_available(): {tf.test.is_gpu_available()}")

    def _train(self):
        print(f"@ TestOptFnTrainable._train() - tf.test.is_gpu_available(): {tf.test.is_gpu_available()}")
        return {"performance": 0}

    def _save(self, checkpoint_dir):
        return os.path.join(checkpoint_dir, "null")  # assume test optimization functions are stateless

    def _restore(self, checkpoint_path):
        pass  # assume test optimization functions are stateless


def main():
    print(f"@ main() - tf.test.is_gpu_available(): {tf.test.is_gpu_available()}")

    # Define tune experiment settings
    tune_exp_cfg = {
        "run": TestOptFnTrainable,
        "num_samples": 1,
        "stop": {
            "training_iteration": 1,  # no need to run inner optimization loop more than once
        },
        "config": {},
        "local_dir": None,
        "max_failures": 0,
        "loggers": [],
        "resources_per_trial": {
            "cpu": 1,
            "gpu": 1,  # <<< CHANGE ME TO 0 TO DISABLE GPU ACCESS ON TRAINABLE
        }
    }

    # Start Ray cluster and run Tune experiment
    ray.init(include_webui=False, logging_level=logging.WARNING, local_mode=False, num_gpus=1)
    tune.run_experiments({"TestOptFn": tune_exp_cfg}, verbose=0)  # set verbose=1/2 to see Tune trial status


if __name__ == "__main__":
    main()