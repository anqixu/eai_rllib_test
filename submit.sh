#!/bin/bash
eai job new --image registry.console.elementai.com/eai.arl.semantic_navigation/eai_rllib_test --cpu 2.0 --gpu 1 --mem 4 --preemptable --restartable -- python3 /train_rllib_pendulum.py