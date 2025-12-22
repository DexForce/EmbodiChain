#!/bin/bash

NUM_PROCESSES=3  # Set this to the number of parallel processes you want

for ((i=0; i<NUM_PROCESSES; i++)); do
    python ./embodichain/lab/scripts/run_env.py \
        --gym_config /home/dyc/workspace/sources/EmbodiChain/configs/gym/pour_water/gym_config_simple.json \
        --action_config /home/dyc/workspace/sources/EmbodiChain/configs/gym/pour_water/action_config.json \
        --headless --enable_rt &
    sleep 10
done

wait  # Wait for all background processes to finish