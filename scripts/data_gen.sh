#!/bin/bash

# A single task should contain 20w trajectories.
# each dataset folder should contain 600 trajectories with the following rule:
# max_episodes: 30
# num_envs: 20
# Total number of trajectories per bash run: LOOP_COUNT * 30 * 20
# If use 4 process to run the bash script, we can set LOOP_COUNT=80.
# totol_run should be around 320


SYNC_LOOP_COUNT=60  # Set this to the number of parallel processes you want
ASYNC_LOOP_COUNT=7   # Number of asynchronous runs per sync run


for ((i=0; i<SYNC_LOOP_COUNT; i++)); do
    for ((j=0; j<ASYNC_LOOP_COUNT; j++)); do
        python -m embodichain.lab.scripts.run_env \
            --gym_config ur10_parallel.json \
            --headless --enable_rt --num_envs 20 --gpu_id 6 &
        sleep 5

    done
    
    wait
done
