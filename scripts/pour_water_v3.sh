# pass arguments to the script
#!/bin/sh
echo $1 $2

projectdir=$(dirname $(cd $(dirname $0); pwd))
demo="pour_water_v3"

# task_type="pour_water_5e-2range "
# robot_name="Aloha"

if [[ "$2" == "headless" || "$2" == "--headless" ]]; then  
    headless_cfg="--headless"  
    headless_msg="ON"
else  
    headless_cfg=""  
    headless_msg="OFF"
fi

echo "Run $demo."
# echo "- Task Type: $task_type"  
# echo "- Robot Type: $robot_name"  
echo "- Headless Mode: $headless_msg"

python3 -m embodichain.lab.scripts.run_env \
    --gym_config ${projectdir}/embodichain/lab/configs/${demo}/gym_config.json  \
    --action_config ${projectdir}/embodichain/lab/configs/${demo}/action_config.json  \
    ${headless_cfg}
    # --task_type "${task_type}" \
    # --robot_name "${robot_name}" \
    # --obj_config ${projectdir}/embodichain/lab/configs/${demo}/object_config.json  \

