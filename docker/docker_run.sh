#!/bin/bash  
# Usage: ./docker_run.sh [container_name] [data_path]  
  
# Check if the correct number of arguments are provided  
if [ "$#" -ne 2 ]; then  
  echo "Usage: $0 [container_name] [data_path]"  
  exit 1  
fi  
  
container_name=$1  
data_path=$2  
  
echo "image_name: $container_name"  
echo "data_path: $data_path"  
  
# Initialize error flag and paths  
error_flag=0  
VK_DRIVER_PATH=""  
GL_DRIVER_PATH=""  
  
# Check Vulkan driver paths  
if [ -f /etc/vulkan/icd.d/nvidia_icd.json ] && [ -f /etc/vulkan/implicit_layer.d/nvidia_layers.json ] &&  
   ! [ -d /etc/vulkan/icd.d/nvidia_icd.json ] && ! [ -d /etc/vulkan/implicit_layer.d/nvidia_layers.json ]; then  
  VK_DRIVER_PATH=/etc  
elif [ -f /usr/share/vulkan/icd.d/nvidia_icd.json ] && [ -f /usr/share/vulkan/implicit_layer.d/nvidia_layers.json ] &&  
     ! [ -d /usr/share/vulkan/icd.d/nvidia_icd.json ] && ! [ -d /usr/share/vulkan/implicit_layer.d/nvidia_layers.json ]; then  
  VK_DRIVER_PATH=/usr/share  
else  
  if [ -d /etc/vulkan/icd.d/nvidia_icd.json ] || [ -d /etc/vulkan/implicit_layer.d/nvidia_layers.json ]; then  
    echo "Warning: /etc/vulkan/icd.d/nvidia_icd.json or /etc/vulkan/implicit_layer.d/nvidia_layers.json is a directory, not a file."  
  elif [ -d /usr/share/vulkan/icd.d/nvidia_icd.json ] || [ -d /usr/share/vulkan/implicit_layer.d/nvidia_layers.json ]; then  
    echo "Warning: /usr/share/vulkan/icd.d/nvidia_icd.json or /usr/share/vulkan/implicit_layer.d/nvidia_layers.json is a directory, not a file."  
  else  
    echo "Warning: Required Vulkan driver files not found."  
  fi  
    error_flag=$((error_flag + 1))  
fi  
echo "VK_DRIVER_PATH: $VK_DRIVER_PATH"  
  
# Check OpenGL driver paths  
if [ -f /etc/glvnd/egl_vendor.d/10_nvidia.json ] && ! [ -d /etc/glvnd/egl_vendor.d/10_nvidia.json ]; then  
  GL_DRIVER_PATH=/etc  
elif [ -f /usr/share/glvnd/egl_vendor.d/10_nvidia.json ] && ! [ -d /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then  
  GL_DRIVER_PATH=/usr/share  
else  
  if [ -d /etc/glvnd/egl_vendor.d/10_nvidia.json ]; then  
    echo "Warning: /etc/glvnd/egl_vendor.d/10_nvidia.json is a directory, not a file."  
  elif [ -d /usr/share/glvnd/egl_vendor.d/10_nvidia.json ]; then  
    echo "Warning: /usr/share/glvnd/egl_vendor.d/10_nvidia.json is a directory, not a file."  
  else  
    echo "Warning: Required OpenGL driver files not found."  
  fi  
  error_flag=$((error_flag + 1))  
fi  
echo "GL_DRIVER_PATH: $GL_DRIVER_PATH"  
  
# Check error flag and exit if necessary  
if [ $error_flag -eq 2 ]; then  
  echo "No driver paths were found to be directories instead of files. Please check the paths."  
  exit 1  
fi  
  
# Build the docker run command with conditional volume mounts  
docker_cmd="docker run -itd --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  --device /dev/dri \
  --name $container_name \
  -v /usr/share/nvidia:/usr/share/nvidia \
  -v $data_path:/root/workspace"
  
# Add Vulkan driver mounts if VK_DRIVER_PATH is set  
if [ -n "$VK_DRIVER_PATH" ]; then  
  docker_cmd="$docker_cmd \
  -v ${VK_DRIVER_PATH}/vulkan/icd.d/nvidia_icd.json:${VK_DRIVER_PATH}/vulkan/icd.d/nvidia_icd.json \
  -v ${VK_DRIVER_PATH}/vulkan/implicit_layer.d/nvidia_layers.json:${VK_DRIVER_PATH}/vulkan/implicit_layer.d/nvidia_layers.json"  
fi  
  
# Add OpenGL driver mounts if GL_DRIVER_PATH is set  
if [ -n "$GL_DRIVER_PATH" ]; then  
  docker_cmd="$docker_cmd \
  -v ${GL_DRIVER_PATH}/glvnd/egl_vendor.d/10_nvidia.json:${GL_DRIVER_PATH}/glvnd/egl_vendor.d/10_nvidia.json"  
fi  
  
# Add remaining common volume mounts and environment variables  
docker_cmd="$docker_cmd \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v ~/.Xauthority:/root/.Xauthority:rw \
  -e DISPLAY=$DISPLAY \
  -e WORK=~/workspace \
  --shm-size 50G \
  --network=host \
  -v /dev/shm:/dev/shm \
  dexforce/embodichain:ubuntu22.04-cuda12.8 /bin/bash"  
  
# Execute the docker run command  
eval $docker_cmd  
  
# Print success message  
if [ $? -eq 0 ]; then  
  echo "Docker container $container_name started successfully."  
else  
  echo "Failed to start Docker container $container_name."  
fi  
