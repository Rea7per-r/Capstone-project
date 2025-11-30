#!/bin/bash

# Activate virtual environment
source ~/yolo_env_py311/bin/activate

# Ensure ROS uses Python 3.11
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
export ROS_PYTHON_VERSION=3
export PYTHONEXECUTABLE=$(which python)

# Run YOLO detection node
rosrun ml_detector detect_node_publisher.py "$@"

