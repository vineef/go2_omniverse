#!/bin/bash
# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-2-Clause

# ==== CONFIGURAÇÃO DO AMBIENTE ROS ====
clear
source /opt/ros/humble/setup.bash

# ==== BUILD DO WORKSPACE DO SDK ====
cd ../ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install
source install/setup.bash

# ==== EXECUTAR O SDK ====
ros2 launch go2_robot_sdk robot.launch.py
# opcional: iniciar rviz2
# nohup rviz2 -d rviz_configs/go2_single_robot.rviz > /dev/null 2>&1 &
# ou:
# nohup rviz2 -d rviz_configs/go2_single_robot.rviz &
