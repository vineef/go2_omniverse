#!/bin/bash
# Copyright (c) 2024, RoboVerse community
# SPDX-License-Identifier: BSD-2-Clause

# ==== CONFIGURAÇÃO DO AMBIENTE ROS ====
clear
source /opt/ros/humble/setup.bash

# ==== BUILD DOS WORKSPACES ====
# cd IsaacSim-ros_workspaces/humble_ws
# rosdep install --from-paths src --ignore-src -r -y
# colcon build
# source install/setup.bash
# cd ../..

cd go2_omniverse_ws
# rosdep install --from-paths src --ignore-src -r -y
# colcon build
source install/setup.bash
cd ..

# ==== ATIVAR CONDA E EXECUTAR SIMULAÇÃO ====
eval "$(conda shell.bash hook)"
conda activate env_isaaclab
# conda activate env_isaaclab_510

export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# ==== EXECUTAR SIMULAÇÃO ====
# cd go2_omniverse
python main.py --robot_amount 1 --robot go2 --device cuda --enable_cameras --custom_env house --use_sim_time #--headless #--rendering_mode performance 
# python main2.py --robot_amount 1 --robot go2 --device cuda --custom_env house
# python test.py
