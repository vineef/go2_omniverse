# Copyright (c) 2024, RoboVerse community
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""Script to play a checkpoint if an RL agent from RSL-RL."""

from __future__ import annotations

import os
import gymnasium as gym
import torch

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from core.ros2 import add_camera, add_realsense, add_rtx_lidar, pub_robo_data_ros2, init_ros_nodes

from core.agent_cfg import unitree_go2_agent_cfg
import core.custom_rl_env as rl_env
from core.omnigraph import *#create_front_cam_omnigraph, create_pointcloud_omnigraph

def setup_custom_env(custom_env: str):
    import omni
    from pxr import UsdGeom
    
    env_map = {
        "warehouse": "./envs/warehouse.usd",
        "office": "./envs/office.usd",
        "office2": "./envs/office2.usd",
        "simple_room": "./envs/simple_room.usd",
        "modern_reception": "./envs/Assets/ArchVis/Commercial/Reception/Modern.usd",
        "house": "./envs/basic_house_map.usdz",
    }

    if custom_env not in env_map:
        print(f"Environment '{custom_env}' not found. Available: {list(env_map.keys())}")
        return

    usd_path = env_map[custom_env]

    try:
        # Obtém o Stage atual
        stage = omni.usd.get_context().get_stage()

        # Cria um prim root para o ambiente
        prim_path = f"/World/{custom_env}"
        if not stage.GetPrimAtPath(prim_path):
            root_prim = stage.DefinePrim(prim_path, "Xform")

        xform = UsdGeom.XformCommonAPI(root_prim)
        xform.SetTranslate((2.0, 0.0, 0.0))
        xform.SetScale((0.00015, 0.00015, 0.00015))
        xform.SetRotate((-90.0, 0.0, 0.0))

        # Carrega o USD externo como referência (Reference) dentro do prim
        root_prim = stage.GetPrimAtPath(prim_path)
        root_prim.GetReferences().AddReference(usd_path)

        print(f"Environment '{custom_env}' loaded successfully at {prim_path}")

    except Exception as e:
        print(f"Error loading environment '{custom_env}': {e}")


def env_config():
    env_cfg = rl_env.UnitreeGo2CustomEnvCfg()
    env_cfg.scene.num_envs = 1
    rl_env.base_command["0"] = [0, 0, 0]

    return env_cfg


def create_env(env_cfg, task):
    env = gym.make(task, cfg=env_cfg) # Create isaac environment
    env = RslRlVecEnvWrapper(env) # Wrap around environment for rsl-rl

    return env


def load_checkpoint(env):
    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_agent_cfg

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg["experiment_name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(log_root_path, agent_cfg["load_run"], agent_cfg["load_checkpoint"])

    # Load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg, log_dir=None, device=agent_cfg["device"])
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    return ppo_runner
    

def run_sim(simulation_app, args_cli):
    """Play with RSL-RL agent."""

    from core.keyboard_input import keyboard_config
    keyboard_config()

    env_cfg = env_config()
    env = create_env(env_cfg, args_cli.task)
    ppo_runner = load_checkpoint(env)

    # Obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)    

    # base_node = RobotBaseNode(env_cfg.scene.num_envs)

    annotator_lst = None
    UnitreeL1_annotator_lst = add_rtx_lidar(env_cfg.scene.num_envs, "UnitreeL1", debug=False)
    Robosense_annotator_lst = add_rtx_lidar(env_cfg.scene.num_envs, "Robosense", debug=False)
    annotator_lst = UnitreeL1_annotator_lst + Robosense_annotator_lst
    camera_objects = add_camera(env_cfg.scene.num_envs)
    add_realsense()

    # create_front_cam_omnigraph(0) # Create ros2 camera stream omnigraph
    create_d455_rgb_depth_graph()

    if args_cli.use_sim_time:
        create_ros2_clock_publisher()
    
    setup_custom_env(args_cli.custom_env)

    # Initialize ROS2 node
    # rclpy.init()

    try:
        base_node = init_ros_nodes(env_cfg.scene.num_envs, env, env_cfg, annotator_lst, args_cli.use_sim_time)
        obs, _ = env.get_observations() # Reset environment

        # Simulate environment
        while simulation_app.is_running():
            with torch.inference_mode(): # Run everything in inference mode
                actions = policy(obs) # Agent stepping
                obs, _, _, _ = env.step(actions) # env stepping
                
                try:
                    pub_robo_data_ros2(
                        env_cfg.scene.num_envs,
                        base_node,
                        env,
                        annotator_lst,
                    )
                except Exception as e:
                    print("Erro ao publicar ROS2:", e)

                # --- Conditional ROS2 spinning ---
                # if current_time - last_ros2_update_time >= ros2_update_period:
                #     rclpy.spin_once(base_node, timeout_sec=0.0) # No timeout needed if we're controlling frequency
                #     last_ros2_update_time = current_time
                # ---------------------------------

    except KeyboardInterrupt:
        print("Keyboard interrupt received - exiting gracefully.")
    finally:
        print("Closing simulation app...")
        env.close()
        simulation_app.close()

