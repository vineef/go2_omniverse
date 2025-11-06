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


"""Launch Isaac Sim Simulator first."""
import argparse
from isaaclab.app import AppLauncher


import cli_args
import time
import os
import threading


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
# parser.add_argument("--device", type=str, default="cpu", help="Use CPU pipeline.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of environments to simulate."
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Velocity-Rough-Unitree-Go2-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--custom_env", type=str, default="", help="Setup the environment"
)
parser.add_argument("--robot", type=str, default="go2", help="Setup the robot")
parser.add_argument(
    "--robot_amount", type=int, default=1, help="Setup the robot amount"
)


# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import omni


ext_manager = omni.kit.app.get_app().get_extension_manager()
ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

# FOR VR SUPPORT
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.core", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.steamvr", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.simulatedxr", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.system.openxr", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.telemetry", True)
# ext_manager.set_extension_enabled_immediate("omni.kit.xr.profile.vr", True)


"""Rest everything follows."""
import gymnasium as gym
import torch
import carb


from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
)
import isaaclab.sim as sim_utils
import omni.appwindow
from rsl_rl.runners import OnPolicyRunner


import rclpy
from rclpy.executors import MultiThreadedExecutor
from ros2 import (
    RobotBaseNode,
    add_camera,
    add_rtx_lidar,
    pub_robo_data_ros2
)
from geometry_msgs.msg import Twist


from agent_cfg import unitree_go2_agent_cfg
from custom_rl_env import UnitreeGo2CustomEnvCfg
import custom_rl_env

from isaaclab.assets import Articulation


from omnigraph import create_front_cam_omnigraph

def launch_rviz(rviz_config_file_path):
    """
    Lança o RViz com o arquivo de configuração especificado.
    """
    print(f"[INFO] Lançando RViz com o arquivo de configuração: {rviz_config_file_path}")
    try:
        # Usamos nohup e & para que o RViz continue rodando mesmo se o script Python for fechado
        # e para que o script Python não espere o RViz terminar.
        command = f"nohup rviz2 -d {rviz_config_file_path} > /dev/null 2>&1 &"
        subprocess.Popen(command, shell=True)
        print("[INFO] Comando RViz enviado.")
    except Exception as e:
        print(f"[ERROR] Falha ao lançar RViz: {e}")


def sub_keyboard_event(event, *args, **kwargs) -> bool:
    linear_velocity = 1.5
    angular_velocity = 2.0

    if len(custom_rl_env.base_command) > 0:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "W":
                custom_rl_env.base_command["0"] = [linear_velocity, 0, 0]
            if event.input.name == "S":
                custom_rl_env.base_command["0"] = [-linear_velocity, 0, 0]
            if event.input.name == "A":
                custom_rl_env.base_command["0"] = [0, linear_velocity, 0]
            if event.input.name == "D":
                custom_rl_env.base_command["0"] = [0, -linear_velocity, 0]
            if event.input.name == "Q":
                custom_rl_env.base_command["0"] = [0, 0, angular_velocity]
            if event.input.name == "E":
                custom_rl_env.base_command["0"] = [0, 0, -angular_velocity]

            if len(custom_rl_env.base_command) > 1:
                if event.input.name == "I":
                    custom_rl_env.base_command["1"] = [1, 0, 0]
                if event.input.name == "K":
                    custom_rl_env.base_command["1"] = [-1, 0, 0]
                if event.input.name == "J":
                    custom_rl_env.base_command["1"] = [0, 1, 0]
                if event.input.name == "L":
                    custom_rl_env.base_command["1"] = [0, -1, 0]
                if event.input.name == "U":
                    custom_rl_env.base_command["1"] = [0, 0, 1]
                if event.input.name == "O":
                    custom_rl_env.base_command["1"] = [0, 0, -1]
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            for i in range(len(custom_rl_env.base_command)):
                custom_rl_env.base_command[str(i)] = [0, 0, 0]
    return True


def setup_custom_env():
    try:
        if args_cli.custom_env == "warehouse":
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/warehouse.usd")
            cfg_scene.func("/World/warehouse", cfg_scene, translation=(0.0, 0.0, 0.0))

        if args_cli.custom_env == "office":
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/office.usd")
            cfg_scene.func("/World/office", cfg_scene, translation=(0.0, 0.0, 0.0))

        if args_cli.custom_env == "office2":
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/office2.usd")
            cfg_scene.func("/World/office2", cfg_scene, translation=(0.0, 0.0, 0.0))

        if args_cli.custom_env == "simple_room":
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/simple_room.usd")
            cfg_scene.func("/World/simple_room", cfg_scene, translation=(0.0, 0.0, 0.0))

        if args_cli.custom_env == "house":
            cfg_scene = sim_utils.UsdFileCfg(usd_path="./envs/basic_house_map.usdz")
            cfg_scene.func("/World/house", cfg_scene, translation=(0.0, 0.0, 0.0))

    except:
        print(
            "Error loading custom environment. You should download custom envs folder from: https://drive.google.com/drive/folders/1vVGuO1KIX1K6mD6mBHDZGm9nk2vaRyj3?usp=sharing"
        )


from pxr import Usd, UsdGeom, PhysxSchema, Gf

def setup_custom_env(custom_env: str):
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

        # 🔽 Após carregar, adiciona colisores básicos
        # def add_collision_to_meshes(prim):
        #     if prim.IsA(UsdGeom.Mesh):
        #         # Cria colisor se não existir
        #         if not PhysxSchema.PhysxCollisionAPI(prim):
        #             PhysxSchema.PhysxCollisionAPI.Apply(prim)
        #             print(f"Added collider to: {prim.GetPath()}")
        #     for child in prim.GetChildren():
        #         add_collision_to_meshes(child)

        # add_collision_to_meshes(root_prim)

        print(f"Environment '{custom_env}' loaded successfully at {prim_path}")

    except Exception as e:
        print(f"Error loading environment '{custom_env}': {e}")
        print(
            "You should download the custom envs folder from: "
            "https://drive.google.com/drive/folders/1vVGuO1KIX1K6mD6mBHDZGm9nk2vaRyj3?usp=sharing"
        )



def cmd_vel_cb(msg, num_robot):
    scale = 1.0
    x = msg.linear.x * scale
    y = msg.linear.y * scale
    z = msg.angular.z * scale
    custom_rl_env.base_command[str(num_robot)] = [x, y, z]


def add_cmd_sub(num_envs):
    node_test = rclpy.create_node("position_velocity_publisher")
    for i in range(num_envs):
        node_test.create_subscription(
            # Twist, f"cmd_vel", lambda msg, i=i: cmd_vel_cb(msg, str(i)), 10
            # Twist, f"/cmd_vel_out", lambda msg, i=i: cmd_vel_cb(msg, str(i)), 10
            Twist, f"/cmd_vel_out", lambda msg: cmd_vel_cb(msg, "0"), 10
            
        )
    return node_test


def specify_cmd_for_robots(numv_envs):
    for i in range(numv_envs):
        custom_rl_env.base_command[str(i)] = [0, 0, 0]

def enable_collisions_for_all(stage):
    """
    Percorre todos os prims e aplica PhysXCollisionAPI em meshes que ainda não possuem.
    """
    def apply_collision(prim):
        if prim.IsA(UsdGeom.Mesh):
            if not PhysxSchema.PhysxCollisionAPI(prim):
                PhysxSchema.PhysxCollisionAPI.Apply(prim)
                print(f"✅ Collision enabled for: {prim.GetPath()}")
        for child in prim.GetChildren():
            apply_collision(child)

    root = stage.GetPrimAtPath("/World")
    apply_collision(root)


def run_sim():

    # acquire input interface
    _input = carb.input.acquire_input_interface()
    _appwindow = omni.appwindow.get_default_app_window()
    _keyboard = _appwindow.get_keyboard()
    _sub_keyboard = _input.subscribe_to_keyboard_events(_keyboard, sub_keyboard_event)

    """Play with RSL-RL agent."""
    # parse configuration

    env_cfg = UnitreeGo2CustomEnvCfg()

    # add N robots to env
    env_cfg.scene.num_envs = args_cli.robot_amount

    specify_cmd_for_robots(env_cfg.scene.num_envs)

    agent_cfg: RslRlOnPolicyRunnerCfg = unitree_go2_agent_cfg

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg["experiment_name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(
        log_root_path, agent_cfg["load_run"], agent_cfg["load_checkpoint"]
    )

    # load previously trained model
    ppo_runner = OnPolicyRunner(
        env, agent_cfg, log_dir=None, device=agent_cfg["device"]
    )
    ppo_runner.load(resume_path)
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # reset environment
    obs, _ = env.get_observations()

    # initialize ROS2 node
    rclpy.init()
    base_node = RobotBaseNode(env_cfg.scene.num_envs)
    node_test = add_cmd_sub(env_cfg.scene.num_envs)

    executor = MultiThreadedExecutor()
    # executor.add_node(base_node)
    executor.add_node(node_test) # DESCOMENTAR PARA JOYSTICK
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    annotator_lst = None
    UnitreeL1_annotator_lst = add_rtx_lidar(env_cfg.scene.num_envs, args_cli.robot, "UnitreeL1", False)
    Robosense_annotator_lst = add_rtx_lidar(env_cfg.scene.num_envs, args_cli.robot, "Robosense", False)
    annotator_lst = UnitreeL1_annotator_lst + Robosense_annotator_lst
    # annotator_lst = Robosense_annotator_lst
    camera_objects = add_camera(env_cfg.scene.num_envs, args_cli.robot)

    # create ros2 camera stream omnigraph
    for i in range(env_cfg.scene.num_envs):
        create_front_cam_omnigraph(i)

    # setup_custom_env()
    setup_custom_env(args_cli.custom_env)

    # stage = omni.usd.get_context().get_stage()
    # enable_collisions_for_all(stage)

    try:
        # simulate environment
        while simulation_app.is_running():
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, _, _ = env.step(actions)
                pub_robo_data_ros2(
                    args_cli.robot,
                    env_cfg.scene.num_envs,
                    base_node,
                    env,
                    annotator_lst,
                )

                # --- Conditional ROS2 spinning ---
                # if current_time - last_ros2_update_time >= ros2_update_period:
                #     rclpy.spin_once(base_node, timeout_sec=0.0) # No timeout needed if we're controlling frequency
                #     last_ros2_update_time = current_time
                # ---------------------------------
    finally:
        print("Closing simulation app...")
        env.close()
        simulation_app.close()

    env.close()
    rclpy.shutdown()
    executor_thread.join()

