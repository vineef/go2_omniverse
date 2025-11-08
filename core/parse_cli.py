import cli_args
import argparse

from isaaclab.app import AppLauncher

def parse_cli_args():
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
    parser.add_argument(
        "--use_sim_time", action="store_true", default=False, help="Use simulated time in ROS2"
    )


    # append RSL-RL cli arguments
    cli_args.add_rsl_rl_args(parser)


    # append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    return args_cli, parser