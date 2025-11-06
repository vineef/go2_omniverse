"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher
import omni

def launch_simulator(args_cli=None, parser=None):
    """Launch the Isaac Sim Simulator app."""

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)

    return simulation_app