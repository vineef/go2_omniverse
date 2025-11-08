"""Launch Isaac Sim Simulator first."""
from isaaclab.app import AppLauncher

def launch_simulator(args_cli=None, parser=None):
    """Launch the Isaac Sim Simulator app."""

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import omni
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("isaacsim.ros2.bridge", True)
    ext_manager.set_extension_enabled_immediate("omni.graph.window.action", True)

    return simulation_app