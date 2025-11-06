import carb
import omni
import core.custom_rl_env as rl_env

def keyboard_config():
    """Configure keyboard input interface."""

    # acquire input interface
    _input = carb.input.acquire_input_interface()
    _appwindow = omni.appwindow.get_default_app_window()
    _keyboard = _appwindow.get_keyboard()
    _sub_keyboard = _input.subscribe_to_keyboard_events(_keyboard, sub_keyboard_event)

def sub_keyboard_event(event, *args, **kwargs) -> bool:
    linear_velocity = 1.5
    angular_velocity = 2.0

    if len(rl_env.base_command) > 0:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "W":
                rl_env.base_command["0"] = [linear_velocity, 0, 0]
            if event.input.name == "S":
                rl_env.base_command["0"] = [-linear_velocity, 0, 0]
            if event.input.name == "A":
                rl_env.base_command["0"] = [0, linear_velocity, 0]
            if event.input.name == "D":
                rl_env.base_command["0"] = [0, -linear_velocity, 0]
            if event.input.name == "Q":
                rl_env.base_command["0"] = [0, 0, angular_velocity]
            if event.input.name == "E":
                rl_env.base_command["0"] = [0, 0, -angular_velocity]

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            rl_env.base_command["0"] = [0, 0, 0]
    
    return True