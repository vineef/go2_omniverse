# test_minimal.py
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicCuboid

# Criar mundo simples
world = World()
world.scene.add_default_ground_plane()

# Adicionar apenas um cubo
cube = DynamicCuboid(
    prim_path="/World/cube",
    name="cube",
    position=[0, 0, 1.0],
    size=0.5,
)

world.reset()

for i in range(100):
    world.step(render=True)

simulation_app.close()