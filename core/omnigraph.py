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


import omni
import omni.graph.core as og


def create_front_cam_omnigraph(robot_num):
    """Define the OmniGraph for the Isaac Sim environment."""

    keys = og.Controller.Keys

    graph_path = f"/ROS_" + f"front_cam{robot_num}"
    og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                (
                    "IsaacCreateRenderProduct",
                    "isaacsim.core.nodes.IsaacCreateRenderProduct",
                ),
                ("ROS2CameraHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.SET_VALUES: [
                (
                    "IsaacCreateRenderProduct.inputs:cameraPrim",
                    f"/World/envs/env_{robot_num}/Robot/base/front_cam",
                ),
                ("IsaacCreateRenderProduct.inputs:enabled", True),
                ("ROS2CameraHelper.inputs:type", "rgb"),
                (
                    "ROS2CameraHelper.inputs:topicName",
                    f"robot{robot_num}/front_cam/rgb",
                ),
                ("ROS2CameraHelper.inputs:frameId", f"robot{robot_num}"),
            ],
            keys.CONNECT: [
                (
                    "OnPlaybackTick.outputs:tick",
                    "IsaacCreateRenderProduct.inputs:execIn",
                ),
                (
                    "IsaacCreateRenderProduct.outputs:execOut",
                    "ROS2CameraHelper.inputs:execIn",
                ),
                (
                    "IsaacCreateRenderProduct.outputs:renderProductPath",
                    "ROS2CameraHelper.inputs:renderProductPath",
                ),
            ],
        },
    )


def create_d455_rgb_depth_graph():
    """Cria OmniGraph para publicar RGB, Depth e CameraInfo da Realsense D455."""

    keys = og.Controller.Keys
    graph_path = f"/ROS_CameraGraph_D455_robot0"

    color_cam_path = f"/World/envs/env_0/Robot/base/d455/RSD455/Camera_OmniVision_OV9782_Color"
    depth_cam_path = f"/World/envs/env_0/Robot/base/d455/RSD455/Camera_Pseudo_Depth"

    width_rgb = 640
    height_rgb = 480
    width_depth = int(width_rgb/2)
    height_depth = int(height_rgb/2)

    og.Controller.edit(
        {
            "graph_path": graph_path,
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),

                ("RenderProductDepthColor", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("RenderProductDepth", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("RenderProductRGB", "isaacsim.core.nodes.IsaacCreateRenderProduct"),

                ("ROS2CameraDepthColor", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("ROS2CameraDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                # ("ROS2CameraInfoColor", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("ROS2CameraDepthInfo", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("ROS2CameraRGB", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.SET_VALUES: [
                # RenderProducts
                ("RenderProductDepthColor.inputs:cameraPrim", color_cam_path),
                ("RenderProductDepthColor.inputs:enabled", True),
                ("RenderProductDepthColor.inputs:width", width_depth),
                ("RenderProductDepthColor.inputs:height", height_depth),
                
                ("RenderProductDepth.inputs:cameraPrim", depth_cam_path),
                ("RenderProductDepth.inputs:enabled", True),
                ("RenderProductDepth.inputs:width", width_depth),
                ("RenderProductDepth.inputs:height", height_depth),

                ("RenderProductRGB.inputs:cameraPrim", depth_cam_path),
                ("RenderProductRGB.inputs:enabled", True),
                ("RenderProductRGB.inputs:width", width_rgb),
                ("RenderProductRGB.inputs:height", height_rgb),

                # ROS2 publishers RGB
                ("ROS2CameraDepthColor.inputs:type", "rgb"),
                ("ROS2CameraDepthColor.inputs:topicName", f"robot0/d455/depth_rgb"),
                ("ROS2CameraDepthColor.inputs:frameId", f"robot0/Realsense_link"),
                # ("ROS2CameraDepthColor.inputs:useSystemTime", True),

                # ROS2 publishers Depth
                ("ROS2CameraDepth.inputs:type", "depth"),
                ("ROS2CameraDepth.inputs:topicName", f"robot0/d455/depth"),
                ("ROS2CameraDepth.inputs:frameId", f"robot0/Realsense_link"),
                # ("ROS2CameraDepth.inputs:useSystemTime", True),

                # ROS2 publishers RGB
                ("ROS2CameraRGB.inputs:type", "rgb"),
                ("ROS2CameraRGB.inputs:topicName", f"robot0/d455/rgb"),
                ("ROS2CameraRGB.inputs:frameId", f"robot0/Realsense_link"),
                # ("ROS2CameraRGB.inputs:useSystemTime", True),

                # ROS2 CameraInfo RGB
                # ("ROS2CameraInfoColor.inputs:type", "camera_info"),
                # ("ROS2CameraInfoColor.inputs:topicName", f"robot0/d455/rgb/camera_info"),
                # ("ROS2CameraInfoColor.inputs:frameId", f"robot0/Realsense_link"),

                # ROS2 CameraInfo Depth
                ("ROS2CameraDepthInfo.inputs:type", "camera_info"),
                ("ROS2CameraDepthInfo.inputs:topicName", f"robot0/d455/camera_info"),
                ("ROS2CameraDepthInfo.inputs:frameId", f"robot0/Realsense_link"),
                # ("ROS2CameraDepthInfo.inputs:useSystemTime", True),
            ],
            keys.CONNECT: [
                # Conecta execução
                ("OnPlaybackTick.outputs:tick", "RenderProductDepthColor.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "RenderProductDepth.inputs:execIn"),
                ("OnPlaybackTick.outputs:tick", "RenderProductRGB.inputs:execIn"),

                ("RenderProductDepthColor.outputs:execOut", "ROS2CameraDepthColor.inputs:execIn"),
                ("RenderProductDepth.outputs:execOut", "ROS2CameraDepth.inputs:execIn"),
                ("RenderProductRGB.outputs:execOut", "ROS2CameraRGB.inputs:execIn"),

                # ("RenderProductDepthColor.outputs:execOut", "ROS2CameraInfoColor.inputs:execIn"),
                ("RenderProductDepth.outputs:execOut", "ROS2CameraDepthInfo.inputs:execIn"),

                # Conecta renderProductPath
                ("RenderProductDepthColor.outputs:renderProductPath", "ROS2CameraDepthColor.inputs:renderProductPath"),
                ("RenderProductDepth.outputs:renderProductPath", "ROS2CameraDepth.inputs:renderProductPath"),
                ("RenderProductRGB.outputs:renderProductPath", "ROS2CameraRGB.inputs:renderProductPath"),

                # Conecta renderProductPath para CameraInfo
                # ("RenderProductDepthColor.outputs:renderProductPath", "ROS2CameraInfoColor.inputs:renderProductPath"),
                ("RenderProductDepth.outputs:renderProductPath", "ROS2CameraDepthInfo.inputs:renderProductPath"),
            ],
        },
    )


def create_ros2_clock_publisher():
    """
    Cria um nó OmniGraph que publica o tempo de simulação em /clock.
    """
    print("[VINICIUS] Creating ROS2 clock publisher...")
    
    graph_path = "/ROS_ClockGraph"

    # Cria o grafo se não existir
    # if not og.Controller.exists(graph_path):
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ROS2Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ROS2PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            og.Controller.Keys.CONNECT: [
                # Conecta o tempo simulado ao publicador
                ("OnPlaybackTick.outputs:tick", "ROS2PublishClock.inputs:execIn"),
                ("ROS2Context.outputs:context", "ROS2PublishClock.inputs:context"),
                ("ReadSimTime.outputs:simulationTime", "ROS2PublishClock.inputs:timeStamp"),
            ],
        },
    )
    print("[INFO] ROS2 clock publisher criado com sucesso.")
    # else:
    #     print("[INFO] ROS2 clock publisher já existe.")