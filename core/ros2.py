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

import asyncio
import time
import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.parameter import Parameter
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from tf2_ros import TransformBroadcaster
from go2_interfaces.msg import Go2State
from std_msgs.msg import Header

from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, PointField, Imu
from sensor_msgs_py import point_cloud2

from rosgraph_msgs.msg import Clock

from isaaclab.sensors import CameraCfg, Camera
import omni.kit.commands
import omni.replicator.core as rep
import isaaclab.sim as sim_utils

from pxr import UsdGeom, UsdPhysics, Gf, Sdf, UsdPhysics, Usd
from tf_transformations import quaternion_from_euler

import rclpy
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
import threading
import core.custom_rl_env as rl_env

import isaacsim.core.utils.nucleus as nucleus_utils
from isaacsim.core.utils.stage import add_reference_to_stage

UnitreeL1_translation = (0.293, 0.0, -0.08)
UnitreeL1_quat = quaternion_from_euler(0, 165 * 3.14159265 / 180, 0) # 165° y axis

Robosense_translation = (0.15, 0.0, 0.18)
Robosense_quat = quaternion_from_euler(0, 0, 0) 

Realsense_translation = (0.35, 0.0, 0.075)
Realsense_quat = quaternion_from_euler(-80 * 3.14159265 / 180, 0, -90 * 3.14159265 / 180) # 10graus virado pra cima

def cmd_vel_cb(msg):
    scale = 1.0
    rl_env.base_command["0"] = [msg.linear.x * scale, msg.linear.y * scale, msg.angular.z * scale]

def ros2_pub_loop(node, env, env_cfg, annotator_lst, publish_rate_hz=20):
    """Loop que publica dados do robô via ROS2."""
    rate_sec = 1.0 / publish_rate_hz
    while rclpy.ok():
        try:
            # publica dados do robô
            pub_robo_data_ros2(
                env_cfg.scene.num_envs,
                node,
                env,
                annotator_lst,
            )
        except Exception as e:
            print("Erro ao publicar ROS2:", e)
        rclpy.spin_once(node, timeout_sec=rate_sec)


def ros2_sub_loop(sub_node, rate_hz=10):
    """Loop que processa callbacks do subscriber."""
    rate_sec = 1.0 / rate_hz
    while rclpy.ok():
        try:
            rclpy.spin_once(sub_node, timeout_sec=rate_sec)
        except IndexError:
            print("[ROS2] Executor finalizado ou desconectado.")
            break


def init_ros_nodes(num_envs, env, env_cfg, annotator_lst, use_sim_time):
    rclpy.init()

    from core.ros2 import RobotBaseNode
    node = RobotBaseNode(num_envs, use_sim_time)

    # Node de subscrição
    sub_node = rclpy.create_node("cmd_vel_listener")
    sub_node.create_subscription(Twist, "/cmd_vel_out", cmd_vel_cb, 10)
    sub_node.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, use_sim_time)])

    # Thread para publicação (node principal)
    pub_thread = threading.Thread(
        target=ros2_pub_loop,
        args=(node, env, env_cfg, annotator_lst),
        daemon=True
    )
    # pub_thread.start()

    # Thread para assinaturas (cmd_vel_listener)
    sub_thread = threading.Thread(
        target=ros2_sub_loop,
        args=(sub_node,),
        daemon=True
    )
    sub_thread.start()

    return node

def add_rtx_lidar(num_envs, lidar_type, debug=False):
    if lidar_type == "UnitreeL1":
        trans = UnitreeL1_translation
        quat = UnitreeL1_quat
        config = "Unitree_L1"
    if lidar_type == "Robosense":
        trans = Robosense_translation
        quat = Robosense_quat
        config = "Robosense"
    
    annotator_lst = []
    for i in range(num_envs):

        _, lidar_sensor = omni.kit.commands.execute(
            "IsaacSensorCreateRtxLidar",
            path=f"/World/envs/env_{i}/Robot/base/lidar_sensor",
            parent=None,
            translation=trans,
            orientation=Gf.Quatd(quat[3], quat[0], quat[1], quat[2]),
            config=config,
        )
        # if lidar_type == "Robosense":
        #     attach_usd_to_sensor(lidar_sensor.GetPath(), "./lidar/os2_mesh.usd")

        lidar_texture = rep.create.render_product(lidar_sensor.GetPath(), [1, 1], name="UnitreeL1")
        if debug:
            # Create the debug draw pipeline in the post process graph
            writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloudBuffer")
            writer.attach([lidar_texture])

        # writer = rep.writers.get("RtxLidar" + "ROS2PublishPointCloud")
        # writer.initialize(topicName=f"robot{i}/point_cloud2", frameId=f"robot{i}/base_link")
        # writer.attach([lidar_texture])

        # annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacCreateRTXLidarScanBuffer")
        # annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorGpuIsaacReadRTXLidarData")
        # annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacReadRTXLidarData")
        annotator = rep.AnnotatorRegistry.get_annotator("RtxSensorCpuIsaacComputeRTXLidarPointCloud")
        annotator.attach(lidar_texture)

        annotator_info = {
            "type": lidar_type,
            "annotator_object": annotator
        }

        annotator_lst.append(annotator_info)
    
    return annotator_lst

def attach_usd_to_sensor(sensor_path: str, usd_path: str, visible=True):
    stage = omni.usd.get_context().get_stage()
    visual_path = f"{sensor_path}/visual"
    UsdGeom.Xform.Define(stage, Sdf.Path(visual_path))
    mesh_prim = stage.DefinePrim(Sdf.Path(f"{visual_path}/mesh"), "Xform")
    mesh_prim.GetReferences().AddReference(usd_path)
    
    UsdPhysics.CollisionAPI.Apply(mesh_prim).CreateCollisionEnabledAttr(False)
    
    # rt_vis_api = UsdGeom.Tokens
    mesh_prim.CreateAttribute("visibility:raytracing:camera", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:transmission", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:shadow", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:diffuse", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:glossy", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:specular", Sdf.ValueTypeNames.Token).Set("invisible")
    mesh_prim.CreateAttribute("visibility:raytracing:scatter", Sdf.ValueTypeNames.Token).Set("invisible")


    if visible:
        path = Sdf.Path(sensor_path)
        prim = stage.GetPrimAtPath(path)
        prim.GetAttribute("visibility").Set("inherited")

def add_camera(num_envs):
    cameras_list = []

    for i in range(num_envs):
        cameraCfg = CameraCfg(
            prim_path=f"/World/envs/env_{i}/Robot/base/front_cam",
            update_period=0.05,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=16.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.32487, -0.00095, 0.05362),
                rot=(0.5, -0.5, 0.5, -0.5),
                convention="ros",
            ),
        )

        cameras_list.append(Camera(cameraCfg))
    
    return cameras_list

def add_realsense():
    nucleus_server = nucleus_utils.get_assets_root_path()
    
    d455_usd_path = nucleus_server + "/Isaac/Sensors/Intel/RealSense/rsd455.usd"
    
    d455:Usd.Prim = add_reference_to_stage(usd_path=d455_usd_path, prim_path="/World/envs/env_0/Robot/base/d455")

    stage = omni.usd.get_context().get_stage()

    d455 = stage.GetPrimAtPath("/World/envs/env_0/Robot/base/d455")

    d455_xform = UsdGeom.XformCommonAPI(d455)
    d455_xform.SetTranslate((0.35, 0, 0.075))
    d455_xform.SetScale((0.8, 0.7, 0.85))
    d455_xform.SetRotate((0.0, -10.0, 0.0))

def pub_robo_data_ros2(num_envs, base_node, env, annotator_lst):
    rclpy.spin_once(base_node, timeout_sec=0.0)

    for i in range(num_envs):
        # publish ros2 info
        base_node.publish_joints(
            env.unwrapped.scene["robot"].data.joint_names,
            env.unwrapped.scene["robot"].data.joint_pos[i],
            i,
        )
        base_node.publish_odom(
            env.unwrapped.scene["robot"].data.root_state_w[i, :3],
            env.unwrapped.scene["robot"].data.root_state_w[i, 3:7],
            i,
        )
        base_node.publish_imu(
            env.unwrapped.scene["robot"].data.root_state_w[i, 3:7],
            env.unwrapped.scene["robot"].data.root_lin_vel_b[i, :],
            env.unwrapped.scene["robot"].data.root_ang_vel_b[i, :],
            i,
        )

        base_node.publish_robot_state(
            [
                env.unwrapped.scene["contact_forces"].data.net_forces_w[i][4][2],
                env.unwrapped.scene["contact_forces"].data.net_forces_w[i][8][2],
                env.unwrapped.scene["contact_forces"].data.net_forces_w[i][14][2],
                env.unwrapped.scene["contact_forces"].data.net_forces_w[i][18][2],
            ], i,
        )

        if annotator_lst!= None:
            try:
                for lidar_id in range(2): 
                    index_annotator = (i * 2) + lidar_id
                    base_node.publish_lidar(annotator_lst[index_annotator], i)

            except Exception as e:
                print(f"Erro ao publicar LiDAR para o ambiente {i}: {e}")

class RobotBaseNode(Node):
    def __init__(self, num_envs, use_sim_time):
        super().__init__("go2_driver_node")

        self.set_parameters([rclpy.parameter.Parameter("use_sim_time", rclpy.Parameter.Type.BOOL, use_sim_time)])

        qos_profile = QoSProfile(depth=10)

        self.joint_pub = []
        self.go2_state_pub = []
        self.go2_lidar_L1_pub = []
        self.go2_lidar_robosense_pub = []
        self.odom_pub = []
        self.imu_pub = []

        for i in range(num_envs):
            self.joint_pub.append(self.create_publisher(JointState, f"robot{i}/joint_states", qos_profile))
            self.go2_state_pub.append(self.create_publisher(Go2State, f"robot{i}/go2_states", qos_profile))
            self.odom_pub.append(self.create_publisher(Odometry, f"robot{i}/odom", qos_profile))
            self.imu_pub.append(self.create_publisher(Imu, f"robot{i}/imu", qos_profile))
            self.go2_lidar_L1_pub.append(self.create_publisher(PointCloud2, f"robot{i}/point_cloud2_l1", qos_profile))
            self.go2_lidar_robosense_pub.append(self.create_publisher(PointCloud2, f"robot{i}/point_cloud2_robosense", qos_profile))
            
        self.broadcaster = TransformBroadcaster(self, qos=qos_profile)

    def clock_callback(self, msg):
        self.sim_time = msg.clock
        print("clock_callback")

    def get_sim_time(self):
        return self.get_clock().now().to_msg()

    def publish_joints(self, joint_names_lst, joint_state_lst, robot_num):
        # Create message
        joint_state = JointState()
        joint_state.header.stamp = self.get_sim_time()

        joint_state_names_formated = []
        for joint_name in joint_names_lst:
            joint_state_names_formated.append(f"robot{robot_num}/" + joint_name)

        joint_state_formated = []
        for joint_state_val in joint_state_lst:
            joint_state_formated.append(joint_state_val.item())

        joint_state.name = joint_state_names_formated
        joint_state.position = joint_state_formated
        self.joint_pub[robot_num].publish(joint_state)

    def publish_odom(self, base_pos, base_rot, robot_num):
        now = self.get_sim_time()

        odom_trans = TransformStamped()
        odom_trans.header.stamp = now
        odom_trans.header.frame_id = "odom"
        odom_trans.child_frame_id = f"robot{robot_num}/base_link"
        odom_trans.transform.translation.x = base_pos[0].item()
        odom_trans.transform.translation.y = base_pos[1].item()
        odom_trans.transform.translation.z = base_pos[2].item()
        odom_trans.transform.rotation.x = base_rot[1].item()
        odom_trans.transform.rotation.y = base_rot[2].item()
        odom_trans.transform.rotation.z = base_rot[3].item()
        odom_trans.transform.rotation.w = base_rot[0].item()
        self.broadcaster.sendTransform(odom_trans)

        UnitreeL1_trans = TransformStamped()
        UnitreeL1_trans.header.stamp = now
        UnitreeL1_trans.header.frame_id = f"robot{robot_num}/base_link"
        UnitreeL1_trans.child_frame_id = f"robot{robot_num}/UnitreeL1_link"
        UnitreeL1_trans.transform.translation = Vector3(x=UnitreeL1_translation[0], y=UnitreeL1_translation[1], z=UnitreeL1_translation[2])
        UnitreeL1_trans.transform.rotation = Quaternion(x=UnitreeL1_quat[0], y=UnitreeL1_quat[1], z=UnitreeL1_quat[2], w=UnitreeL1_quat[3])
        self.broadcaster.sendTransform(UnitreeL1_trans)

        Robosense_trans = TransformStamped()
        Robosense_trans.header.stamp = now
        Robosense_trans.header.frame_id = f"robot{robot_num}/base_link"
        Robosense_trans.child_frame_id = f"robot{robot_num}/Robosense_link"
        Robosense_trans.transform.translation = Vector3(x=Robosense_translation[0], y=Robosense_translation[1], z=Robosense_translation[2])
        Robosense_trans.transform.rotation = Quaternion(x=Robosense_quat[0], y=Robosense_quat[1], z=Robosense_quat[2], w=Robosense_quat[3])
        self.broadcaster.sendTransform(Robosense_trans)

        Realsense_trans = TransformStamped()
        Realsense_trans.header.stamp = now
        Realsense_trans.header.frame_id = f"robot{robot_num}/base_link"
        Realsense_trans.child_frame_id = f"robot{robot_num}/Realsense_link"
        Realsense_trans.transform.translation = Vector3(x=Realsense_translation[0], y=Realsense_translation[1], z=Realsense_translation[2])
        Realsense_trans.transform.rotation = Quaternion(x=Realsense_quat[0], y=Realsense_quat[1], z=Realsense_quat[2], w=Realsense_quat[3])
        self.broadcaster.sendTransform(Realsense_trans)

        odom_topic = Odometry()
        odom_topic.header.stamp = now
        odom_topic.header.frame_id = "odom"
        odom_topic.child_frame_id = f"robot{robot_num}/base_link"
        odom_topic.pose.pose.position.x = base_pos[0].item()
        odom_topic.pose.pose.position.y = base_pos[1].item()
        odom_topic.pose.pose.position.z = base_pos[2].item()
        odom_topic.pose.pose.orientation.x = base_rot[1].item()
        odom_topic.pose.pose.orientation.y = base_rot[2].item()
        odom_topic.pose.pose.orientation.z = base_rot[3].item()
        odom_topic.pose.pose.orientation.w = base_rot[0].item()
        self.odom_pub[robot_num].publish(odom_topic)
        
    def publish_imu(self, base_rot, base_lin_vel, base_ang_vel, robot_num):
        imu_trans = Imu()
        imu_trans.header.stamp = self.get_sim_time()
        imu_trans.header.frame_id = f"robot{robot_num}/base_link"

        imu_trans.linear_acceleration.x = base_lin_vel[0].item()
        imu_trans.linear_acceleration.y = base_lin_vel[1].item()
        imu_trans.linear_acceleration.z = base_lin_vel[2].item()

        imu_trans.angular_velocity.x = base_ang_vel[0].item()
        imu_trans.angular_velocity.y = base_ang_vel[1].item()
        imu_trans.angular_velocity.z = base_ang_vel[2].item()

        imu_trans.orientation.x = base_rot[1].item()
        imu_trans.orientation.y = base_rot[2].item()
        imu_trans.orientation.z = base_rot[3].item()
        imu_trans.orientation.w = base_rot[0].item()

        self.imu_pub[robot_num].publish(imu_trans)

    def publish_robot_state(self, foot_force_lst, robot_num):

        go2_state = Go2State()
        # go2_state.header.stamp = self.get_sim_time()
        go2_state.foot_force = [
            int(foot_force_lst[0].item()),
            int(foot_force_lst[1].item()),
            int(foot_force_lst[2].item()),
            int(foot_force_lst[3].item()),
        ]
        self.go2_state_pub[robot_num].publish(go2_state)

    def publish_lidar(self, data, robot_num):
        """Publica o PointCloud2 do LiDAR no tópico correspondente."""
        annotator = data.get("annotator_object")
        if annotator is None:
            return

        cloud_data = annotator.get_data().get("data")
        if cloud_data is None or not isinstance(cloud_data, np.ndarray) or cloud_data.size == 0:
            return

        # Determina o tipo e publisher
        lidar_type = data.get("type")
        if lidar_type == "UnitreeL1":
            frame_id = f"robot{robot_num}/UnitreeL1_link"
            publisher = self.go2_lidar_L1_pub[robot_num]
        elif lidar_type == "Robosense":
            frame_id = f"robot{robot_num}/Robosense_link"
            publisher = self.go2_lidar_robosense_pub[robot_num]
        else:
            print(f"[WARN] Tipo de LiDAR desconhecido: {lidar_type}")
            return

        # Cria o header com timestamp simulado
        header = Header()
        header.stamp = self.get_sim_time()
        header.frame_id = frame_id

        # Campos XYZ — se quiser incluir intensidade, basta adicionar outro campo
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]

        # Cria a mensagem PointCloud2
        try:
            cloud_msg = point_cloud2.create_cloud(header, fields, cloud_data)
            publisher.publish(cloud_msg)
        except Exception as e:
            print(f"[ERROR] Falha ao publicar PointCloud2 para {frame_id}: {e}")

    # async def run(self):
    #     while True:
    #         self.publish_lidar()
    #         await asyncio.sleep(0.1)
