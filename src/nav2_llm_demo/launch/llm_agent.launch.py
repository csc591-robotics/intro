"""Launch file for the vision-based LLM agent navigation.

Two flows are supported:

* **world_to_map (sidecar) flow** — when ``world`` is non-empty, this launch
  file delegates Gazebo + robot spawn + map_server + ``map -> odom`` static TF
  + RViz to ``world_to_map.launch.py``.  The robot is spawned at the source
  pose declared in ``custom_map_builder/maps/map_poses.yaml`` (Gazebo world
  frame), and the static TF translation is set to ``spawn_xy + sidecar_offset``
  so that ``map -> base_link`` matches the desired map-frame coordinates at
  startup.  Only ``llm_agent_node`` is added on top.

* **Hand-crafted (legacy) flow** — when ``world`` is empty, this launch file
  starts ``map_server``, an identity ``map -> odom`` static TF, optionally
  RViz, and ``llm_agent_node``.  Gazebo (typically an empty world) must be
  started separately or via ``run_llm_nav.sh``.

The ``llm_agent_node`` itself is unchanged: it consumes ``source_x/y/yaw``
and ``dest_x/y/yaw`` in the **map frame** (because the agent compares pose
against ``map -> base_link`` TF lookups).  The runner script is responsible
for converting Gazebo-world-frame coordinates from ``map_poses.yaml`` into
map-frame coordinates by adding the sidecar offset.
"""

from __future__ import annotations

import os
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *_args, **_kwargs) -> List:
    pkg_share = get_package_share_directory("nav2_llm_demo")
    default_rviz_config = os.path.join(pkg_share, "rviz", "llm_agent.rviz")

    map_yaml = LaunchConfiguration("map_yaml").perform(context).strip()
    world = LaunchConfiguration("world").perform(context).strip()
    use_sim_time = LaunchConfiguration("use_sim_time").perform(context)
    launch_rviz = LaunchConfiguration("launch_rviz").perform(context)
    rviz_config = LaunchConfiguration("rviz_config").perform(context).strip()
    flow = LaunchConfiguration("flow").perform(context).strip() or "1"
    if flow not in {"1", "2", "3", "4", "5"}:
        raise RuntimeError(
            f"flow must be 1, 2, 3, 4, or 5 (got {flow!r})"
        )
    if not rviz_config:
        rviz_config = default_rviz_config

    source_x = LaunchConfiguration("source_x").perform(context)
    source_y = LaunchConfiguration("source_y").perform(context)
    source_yaw = LaunchConfiguration("source_yaw").perform(context)
    dest_x = LaunchConfiguration("dest_x").perform(context)
    dest_y = LaunchConfiguration("dest_y").perform(context)
    dest_yaw = LaunchConfiguration("dest_yaw").perform(context)

    spawn_x = LaunchConfiguration("spawn_x").perform(context)
    spawn_y = LaunchConfiguration("spawn_y").perform(context)
    spawn_z = LaunchConfiguration("spawn_z").perform(context)
    spawn_yaw = LaunchConfiguration("spawn_yaw").perform(context)
    static_tf_x = LaunchConfiguration("static_tf_x").perform(context)
    static_tf_y = LaunchConfiguration("static_tf_y").perform(context)
    gazebo_model_path = LaunchConfiguration("gazebo_model_path").perform(
        context).strip()

    use_sim_time_bool_str = use_sim_time
    launch_rviz_bool = launch_rviz.lower() == "true"

    if not map_yaml:
        raise RuntimeError("map_yaml argument is required")

    actions: List = []

    actions.append(SetEnvironmentVariable("LLM_FLOW", flow))

    agent_node = Node(
        package="nav2_llm_demo",
        executable="llm_agent_node",
        name="llm_agent_node",
        output="screen",
        emulate_tty=True,
        arguments=["--ros-args", "--log-level", "info"],
        parameters=[
            {"source_x": float(source_x)},
            {"source_y": float(source_y)},
            {"source_yaw": float(source_yaw)},
            {"dest_x": float(dest_x)},
            {"dest_y": float(dest_y)},
            {"dest_yaw": float(dest_yaw)},
            {"map_yaml_path": map_yaml},
            {"use_sim_time": use_sim_time_bool_str.lower() == "true"},
        ],
    )

    if world:
        # ── sidecar flow: delegate Gazebo + spawn + map_server + TF + RViz ──
        if not os.path.isfile(world):
            raise FileNotFoundError(f"world file not found: {world}")

        world_to_map_share = get_package_share_directory("world_to_map")
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    world_to_map_share, "launch", "world_to_map.launch.py"
                )
            ),
            launch_arguments={
                "world": world,
                "map_yaml": map_yaml,
                "x_pose": spawn_x,
                "y_pose": spawn_y,
                "z_pose": spawn_z,
                "yaw": spawn_yaw,
                "map_offset_x": static_tf_x,
                "map_offset_y": static_tf_y,
                "launch_rviz": "true" if launch_rviz_bool else "false",
                "use_sim_time": use_sim_time_bool_str,
                "gazebo_model_path": gazebo_model_path,
            }.items(),
        ))
        actions.append(agent_node)
        return actions

    # ── legacy hand-crafted-map flow ────────────────────────────────────────
    if gazebo_model_path:
        existing = os.environ.get("GAZEBO_MODEL_PATH", "")
        merged = gazebo_model_path
        if existing:
            merged = gazebo_model_path + os.pathsep + existing
        actions.append(SetEnvironmentVariable("GAZEBO_MODEL_PATH", merged))

    actions.append(Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {"yaml_filename": map_yaml},
            {"use_sim_time": use_sim_time_bool_str.lower() == "true"},
        ],
    ))

    actions.append(Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_map",
        output="screen",
        parameters=[
            {"node_names": ["map_server"]},
            {"autostart": True},
            {"use_sim_time": use_sim_time_bool_str.lower() == "true"},
        ],
    ))

    actions.append(Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_odom_static",
        arguments=[
            "--x", static_tf_x,
            "--y", static_tf_y,
            "--z", "0",
            "--qx", "0", "--qy", "0", "--qz", "0", "--qw", "1",
            "--frame-id", "map",
            "--child-frame-id", "odom",
        ],
    ))

    actions.append(agent_node)

    if launch_rviz_bool:
        actions.append(Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
            parameters=[{"use_sim_time": use_sim_time_bool_str.lower() == "true"}],
        ))

    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("map_yaml",
                              description="Absolute path to Nav2 map YAML"),
        DeclareLaunchArgument("world", default_value="",
                              description="Absolute path to a Gazebo .world "
                                          "file. Empty = legacy flow."),
        DeclareLaunchArgument("source_x", default_value="0.0",
                              description="Source x in MAP frame "
                                          "(= world_x + sidecar.offset_x)"),
        DeclareLaunchArgument("source_y", default_value="0.0",
                              description="Source y in MAP frame"),
        DeclareLaunchArgument("source_yaw", default_value="0.0"),
        DeclareLaunchArgument("dest_x", default_value="0.0",
                              description="Destination x in MAP frame"),
        DeclareLaunchArgument("dest_y", default_value="0.0",
                              description="Destination y in MAP frame"),
        DeclareLaunchArgument("dest_yaw", default_value="0.0"),
        DeclareLaunchArgument("spawn_x", default_value="0.0",
                              description="Robot spawn x in Gazebo world "
                                          "frame (passed to spawn_entity.py)."),
        DeclareLaunchArgument("spawn_y", default_value="0.0",
                              description="Robot spawn y in Gazebo world "
                                          "frame."),
        DeclareLaunchArgument("spawn_z", default_value="0.05"),
        DeclareLaunchArgument("spawn_yaw", default_value="0.0"),
        DeclareLaunchArgument("static_tf_x", default_value="0.0",
                              description="map -> odom static translation x. "
                                          "For sidecar maps this is "
                                          "spawn_x + sidecar.offset_x."),
        DeclareLaunchArgument("static_tf_y", default_value="0.0",
                              description="map -> odom static translation y."),
        DeclareLaunchArgument("gazebo_model_path", default_value="",
                              description="Extra dirs prepended to "
                                          "GAZEBO_MODEL_PATH for the "
                                          "delegated Gazebo launch."),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("launch_rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=""),
        DeclareLaunchArgument("flow", default_value="1",
                              description="LLM agent flow: 1 = custom loop "
                                          "(default), 2 = LangGraph "
                                          "create_react_agent, 3 = same "
                                          "as 2 + LiDAR summary tool, "
                                          "4 = fixed gather/decide/execute/"
                                          "check graph (no ReAct freedom), "
                                          "5 = A* path planner + LLM "
                                          "follower (most reliable)."),
        OpaqueFunction(function=_launch_setup),
    ])
