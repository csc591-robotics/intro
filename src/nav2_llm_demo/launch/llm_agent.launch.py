"""Launch file for the vision-based LLM agent navigation.

Starts:
  - map_server (loads the custom PGM/YAML map)
  - lifecycle_manager (activates map_server)
  - static TF: map -> odom (identity)
  - llm_agent_node (the LangGraph agent)
  - optionally RViz2 with a pre-configured display layout

Does NOT start Nav2 bringup -- the agent controls the robot directly via
cmd_vel.  Gazebo is started separately by run_llm_nav.sh.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg_share = get_package_share_directory("nav2_llm_demo")
    default_rviz_config = os.path.join(pkg_share, "rviz", "llm_agent.rviz")

    map_yaml = LaunchConfiguration("map_yaml")
    source_x = LaunchConfiguration("source_x")
    source_y = LaunchConfiguration("source_y")
    source_yaw = LaunchConfiguration("source_yaw")
    dest_x = LaunchConfiguration("dest_x")
    dest_y = LaunchConfiguration("dest_y")
    dest_yaw = LaunchConfiguration("dest_yaw")
    use_sim_time = LaunchConfiguration("use_sim_time")
    launch_rviz = LaunchConfiguration("launch_rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    map_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="map_server",
        output="screen",
        parameters=[
            {"yaml_filename": map_yaml},
            {"use_sim_time": use_sim_time},
        ],
    )

    lifecycle_mgr = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_map",
        output="screen",
        parameters=[
            {"node_names": ["map_server"]},
            {"autostart": True},
            {"use_sim_time": use_sim_time},
        ],
    )

    static_tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="map_to_odom_static",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"],
    )

    agent_node = Node(
        package="nav2_llm_demo",
        executable="llm_agent_node",
        name="llm_agent_node",
        output="screen",
        parameters=[
            {"source_x": source_x},
            {"source_y": source_y},
            {"source_yaw": source_yaw},
            {"dest_x": dest_x},
            {"dest_y": dest_y},
            {"dest_yaw": dest_yaw},
            {"map_yaml_path": map_yaml},
            {"use_sim_time": use_sim_time},
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config],
        condition=IfCondition(launch_rviz),
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription([
        DeclareLaunchArgument("map_yaml", description="Absolute path to Nav2 map YAML"),
        DeclareLaunchArgument("source_x", default_value="0.0"),
        DeclareLaunchArgument("source_y", default_value="0.0"),
        DeclareLaunchArgument("source_yaw", default_value="0.0"),
        DeclareLaunchArgument("dest_x", default_value="0.0"),
        DeclareLaunchArgument("dest_y", default_value="0.0"),
        DeclareLaunchArgument("dest_yaw", default_value="0.0"),
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("launch_rviz", default_value="false"),
        DeclareLaunchArgument("rviz_config", default_value=default_rviz_config),
        map_server,
        lifecycle_mgr,
        static_tf,
        agent_node,
        rviz_node,
    ])
