"""Gazebo + TurtleBot3 Burger, map_server, RViz, and click coordinate echo."""

from __future__ import annotations

import os
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _launch_setup(context, *_args, **_kwargs) -> List:
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    map_yaml_arg = LaunchConfiguration('map_yaml').perform(context).strip()
    launch_gazebo = LaunchConfiguration('launch_gazebo').perform(context).lower() == 'true'
    launch_rviz = LaunchConfiguration('launch_rviz').perform(context).lower() == 'true'

    pkg_share = get_package_share_directory('custom_map_builder')
    if not map_yaml_arg:
        map_yaml = os.path.join(pkg_share, 'maps', 'default.yaml')
    elif os.path.isabs(map_yaml_arg):
        map_yaml = map_yaml_arg
    else:
        map_yaml = os.path.join(pkg_share, 'maps', map_yaml_arg)
    map_yaml = os.path.abspath(map_yaml)
    if not os.path.isfile(map_yaml):
        raise FileNotFoundError(f'map yaml not found: {map_yaml}')

    rviz_cfg = os.path.join(pkg_share, 'rviz', 'map_builder.rviz')

    actions = []

    if launch_gazebo:
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [
                                FindPackageShare('turtlebot3_gazebo'),
                                'launch',
                                'turtlebot3_world.launch.py',
                            ]
                        )
                    ]
                ),
                launch_arguments={
                    'use_sim_time': LaunchConfiguration('use_sim_time'),
                    # Avoid a second RViz if this launch file supports the arg (Humble TurtleBot3 does).
                    'use_rviz': 'false',
                }.items(),
            )
        )

    actions.extend(
        [
            Node(
                package='nav2_map_server',
                executable='map_server',
                name='map_server',
                output='screen',
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'yaml_filename': map_yaml},
                ],
            ),
            Node(
                package='nav2_lifecycle_manager',
                executable='lifecycle_manager',
                name='lifecycle_manager',
                output='screen',
                parameters=[
                    {'use_sim_time': use_sim_time},
                    {'autostart': True},
                    {'node_names': ['map_server']},
                    {'bond_timeout': 20.0},
                ],
            ),
            Node(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='static_map_to_odom',
                arguments=[
                    '--x',
                    '0',
                    '--y',
                    '0',
                    '--z',
                    '0',
                    '--qx',
                    '0',
                    '--qy',
                    '0',
                    '--qz',
                    '0',
                    '--qw',
                    '1',
                    '--frame-id',
                    'map',
                    '--child-frame-id',
                    'odom',
                ],
                parameters=[{'use_sim_time': use_sim_time}],
            ),
            Node(
                package='custom_map_builder',
                executable='click_echo',
                name='map_click_echo',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
            ),
        ]
    )

    if launch_rviz:
        actions.append(
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                arguments=['-d', rviz_cfg],
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen',
            )
        )

    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'use_sim_time',
                default_value='true',
                description='Must be true when Gazebo is the clock source.',
            ),
            DeclareLaunchArgument(
                'map_yaml',
                default_value='',
                description=(
                    'Empty = share/custom_map_builder/maps/default.yaml. '
                    'Otherwise absolute path to map YAML, or a filename under maps/.'
                ),
            ),
            DeclareLaunchArgument(
                'launch_gazebo',
                default_value='true',
                description='Set false to only run map_server + RViz + click echo.',
            ),
            DeclareLaunchArgument(
                'launch_rviz',
                default_value='true',
                description='Set false for headless (coordinates only via echo if another RViz publishes).',
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
