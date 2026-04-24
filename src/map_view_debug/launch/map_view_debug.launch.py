"""Launch Gazebo + map_server + RViz aligned via the world_to_map sidecar,
plus a ``map_view_debug_node`` that renders the LLM map view from each
"2D Pose Estimate" click in RViz.

This is intentionally a stripped-down sibling of
``world_to_map.launch.py``: no robot is spawned, no ``robot_state_publisher``
is started, and no ``TURTLEBOT3_MODEL`` is honoured. The only things on the
TF tree are the static ``map -> odom`` transform from the sidecar offset and
whatever Gazebo emits internally (Gazebo state is irrelevant for the debug
renderer; we keep gzserver/gzclient up so the user can visually compare the
two worlds while clicking).
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _patch_rviz_focus(template_path: str, focus_x: float, focus_y: float,
                      distance: float = 12.0) -> str:
    """Copy the RViz config to a temp file with Focal Point / Distance set."""
    with open(template_path, 'r') as fh:
        body = fh.read()

    body = re.sub(
        r'(Focal Point:\s*\n\s*X:\s*)[-\d.eE+]+',
        rf'\g<1>{focus_x}', body, count=1,
    )
    body = re.sub(
        r'(Focal Point:\s*\n\s*X:\s*[-\d.eE+]+\s*\n\s*Y:\s*)[-\d.eE+]+',
        rf'\g<1>{focus_y}', body, count=1,
    )
    body = re.sub(r'(Distance:\s*)\d+(?:\.\d+)?', rf'\g<1>{distance}',
                  body, count=1)

    out = tempfile.NamedTemporaryFile(
        mode='w', delete=False, prefix='map_view_debug_', suffix='.rviz')
    out.write(body)
    out.close()
    return out.name


def _launch_setup(context, *_args, **_kwargs) -> List:
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    world = LaunchConfiguration('world').perform(context).strip()
    map_yaml = LaunchConfiguration('map_yaml').perform(context).strip()
    launch_rviz = LaunchConfiguration('launch_rviz').perform(context).lower() == 'true'
    launch_gazebo = LaunchConfiguration('launch_gazebo').perform(context).lower() == 'true'
    map_offset_x = LaunchConfiguration('map_offset_x').perform(context)
    map_offset_y = LaunchConfiguration('map_offset_y').perform(context)
    gazebo_model_path = LaunchConfiguration('gazebo_model_path').perform(context).strip()

    source_x = LaunchConfiguration('source_x').perform(context)
    source_y = LaunchConfiguration('source_y').perform(context)
    dest_x = LaunchConfiguration('dest_x').perform(context)
    dest_y = LaunchConfiguration('dest_y').perform(context)
    crop_radius_m = LaunchConfiguration('crop_radius_m').perform(context)
    output_size = LaunchConfiguration('output_size').perform(context)
    output_dir = LaunchConfiguration('output_dir').perform(context)

    if not os.path.isfile(map_yaml):
        raise FileNotFoundError(f'map yaml not found: {map_yaml}')

    actions: List = []

    if launch_gazebo:
        if not world:
            raise RuntimeError(
                'launch_gazebo:=true but no world: argument provided'
            )
        if not os.path.isfile(world):
            raise FileNotFoundError(f'world file not found: {world}')

        if gazebo_model_path:
            existing = os.environ.get('GAZEBO_MODEL_PATH', '')
            merged = gazebo_model_path
            if existing:
                merged = gazebo_model_path + os.pathsep + existing
            actions.append(SetEnvironmentVariable('GAZEBO_MODEL_PATH', merged))

        gazebo_share = get_package_share_directory('gazebo_ros')
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_share, 'launch', 'gzserver.launch.py')
            ),
            launch_arguments={'world': world}.items(),
        ))
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(gazebo_share, 'launch', 'gzclient.launch.py')
            )
        ))

    actions.append(Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'yaml_filename': map_yaml},
        ],
    ))

    actions.append(Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_map_view_debug',
        output='screen',
        parameters=[
            {'use_sim_time': use_sim_time},
            {'autostart': True},
            {'node_names': ['map_server']},
            {'bond_timeout': 20.0},
        ],
    ))

    actions.append(Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_map_to_odom',
        arguments=[
            '--x', str(map_offset_x),
            '--y', str(map_offset_y),
            '--z', '0',
            '--qx', '0', '--qy', '0', '--qz', '0', '--qw', '1',
            '--frame-id', 'map',
            '--child-frame-id', 'odom',
        ],
        parameters=[{'use_sim_time': use_sim_time}],
    ))

    if launch_rviz:
        try:
            offx = float(map_offset_x)
            offy = float(map_offset_y)
            sx = float(source_x)
            sy = float(source_y)
            focus_x = sx if sx else offx
            focus_y = sy if sy else offy
        except ValueError:
            focus_x = 0.0
            focus_y = 0.0
        rviz_share = get_package_share_directory('world_to_map')
        rviz_cfg = os.path.join(rviz_share, 'rviz', 'world_to_map.rviz')
        rviz_runtime = _patch_rviz_focus(rviz_cfg, focus_x, focus_y,
                                         distance=18.0)
        actions.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_runtime],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ))

    debug_params = {
        'map_yaml_path': map_yaml,
        'source_x': float(source_x) if source_x else 0.0,
        'source_y': float(source_y) if source_y else 0.0,
        'dest_x': float(dest_x) if dest_x else 0.0,
        'dest_y': float(dest_y) if dest_y else 0.0,
        'crop_radius_m': float(crop_radius_m) if crop_radius_m else 18.0,
        'output_size': int(output_size) if output_size else 512,
        'output_dir': output_dir,
        'use_sim_time': use_sim_time,
    }
    actions.append(Node(
        package='map_view_debug',
        executable='map_view_debug_node',
        name='map_view_debug_node',
        output='screen',
        emulate_tty=True,
        parameters=[debug_params],
        arguments=['--ros-args', '--log-level', 'info'],
    ))

    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            'world', default_value='',
            description='Absolute path to the .world file (required when '
                        'launch_gazebo:=true).'),
        DeclareLaunchArgument(
            'map_yaml',
            description='Absolute path to the Nav2 map YAML.'),
        DeclareLaunchArgument(
            'use_sim_time', default_value='true'),
        DeclareLaunchArgument(
            'launch_rviz', default_value='true'),
        DeclareLaunchArgument(
            'launch_gazebo', default_value='true',
            description='Set to false to skip Gazebo entirely (RViz + '
                        'debug node only).'),
        DeclareLaunchArgument(
            'map_offset_x', default_value='0.0',
            description='Static TF map->odom x translation '
                        '(from sidecar world_to_map_offset).'),
        DeclareLaunchArgument(
            'map_offset_y', default_value='0.0',
            description='Static TF map->odom y translation '
                        '(from sidecar world_to_map_offset).'),
        DeclareLaunchArgument(
            'gazebo_model_path', default_value='',
            description='Colon-separated extra dirs to prepend to '
                        'GAZEBO_MODEL_PATH.'),
        DeclareLaunchArgument(
            'source_x', default_value='0.0',
            description='Source X (map frame) - drawn as the blue marker '
                        'on the rendered debug image.'),
        DeclareLaunchArgument(
            'source_y', default_value='0.0'),
        DeclareLaunchArgument(
            'dest_x', default_value='0.0',
            description='Destination X (map frame) - drawn as the green '
                        'marker on the rendered debug image.'),
        DeclareLaunchArgument(
            'dest_y', default_value='0.0'),
        DeclareLaunchArgument(
            'crop_radius_m', default_value='18.0',
            description='Crop radius (meters) passed to '
                        'render_annotated_map; mirrors get_map_b64.'),
        DeclareLaunchArgument(
            'output_size', default_value='512',
            description='Output PNG side length in pixels; mirrors '
                        'get_map_b64.'),
        DeclareLaunchArgument(
            'output_dir', default_value='',
            description='Directory to write pose_NNN.png/.json into. '
                        'Default: ~/map_view_debug_runs/<timestamp>/'),
        OpaqueFunction(function=_launch_setup),
    ])
