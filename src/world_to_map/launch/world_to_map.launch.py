"""Launch Gazebo (custom world) + map_server + RViz + a TurtleBot3 spawn.

This launch file also writes a per-run RViz config to a temp directory whose
camera is focused on the robot's spawn point in the map frame, so the robot
is visible in the default view (otherwise the rasterized map sits at, e.g.,
(4, 3) in bottom-left mode while RViz starts looking at (0, 0)).

The static ``map -> odom`` transform applies the offset reported by the
rasterizer's sidecar so that:

- With ``origin-mode = world``: offset is (0, 0) and ``map`` frame == Gazebo
  ``world`` frame.
- With ``origin-mode = bottom-left``: offset is ``(-pmin_x, -pmin_y)`` so the
  bottom-left of the rasterized map sits at ``map (0, 0)`` while the robot's
  pose still moves 1:1 with Gazebo (just shifted into the positive quadrant).

``GAZEBO_MODEL_PATH`` is propagated as a launch argument so the runner can
point Gazebo at the asset collection without polluting the user's shell env.

Robot spawning notes:

- Gazebo gets the **SDF** from ``turtlebot3_gazebo/models/turtlebot3_<model>/
  model.sdf``. This file embeds the diff_drive_controller, IMU, joint state
  and (for waffle) camera/lidar plugins, so the robot actually drives when
  ``/cmd_vel`` is published and publishes ``odom -> base_footprint``.
- ``robot_state_publisher`` is fed the **URDF** from ``turtlebot3_description/
  urdf/turtlebot3_<model>.urdf`` so RViz's ``RobotModel`` display has a
  description to render. Joint TFs come from the ``joint_states`` topic the
  Gazebo SDF plugin already publishes.

If we instead spawned the description URDF directly into Gazebo, no Gazebo
plugins would load: the robot would appear (sometimes) but never move, and
RViz would have no ``odom -> base_footprint`` to anchor it.
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


def _model_name() -> str:
    return os.environ.get('TURTLEBOT3_MODEL', 'burger')


def _read_burger_urdf() -> str:
    description_share = get_package_share_directory('turtlebot3_description')
    urdf_path = os.path.join(
        description_share, 'urdf', f'turtlebot3_{_model_name()}.urdf'
    )
    with open(urdf_path, 'r') as fh:
        return fh.read()


def _gazebo_model_sdf_path() -> str:
    """Return the path to the gazebo-plugin-bundled SDF for the active model."""
    gazebo_share = get_package_share_directory('turtlebot3_gazebo')
    return os.path.join(
        gazebo_share, 'models', f'turtlebot3_{_model_name()}', 'model.sdf'
    )


def _patch_rviz_focus(template_path: str, focus_x: float, focus_y: float,
                      distance: float = 12.0) -> str:
    """Copy the RViz config to a temp file with Focal Point / Distance set.

    Default RViz config focuses at (0, 0). With ``bottom-left`` origin mode the
    rasterized map lives in the positive quadrant, so the robot is several
    metres away from (0, 0) and the user sees an empty grid. Patching the
    focal point puts the robot dead center on launch.
    """
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
        mode='w', delete=False, prefix='world_to_map_', suffix='.rviz')
    out.write(body)
    out.close()
    return out.name


def _launch_setup(context, *_args, **_kwargs) -> List:
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    world = LaunchConfiguration('world').perform(context).strip()
    map_yaml = LaunchConfiguration('map_yaml').perform(context).strip()
    x_pose = LaunchConfiguration('x_pose').perform(context)
    y_pose = LaunchConfiguration('y_pose').perform(context)
    z_pose = LaunchConfiguration('z_pose').perform(context)
    yaw = LaunchConfiguration('yaw').perform(context)
    launch_rviz = LaunchConfiguration('launch_rviz').perform(context).lower() == 'true'
    map_offset_x = LaunchConfiguration('map_offset_x').perform(context)
    map_offset_y = LaunchConfiguration('map_offset_y').perform(context)
    gazebo_model_path = LaunchConfiguration('gazebo_model_path').perform(context).strip()

    pkg_share = get_package_share_directory('world_to_map')
    rviz_cfg = os.path.join(pkg_share, 'rviz', 'world_to_map.rviz')

    if not os.path.isfile(world):
        raise FileNotFoundError(f'world file not found: {world}')
    if not os.path.isfile(map_yaml):
        raise FileNotFoundError(f'map yaml not found: {map_yaml}')

    actions: List = []

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

    robot_description = _read_burger_urdf()
    actions.append(Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
            'robot_description': robot_description,
        }],
    ))

    sdf_path = _gazebo_model_sdf_path()
    if not os.path.isfile(sdf_path):
        raise FileNotFoundError(
            'turtlebot3_gazebo SDF not found at: ' + sdf_path +
            '. Install ros-humble-turtlebot3-gazebo or set TURTLEBOT3_MODEL.'
        )

    actions.append(Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_turtlebot3',
        output='screen',
        arguments=[
            '-entity', 'turtlebot3_' + _model_name(),
            '-file', sdf_path,
            '-x', str(x_pose),
            '-y', str(y_pose),
            '-z', str(z_pose),
            '-Y', str(yaw),
        ],
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
        name='lifecycle_manager_world_to_map',
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
            spx = float(x_pose)
            spy = float(y_pose)
            focus_x = offx + spx
            focus_y = offy + spy
        except ValueError:
            focus_x = 0.0
            focus_y = 0.0
        rviz_runtime = _patch_rviz_focus(rviz_cfg, focus_x, focus_y,
                                         distance=12.0)
        actions.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_runtime],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
        ))

    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument('world', description='Absolute path to the .world file.'),
        DeclareLaunchArgument('map_yaml', description='Absolute path to the Nav2 map YAML.'),
        DeclareLaunchArgument('x_pose', default_value='0.0',
                              description='Robot spawn x in Gazebo world frame.'),
        DeclareLaunchArgument('y_pose', default_value='0.0',
                              description='Robot spawn y in Gazebo world frame.'),
        DeclareLaunchArgument('z_pose', default_value='0.05'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('launch_rviz', default_value='true'),
        DeclareLaunchArgument('map_offset_x', default_value='0.0',
                              description='Static TF map->odom x translation '
                                          '(from sidecar world_to_map_offset).'),
        DeclareLaunchArgument('map_offset_y', default_value='0.0',
                              description='Static TF map->odom y translation '
                                          '(from sidecar world_to_map_offset).'),
        DeclareLaunchArgument('gazebo_model_path', default_value='',
                              description='Colon-separated extra dirs to '
                                          'prepend to GAZEBO_MODEL_PATH.'),
        OpaqueFunction(function=_launch_setup),
    ])
