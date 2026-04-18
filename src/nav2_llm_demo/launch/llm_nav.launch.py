"""Launch the full LLM-controlled Nav2 demo end-to-end.

Single launch file that, given a ``map_poses.yaml`` and a ``map_name`` key,
brings up everything the user needs for an LLM-driven navigation mission so
that **Gazebo and RViz both show the robot at the same place from the very
first frame**:

  1. Reads the entry for ``map_name`` from ``map_poses.yaml``.
  2. Reads the per-map ``.world_map.yaml`` *sidecar* (referenced from
     ``map_poses.yaml``) to discover:
       - the path to the ``.world`` file Gazebo should load,
       - the Nav2 ``map.yaml`` rasterized from that world,
       - the ``world_to_map_offset`` produced by the rasterizer.
  3. Launches Gazebo with that ``.world``.
  4. Spawns the TurtleBot3 at the ``source`` pose recorded in
     ``map_poses.yaml`` (Gazebo world frame).
  5. Publishes a static ``map → odom`` transform whose translation equals
     ``source + world_to_map_offset`` so the robot's pose in the map frame
     matches its position in Gazebo to the millimetre.
  6. Brings up Nav2's *navigation* stack only (planner / controller /
     bt_navigator / behavior / waypoint / smoother). We deliberately do NOT
     run AMCL: Gazebo's diff-drive plugin gives us perfect odometry and our
     static TF gives us perfect localization, so AMCL would only fight us.
  7. Opens RViz with the rasterized PGM and the camera focused on the
     robot's spawn position.
  8. Starts ``llm_nav_node`` parameterized with the same
     ``map_poses_path`` + ``map_name`` so the LLM picks goals from the
     ``route_graph`` block.

Steps 3–7 are delegated to ``world_to_map.launch.py`` which already knows
how to spawn the robot, publish the static TF, run ``map_server`` +
``lifecycle_manager``, and open RViz with a focus-patched config.

For hand-crafted maps that have no sidecar (``sidecar: null`` in
``map_poses.yaml``), there is no Gazebo world to launch. In that case the
launch file falls back to plain ``nav2_bringup`` (full bringup, AMCL
included) using the ``map_yaml_fallback`` argument the user must supply.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, List

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile


def _share_dir() -> str:
    return get_package_share_directory('nav2_llm_demo')


def _default_llm_params_file() -> str:
    return os.path.join(_share_dir(), 'config', 'llm_nav_params.yaml')


def _resolve_path(path_str: str) -> Path:
    """Resolve a path that may be absolute or workspace-relative."""
    p = Path(path_str).expanduser()
    if p.is_file():
        return p

    # Try resolving relative to the workspace root (parent of install/).
    # __file__ lives at install/.../share/nav2_llm_demo/launch/llm_nav.launch.py
    # so parents[5] is the workspace root.
    here = Path(__file__).resolve()
    for n in range(3, 8):
        candidate = here.parents[n] / path_str
        if candidate.is_file():
            return candidate

    # Last-ditch: try CWD.
    cwd_candidate = Path.cwd() / path_str
    if cwd_candidate.is_file():
        return cwd_candidate

    raise FileNotFoundError(
        f'Could not resolve path: {path_str!r}. Tried as absolute, '
        'workspace-relative, and CWD-relative.'
    )


def _coerce_pose(source: dict[str, Any] | None) -> tuple[float, float, float, float]:
    """Return (x, y, z, yaw) from a map_poses ``source`` block (with defaults)."""
    if not source:
        return 0.0, 0.0, 0.05, 0.0
    pos = source.get('position') or {}
    x = float(pos.get('x', 0.0))
    y = float(pos.get('y', 0.0))
    z = float(pos.get('z', 0.05))
    if z <= 0.0:
        z = 0.05  # avoid spawning the robot inside the floor plane
    yaw = float(source.get('yaw_rad', 0.0))
    return x, y, z, yaw


def _launch_setup(context, *_args, **_kwargs) -> List:
    map_poses_path = LaunchConfiguration('map_poses_path').perform(context).strip()
    map_name = LaunchConfiguration('map_name').perform(context).strip()
    use_sim_time = LaunchConfiguration('use_sim_time').perform(context).lower() == 'true'
    autostart = LaunchConfiguration('autostart').perform(context)
    nav2_params_file = LaunchConfiguration('nav2_params_file').perform(context).strip()
    llm_params_file = LaunchConfiguration('llm_params_file').perform(context).strip()
    launch_rviz = LaunchConfiguration('launch_rviz').perform(context).lower() == 'true'
    map_yaml_fallback = LaunchConfiguration('map_yaml_fallback').perform(context).strip()
    gazebo_model_path = LaunchConfiguration('gazebo_model_path').perform(context).strip()

    if not map_poses_path:
        raise RuntimeError('map_poses_path is required')
    if not map_name:
        raise RuntimeError('map_name is required')
    if not nav2_params_file:
        raise RuntimeError('nav2_params_file is required')

    poses_path = _resolve_path(map_poses_path)
    with poses_path.open() as fh:
        poses_data = yaml.safe_load(fh) or {}

    map_key = map_name if map_name.endswith('.pgm') else map_name + '.pgm'
    maps = poses_data.get('maps') or {}
    if map_key not in maps:
        raise RuntimeError(
            f"map_poses.yaml has no entry for {map_key!r}. "
            f'Available: {sorted(maps.keys())}'
        )
    entry = maps[map_key] or {}

    sidecar_rel = entry.get('sidecar')
    sx, sy, sz, syaw = _coerce_pose(entry.get('source'))

    nav2_share = get_package_share_directory('nav2_bringup')

    actions: List = []

    # ── Common parameters for the LLM node (both code paths share them) ────
    llm_node = Node(
        package='nav2_llm_demo',
        executable='llm_nav_node',
        name='llm_nav_node',
        output='screen',
        parameters=[
            ParameterFile(llm_params_file, allow_substs=True),
            {
                'use_sim_time': use_sim_time,
                'map_poses_path': str(poses_path),
                'map_name': map_key,
            },
        ],
    )

    if sidecar_rel:
        # ── Path A: world_to_map-generated map ───────────────────────────
        # Gazebo world + spawn at source + static TF + map_server + RViz
        # come from world_to_map.launch.py. We add only the navigation
        # stack on top (no map_server, no AMCL).
        sidecar_path = _resolve_path(sidecar_rel)
        with sidecar_path.open() as fh:
            sidecar = yaml.safe_load(fh) or {}

        world_path = sidecar.get('source_world')
        if not world_path:
            raise RuntimeError(
                f'sidecar {sidecar_path} is missing source_world'
            )
        if not Path(world_path).is_file():
            raise FileNotFoundError(
                f'.world file referenced by sidecar does not exist: '
                f'{world_path}. Re-run rasterize_world inside the '
                'container so the path matches your filesystem.'
            )

        offset = sidecar.get('world_to_map_offset', [0.0, 0.0]) or [0.0, 0.0]
        offset_x = float(offset[0])
        offset_y = float(offset[1])

        # map_yaml is sibling of the sidecar by convention.
        map_yaml_rel = sidecar.get('map_yaml', '')
        if not map_yaml_rel:
            raise RuntimeError(
                f'sidecar {sidecar_path} is missing map_yaml'
            )
        map_yaml_path = (sidecar_path.parent / map_yaml_rel).resolve()
        if not map_yaml_path.is_file():
            raise FileNotFoundError(
                f'map yaml referenced by sidecar does not exist: '
                f'{map_yaml_path}'
            )

        # Static TF map → odom translation must equal source + offset so
        # that the robot's odom origin (which Gazebo plants at the spawn
        # location) lines up with the right map-frame coordinate.
        map_to_odom_x = sx + offset_x
        map_to_odom_y = sy + offset_y

        world_to_map_share = get_package_share_directory('world_to_map')
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    world_to_map_share, 'launch', 'world_to_map.launch.py'
                )
            ),
            launch_arguments={
                'world': str(world_path),
                'map_yaml': str(map_yaml_path),
                'x_pose': f'{sx:.6f}',
                'y_pose': f'{sy:.6f}',
                'z_pose': f'{sz:.6f}',
                'yaw': f'{syaw:.6f}',
                'map_offset_x': f'{map_to_odom_x:.6f}',
                'map_offset_y': f'{map_to_odom_y:.6f}',
                'launch_rviz': 'true' if launch_rviz else 'false',
                'use_sim_time': 'true' if use_sim_time else 'false',
                'gazebo_model_path': gazebo_model_path,
            }.items(),
        ))

        # Nav2 navigation stack only (no map_server, no AMCL).
        navigation_launch = os.path.join(
            nav2_share, 'launch', 'navigation_launch.py'
        )
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(navigation_launch),
            launch_arguments={
                'use_sim_time': 'true' if use_sim_time else 'false',
                'params_file': nav2_params_file,
                'autostart': autostart,
            }.items(),
        ))

    else:
        # ── Path B: hand-crafted map (no Gazebo world) ───────────────────
        # Fall back to full nav2_bringup (map_server + AMCL + navigation)
        # so the user can still drive a click-built map without Gazebo.
        if not map_yaml_fallback:
            raise RuntimeError(
                f"map entry {map_key!r} has no sidecar (hand-crafted map). "
                'Provide map_yaml_fallback:=<absolute path to map.yaml> '
                'so nav2_bringup has something to load.'
            )
        map_yaml_path = _resolve_path(map_yaml_fallback)

        bringup_launch = os.path.join(
            nav2_share, 'launch', 'bringup_launch.py'
        )
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(bringup_launch),
            launch_arguments={
                'slam': 'False',
                'map': str(map_yaml_path),
                'use_sim_time': 'true' if use_sim_time else 'false',
                'params_file': nav2_params_file,
                'autostart': autostart,
            }.items(),
        ))

    actions.append(llm_node)
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            'map_poses_path',
            description=(
                'Absolute (or workspace-relative) path to map_poses.yaml '
                '(src/custom_map_builder/maps/map_poses.yaml). The single '
                'shared file with custom_map_builder — contains the '
                'source/destination poses, the route_graph, and a sidecar '
                'reference per map.'
            ),
        ),
        DeclareLaunchArgument(
            'map_name',
            description=(
                'PGM filename key in map_poses.yaml, e.g. "warehouse" or '
                '"warehouse.pgm". Must match an entry under maps:.'
            ),
        ),
        DeclareLaunchArgument(
            'nav2_params_file',
            description='Absolute path to the Nav2 parameters YAML file.',
        ),
        DeclareLaunchArgument(
            'llm_params_file',
            default_value=_default_llm_params_file(),
            description='Path to the llm_nav_node parameter file.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description=(
                'Use Gazebo /clock. Defaults to true because the world_to_map '
                'flow always runs Gazebo.'
            ),
        ),
        DeclareLaunchArgument(
            'autostart',
            default_value='true',
            description='Automatically transition Nav2 lifecycle nodes.',
        ),
        DeclareLaunchArgument(
            'launch_rviz',
            default_value='true',
            description='Open RViz focused on the robot spawn point.',
        ),
        DeclareLaunchArgument(
            'map_yaml_fallback',
            default_value='',
            description=(
                'Only used when the chosen map entry has sidecar: null '
                '(hand-crafted maps). Path to the Nav2 map yaml for the '
                'fallback nav2_bringup full-bringup path.'
            ),
        ),
        DeclareLaunchArgument(
            'gazebo_model_path',
            default_value='',
            description=(
                'Colon-separated extra dirs to prepend to GAZEBO_MODEL_PATH '
                'so Gazebo can resolve <include><uri>model://…</uri> tags '
                'in the chosen .world file.'
            ),
        ),
        OpaqueFunction(function=_launch_setup),
    ])
