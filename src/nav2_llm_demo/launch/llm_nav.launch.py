from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _share_dir() -> str:
    """Return the installed package share directory."""
    return get_package_share_directory('nav2_llm_demo')


def _params_file() -> str:
    """Resolve the default YAML parameter file for the node."""
    share_dir = _share_dir()
    return f'{share_dir}/config/llm_nav_params.yaml'


def _route_graph_file() -> str:
    """Resolve the route graph JSON used by the planner node."""
    share_dir = get_package_share_directory('nav2_llm_demo')
    return f'{share_dir}/config/route_graph.json'


def generate_launch_description() -> LaunchDescription:
    """Launch Nav2 bringup plus the LLM navigation node."""
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    map_yaml_file = LaunchConfiguration('map')
    nav2_params_file = LaunchConfiguration('nav2_params_file')
    llm_params_file = LaunchConfiguration('llm_params_file')
    route_graph_path = LaunchConfiguration('route_graph_path')

    nav2_bringup_dir = get_package_share_directory('nav2_bringup')
    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            f'{nav2_bringup_dir}/launch/bringup_launch.py'
        ),
        launch_arguments={
            'slam': 'False',
            'map': map_yaml_file,
            'use_sim_time': use_sim_time,
            'params_file': nav2_params_file,
            'autostart': autostart,
        }.items(),
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            description='Absolute path to the Nav2 map YAML file.',
        ),
        DeclareLaunchArgument(
            'nav2_params_file',
            description='Absolute path to the Nav2 parameters YAML file.',
        ),
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='false',
            description='Use simulation time if true.',
        ),
        DeclareLaunchArgument(
            'autostart',
            default_value='true',
            description='Automatically transition Nav2 lifecycle nodes.',
        ),
        DeclareLaunchArgument(
            'llm_params_file',
            default_value=_params_file(),
            description='Path to the llm_nav_node parameter file.',
        ),
        DeclareLaunchArgument(
            'route_graph_path',
            default_value=_route_graph_file(),
            description='Path to the route graph JSON file.',
        ),
        nav2_bringup,
        Node(
            package='nav2_llm_demo',
            executable='llm_nav_node',
            name='llm_nav_node',
            output='screen',
            parameters=[
                ParameterFile(llm_params_file, allow_substs=True),
                {'route_graph_path': route_graph_path},
            ],
        ),
    ])
