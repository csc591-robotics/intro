from launch import LaunchDescription
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def _share_dir() -> str:
    return get_package_share_directory('nav2_llm_demo')


def _params_file() -> str:
    share_dir = _share_dir()
    return f'{share_dir}/config/llm_nav_params.yaml'


def _route_graph_file() -> str:
    share_dir = get_package_share_directory('nav2_llm_demo')
    return f'{share_dir}/config/route_graph.json'


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        Node(
            package='nav2_llm_demo',
            executable='llm_nav_node',
            name='llm_nav_node',
            output='screen',
            parameters=[
                ParameterFile(_params_file(), allow_substs=True),
                {'route_graph_path': _route_graph_file()},
            ],
        ),
    ])
