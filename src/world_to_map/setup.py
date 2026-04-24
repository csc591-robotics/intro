from pathlib import Path

from setuptools import find_packages, setup

package_name = 'world_to_map'

_pkg_root = Path(__file__).resolve().parent
_maps_dir = _pkg_root / 'maps'
_map_install_paths = sorted(
    str(p.relative_to(_pkg_root))
    for p in _maps_dir.rglob('*')
    if p.is_file() and not any(part.startswith('.') for part in p.relative_to(_maps_dir).parts)
)

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/world_to_map.launch.py']),
        ('share/' + package_name + '/rviz', ['rviz/world_to_map.rviz']),
        ('share/' + package_name + '/maps', _map_install_paths),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='NCSU CSC591',
    maintainer_email='akaluri@ncsu.edu',
    description=(
        'Rasterize Gazebo .world files to Nav2 PGM/YAML and launch '
        'Gazebo + map_server + RViz + TurtleBot3 with 1:1 frame parity.'
    ),
    license='MIT-0',
    entry_points={
        'console_scripts': [
            'rasterize_world = world_to_map.rasterize_world:main',
            'teleop_wasdx = world_to_map.teleop_wasdx:main',
        ],
    },
)
