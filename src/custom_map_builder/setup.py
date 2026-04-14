from pathlib import Path

from setuptools import find_packages, setup

package_name = 'custom_map_builder'

_pkg_root = Path(__file__).resolve().parent
_maps_dir = _pkg_root / 'maps'
# Every regular file under maps/ (any depth) is installed under share/.../maps/ on
# colcon build / pip install — not at ros2 launch time.
_map_install_paths = sorted(
    str(p.relative_to(_pkg_root))
    for p in _maps_dir.rglob('*')
    if p.is_file() and not any(part.startswith('.') for part in p.relative_to(_maps_dir).parts)
)
if not _map_install_paths:
    raise RuntimeError(
        f'Expected at least one file in {_maps_dir} (e.g. default.yaml).'
    )

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/map_builder.launch.py']),
        ('share/' + package_name + '/rviz', ['rviz/map_builder.rviz']),
        ('share/' + package_name + '/maps', _map_install_paths),
        (
            'share/' + package_name + '/scripts',
            ['scripts/generate_y_corridor_map.py'],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='NCSU CSC591',
    maintainer_email='akaluri@ncsu.edu',
    description='Map builder: Gazebo + Burger + RViz + map click coordinates.',
    license='MIT-0',
    entry_points={
        'console_scripts': [
            'click_echo = custom_map_builder.click_echo_node:main',
        ],
    },
)
