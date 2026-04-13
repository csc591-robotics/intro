from setuptools import find_packages, setup

package_name = 'custom_map_builder'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/map_builder.launch.py']),
        ('share/' + package_name + '/rviz', ['rviz/map_builder.rviz']),
        (
            'share/' + package_name + '/maps',
            ['maps/default.pgm', 'maps/default.yaml'],
        ),
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
