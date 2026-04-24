from setuptools import find_packages, setup

package_name = 'map_view_debug'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/map_view_debug.launch.py',
        ]),
        ('share/' + package_name + '/scripts', [
            'scripts/run_map_view_debug.sh',
        ]),
    ],
    install_requires=[
        'setuptools',
        'Pillow',
        'numpy',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='Anirudh, Brennen, Miles',
    maintainer_email='akaluri@ncsu.edu',
    description=(
        'Interactive debug tool: bring up Gazebo + RViz aligned via the '
        'world_to_map sidecar (no robot) and render the LLM map view from '
        'a 2D Pose Estimate click in RViz.'
    ),
    license='MIT-0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'map_view_debug_node = map_view_debug.map_view_debug_node:main',
        ],
    },
)
