from setuptools import find_packages, setup

package_name = 'nav2_llm_demo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/llm_nav.launch.py']),
        (
            'share/' + package_name + '/config',
            ['config/llm_nav_params.yaml', 'config/route_graph.json'],
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Groq-driven route decision layer for Nav2 waypoint missions.',
    license='MIT-0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'llm_nav_node = nav2_llm_demo.llm_nav_node:main',
        ],
    },
)
