import os

from setuptools import find_packages, setup

package_name = 'nav2_llm_demo'


def _collect_maps() -> list[str]:
    """Gather all non-dotfiles under maps/ for installation."""
    maps_dir = os.path.join(os.path.dirname(__file__), 'maps')
    if not os.path.isdir(maps_dir):
        return []
    return [
        os.path.join('maps', f)
        for f in os.listdir(maps_dir)
        if os.path.isfile(os.path.join(maps_dir, f)) and not f.startswith('.')
    ]


setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/llm_agent.launch.py',
        ]),
        ('share/' + package_name + '/config', [
            'config/llm_nav_params.yaml',
            'config/nav2_params_flow6.yaml',
        ]),
        ('share/' + package_name + '/maps', _collect_maps()),
        ('share/' + package_name + '/scripts', [
            'scripts/parse_nav_config.py',
        ]),
        ('share/' + package_name + '/rviz', [
            'rviz/llm_agent.rviz',
        ]),
    ],
    install_requires=[
        'setuptools',
        'langchain',
        'langchain-core',
        'langgraph',
        'Pillow',
        'numpy',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='Anirudh, Brennen, Miles',
    maintainer_email='akaluri@ncsu.edu',
    description='Vision-based LLM agent that navigates a robot using tool calling.',
    license='MIT-0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'llm_agent_node = nav2_llm_demo.llm_agent_node:main',
            'llm_route_agent_node = nav2_llm_demo.llm_route_agent_node:main',
        ],
    },
)
