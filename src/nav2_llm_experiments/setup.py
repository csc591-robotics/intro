from setuptools import find_packages, setup

package_name = 'nav2_llm_experiments'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', [
            'config/experiments.yaml',
        ]),
        ('share/' + package_name + '/scripts', [
            'scripts/run_experiments.sh',
        ]),
    ],
    install_requires=[
        'setuptools',
        'pyyaml',
    ],
    zip_safe=True,
    maintainer='Anirudh, Brennen, Miles',
    maintainer_email='akaluri@ncsu.edu',
    description='Batch experiment runner on top of nav2_llm_demo.',
    license='MIT-0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'recorder_node = nav2_llm_experiments.recorder_node:main',
            'run_experiments = nav2_llm_experiments.run_experiments:main',
        ],
    },
)
