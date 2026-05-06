import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'homeostatic_bot'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), 
            glob('launch/*.py')),
        # Include world files
        (os.path.join('share', package_name, 'worlds'), 
            glob('worlds/*.sdf')),
        # Include AprilTag model
        (os.path.join('share', package_name, 'models', 'apriltag_0'),
            ['models/apriltag_0/model.config',
             'models/apriltag_0/model.sdf']),
        (os.path.join('share', package_name, 'models', 'apriltag_0', 'materials', 'textures'),
            ['models/apriltag_0/materials/textures/tag36_11_00000.png']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='amron',
    maintainer_email='amron@todo.todo',
    description='Homeostatic energy management for TurtleBot3',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'patrol_node = homeostatic_bot.patrol_node:main',
            'battery_node = homeostatic_bot.battery_node:main',
            'docking_controller = homeostatic_bot.docking_controller:main',
            'test_env_smoke = homeostatic_bot.test_env_smoke:main',
            'test_homeostatic_reward = homeostatic_bot.test_homeostatic_reward:main',
            'threshold_baseline = homeostatic_bot.threshold_baseline:main',
            'train_flat = homeostatic_bot.train_flat:main', 
        ],
    },
)