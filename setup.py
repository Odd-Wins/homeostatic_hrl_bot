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
        ],
    },
)
