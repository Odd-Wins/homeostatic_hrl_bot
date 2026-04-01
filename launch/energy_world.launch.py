import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():
    # Package directories
    pkg_homeostatic = get_package_share_directory('homeostatic_bot')
    pkg_turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')
    
    # Path to your custom world
    world_file = os.path.join(pkg_homeostatic, 'worlds', 'energy_world.sdf')
    
    # Set Gazebo resource path (include BOTH TurtleBot3 AND your models)
    gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.join(pkg_homeostatic, 'models') + ':' +
              os.path.join(pkg_turtlebot3_gazebo, 'models')
    )
    
    # Launch Gazebo with custom world
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': f'-r -v2 {world_file}',
            'on_exit_shutdown': 'true'
        }.items()
    )
    
    # Spawn TurtleBot3 (DELAYED to let Gazebo start)
    spawn_robot = TimerAction(
        period=5.0,  # Wait 5 seconds for Gazebo to be ready
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(pkg_turtlebot3_gazebo, 'launch', 'spawn_turtlebot3.launch.py')
                ),
                launch_arguments={
                    'x_pose': '0.0',
                    'y_pose': '0.0'
                }.items()
            )
        ]
    )
    
    # Robot state publisher
    robot_state_publisher = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_turtlebot3_gazebo, 'launch', 'robot_state_publisher.launch.py')
        ),
        launch_arguments={'use_sim_time': 'true'}.items()
    )
    
    # Image bridge (Gazebo camera → ROS2)
    image_bridge = Node(
        package='ros_gz_image',
        executable='image_bridge',
        arguments=['/camera/image_raw'],
        output='screen',
        parameters=[{'use_sim_time': True}],
    )
    
    # Parameter bridge (camera_info and other topics)
    parameter_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        output='screen',
        parameters=[{'use_sim_time': True}],
    )
    
    return LaunchDescription([
        gz_resource_path,
        gazebo,
        spawn_robot,
        robot_state_publisher,
        image_bridge,
        parameter_bridge,
    ])