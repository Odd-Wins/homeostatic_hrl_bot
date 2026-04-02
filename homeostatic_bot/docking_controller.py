#!/usr/bin/env python3
"""
Docking Controller Node
- Uses AprilTag TF transform to navigate to charging station
- Publishes /charging/detected when robot is in charging zone
"""

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException


class DockingController(Node):
    def __init__(self):
        super().__init__('docking_controller')
        
        # Parameters
        self.declare_parameter('target_frame', 'tag36h11:0')
        self.declare_parameter('camera_frame', 'camera_rgb_frame')
        self.declare_parameter('docking_distance', 0.5)  # Stop 0.5m from tag
        self.declare_parameter('kp_angular', 1.0)        # Turn speed gain
        self.declare_parameter('kp_linear', 0.3)         # Forward speed gain
        self.declare_parameter('max_angular', 0.5)       # Max turn speed
        self.declare_parameter('max_linear', 0.15)       # Max forward speed
        
        # Get parameters
        self.target_frame = self.get_parameter('target_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.docking_distance = self.get_parameter('docking_distance').value
        self.kp_angular = self.get_parameter('kp_angular').value
        self.kp_linear = self.get_parameter('kp_linear').value
        self.max_angular = self.get_parameter('max_angular').value
        self.max_linear = self.get_parameter('max_linear').value
        
        # TF2 for getting tag position
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Publishers
        self.cmd_pub = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        self.charging_pub = self.create_publisher(Bool, '/charging/detected', 10)
        
        # Control loop timer (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        # State
        self.is_docked = False
        
        self.get_logger().info('Docking Controller started!')
        self.get_logger().info(f'Looking for tag: {self.target_frame}')
        self.get_logger().info(f'Will stop at: {self.docking_distance}m from tag')

    def control_loop(self):
        """Main control loop - runs at 10 Hz"""
        twist = TwistStamped()
        twist.header.stamp = self.get_clock().now().to_msg()
        twist.header.frame_id = 'base_link'
        charging_msg = Bool()
        
        try:
            # Look up transform: where is the tag relative to camera?
            transform = self.tf_buffer.lookup_transform(
                self.camera_frame,
                self.target_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1)
            )
            
            # Extract position (in camera frame)
            # x = left/right (negative = left, positive = right)
            # y = up/down
            # z = forward distance
            tag_x = transform.transform.translation.x
            tag_z = transform.transform.translation.z
            
            self.get_logger().info(
                f'Tag detected! x={tag_x:.2f}m (lateral), z={tag_z:.2f}m (forward)',
                throttle_duration_sec=1.0
            )
            
            # Check if we're close enough to dock
            if tag_z <= self.docking_distance:
                # DOCKED!
                self.is_docked = True
                charging_msg.data = True
                twist.twist.linear.x = 0.0
                twist.twist.angular.z = 0.0
                self.get_logger().info('DOCKED! Charging...', throttle_duration_sec=2.0)
            else:
                # Not docked yet - navigate toward tag
                self.is_docked = False
                charging_msg.data = False
                
                # Angular control: center the tag in camera view
                # If tag_x is positive (tag on right), turn right (negative angular)
                angular_error = -tag_x
                twist.twist.angular.z = max(-self.max_angular, 
                                     min(self.max_angular, 
                                         self.kp_angular * angular_error))
                
                # Linear control: move forward based on distance
                distance_error = tag_z - self.docking_distance
                twist.twist.linear.x = max(0.0,  # Don't reverse
                                    min(self.max_linear,
                                        self.kp_linear * distance_error))
                
                # Slow down forward speed if not centered
                if abs(tag_x) > 0.1:
                    twist.twist.linear.x *= 0.5  # Half speed when turning
                    
        except (LookupException, ConnectivityException, ExtrapolationException):
            # Can't see the tag
            if self.is_docked:
                # Was docked, might have lost sight briefly - keep charging
                charging_msg.data = True
            else:
                # Not docked and can't see tag - stop
                charging_msg.data = False
                twist.twist.linear.x = 0.0
                twist.twist.angular.z = 0.0
                self.get_logger().info('Tag not visible...', throttle_duration_sec=2.0)
        
        # Publish commands
        self.cmd_pub.publish(twist)
        self.charging_pub.publish(charging_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DockingController()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop robot on shutdown
        stop_msg = TwistStamped()
        node.cmd_pub.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()