import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped


class PatrolNode(Node):
    def __init__(self):
        super().__init__('patrol_node')
        
        # Publisher for velocity commands (TwistStamped for Gazebo)
        self.publisher = self.create_publisher(TwistStamped, '/cmd_vel', 10)
        
        # Timer - runs every 0.1 seconds
        self.timer = self.create_timer(0.1, self.timer_callback)
        
        # State machine
        self.state = 'FORWARD'
        self.counter = 0
        
        self.get_logger().info('Patrol Node Started! Controlling TurtleBot3...')

    def timer_callback(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        
        if self.state == 'FORWARD':
            msg.twist.linear.x = 0.2
            msg.twist.angular.z = 0.0
            self.counter += 1
            if self.counter > 50:
                self.state = 'TURN'
                self.counter = 0
                self.get_logger().info('Turning...')
                
        elif self.state == 'TURN':
            msg.twist.linear.x = 0.0
            msg.twist.angular.z = 0.5
            self.counter += 1
            if self.counter > 31:
                self.state = 'FORWARD'
                self.counter = 0
                self.get_logger().info('Moving forward...')
        
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = PatrolNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_msg = TwistStamped()
        node.publisher.publish(stop_msg)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
