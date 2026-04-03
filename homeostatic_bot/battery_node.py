import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import TwistStamped


class BatteryNode(Node):
    def __init__(self):
        super().__init__('battery_node')
        
        # Battery parameters
        self.soc = 100.0  # State of Charge (0-100%)
        self.soh = 100.0  # State of Health (0-100%) - for degradation
        self.max_capacity = 100.0  # Maximum capacity at 100% SOH
        
        # Rates
        self.drain_rate_moving = 0.5    # % per second when moving
        self.drain_rate_idle = 0.05     # % per second when idle
        self.charge_rate = 2.0          # % per second when charging
        
        # State tracking
        self.is_moving = False
        self.is_charging = False  # Now from /charging/detected topic
        
        # Publishers
        self.soc_publisher = self.create_publisher(Float32, '/battery/soc', 10)
        self.soh_publisher = self.create_publisher(Float32, '/battery/soh', 10)
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            TwistStamped,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        self.charging_sub = self.create_subscription(
            Bool,
            '/charging/detected',
            self.charging_callback,
            10
        )
        
        # Timer for battery updates (10 Hz)
        self.timer = self.create_timer(0.1, self.update_battery)
        
        self.get_logger().info('Battery Node Started!')
        self.get_logger().info('Listening for charging status on /charging/detected')

    def cmd_vel_callback(self, msg):
        """Check if robot is moving based on velocity commands."""
        linear = abs(msg.twist.linear.x) + abs(msg.twist.linear.y)
        angular = abs(msg.twist.angular.z)
        self.is_moving = (linear > 0.01) or (angular > 0.01)

    def charging_callback(self, msg):
        """Update charging state from docking controller."""
        self.is_charging = msg.data

    def update_battery(self):
        """Update battery state every 0.1 seconds."""
        dt = 0.1  # Time step in seconds
        
        if self.is_charging and self.soc < 100.0:
            # Charging
            self.soc += self.charge_rate * dt
            self.soc = min(self.soc, 100.0)  # Cap at 100%
            status = "CHARGING"
        elif self.is_moving:
            # Draining (moving)
            self.soc -= self.drain_rate_moving * dt
            self.soc = max(self.soc, 0.0)  # Don't go below 0%
            status = "DRAINING (moving)"
        else:
            # Draining (idle)
            self.soc -= self.drain_rate_idle * dt
            self.soc = max(self.soc, 0.0)
            status = "DRAINING (idle)"
        
        # Publish battery state
        soc_msg = Float32()
        soc_msg.data = self.soc
        self.soc_publisher.publish(soc_msg)
        
        soh_msg = Float32()
        soh_msg.data = self.soh
        self.soh_publisher.publish(soh_msg)
        
        # Log every 5 seconds
        if int(self.get_clock().now().nanoseconds / 1e9) % 5 == 0:
            if not hasattr(self, '_last_log_time'):
                self._last_log_time = 0
            current_time = int(self.get_clock().now().nanoseconds / 1e9)
            if current_time != self._last_log_time:
                self._last_log_time = current_time
                self.get_logger().info(f'SOC: {self.soc:.1f}% | SOH: {self.soh:.1f}% | {status}')
        
        # Warning at low battery
        if self.soc <= 20.0 and self.soc > 0:
            self.get_logger().warn(f'LOW BATTERY: {self.soc:.1f}%')
        elif self.soc <= 0:
            self.get_logger().error('BATTERY DEAD!')


def main(args=None):
    rclpy.init(args=args)
    node = BatteryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()