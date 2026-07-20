"""Battery node - tracks SOC and SOH capacity fade + 1/SOH power fade)."""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from geometry_msgs.msg import TwistStamped


class BatteryNode(Node):
    # Publishes /battery/soc and /battery/soh

    def __init__(self):
        super().__init__('battery_node')

        # Configurable parameters
        self.declare_parameter('drain_rate_moving', 0.5)    # % per sec
        self.declare_parameter('drain_rate_idle', 0.05)     # % per sec
        self.declare_parameter('charge_rate', 2.0)          # % per sec
        self.declare_parameter('alpha', 0.001)              # Degradation rate
        self.declare_parameter('beta', 1.2)                 # Degradation acceleration
        self.declare_parameter('cycle_threshold', 20.0)     # SOC % - triggers new cycle

        # Get parameters
        self.drain_rate_moving = self.get_parameter('drain_rate_moving').value
        self.drain_rate_idle = self.get_parameter('drain_rate_idle').value
        self.charge_rate = self.get_parameter('charge_rate').value
        self.alpha = self.get_parameter('alpha').value
        self.beta = self.get_parameter('beta').value
        self.cycle_threshold = self.get_parameter('cycle_threshold').value

        # Battery state
        self.soc = 100.0         
        self.soh = 100.0          
        self.charge_cycles = 0    # no. of completed charge cycles

        # Cycle tracking
        self.was_below_threshold = False  # Track if dipped below threshold

        # State tracking
        self.is_moving = False
        self.is_charging = False

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
        self.get_logger().info(f'Degradation model: SOH = 1 - {self.alpha} × n^{self.beta}')
        self.get_logger().info(f'Cycle threshold: {self.cycle_threshold}%')

    def cmd_vel_callback(self, msg):
        # Check if robot is moving based on velocity commands
        linear = abs(msg.twist.linear.x) + abs(msg.twist.linear.y)
        angular = abs(msg.twist.angular.z)
        self.is_moving = (linear > 0.01) or (angular > 0.01)

    def charging_callback(self, msg):
        # Update charging state from docking controller
        self.is_charging = msg.data

    def update_soh(self):
        # Huang et al. degradation model
        degradation = self.alpha * (self.charge_cycles ** self.beta)
        self.soh = max(0.0, (1.0 - degradation) * 100.0)  # converts to percentage

    def update_battery(self):
        # Update battery state every 0.1 seconds
        dt = 0.1  # Time step per seconds

        # Power fade multiplier - degraded battery  drains faster
        # at 100% soh: factor = 1.00x (baseline)
        # at  80% soh: factor = 1.25x
        # at  60% soh: factor = 1.67x
        # at  40% soh: factor = 2.50x
        soh_factor = 100.0 / max(self.soh, 1.0)

        if self.is_charging and self.soc < 100.0:
            self.soc += self.charge_rate * dt
            self.soc = min(self.soc, 100.0)
            status = "CHARGING"

            # Track charge cycles
            if self.was_below_threshold and self.soc >= 99.0:
                self.charge_cycles += 1
                self.was_below_threshold = False
                self.update_soh()
                self.get_logger().info(
                    f'Charge cycle {self.charge_cycles} complete! '
                    f'SOH: {self.soh:.1f}%'
                )
        elif self.is_moving:
            self.soc -= self.drain_rate_moving * soh_factor * dt
            self.soc = max(self.soc, 0.0)
            status = "DRAINING (moving)"
        else:
            self.soc -= self.drain_rate_idle * soh_factor * dt
            self.soc = max(self.soc, 0.0)
            status = "DRAINING (idle)"

        # Tracks if soc drops below cycle threshold
        if self.soc <= self.cycle_threshold:
            self.was_below_threshold = True

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
                self.get_logger().info(
                    f'SOC: {self.soc:.1f}% | SOH: {self.soh:.1f}% | '
                    f'Cycles: {self.charge_cycles} | DrainMult: {soh_factor:.2f}x | {status}'
                )

        # low battery warning
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
