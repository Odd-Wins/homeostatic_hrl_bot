"""RViz2 battery status display for markers showing SOC and SOH.
Subscribes to a custom topic published by t GUI demo script. Ran alongside
Gazebo for video demos used in presentation.
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Float32MultiArray
from builtin_interfaces.msg import Duration


class BatteryDisplay(Node):
    def __init__(self):
        super().__init__("battery_display")
        self._pub = self.create_publisher(MarkerArray, "/battery_display", 10)
        self._sub = self.create_subscription(
            Float32MultiArray, "/battery_status", self._battery_cb, 10
        )
        self._timer = self.create_timer(0.5, self._publish_markers)  # 2 Hz

        self._soc = None
        self._soh = None
        self.get_logger().info("Battery display node started. Waiting for /battery_status...")

    def _battery_cb(self, msg):
        if len(msg.data) >= 2:
            self._soc = msg.data[0]
            self._soh = msg.data[1]

    def _soc_color(self, soc):
        """Green >60%, yellow 30-60%, red <30%."""
        if soc > 60:
            return (0.0, 1.0, 0.0, 1.0)  # green
        elif soc > 30:
            return (1.0, 1.0, 0.0, 1.0)  # yellow
        else:
            return (1.0, 0.0, 0.0, 1.0)  # red

    def _make_text_marker(self, id, text, x, y, z, r, g, b, a=1.0, scale=0.4):
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = "battery"
        m.id = id
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.scale.z = scale  # text height in metres
        m.color.r = r
        m.color.g = g
        m.color.b = b
        m.color.a = a
        m.text = text
        m.lifetime = Duration(sec=1, nanosec=0)  # auto-expire if node stops
        return m

    def _publish_markers(self):
        markers = MarkerArray()

        if self._soc is None:
            m = self._make_text_marker(0, "WAITING...", 0.0, 0.0, 2.0, 1.0, 1.0, 1.0)
            markers.markers.append(m)
        else:
            # SOC marker - floats above robot position 
            r, g, b, a = self._soc_color(self._soc)
            soc_text = f"SOC: {self._soc:.1f}%"
            m_soc = self._make_text_marker(0, soc_text, -4.0, 4.5, 1.5, r, g, b, scale=0.5)
            markers.markers.append(m_soc)

            # SOH marker
            soh_text = f"SOH: {self._soh:.1f}%"
            m_soh = self._make_text_marker(1, soh_text, -4.0, 4.5, 1.0, 0.8, 0.8, 1.0, scale=0.5)
            markers.markers.append(m_soh)

            # Drain multiplier
            if self._soh > 0:
                mult = 100.0 / self._soh
                mult_text = f"Drain: {mult:.2f}x"
                m_mult = self._make_text_marker(2, mult_text, -4.0, 4.5, 0.5, 0.8, 0.6, 0.6, scale=0.35)
                markers.markers.append(m_mult)

        self._pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = BatteryDisplay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
