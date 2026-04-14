"""Subscribe to RViz Publish Point (/clicked_point) and log coordinates."""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped


class MapClickEcho(Node):
    """Print map-frame coordinates for each clicked point."""

    def __init__(self) -> None:
        super().__init__('map_click_echo')
        self._sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self._on_point,
            10,
        )
        self.get_logger().info(
            'Listening on /clicked_point. In RViz use the "Publish Point" tool '
            'and click on the map; coordinates print below (check frame_id).'
        )

    def _on_point(self, msg: PointStamped) -> None:
        f = msg.header.frame_id
        self.get_logger().info(
            f'clicked_point  frame="{f}"  '
            f'x={msg.point.x:.4f}  y={msg.point.y:.4f}  z={msg.point.z:.4f}'
        )


def main() -> None:
    rclpy.init()
    node = MapClickEcho()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
