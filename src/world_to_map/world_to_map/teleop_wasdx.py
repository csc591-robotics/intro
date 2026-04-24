"""Minimal WASDX keyboard teleop publishing geometry_msgs/Twist on /cmd_vel.

Self-contained: depends only on rclpy + geometry_msgs (both ship with any
ros-humble-ros-base install), so it works even when turtlebot3_teleop
isn't installed in the container.

Key bindings:

    w  forward          a  turn left
    x  backward         d  turn right
    s  full stop        q  quit (also Ctrl-C)

Linear/angular speeds bump up/down with:

    +  faster (linear)        ]  faster (angular)
    -  slower (linear)        [  slower (angular)

Each key press accelerates the robot by one tick; the node publishes the
current Twist at ~10 Hz so the velocity is held steady between presses.
Hit ``s`` (or release the throttle) to stop. The node restores the TTY
on exit.
"""

from __future__ import annotations

import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node

LINEAR_STEP = 0.05
ANGULAR_STEP = 0.1
LINEAR_MAX = 0.4
ANGULAR_MAX = 2.5

HELP = """\
WASDX teleop -> /cmd_vel  (Twist)

  w / x : forward / backward (one step each press)
  a / d : turn left / right
  s     : stop
  + / - : increase / decrease linear speed step
  ] / [ : increase / decrease angular speed step
  q     : quit
"""


class WasdxTeleop(Node):
    def __init__(self) -> None:
        super().__init__('wasdx_teleop')
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.linear_step = LINEAR_STEP
        self.angular_step = ANGULAR_STEP
        self.lin = 0.0
        self.ang = 0.0
        self.timer = self.create_timer(0.1, self._tick)

    def _tick(self) -> None:
        msg = Twist()
        msg.linear.x = self.lin
        msg.angular.z = self.ang
        self.pub.publish(msg)

    def apply_key(self, key: str) -> bool:
        if key == 'w':
            self.lin = min(LINEAR_MAX, self.lin + self.linear_step)
        elif key == 'x':
            self.lin = max(-LINEAR_MAX, self.lin - self.linear_step)
        elif key == 'a':
            self.ang = min(ANGULAR_MAX, self.ang + self.angular_step)
        elif key == 'd':
            self.ang = max(-ANGULAR_MAX, self.ang - self.angular_step)
        elif key == 's':
            self.lin = 0.0
            self.ang = 0.0
        elif key == '+' or key == '=':
            self.linear_step = min(0.2, self.linear_step + 0.01)
        elif key == '-' or key == '_':
            self.linear_step = max(0.01, self.linear_step - 0.01)
        elif key == ']':
            self.angular_step = min(1.0, self.angular_step + 0.05)
        elif key == '[':
            self.angular_step = max(0.05, self.angular_step - 0.05)
        elif key == 'q' or key == '\x03':
            return False
        else:
            return True
        self.get_logger().info(
            f'lin={self.lin:+.2f} m/s  ang={self.ang:+.2f} rad/s  '
            f'(lin_step={self.linear_step:.2f}, ang_step={self.angular_step:.2f})'
        )
        return True


def _read_key(timeout: float = 0.05) -> str:
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if not rlist:
        return ''
    return sys.stdin.read(1)


def main(argv=None) -> int:
    rclpy.init(args=argv)
    node = WasdxTeleop()
    print(HELP, flush=True)

    fd = sys.stdin.fileno()
    old_attr = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        running = True
        while running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            key = _read_key(timeout=0.05)
            if key:
                running = node.apply_key(key)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attr)
        stop = Twist()
        node.pub.publish(stop)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
