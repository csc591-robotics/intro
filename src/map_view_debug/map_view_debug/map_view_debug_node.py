"""Render the LLM's annotated map view from a "2D Pose Estimate" click in RViz.

This node is a thin wrapper around ``nav2_llm_demo.llm.map_renderer.render_annotated_map``
intended to debug the image that the LLM agent receives via ``get_map_b64``.
It does not control any robot - the only input is RViz's ``/initialpose``
topic (published by the "2D Pose Estimate" tool, frame_id="map").

For every received pose the node:

1. Calls ``render_annotated_map(...)`` with the **same arguments** as
   ``llm_agent.LLMAgent.get_map_b64``: source / destination / crop_radius_m /
   output_size are passed through from launch params, and (robot_x, robot_y,
   robot_yaw) are taken from the click in the map frame.
2. Decodes the returned base64 PNG and writes it to disk as
   ``pose_NNN.png`` plus a ``pose_NNN.json`` sidecar describing the inputs.
3. Logs the path so the user can ``xdg-open`` it immediately.
"""

from __future__ import annotations

import base64
import json
import math
import os
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node

from nav2_llm_demo.llm.map_renderer import render_annotated_map


def _yaw_from_quat(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


class MapViewDebugNode(Node):
    """Subscribe to /initialpose and dump rendered map images per click."""

    def __init__(self) -> None:
        super().__init__('map_view_debug_node')

        self.declare_parameter('map_yaml_path', '')
        self.declare_parameter('source_x', 0.0)
        self.declare_parameter('source_y', 0.0)
        self.declare_parameter('dest_x', 0.0)
        self.declare_parameter('dest_y', 0.0)
        self.declare_parameter('crop_radius_m', 18.0)
        self.declare_parameter('output_size', 512)
        self.declare_parameter('output_dir', '')

        self._map_yaml_path = self.get_parameter('map_yaml_path').value
        self._source_x = float(self.get_parameter('source_x').value)
        self._source_y = float(self.get_parameter('source_y').value)
        self._dest_x = float(self.get_parameter('dest_x').value)
        self._dest_y = float(self.get_parameter('dest_y').value)
        self._crop_radius_m = float(self.get_parameter('crop_radius_m').value)
        self._output_size = int(self.get_parameter('output_size').value)

        if not self._map_yaml_path or not os.path.isfile(self._map_yaml_path):
            raise RuntimeError(
                f'map_yaml_path is required and must exist '
                f'(got {self._map_yaml_path!r})'
            )

        out_dir_param = str(self.get_parameter('output_dir').value or '').strip()
        if not out_dir_param:
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Mirror nav2_llm_demo: default to <WORKSPACE_DIR>/map_view_debug_runs/<stamp>/.
            # WORKSPACE_DIR defaults to /workspace which (inside this project's Docker
            # setup) is bind-mounted to the host's intro/ folder, so the PNGs land
            # next to llm_agent_runs/ and are visible from the host IDE.
            workspace = os.environ.get('WORKSPACE_DIR', '/workspace')
            out_dir_param = str(Path(workspace) / 'map_view_debug_runs' / stamp)
        self._out_dir = Path(out_dir_param).expanduser().resolve()
        self._out_dir.mkdir(parents=True, exist_ok=True)

        self._n = 0

        self._sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self._on_pose,
            10,
        )

        self.get_logger().info('=' * 70)
        self.get_logger().info('map_view_debug_node ready.')
        self.get_logger().info(
            'In RViz: pick the "2D Pose Estimate" tool (shortcut: E) and '
            'drag an arrow on the map to render an LLM-style map view.'
        )
        self.get_logger().info(f'Map YAML       : {self._map_yaml_path}')
        self.get_logger().info(
            f'Source (map)   : ({self._source_x:.3f}, {self._source_y:.3f})'
        )
        self.get_logger().info(
            f'Dest   (map)   : ({self._dest_x:.3f}, {self._dest_y:.3f})'
        )
        self.get_logger().info(
            f'crop_radius_m  : {self._crop_radius_m}  '
            f'output_size: {self._output_size}'
        )
        self.get_logger().info(f'Saving images  : {self._out_dir}')
        self.get_logger().info('=' * 70)

    def _on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        frame = (msg.header.frame_id or '').strip()
        if frame and frame != 'map':
            self.get_logger().warning(
                f'Ignoring pose in frame {frame!r}; expected "map" '
                '(set RViz Fixed Frame to "map" before clicking).'
            )
            return

        x = float(msg.pose.pose.position.x)
        y = float(msg.pose.pose.position.y)
        q = msg.pose.pose.orientation
        yaw = _yaw_from_quat(
            float(q.x), float(q.y), float(q.z), float(q.w),
        )

        try:
            img_b64 = render_annotated_map(
                map_yaml_path=self._map_yaml_path,
                robot_x=x, robot_y=y, robot_yaw=yaw,
                dest_x=self._dest_x, dest_y=self._dest_y,
                source_x=self._source_x, source_y=self._source_y,
                crop_radius_m=self._crop_radius_m,
                output_size=self._output_size,
            )
        except Exception as exc:
            self.get_logger().error(f'render_annotated_map failed: {exc}')
            return

        self._n += 1
        png_path = self._out_dir / f'pose_{self._n:03d}.png'
        meta_path = self._out_dir / f'pose_{self._n:03d}.json'

        try:
            png_path.write_bytes(base64.b64decode(img_b64))
            meta_path.write_text(json.dumps({
                'click': self._n,
                'input_pose_map': {
                    'x': x,
                    'y': y,
                    'yaw_rad': yaw,
                    'yaw_deg': math.degrees(yaw),
                },
                'map_yaml': self._map_yaml_path,
                'source': [self._source_x, self._source_y],
                'destination': [self._dest_x, self._dest_y],
                'crop_radius_m': self._crop_radius_m,
                'output_size': self._output_size,
            }, indent=2))
        except OSError as exc:
            self.get_logger().error(f'Failed to write outputs: {exc}')
            return

        self.get_logger().info(
            f'pose#{self._n}  map=({x:+.3f}, {y:+.3f}, '
            f'yaw={math.degrees(yaw):+.1f} deg)  ->  {png_path}'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapViewDebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
