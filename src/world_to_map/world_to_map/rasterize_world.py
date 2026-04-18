"""Rasterize a Gazebo .world file into a Nav2-compatible PGM + YAML map.

v2 capabilities:

- Inline ``<box>`` (yaw-rotated) and ``<cylinder>`` collision/visual primitives.
- ``<include>`` resolution against ``--model-paths`` (or ``GAZEBO_MODEL_PATH``);
  each included model's ``model.sdf`` is parsed recursively, so a world that
  is "all includes" (workshop, warehouse, ...) still produces an occupancy
  grid as long as its model directories are on disk.
- Pose composition is translation + yaw only; roll/pitch and 3D rotations are
  treated as identity. This is fine for "stuff sitting on the floor".
- ``<mesh>``, ``<polyline>``, ``<sphere>``, ``<heightmap>`` are skipped with
  a one-line warning. Most models in the Gazebo collection use boxes /
  cylinders for ``<collision>`` (visuals can be meshes -- we ignore visuals
  whenever a collision is present).

Outputs three sibling files (``--out`` is a path prefix without extension):

- ``<prefix>.pgm``               occupancy grid (white = free, black = occupied)
- ``<prefix>.yaml``              Nav2 ``map_server`` config
- ``<prefix>.world_map.yaml``    reusable scaling/transform sidecar (resolution,
                                  origin, AABB, padding, frames, world->map
                                  offset for the launch's static TF)

Origin modes (``--origin-mode``):

- ``world`` (default in earlier v1): map.yaml ``origin = [min_x_padded,
  min_y_padded, 0]``, so the **map frame is identical to Gazebo's world
  frame** and the static TF map -> odom is identity.
- ``bottom-left``: map.yaml ``origin = [0, 0, 0]``, geometry is shifted into
  the positive quadrant so pixel (0, 0) of the PGM lands at map (0, 0). The
  launch then uses a non-identity static TF map -> odom of
  ``(-min_x_padded, -min_y_padded)`` to keep robot poses 1:1 with Gazebo.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image, ImageDraw

PGM_FREE = 254
PGM_OCCUPIED = 0


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0

    @classmethod
    def parse(cls, text: Optional[str]) -> "Pose":
        if not text:
            return cls()
        parts = text.strip().split()
        vals = [float(p) for p in parts]
        while len(vals) < 6:
            vals.append(0.0)
        return cls(*vals[:6])

    def compose(self, child: "Pose") -> "Pose":
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        x = self.x + cy * child.x - sy * child.y
        y = self.y + sy * child.x + cy * child.y
        z = self.z + child.z
        return Pose(x, y, z, 0.0, 0.0, self.yaw + child.yaw)


@dataclass
class BoxShape:
    pose: Pose
    sx: float
    sy: float
    sz: float


@dataclass
class CylinderShape:
    pose: Pose
    radius: float
    length: float


@dataclass
class ParseStats:
    boxes: int = 0
    cylinders: int = 0
    skipped_mesh: int = 0
    skipped_other: int = 0
    skipped_floor: int = 0
    includes_resolved: int = 0
    includes_unresolved: List[str] = field(default_factory=list)
    skipped_examples: List[str] = field(default_factory=list)


def _findtext(elem: ET.Element, tag: str) -> Optional[str]:
    child = elem.find(tag)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _parse_size3(text: Optional[str]) -> Optional[Tuple[float, float, float]]:
    if not text:
        return None
    parts = text.strip().split()
    if len(parts) < 3:
        return None
    return float(parts[0]), float(parts[1]), float(parts[2])


def _z_band_overlaps(center_z: float, half_height: float,
                     z_min: float, z_max: float) -> bool:
    lo = center_z - half_height
    hi = center_z + half_height
    return hi >= z_min and lo <= z_max


def _resolve_model_dir(uri: str, model_paths: List[Path]) -> Optional[Path]:
    """Resolve a ``model://name[/...]`` URI to a directory on disk."""
    if not uri.startswith('model://'):
        return None
    rel = uri[len('model://'):].strip('/')
    if not rel:
        return None
    name = rel.split('/', 1)[0]
    for base in model_paths:
        candidate = base / name
        if candidate.is_dir():
            return candidate
    return None


def _model_sdf_path(model_dir: Path) -> Optional[Path]:
    """Return the SDF file inside ``model_dir`` (preferring model.config hint)."""
    cfg = model_dir / 'model.config'
    if cfg.is_file():
        try:
            cfg_root = ET.parse(cfg).getroot()
            sdf_el = cfg_root.find('sdf')
            if sdf_el is not None and sdf_el.text:
                candidate = model_dir / sdf_el.text.strip()
                if candidate.is_file():
                    return candidate
        except ET.ParseError:
            pass
    fallback = model_dir / 'model.sdf'
    if fallback.is_file():
        return fallback
    for f in sorted(model_dir.glob('*.sdf')):
        if f.is_file():
            return f
    return None


def _is_floor_slab(top_z: float, sz: float, footprint_max: float,
                   floor_clearance: float) -> bool:
    """Heuristic to detect floor / ground plate boxes.

    A "floor slab" is any thin horizontal box that sits at (or below) ground
    level. Stamping it would fill the whole interior of a room with a single
    occupied rectangle, hiding every real obstacle inside.
    """
    if top_z > floor_clearance:
        return False
    if sz <= floor_clearance:
        return True
    if footprint_max >= 3.0 and sz <= 0.10:
        return True
    return False


def _collect_geometry(geom: ET.Element, link_pose_world: Pose,
                      collision_pose: Pose,
                      z_min: float, z_max: float,
                      floor_clearance: float,
                      boxes: List[BoxShape],
                      cylinders: List[CylinderShape],
                      stats: ParseStats) -> None:
    shape_pose = link_pose_world.compose(collision_pose)
    box = geom.find('box')
    cyl = geom.find('cylinder')
    if box is not None:
        size = _parse_size3(_findtext(box, 'size'))
        if size is None:
            stats.skipped_other += 1
            return
        sx, sy, sz = size
        if not _z_band_overlaps(shape_pose.z, sz / 2.0, z_min, z_max):
            return
        top_z = shape_pose.z + sz / 2.0
        if _is_floor_slab(top_z, sz, max(sx, sy), floor_clearance):
            stats.skipped_floor += 1
            return
        boxes.append(BoxShape(pose=shape_pose, sx=sx, sy=sy, sz=sz))
        stats.boxes += 1
        return
    if cyl is not None:
        radius_t = _findtext(cyl, 'radius')
        length_t = _findtext(cyl, 'length')
        if radius_t is None or length_t is None:
            stats.skipped_other += 1
            return
        radius = float(radius_t)
        length = float(length_t)
        if not _z_band_overlaps(shape_pose.z, length / 2.0, z_min, z_max):
            return
        top_z = shape_pose.z + length / 2.0
        if _is_floor_slab(top_z, length, 2.0 * radius, floor_clearance):
            stats.skipped_floor += 1
            return
        cylinders.append(CylinderShape(pose=shape_pose, radius=radius,
                                       length=length))
        stats.cylinders += 1
        return
    if geom.find('mesh') is not None:
        stats.skipped_mesh += 1
        if len(stats.skipped_examples) < 5:
            stats.skipped_examples.append('<mesh> (visual or collision)')
        return
    for tag in ('plane', 'polyline', 'sphere', 'heightmap'):
        if geom.find(tag) is not None:
            stats.skipped_other += 1
            if len(stats.skipped_examples) < 5:
                stats.skipped_examples.append(f'<{tag}>')
            return
    stats.skipped_other += 1


def _process_links(links: Iterable[ET.Element], parent_pose: Pose,
                   z_min: float, z_max: float, floor_clearance: float,
                   boxes: List[BoxShape],
                   cylinders: List[CylinderShape],
                   stats: ParseStats) -> None:
    for link in links:
        link_pose = Pose.parse(_findtext(link, 'pose'))
        link_world = parent_pose.compose(link_pose)
        collisions = link.findall('collision')
        targets = collisions if collisions else link.findall('visual')
        for shape in targets:
            geom = shape.find('geometry')
            if geom is None:
                continue
            shape_pose = Pose.parse(_findtext(shape, 'pose'))
            _collect_geometry(geom, link_world, shape_pose, z_min, z_max,
                              floor_clearance, boxes, cylinders, stats)


def _process_model(model: ET.Element, parent_pose: Pose,
                   model_paths: List[Path], visited_models: set,
                   z_min: float, z_max: float, floor_clearance: float,
                   boxes: List[BoxShape],
                   cylinders: List[CylinderShape],
                   stats: ParseStats) -> None:
    model_pose = Pose.parse(_findtext(model, 'pose'))
    model_world = parent_pose.compose(model_pose)
    _process_links(model.findall('link'), model_world, z_min, z_max,
                   floor_clearance, boxes, cylinders, stats)
    for nested_model in model.findall('model'):
        _process_model(nested_model, model_world, model_paths, visited_models,
                       z_min, z_max, floor_clearance,
                       boxes, cylinders, stats)
    for include in model.findall('include'):
        _process_include(include, model_world, model_paths, visited_models,
                         z_min, z_max, floor_clearance,
                         boxes, cylinders, stats)


def _process_include(include: ET.Element, parent_pose: Pose,
                     model_paths: List[Path], visited_models: set,
                     z_min: float, z_max: float, floor_clearance: float,
                     boxes: List[BoxShape],
                     cylinders: List[CylinderShape],
                     stats: ParseStats) -> None:
    uri = _findtext(include, 'uri') or ''
    include_pose = Pose.parse(_findtext(include, 'pose'))
    world_pose = parent_pose.compose(include_pose)
    if uri.startswith('model://'):
        bare_name = uri[len('model://'):].strip('/').split('/')[0]
        if bare_name in {'ground_plane', 'sun'}:
            stats.includes_resolved += 1
            return
    model_dir = _resolve_model_dir(uri, model_paths)
    if model_dir is None:
        stats.includes_unresolved.append(uri)
        return
    sdf_path = _model_sdf_path(model_dir)
    if sdf_path is None:
        stats.includes_unresolved.append(uri + ' (no model.sdf)')
        return
    sdf_key = str(sdf_path.resolve())
    if sdf_key in visited_models:
        stats.includes_resolved += 1
        return
    visited_models.add(sdf_key)
    try:
        sdf_root = ET.parse(sdf_path).getroot()
    except ET.ParseError as e:
        stats.includes_unresolved.append(f'{uri} (parse error: {e})')
        return
    stats.includes_resolved += 1
    for model_el in sdf_root.findall('model'):
        _process_model(model_el, world_pose, model_paths, visited_models,
                       z_min, z_max, floor_clearance,
                       boxes, cylinders, stats)
    for nested_inc in sdf_root.findall('include'):
        _process_include(nested_inc, world_pose, model_paths, visited_models,
                         z_min, z_max, floor_clearance,
                         boxes, cylinders, stats)


def _parse_world(world_path: Path, model_paths: List[Path],
                 z_min: float, z_max: float, floor_clearance: float,
                 ) -> Tuple[List[BoxShape], List[CylinderShape], ParseStats]:
    tree = ET.parse(world_path)
    root = tree.getroot()
    world = root.find('world')
    if world is None:
        raise ValueError(f'No <world> element in {world_path}')

    boxes: List[BoxShape] = []
    cylinders: List[CylinderShape] = []
    stats = ParseStats()
    visited_models: set = set()

    for model in world.findall('model'):
        if model.get('name', '') == 'ground_plane':
            continue
        _process_model(model, Pose(), model_paths, visited_models,
                       z_min, z_max, floor_clearance,
                       boxes, cylinders, stats)
    for include in world.findall('include'):
        _process_include(include, Pose(), model_paths, visited_models,
                         z_min, z_max, floor_clearance,
                         boxes, cylinders, stats)

    return boxes, cylinders, stats


def _shape_aabb(boxes: Iterable[BoxShape], cylinders: Iterable[CylinderShape]
                ) -> Optional[Tuple[float, float, float, float]]:
    xs_lo: List[float] = []
    ys_lo: List[float] = []
    xs_hi: List[float] = []
    ys_hi: List[float] = []
    for b in boxes:
        cy, sy = math.cos(b.pose.yaw), math.sin(b.pose.yaw)
        hx, hy = b.sx / 2.0, b.sy / 2.0
        for lx, ly in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]:
            wx = b.pose.x + cy * lx - sy * ly
            wy = b.pose.y + sy * lx + cy * ly
            xs_lo.append(wx); ys_lo.append(wy)
            xs_hi.append(wx); ys_hi.append(wy)
    for c in cylinders:
        xs_lo.append(c.pose.x - c.radius)
        ys_lo.append(c.pose.y - c.radius)
        xs_hi.append(c.pose.x + c.radius)
        ys_hi.append(c.pose.y + c.radius)
    if not xs_lo:
        return None
    return min(xs_lo), min(ys_lo), max(xs_hi), max(ys_hi)


def _rasterize(boxes: List[BoxShape], cylinders: List[CylinderShape],
               origin_x: float, origin_y: float, resolution: float,
               width_px: int, height_px: int) -> np.ndarray:
    """Stamp shapes into a PGM. ``origin_x/y`` is the world coord of pixel (0,0)."""
    img = Image.new('L', (width_px, height_px), color=PGM_FREE)
    draw = ImageDraw.Draw(img)

    def to_pixel(wx: float, wy: float) -> Tuple[float, float]:
        col = (wx - origin_x) / resolution
        row = (height_px - 1) - (wy - origin_y) / resolution
        return col, row

    for b in boxes:
        cy, sy = math.cos(b.pose.yaw), math.sin(b.pose.yaw)
        hx, hy = b.sx / 2.0, b.sy / 2.0
        pts = []
        for lx, ly in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]:
            wx = b.pose.x + cy * lx - sy * ly
            wy = b.pose.y + sy * lx + cy * ly
            pts.append(to_pixel(wx, wy))
        draw.polygon(pts, fill=PGM_OCCUPIED)

    for c in cylinders:
        cx_px, cy_px = to_pixel(c.pose.x, c.pose.y)
        r_px = c.radius / resolution
        draw.ellipse([cx_px - r_px, cy_px - r_px,
                      cx_px + r_px, cy_px + r_px],
                     fill=PGM_OCCUPIED)

    return np.array(img, dtype=np.uint8)


def _write_pgm(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(arr, mode='L').save(path, format='PPM')


def _write_yaml(path: Path, image_filename: str, resolution: float,
                origin: Tuple[float, float, float]) -> None:
    body = {
        'image': image_filename,
        'resolution': float(resolution),
        'origin': [float(origin[0]), float(origin[1]), float(origin[2])],
        'occupied_thresh': 0.65,
        'free_thresh': 0.196,
        'negate': 0,
    }
    with path.open('w') as fh:
        yaml.safe_dump(body, fh, sort_keys=False)


def _write_sidecar(path: Path, *, source_world: Path, map_yaml: str,
                   map_pgm: str, resolution: float,
                   origin: Tuple[float, float, float],
                   image_size_px: Tuple[int, int],
                   raw_aabb: Tuple[float, float, float, float],
                   padded_aabb: Tuple[float, float, float, float],
                   padding: float, z_band: Tuple[float, float],
                   origin_mode: str,
                   world_to_map_offset: Tuple[float, float],
                   model_paths: List[Path],
                   stats: ParseStats) -> None:
    body = {
        'schema_version': 2,
        'generated_by': 'world_to_map.rasterize_world',
        'source_world': str(source_world),
        'map_yaml': map_yaml,
        'map_pgm': map_pgm,
        'resolution': float(resolution),
        'origin': [float(origin[0]), float(origin[1]), float(origin[2])],
        'image_size_px': [int(image_size_px[0]), int(image_size_px[1])],
        'world_extent_m': [
            float(image_size_px[0] * resolution),
            float(image_size_px[1] * resolution),
        ],
        'world_aabb': {
            'min': [float(raw_aabb[0]), float(raw_aabb[1])],
            'max': [float(raw_aabb[2]), float(raw_aabb[3])],
        },
        'padded_aabb': {
            'min': [float(padded_aabb[0]), float(padded_aabb[1])],
            'max': [float(padded_aabb[2]), float(padded_aabb[3])],
        },
        'padding_m': float(padding),
        'z_band_m': [float(z_band[0]), float(z_band[1])],
        'origin_mode': origin_mode,
        'world_to_map_offset': [
            float(world_to_map_offset[0]),
            float(world_to_map_offset[1]),
        ],
        'frames': {
            'map_frame': 'map',
            'world_frame': 'world',
            'map_to_odom_translation': [
                float(world_to_map_offset[0]),
                float(world_to_map_offset[1]),
                0.0,
            ],
        },
        'model_paths': [str(p) for p in model_paths],
        'stats': {
            'boxes': stats.boxes,
            'cylinders': stats.cylinders,
            'includes_resolved': stats.includes_resolved,
            'includes_unresolved': stats.includes_unresolved[:50],
            'skipped_mesh': stats.skipped_mesh,
            'skipped_floor': stats.skipped_floor,
            'skipped_other': stats.skipped_other,
        },
    }
    with path.open('w') as fh:
        yaml.safe_dump(body, fh, sort_keys=False)


def _split_model_paths(arg: str) -> List[Path]:
    out: List[Path] = []
    for chunk in arg.split(os.pathsep):
        chunk = chunk.strip()
        if not chunk:
            continue
        p = Path(chunk).expanduser().resolve()
        if p.is_dir():
            out.append(p)
    return out


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--world', required=True, type=Path,
                   help='Path to the Gazebo .world file.')
    p.add_argument('--out', required=True, type=Path,
                   help='Output prefix (no extension). Three files are written.')
    p.add_argument('--resolution', type=float, default=0.05,
                   help='Meters per pixel (the "scale"). Default 0.05.')
    p.add_argument('--padding', type=float, default=1.0,
                   help='Empty border (meters) around the world AABB. Default 1.0.')
    p.add_argument('--z-min', dest='z_min', type=float, default=0.0,
                   help='Robot height band lower bound (m). Default 0.0.')
    p.add_argument('--z-max', dest='z_max', type=float, default=0.4,
                   help='Robot height band upper bound (m). Default 0.4.')
    p.add_argument('--floor-clearance', dest='floor_clearance', type=float,
                   default=0.05,
                   help='Skip box/cylinder collisions whose top is at or '
                        'below this height (m). Filters Workshop-style floor '
                        'slabs that would otherwise fill the entire room. '
                        'Default 0.05.')
    p.add_argument('--model-paths', default=os.environ.get('GAZEBO_MODEL_PATH', ''),
                   help='Colon-separated list of directories containing model '
                        'subfolders. Defaults to $GAZEBO_MODEL_PATH.')
    p.add_argument('--origin-mode', choices=['world', 'bottom-left'],
                   default='bottom-left',
                   help='"world": map frame == Gazebo world frame, origin = '
                        'padded AABB min. "bottom-left" (default): map.yaml '
                        'origin = (0, 0); geometry shifted into the positive '
                        'quadrant; launch must apply the offset to map->odom.')
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    world_path: Path = args.world.expanduser().resolve()
    out_prefix: Path = args.out.expanduser().resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if not world_path.is_file():
        print(f'world file not found: {world_path}', file=sys.stderr)
        return 2

    if args.resolution <= 0.0:
        print('--resolution must be > 0', file=sys.stderr)
        return 2
    if args.z_max < args.z_min:
        print('--z-max must be >= --z-min', file=sys.stderr)
        return 2

    model_paths = _split_model_paths(args.model_paths)
    if not model_paths:
        print('warning: no model paths provided; <include> URIs will not '
              'resolve. Pass --model-paths or set GAZEBO_MODEL_PATH.',
              file=sys.stderr)

    boxes, cylinders, stats = _parse_world(world_path, model_paths,
                                           args.z_min, args.z_max,
                                           args.floor_clearance)

    aabb = _shape_aabb(boxes, cylinders)
    if aabb is None:
        print('no rasterisable geometry found; writing a small empty map.',
              file=sys.stderr)
        aabb = (-1.0, -1.0, 1.0, 1.0)
    raw_aabb = aabb

    pmin_x = aabb[0] - args.padding
    pmin_y = aabb[1] - args.padding
    pmax_x = aabb[2] + args.padding
    pmax_y = aabb[3] + args.padding
    padded_aabb = (pmin_x, pmin_y, pmax_x, pmax_y)

    width_px = max(1, int(math.ceil((pmax_x - pmin_x) / args.resolution)))
    height_px = max(1, int(math.ceil((pmax_y - pmin_y) / args.resolution)))

    if args.origin_mode == 'world':
        origin_xy = (pmin_x, pmin_y)
        rasterize_origin = (pmin_x, pmin_y)
        world_to_map_offset = (0.0, 0.0)
    else:
        origin_xy = (0.0, 0.0)
        rasterize_origin = (pmin_x, pmin_y)
        world_to_map_offset = (-pmin_x, -pmin_y)

    arr = _rasterize(boxes, cylinders, rasterize_origin[0], rasterize_origin[1],
                     args.resolution, width_px, height_px)

    pgm_path = out_prefix.with_suffix('.pgm')
    yaml_path = out_prefix.with_suffix('.yaml')
    sidecar_path = out_prefix.parent / (out_prefix.name + '.world_map.yaml')

    _write_pgm(pgm_path, arr)
    _write_yaml(yaml_path, pgm_path.name, args.resolution,
                (origin_xy[0], origin_xy[1], 0.0))
    _write_sidecar(
        sidecar_path,
        source_world=world_path,
        map_yaml=yaml_path.name,
        map_pgm=pgm_path.name,
        resolution=args.resolution,
        origin=(origin_xy[0], origin_xy[1], 0.0),
        image_size_px=(width_px, height_px),
        raw_aabb=raw_aabb,
        padded_aabb=padded_aabb,
        padding=args.padding,
        z_band=(args.z_min, args.z_max),
        origin_mode=args.origin_mode,
        world_to_map_offset=world_to_map_offset,
        model_paths=model_paths,
        stats=stats,
    )

    print(f'world             : {world_path}')
    print(f'model paths       : {len(model_paths)}')
    for p in model_paths:
        print(f'                    {p}')
    print(f'origin_mode       : {args.origin_mode}')
    print(f'boxes             : {stats.boxes}')
    print(f'cylinders         : {stats.cylinders}')
    print(f'includes_resolved : {stats.includes_resolved}')
    print(f'includes_unresolved: {len(stats.includes_unresolved)}')
    for u in stats.includes_unresolved[:5]:
        print(f'  - {u}')
    if len(stats.includes_unresolved) > 5:
        print(f'  ... ({len(stats.includes_unresolved) - 5} more)')
    print(f'skipped (mesh)    : {stats.skipped_mesh}')
    print(f'skipped (floor)   : {stats.skipped_floor}')
    print(f'skipped (other)   : {stats.skipped_other}')
    print(f'resolution        : {args.resolution} m/px')
    print(f'image_size_px     : {width_px} x {height_px}')
    print(f'world_extent_m    : {width_px * args.resolution:.3f} x '
          f'{height_px * args.resolution:.3f}')
    print(f'origin (map yaml) : ({origin_xy[0]:.3f}, {origin_xy[1]:.3f}, 0.0)')
    print(f'world->map offset : ({world_to_map_offset[0]:.3f}, '
          f'{world_to_map_offset[1]:.3f})  '
          '(applied as static TF map->odom translation)')
    print(f'wrote             : {pgm_path}')
    print(f'wrote             : {yaml_path}')
    print(f'wrote             : {sidecar_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
