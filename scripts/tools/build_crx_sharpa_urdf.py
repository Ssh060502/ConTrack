"""Merge the two single-arm CRX-5iA + Sharpa URDFs into one dual-arm training asset.

The RB-Y1A task drives ONE articulation holding both arms. This writes the CRX
equivalent: a fixed root link with the left and right arm bases attached at
their scene poses, so the CRX task can reuse that structure.

    assets/urdf/crx_sharpa.urdf
    assets/meshes/crx_sharpa/*

Arm links and joints get a side prefix (J1 -> left_J1 / right_J1, the names used
by dual_crx_control). Sharpa names already carry one and are unchanged. Needs
NumPy only: no Isaac Lab. Convert the result with scripts/tools/convert_urdf.py.

The base poses are baked into the asset. A reference H5 retargeted with a
different right base does not match it: rebuild and reconvert.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT_LINK = 'crx_base'
ARM_JOINTS = [f'J{i}' for i in range(1, 7)]


def quaternion_rpy(wxyz):
    """Unit WXYZ quaternion -> fixed-axis roll/pitch/yaw (URDF origin convention)."""
    w, x, y, z = np.asarray(wxyz, dtype=float) / np.linalg.norm(wxyz)
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1., 1.))
    return np.array([np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)), pitch,
                     np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))])


def prefixed(name, side):
    name = name if name.startswith(side + '_') else f'{side}_{name}'
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def side_elements(urdf, side, mesh_dir, mesh_prefix):
    """Renamed links/joints of one side, without its world wrapper."""
    root = ET.parse(urdf).getroot()
    wrapper = root.find("joint[@name='base_joint']")
    if (root.find("link[@name='world']") is None or wrapper is None
            or wrapper.find('parent').get('link') != 'world'
            or wrapper.find('child').get('link') != 'base_link'):
        raise ValueError(f'{side}: expected a world -> base_link wrapper joint')
    origin = wrapper.find('origin')
    if origin is not None and not np.allclose(
            [float(v) for f in ('xyz', 'rpy') for v in origin.get(f, '0 0 0').split()], 0):
        raise ValueError(f'{side}: base pose must not already be baked into the source URDF')
    movable = [j.get('name') for j in root.findall('joint') if j.get('type') != 'fixed']
    if movable[:6] != ARM_JOINTS or len(movable) != 28 or any(
            not n.startswith(side + '_') for n in movable[6:]):
        raise ValueError(f'{side}: expected J1..J6 followed by 22 {side}_ Sharpa joints')
    elements = []
    for element in root:
        if element.tag not in ('link', 'joint') or element is wrapper or (
                element.tag == 'link' and element.get('name') == 'world'):
            continue
        element = copy.deepcopy(element)
        element.set('name', prefixed(element.get('name'), side))
        for tag in ('parent', 'child'):
            for reference in element.findall(tag):
                reference.set('link', prefixed(reference.get('link'), side))
        for index, material in enumerate(element.iter('material')):
            # Anonymous materials become empty USD identifiers in the importer.
            material.set('name', f"{element.get('name')}_material_{index}")
        for mesh in element.iter('mesh'):
            source = (Path(urdf).parent / mesh.get('filename')).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            # Hashed basenames may start with a digit; USD prim names may not.
            target = mesh_dir / ('mesh_' + re.sub(r'[^a-zA-Z0-9_]', '_', source.stem) + source.suffix.lower())
            if not target.exists():
                shutil.copyfile(source, target)
            mesh.set('filename', f'{mesh_prefix}/{target.name}')
        elements.append(element)
    return elements, [prefixed(n, side) for n in movable]


def build(bundle, output, mesh_dir, bases):
    """bases: side -> xyz + rpy (6). Returns the joint names per side."""
    bundle, output, mesh_dir = Path(bundle), Path(output), Path(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    mesh_prefix = Path(*(['..'] * len(output.parent.relative_to(REPO / 'assets').parts)),
                       mesh_dir.relative_to(REPO / 'assets')).as_posix()
    robot = ET.Element('robot', name='crx5ia_sharpa_wave_dual')
    ET.SubElement(robot, 'link', name=ROOT_LINK)
    names = {}
    for side in ('left', 'right'):
        pose = np.asarray(bases[side], dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError(f'{side}: base must be six finite values')
        joint = ET.SubElement(robot, 'joint', name=f'{side}_base_mount', type='fixed')
        ET.SubElement(joint, 'origin', xyz=' '.join(repr(float(v)) for v in pose[:3]),
                      rpy=' '.join(repr(float(v)) for v in pose[3:]))
        ET.SubElement(joint, 'parent', link=ROOT_LINK)
        ET.SubElement(joint, 'child', link=f'{side}_base_link')
        elements, names[side] = side_elements(bundle / side / 'robot.urdf', side, mesh_dir, mesh_prefix)
        robot.extend(elements)
    link_names = [e.get('name') for e in robot.findall('link')]
    joint_names = [e.get('name') for e in robot.findall('joint')]
    if len(set(link_names)) != len(link_names) or len(set(joint_names)) != len(joint_names):
        raise ValueError('Name collision between the two sides')
    ET.indent(robot)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(robot).write(output, encoding='utf-8', xml_declaration=True)
    return names


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--bundle', type=Path, required=True,
                        help='Prepared scene bundle containing left/ and right/ robot.urdf')
    parser.add_argument('--output', type=Path, default=REPO / 'assets/urdf/crx_sharpa.urdf')
    parser.add_argument('--mesh-dir', type=Path, default=REPO / 'assets/meshes/crx_sharpa')
    for side in ('left', 'right'):
        parser.add_argument(f'--{side}-base', type=float, nargs=6,
                            metavar=('X', 'Y', 'Z', 'ROLL', 'PITCH', 'YAW'),
                            help='Meters/radians; default is the pose recorded in the bundle manifest')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    output, mesh_dir = args.output.resolve(), args.mesh_dir.resolve()
    for path in (output, mesh_dir):
        if not path.is_relative_to(REPO / 'assets'):
            parser.error(f'{path} must be inside {REPO / "assets"}')
    if output.exists() and not args.overwrite:
        parser.error(f'{output} exists; pass --overwrite')
    manifest = json.loads((args.bundle / 'manifest.json').read_text(encoding='utf-8'))
    bases = {}
    for side in ('left', 'right'):
        given = getattr(args, f'{side}_base')
        recorded = manifest['bases_xyz_wxyz'][side]
        bases[side] = np.asarray(given, float) if given is not None else np.r_[
            recorded[:3], quaternion_rpy(recorded[3:])]
    names = build(args.bundle.resolve(), output, mesh_dir, bases)
    record = {'urdf': output.name, 'root_link': ROOT_LINK,
              'bases_xyz_rpy': {s: [float(v) for v in bases[s]] for s in bases},
              'joint_names': names, 'source_bundle_sha256': manifest.get('source_urdf_sha256'),
              'right_base_verified': False}
    output.with_suffix('.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'Wrote {output} (56 movable joints) and {len(list(mesh_dir.iterdir()))} meshes in {mesh_dir}')
    for side in bases:
        print(f'  {side} base xyz/rpy: {np.round(bases[side], 6).tolist()}')


if __name__ == '__main__':
    main()
