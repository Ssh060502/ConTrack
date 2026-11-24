"""Build the CRX-5iA + Sharpa training assets from the two single-arm source URDFs.

The RB-Y1A task drives ONE articulation holding both arms. This writes the CRX
equivalent, plus calibrated single-arm models for offline retargeting:

    assets/urdf/crx_sharpa.urdf          both arms on one fixed root (training)
    assets/urdf/crx_sharpa_left.urdf     one arm, base at identity (retargeting)
    assets/urdf/crx_sharpa_right.urdf
    assets/urdf/crx_sharpa.json          what was baked in, with file hashes
    assets/meshes/crx_sharpa/*

Two lab calibrations are baked in: the right arm base pose relative to the left
base, and each hand's mount on its flange (ee_mount -> hand_flange, the joint
bimanual_manipulation calls <side>_arm_to_hand). Retarget references with the
single-arm files written here so the reference and the simulated robot agree.
When a calibration changes: rebuild, reconvert the USD, retarget again.

In the dual model, arm links and joints get a side prefix (J1 -> left_J1 /
right_J1, the names used by dual_crx_control). Sharpa names already carry one.
Needs NumPy only: no Isaac Lab. Convert crx_sharpa.urdf with convert_urdf.py.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ROOT_LINK = 'crx_base'
ARM_JOINTS = [f'J{i}' for i in range(1, 7)]
DEFAULT_CALIBRATION = REPO / 'assets/calibration/dual_crx_sharpa_d415_20261122_v1.json'


def quaternion_rpy(wxyz):
    """Unit WXYZ quaternion -> fixed-axis roll/pitch/yaw (URDF origin convention)."""
    w, x, y, z = np.asarray(wxyz, dtype=float) / np.linalg.norm(wxyz)
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1., 1.))
    return np.array([np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)), pitch,
                     np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))])


def pose_matrix(xyz, rpy):
    """URDF origin (R = Rz(yaw) @ Ry(pitch) @ Rx(roll)) -> 4x4."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    matrix = np.eye(4)
    matrix[:3, :3] = [[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                      [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                      [-sp, cp*sr, cp*cr]]
    matrix[:3, 3] = xyz
    return matrix


def matrix_xyz_rpy(matrix):
    """4x4 -> xyz, rpy. Hand mounts pitch close to +-90 deg, so handle that branch."""
    r = matrix[:3, :3]
    pitch = np.arcsin(np.clip(-r[2, 0], -1., 1.))
    if abs(r[2, 0]) > 1 - 1e-9:  # gimbal lock: only roll -/+ yaw is defined
        return matrix[:3, 3].copy(), np.array([np.arctan2(-r[1, 2], r[1, 1]), pitch, 0.])
    return matrix[:3, 3].copy(), np.array([np.arctan2(r[2, 1], r[2, 2]), pitch,
                                           np.arctan2(r[1, 0], r[0, 0])])


def origin_matrix(joint):
    origin = joint.find('origin')
    if origin is None:
        return np.eye(4)
    return pose_matrix([float(v) for v in origin.get('xyz', '0 0 0').split()],
                       [float(v) for v in origin.get('rpy', '0 0 0').split()])


def format_values(values):
    return ' '.join(repr(float(v)) for v in values)


def apply_hand_mount(root, side, mount):
    """Set the calibrated ee_mount -> <side>_hand_flange transform. Returns what changed.

    The source model routes the hand through an intermediate fixed frame, so the
    existing mount joint is rewritten relative to its own parent. Both that parent
    and ee_mount are rigidly attached to the flange link.
    """
    joints = {j.find('child').get('link'): j for j in root.findall('joint')}

    def pose_in_flange(link):
        pose = np.eye(4)
        while link != 'flange':
            joint = joints.get(link)
            if joint is None or joint.get('type') != 'fixed':
                raise ValueError(f'{side}: {link} is not rigidly attached to flange')
            pose = origin_matrix(joint) @ pose
            link = joint.find('parent').get('link')
        return pose
    mount_joint = joints[f'{side}_hand_flange']
    parent = mount_joint.find('parent').get('link')
    nominal = np.linalg.inv(pose_in_flange('ee_mount')) @ pose_in_flange(parent) @ origin_matrix(mount_joint)
    calibrated = pose_matrix(mount['xyz_m'], mount['rpy_rad'])
    xyz, rpy = matrix_xyz_rpy(np.linalg.inv(pose_in_flange(parent)) @ pose_in_flange('ee_mount') @ calibrated)
    origin = mount_joint.find('origin')
    if origin is None:
        origin = ET.SubElement(mount_joint, 'origin')
    origin.set('xyz', format_values(xyz))
    origin.set('rpy', format_values(rpy))
    achieved = np.linalg.inv(pose_in_flange('ee_mount')) @ pose_in_flange(parent) @ origin_matrix(mount_joint)
    if not np.allclose(achieved, calibrated, atol=1e-9):
        raise RuntimeError(f'{side}: failed to reproduce the calibrated hand mount')
    delta = nominal[:3, :3].T @ calibrated[:3, :3]
    return {'rotation_from_nominal_deg': float(np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1)))),
            'translation_from_nominal_m': float(np.linalg.norm(calibrated[:3, 3] - nominal[:3, 3]))}


def prefixed(name, side):
    name = name if name.startswith(side + '_') else f'{side}_{name}'
    return re.sub(r'[^a-zA-Z0-9_]', '_', name)


def load_side(urdf, side, mesh_dir, mesh_prefix, mount):
    """Source model of one side with calibrated mount, named materials and repo mesh paths."""
    root = ET.parse(urdf).getroot()
    wrapper = root.find("joint[@name='base_joint']")
    if (root.find("link[@name='world']") is None or wrapper is None
            or wrapper.find('parent').get('link') != 'world'
            or wrapper.find('child').get('link') != 'base_link'):
        raise ValueError(f'{side}: expected a world -> base_link wrapper joint')
    if not np.allclose(origin_matrix(wrapper), np.eye(4)):
        raise ValueError(f'{side}: base pose must not already be baked into the source URDF')
    movable = [j.get('name') for j in root.findall('joint') if j.get('type') != 'fixed']
    if movable[:6] != ARM_JOINTS or len(movable) != 28 or any(
            not n.startswith(side + '_') for n in movable[6:]):
        raise ValueError(f'{side}: expected J1..J6 followed by 22 {side}_ Sharpa joints')
    change = apply_hand_mount(root, side, mount) if mount is not None else None
    for element in root:
        for index, material in enumerate(element.iter('material')):
            # Anonymous materials become empty USD identifiers in the importer.
            material.set('name', f"{prefixed(element.get('name'), side)}_material_{index}")
        for mesh in element.iter('mesh'):
            source = (Path(urdf).parent / mesh.get('filename')).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)
            # Hashed basenames may start with a digit; USD prim names may not.
            target = mesh_dir / ('mesh_' + re.sub(r'[^a-zA-Z0-9_]', '_', source.stem) + source.suffix.lower())
            if not target.exists():
                shutil.copyfile(source, target)
            mesh.set('filename', f'{mesh_prefix}/{target.name}')
    return root, movable, change


def dual_elements(root, side):
    """Renamed links/joints of one side, without its world wrapper."""
    elements = []
    for element in root:
        if element.tag not in ('link', 'joint') or element.get('name') in ('world', 'base_joint'):
            continue
        element = copy.deepcopy(element)
        element.set('name', prefixed(element.get('name'), side))
        for tag in ('parent', 'child'):
            for reference in element.findall(tag):
                reference.set('link', prefixed(reference.get('link'), side))
        elements.append(element)
    return elements


def write(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
    # Hash with LF line endings so a Windows checkout gives the same value.
    return hashlib.sha256(path.read_bytes().replace(bytes([13, 10]), bytes([10]))).hexdigest()


def build(bundle, output, mesh_dir, bases, mounts=None):
    """bases: side -> xyz + rpy (6). mounts: side -> {'xyz_m', 'rpy_rad'} or None."""
    bundle, output, mesh_dir = Path(bundle), Path(output), Path(mesh_dir)
    mesh_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh_prefix = Path(*(['..'] * len(output.parent.relative_to(REPO / 'assets').parts)),
                       mesh_dir.relative_to(REPO / 'assets')).as_posix()
    robot = ET.Element('robot', name='crx5ia_sharpa_wave_dual')
    ET.SubElement(robot, 'link', name=ROOT_LINK)
    result = {'joint_names': {}, 'single_arm_urdf': {}, 'hand_mount_change': {}}
    for side in ('left', 'right'):
        pose = np.asarray(bases[side], dtype=float)
        if pose.shape != (6,) or not np.isfinite(pose).all():
            raise ValueError(f'{side}: base must be six finite values')
        joint = ET.SubElement(robot, 'joint', name=f'{side}_base_mount', type='fixed')
        ET.SubElement(joint, 'origin', xyz=format_values(pose[:3]), rpy=format_values(pose[3:]))
        ET.SubElement(joint, 'parent', link=ROOT_LINK)
        ET.SubElement(joint, 'child', link=f'{side}_base_link')
        root, movable, change = load_side(bundle / side / 'robot.urdf', side, mesh_dir, mesh_prefix,
                                          None if mounts is None else mounts[side])
        robot.extend(dual_elements(root, side))
        single = output.with_name(f'{output.stem}_{side}.urdf')
        result['single_arm_urdf'][side] = {'file': single.name, 'sha256': write(root, single)}
        result['joint_names'][side] = [prefixed(n, side) for n in movable]
        result['hand_mount_change'][side] = change
    link_names = [e.get('name') for e in robot.findall('link')]
    joint_names = [e.get('name') for e in robot.findall('joint')]
    if len(set(link_names)) != len(link_names) or len(set(joint_names)) != len(joint_names):
        raise ValueError('Name collision between the two sides')
    result['sha256'] = write(robot, output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--bundle', type=Path, default=REPO / 'outputs/fanuc_sharpa/hammer_z175_v1',
                        help='Prepared scene bundle containing left/ and right/ robot.urdf')
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION,
                        help='Lab calibration JSON: right base pose and hand mounts')
    parser.add_argument('--no-calibration', action='store_true',
                        help='Nominal hand mounts and the base poses recorded in the bundle')
    parser.add_argument('--output', type=Path, default=REPO / 'assets/urdf/crx_sharpa.urdf')
    parser.add_argument('--mesh-dir', type=Path, default=REPO / 'assets/meshes/crx_sharpa')
    for side in ('left', 'right'):
        parser.add_argument(f'--{side}-base', type=float, nargs=6,
                            metavar=('X', 'Y', 'Z', 'ROLL', 'PITCH', 'YAW'),
                            help='Meters/radians; overrides the calibration file')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    output, mesh_dir = args.output.resolve(), args.mesh_dir.resolve()
    for path in (output, mesh_dir):
        if not path.is_relative_to(REPO / 'assets'):
            parser.error(f'{path} must be inside {REPO / "assets"}')
    if output.exists() and not args.overwrite:
        parser.error(f'{output} exists; pass --overwrite')
    manifest = json.loads((args.bundle / 'manifest.json').read_text(encoding='utf-8'))
    bases = {side: np.r_[manifest['bases_xyz_wxyz'][side][:3],
                         quaternion_rpy(manifest['bases_xyz_wxyz'][side][3:])] for side in ('left', 'right')}
    mounts = calibration_id = None
    if not args.no_calibration:
        calibration = json.loads(args.calibration.read_text(encoding='utf-8'))
        calibration_id = calibration['calibration_id']
        authoritative = calibration['authoritative']
        right = authoritative['dual_crx']['T_leftbase_rightbase']
        bases = {'left': np.zeros(6), 'right': np.r_[right['xyz_m'], right['rpy_rad']]}
        mounts = {side: authoritative['hand_mounts'][side] for side in ('left', 'right')}
    for side in ('left', 'right'):
        if getattr(args, f'{side}_base') is not None:
            bases[side] = np.asarray(getattr(args, f'{side}_base'), float)
    result = build(args.bundle.resolve(), output, mesh_dir, bases, mounts)
    record = {'urdf': output.name, 'sha256': result['sha256'], 'root_link': ROOT_LINK,
              'calibration_id': calibration_id,
              'bases_xyz_rpy': {s: [float(v) for v in bases[s]] for s in bases},
              'hand_mounts_ee_mount_to_hand_flange': mounts,
              'hand_mount_change_from_nominal': result['hand_mount_change'],
              'single_arm_urdf': result['single_arm_urdf'],
              'joint_names': result['joint_names'],
              'source_bundle_sha256': manifest.get('source_urdf_sha256')}
    output.with_suffix('.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(f'Wrote {output} (56 movable joints), two single-arm models and '
          f'{len(list(mesh_dir.iterdir()))} meshes in {mesh_dir}')
    print(f'  calibration: {calibration_id or "none (nominal mounts, bundle bases)"}')
    for side in bases:
        print(f'  {side} base xyz/rpy: {np.round(bases[side], 6).tolist()}')
        change = result['hand_mount_change'][side]
        if change:
            print(f"  {side} hand mount moved {change['rotation_from_nominal_deg']:.2f} deg and "
                  f"{change['translation_from_nominal_m']*1000:.2f} mm from the nominal model")


if __name__ == '__main__':
    main()
