"""Portable Fanuc scene data. This module never imports Isaac Lab or ROS."""
from __future__ import annotations

import hashlib
import copy
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np

HAND_SUFFIXES = (
    'thumb_CMC_FE', 'thumb_CMC_AA', 'thumb_MCP_FE', 'thumb_MCP_AA', 'thumb_IP',
    'index_MCP_FE', 'index_MCP_AA', 'index_PIP', 'index_DIP',
    'middle_MCP_FE', 'middle_MCP_AA', 'middle_PIP', 'middle_DIP',
    'ring_MCP_FE', 'ring_MCP_AA', 'ring_PIP', 'ring_DIP',
    'pinky_CMC', 'pinky_MCP_FE', 'pinky_MCP_AA', 'pinky_PIP', 'pinky_DIP',
)


def joint_names(side):
    return [f'J{i}' for i in range(1, 7)] + [f'{side}_{s}' for s in HAND_SUFFIXES]


def initial_joint_positions(names, values, limits=None):
    """Config scalars must be Python floats, not NumPy float64 tensors-in-waiting.

    Isaac Lab constructs a tensor from the config values and index-assigns it
    into its float32 default-state buffer during sim.reset(). NumPy float64
    elements make that inferred source tensor float64 even with a float32 default.
    """
    values = finite(values, 'initial joint positions')
    if values.shape != (len(names),) or len(set(names)) != len(names):
        raise ValueError('Initial joint names and values must be unique and aligned')
    if limits is not None:
        low = np.array([limits[n]['lower'] for n in names])
        high = np.array([limits[n]['upper'] for n in names])
        if np.any(values < low-1e-6) or np.any(values > high+1e-6):
            raise ValueError('Initial joint position genuinely outside URDF limits')
        # URDF radians -> importer degrees -> float32 radians can move a bound
        # by a few ULPs. Start 1e-5 rad inside, never widen the physical limits.
        margin = np.minimum(1e-5, (high-low)/4)
        values = np.clip(values, low+margin, high-margin)
    return {name: float(value) for name, value in zip(names, values)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(array, name):
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f'{name}: non-finite values')
    return array


def rpy_quaternion(pose):
    """XYZ + fixed-axis RPY -> XYZ + WXYZ (Isaac Lab convention)."""
    x, y, z, r, p, yaw = finite(pose, 'base pose')
    cr, cp, cy = np.cos(np.array([r, p, yaw]) / 2)
    sr, sp, sy = np.sin(np.array([r, p, yaw]) / 2)
    return np.array([x, y, z, cr*cp*cy + sr*sp*sy, sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy, cr*cp*sy - sr*sp*cy])


def matrix_pose(matrix):
    """Rigid 4x4 transform -> XYZ + WXYZ, including rotations near pi."""
    m = finite(matrix, 'object transform')
    r = m[:3, :3]
    if (m.shape != (4, 4) or not np.allclose(m[3], [0, 0, 0, 1], atol=1e-5)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-5)):
        raise ValueError('Invalid rigid transform')
    # Symmetric eigensystem is stable at 180 degrees. Eigenvector order XYZW.
    k = np.array([
        [r[0,0]-r[1,1]-r[2,2], r[0,1]+r[1,0], r[0,2]+r[2,0], r[2,1]-r[1,2]],
        [r[0,1]+r[1,0], r[1,1]-r[0,0]-r[2,2], r[1,2]+r[2,1], r[0,2]-r[2,0]],
        [r[0,2]+r[2,0], r[1,2]+r[2,1], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
        [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], np.trace(r)],
    ]) / 3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1][[3, 0, 1, 2]]
    if q[0] < 0:
        q = -q
    return np.concatenate([m[:3, 3], q])


def sample_motion(qpos, poses, fps, time):
    """Clamped linear joints/translation and shortest-path quaternion SLERP."""
    frame = np.clip(time * fps, 0, len(qpos) - 1)
    i = int(frame)
    j = min(i + 1, len(qpos) - 1)
    a = float(frame - i)
    pose = (1-a)*poses[i] + a*poses[j]
    u, v = poses[i, 3:].copy(), poses[j, 3:].copy()
    dot = np.dot(u, v)
    if dot < 0:
        v, dot = -v, -dot
    if dot > .9995:
        quat = (1-a)*u + a*v
    else:
        theta = np.arccos(np.clip(dot, -1, 1))
        quat = (np.sin((1-a)*theta)*u + np.sin(a*theta)*v) / np.sin(theta)
    pose[3:] = quat / np.linalg.norm(quat)
    return (1-a)*qpos[i] + a*qpos[j], pose


def portable_urdf(source, destination, side):
    """Copy only referenced meshes; preserve their explicit scales and origins."""
    source, destination = Path(source), Path(destination)
    root = ET.parse(source).getroot()
    joints = {j.get('name'): j for j in root.findall('joint') if j.get('type') != 'fixed'}
    if set(joints) != set(joint_names(side)):
        raise ValueError(f'{side}: expected exactly 6 CRX + 22 Sharpa joints')
    for j in joints.values():
        if j.get('type') != 'revolute' or j.find('mimic') is not None:
            raise ValueError('Only independent revolute joints supported')
    # The bundled model must be rooted at identity world -> base_link; base
    # placement belongs exclusively to manifest.json, never baked in twice.
    base_joint = root.find("joint[@name='base_joint']")
    if base_joint is None or base_joint.find('parent').get('link') != 'world':
        raise ValueError('Expected a world -> base_link fixed joint')
    origin = base_joint.find('origin')
    if origin is not None:
        for field in ('xyz', 'rpy'):
            if not np.allclose([float(v) for v in origin.get(field, '0 0 0').split()], 0):
                raise ValueError('Nonidentity baked base transform is unsupported')
    destination.parent.mkdir(parents=True, exist_ok=True)
    for mesh in root.iter('mesh'):
        name = mesh.get('filename')
        if '://' in name:
            raise ValueError(f'Expected resolved relative mesh, got {name}')
        path = (source.parent / name).resolve()
        data = path.read_bytes()
        if data.startswith(b'version https://git-lfs'):
            raise ValueError(f'Missing Git LFS payload: {path}')
        # External textures would also need bundling. Do not silently omit them.
        if path.suffix.lower() == '.dae':
            dae = ET.fromstring(data)
            if any(e.tag.endswith('image') for e in dae.iter()):
                raise ValueError(f'DAE textures require explicit bundling: {path}')
        target = destination.parent / 'meshes' / (sha256(path)[:16] + path.suffix.lower())
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)
        mesh.set('filename', target.relative_to(destination.parent).as_posix())
    # Remove MuJoCo-only metadata; no effect on URDF kinematics/inertias.
    for element in list(root):
        if element.tag == 'mujoco':
            root.remove(element)
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding='utf-8', xml_declaration=True)
    return {name: {field: float(joints[name].find('limit').get(field))
                   for field in ('lower', 'upper', 'effort', 'velocity')}
            for name in joint_names(side)}


def stage_import_urdf(source, destination, collision_visuals=False):
    """Normalize importer-sensitive metadata without editing the signed bundle.

    Remove only the identity, massless world wrapper; preserve base_link and
    every physical link/joint transform. Give anonymous materials and hashed
    mesh basenames valid, nonempty USD identifiers for Isaac Sim 5.1.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    root = ET.parse(source).getroot()
    wrapper = root.find("link[@name='world']")
    base = root.find("joint[@name='base_joint']")
    if wrapper is None or len(wrapper) or base is None or base.get('type') != 'fixed':
        raise ValueError('Expected empty world wrapper and fixed base_joint')
    if (base.find('parent').get('link') != 'world'
            or base.find('child').get('link') != 'base_link'):
        raise ValueError('Unexpected base wrapper topology')
    for joint in root.findall('joint'):
        if joint is not base and (joint.find('parent').get('link') == 'world'
                                  or joint.find('child').get('link') == 'world'):
            raise ValueError('World wrapper has additional branches')
    origin = base.find('origin')
    if origin is not None:
        for field in ('xyz', 'rpy'):
            if not np.allclose([float(v) for v in origin.get(field, '0 0 0').split()], 0):
                raise ValueError('Cannot remove nonidentity world wrapper')
    root.remove(wrapper)
    root.remove(base)
    if collision_visuals:
        # DAE files contain nested scene references. For rendering diagnostics,
        # replace ONLY those visuals with the existing collision representation,
        # including its origin and scale (not merely the filename).
        for link in root.findall('link'):
            dae_visuals = [v for v in link.findall('visual') if any(
                Path(m.get('filename')).suffix.lower() == '.dae' for m in v.iter('mesh'))]
            if not dae_visuals:
                continue
            collisions = link.findall('collision')
            if not collisions:
                raise ValueError(f"No collision visual alternative for {link.get('name')}")
            for visual in dae_visuals:
                link.remove(visual)
            for collision in collisions:
                visual = copy.deepcopy(collision)
                visual.tag = 'visual'
                visual.attrib.pop('name', None)
                link.append(visual)
    material_names = {m.get('name') for m in root.iter('material') if m.get('name')}
    for index, material in enumerate(root.iter('material')):
        if not material.get('name'):
            name = f'fanuc_import_material_{index}'
            while name in material_names:
                name += '_'
            material.set('name', name)
            material_names.add(name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for mesh in root.iter('mesh'):
        original = (source.parent / mesh.get('filename')).resolve()
        name = 'mesh_' + re.sub(r'[^a-zA-Z0-9_]', '_', original.stem) + original.suffix.lower()
        target = destination.parent / 'meshes' / name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            shutil.copyfile(original, target)
        mesh.set('filename', target.relative_to(destination.parent).as_posix())
    ET.indent(root)
    ET.ElementTree(root).write(destination, encoding='utf-8', xml_declaration=True)
    return destination


def prepare_bundle(retargeting_root, trajectory, reference, output, table_bounds):
    root, output = Path(retargeting_root).resolve(), Path(output).resolve()
    trajectory, reference = Path(trajectory).resolve(), Path(reference).resolve()
    if output.exists():
        raise FileExistsError(f'Use a new output directory: {output}')
    with np.load(trajectory, allow_pickle=False) as f:
        q = finite(f['qpos56'], 'qpos56')
        if q.ndim != 2 or q.shape[1] != 56 or len(q) < 2:
            raise ValueError('Expected qpos56 (T>=2, 56)')
        if set(f['sides'].tolist()) != {'left', 'right'}:
            raise ValueError('Both retargeted sides required; inactive home poses are not a trajectory')
        fps = float(f['fps'])
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid trajectory fps')
        poses = np.stack([matrix_pose(m) for m in f['object_object_0_scene_poses']])
        if len(poses) != len(q):
            raise ValueError('Robot/object trajectory lengths differ')
        bases = {side: rpy_quaternion(f[f'{side}_base']).tolist() for side in ('left', 'right')}
        height = float(f['table_height'])
        metadata = {k: f[k].tolist() for k in ('scene_transform', 'time_scale', 'source_table_z')}
        ik_summary = {}
        for side in ('left', 'right'):
            diagnostic = finite(f[f'{side}_diagnostics'], f'{side} IK diagnostics')
            if diagnostic.shape != (len(q), 4):
                raise ValueError(f'{side}: invalid IK diagnostic shape')
            if (diagnostic[:, 0].max() > .001 or diagnostic[:, 1].max() > .01
                    or np.any(diagnostic[:, 3] != 0)):
                raise ValueError(f'{side}: failed IK or near-limit trajectory')
            ik_summary[side] = {'max_position_residual_m': float(diagnostic[:, 0].max()),
                                'max_rotation_residual_rad': float(diagnostic[:, 1].max())}
    bounds = finite(table_bounds, 'table bounds')
    if bounds.shape != (4,) or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError('Expected xmin < xmax, ymin < ymax')
    if not math.isfinite(height):
        raise ValueError('Invalid table height')
    with h5py.File(reference, 'r') as f:
        if f['object_tracks/0/log_path'][()].decode() != 's1/hammer_use_1':
            raise ValueError('This initial scene supports the hammer_use_1 object only')
        vertices = finite(f['object_tracks/0/vertices'][:], 'vertices')
        faces = f['object_tracks/0/faces'][:]
        if (vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2
                or faces.shape[1] != 3 or not np.issubdtype(faces.dtype, np.integer)
                or faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError('Invalid triangle mesh')
        reference_frames = len(f['qpos'])
        reference_fps = f['fps'][:].tolist()
    output.mkdir(parents=True)
    try:
        limits, urdf_hashes = {}, {}
        for index, side in enumerate(('left', 'right')):
            urdf = root / f'assets/robots/crx5ia_sharpa_wave_{side}/urdf/crx5ia_sharpa_wave_{side}.urdf'
            urdf_hashes[side] = sha256(urdf)
            limits[side] = portable_urdf(urdf, output / side / 'robot.urdf', side)
            low = np.array([v['lower'] for v in limits[side].values()])
            high = np.array([v['upper'] for v in limits[side].values()])
            values = q[:, index*28:(index+1)*28]
            if np.any(values < low - 1e-6) or np.any(values > high + 1e-6):
                raise ValueError(f'{side}: trajectory violates joint position bounds')
        np.savez_compressed(output / 'motion.npz', qpos56=q, object_pose_wxyz=poses, fps=fps)
        np.savez_compressed(output / 'hammer_mesh.npz', vertices=vertices, faces=faces)
        manifest = {
            'schema_version': 1, 'bases_xyz_wxyz': bases, 'table_height': height,
            'table_bounds_xy': bounds.tolist(), 'table_thickness': .03,
            'table_bounds_measured': False, 'mount_calibrated': False,
            'frames': len(q), 'fps': fps, 'duration_seconds': (len(q)-1)/fps,
            'joint_names': {s: joint_names(s) for s in ('left', 'right')},
            'joint_limits': limits, 'reference_frames': reference_frames,
            'reference_fps': reference_fps, 'source_metadata': metadata,
            'source_sha256': {'trajectory': sha256(trajectory), 'reference': sha256(reference)},
            'source_urdf_sha256': urdf_hashes, 'ik_diagnostics': ik_summary,
            'notes': ['NPZ bases take precedence over upstream launch defaults.',
                      'Object mesh comes from reference H5; object poses come from NPZ.',
                      'Recorded slowdown is preserved, not an original-speed task.',
                      'No physical calibration or grasp success is asserted.'],
        }
        # Preserve licenses from source asset bundles (meshes can have distinct terms).
        for source in (root / 'LICENSE', root / 'THIRD_PARTY_NOTICES.md'):
            if source.is_file():
                shutil.copyfile(source, output / source.name)
        for name in ('crx5ia_leap_paxini', 'sharpa_wave_left', 'sharpa_wave_right'):
            asset = root / 'assets/robots' / name
            for source in asset.rglob('*'):
                if source.is_file() and (source.suffix.lower() in ('.md', '.txt', '.yaml') or 'license' in source.name.lower()):
                    target = output / 'provenance' / name / source.relative_to(asset)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, target)
        manifest['files_sha256'] = {p.relative_to(output).as_posix(): sha256(p)
                                   for p in sorted(output.rglob('*')) if p.is_file()}
        (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    except Exception:
        # Keep failed staging data reviewable, never claim a valid bundle.
        (output / 'PREPARATION_FAILED').write_text('Preparation failed; use a new directory to retry.')
        raise
    return manifest


def load_bundle(path):
    path = Path(path).resolve()
    if (path / 'PREPARATION_FAILED').exists():
        raise ValueError('Incomplete bundle')
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema_version'] != 1:
        raise ValueError('Unsupported bundle version')
    for name, digest in manifest['files_sha256'].items():
        candidate = (path / name).resolve()
        if not candidate.is_relative_to(path) or sha256(candidate) != digest:
            raise ValueError(f'Bundle integrity failure: {name}')
    with np.load(path / 'motion.npz', allow_pickle=False) as f:
        q, poses, fps = f['qpos56'], f['object_pose_wxyz'], float(f['fps'])
    with np.load(path / 'hammer_mesh.npz', allow_pickle=False) as f:
        vertices, faces = f['vertices'], f['faces']
    return manifest, q, poses, fps, vertices, faces
