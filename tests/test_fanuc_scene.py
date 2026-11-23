"""Offline contracts: geometry, timing, joint ordering and portable scene assets."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/tools'))
from fanuc_scene_data import (joint_names, load_bundle, matrix_pose, portable_urdf,
                              prepare_bundle, rpy_quaternion, sample_motion)


class FanucSceneTests(unittest.TestCase):
    def make_inputs(self, root):
        retarget = root / 'retarget'
        for side in ('left', 'right'):
            directory = retarget / f'assets/robots/crx5ia_sharpa_wave_{side}/urdf'
            directory.mkdir(parents=True)
            robot = ET.Element('robot', name=side)
            ET.SubElement(robot, 'link', name='world')
            ET.SubElement(robot, 'link', name='base_link')
            j = ET.SubElement(robot, 'joint', name='base_joint', type='fixed')
            ET.SubElement(j, 'parent', link='world')
            ET.SubElement(j, 'child', link='base_link')
            ET.SubElement(j, 'origin', xyz='0 0 0', rpy='0 0 0')
            parent = 'base_link'
            for index, name in enumerate(joint_names(side)):
                child = f'link_{index}'
                link = ET.SubElement(robot, 'link', name=child)
                geometry = ET.SubElement(ET.SubElement(link, 'collision'), 'geometry')
                ET.SubElement(geometry, 'mesh', filename='piece.stl', scale='.001 .001 .001')
                j = ET.SubElement(robot, 'joint', name=name, type='revolute')
                ET.SubElement(j, 'parent', link=parent)
                ET.SubElement(j, 'child', link=child)
                ET.SubElement(j, 'axis', xyz='0 0 1')
                ET.SubElement(j, 'limit', lower='-2', upper='2', effort='1', velocity='3')
                parent = child
            (directory / 'piece.stl').write_bytes(b'solid test\nendsolid test\n')
            ET.ElementTree(robot).write(directory / f'crx5ia_sharpa_wave_{side}.urdf')
        trajectory = root / 'motion.npz'
        np.savez(trajectory, qpos56=np.zeros((3, 56)), fps=10., sides=['left', 'right'],
                 object_object_0_scene_poses=np.tile(np.eye(4), (3, 1, 1)),
                 left_base=np.zeros(6), right_base=[.03, -.91, .01, 0, 0, -1.5],
                 table_height=-.175, scene_transform=np.zeros(6), time_scale=7., source_table_z=0.,
                 left_diagnostics=np.zeros((3, 4)), right_diagnostics=np.zeros((3, 4)))
        reference = root / 'reference.h5'
        with h5py.File(reference, 'w') as f:
            f['object_tracks/0/log_path'] = np.bytes_('s1/hammer_use_1')
            f['object_tracks/0/vertices'] = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]
            f['object_tracks/0/faces'] = np.array([[0, 1, 2]], dtype=np.uint32)
            f['qpos'] = np.zeros((20, 2, 29))
            f['fps'] = [120., 120.]
        return retarget, trajectory, reference

    def test_slerp_shortest_path_and_physical_time(self):
        q = np.array([[0.]*56, [2.]*56])
        poses = np.array([[0., 0, 0, 1, 0, 0, 0], [2., 0, 0, 0, 0, 0, -1.]])
        values, pose = sample_motion(q, poses, 2., .25)
        np.testing.assert_allclose(values, 1.)
        np.testing.assert_allclose(pose, [1., 0, 0, 2**-.5, 0, 0, -2**-.5], atol=1e-12)
        np.testing.assert_allclose(sample_motion(q, poses, 2., 10)[0], q[-1])

    def test_antipodal_quaternions_are_same_orientation(self):
        q = np.zeros((2, 56))
        poses = np.array([[0., 0, 0, 1, 0, 0, 0], [0., 0, 0, -1, 0, 0, 0]])
        np.testing.assert_allclose(sample_motion(q, poses, 1., .5)[1][3:], [1, 0, 0, 0])

    def test_transform_conventions_and_pi_rotation(self):
        matrix = np.diag([1., -1, -1, 1])
        matrix[:3, 3] = [1, 2, 3]
        np.testing.assert_allclose(matrix_pose(matrix), [1, 2, 3, 0, 1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(rpy_quaternion([1, 2, 3, np.pi, 0, 0]),
                                   [1, 2, 3, 0, 1, 0, 0], atol=1e-12)
        matrix[0, 0] = 2
        with self.assertRaises(ValueError):
            matrix_pose(matrix)

    def test_portable_bundle_keeps_npz_bases_and_independent_timebases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self.make_inputs(root)
            output = root / 'bundle'
            manifest = prepare_bundle(*inputs, output, [0, 1, -1, 0])
            loaded, q, poses, fps, vertices, faces = load_bundle(output)
            self.assertEqual(q.shape, (3, 56))
            self.assertEqual(loaded['reference_frames'], 20)
            self.assertEqual(fps, 10.)
            self.assertAlmostEqual(loaded['duration_seconds'], .2)
            np.testing.assert_allclose(loaded['bases_xyz_wxyz']['right'][:3], [.03, -.91, .01])
            urdf = ET.parse(output / 'right/robot.urdf')
            for mesh in urdf.iter('mesh'):
                self.assertTrue((output / 'right' / mesh.get('filename')).is_file())
                self.assertEqual(mesh.get('scale'), '.001 .001 .001')
            self.assertFalse(manifest['mount_calibrated'])
            with self.assertRaises(FileExistsError):
                prepare_bundle(*inputs, output, [0, 1, -1, 0])

    def test_tampered_mesh_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self.make_inputs(root)
            output = root / 'bundle'
            prepare_bundle(*inputs, output, [0, 1, -1, 0])
            next((output / 'left/meshes').iterdir()).write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                load_bundle(output)

    def test_invalid_ik_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = self.make_inputs(root)
            with np.load(inputs[1]) as f:
                values = {k: f[k] for k in f.files}
            values['right_diagnostics'][1, 0] = .1
            np.savez(inputs[1], **values)
            with self.assertRaisesRegex(ValueError, 'failed IK'):
                prepare_bundle(*inputs, root / 'bundle', [0, 1, -1, 0])
            self.assertFalse((root / 'bundle').exists())

    def test_nonidentity_baked_base_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            retarget, _, _ = self.make_inputs(root)
            source = retarget / 'assets/robots/crx5ia_sharpa_wave_left/urdf/crx5ia_sharpa_wave_left.urdf'
            tree = ET.parse(source)
            tree.find("joint[@name='base_joint']/origin").set('xyz', '1 0 0')
            tree.write(source)
            with self.assertRaisesRegex(ValueError, 'baked base'):
                portable_urdf(source, root / 'new/robot.urdf', 'left')


if __name__ == '__main__':
    unittest.main()
