"""Offline regression tests; numpy/h5py only, no Isaac Sim or torch required."""
import importlib.util
import inspect
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import h5py
import numpy as np


spec = importlib.util.spec_from_file_location(
    "collection_under_test", Path(__file__).resolve().parents[1] / "scripts/tools/trajectory_collection.py")
collection = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
    spec.loader.exec_module(collection)


class Tensor:
    def __init__(self, value): self.value = np.asarray(value)
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.value


class Actor:
    def get_actor_obs(self, obs):
        return Tensor(np.concatenate([obs["second"].value, obs["first"].value], axis=-1))


class ObservationTests(unittest.TestCase):
    def test_checker_reads_diagnostic_thresholds_from_hdf5(self):
        checker_spec = importlib.util.spec_from_file_location(
            "checker_under_test", Path(__file__).resolve().parents[1] / "scripts/tools/check_trajectories.py")
        checker = importlib.util.module_from_spec(checker_spec)
        checker_spec.loader.exec_module(checker)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.h5"
            with h5py.File(path, "w") as hf:
                hf.attrs.update(schema_version="contrack_episode_v1", collection_complete=True, completed_episodes=1)
                g = hf.create_group("episodes/000000")
                g.attrs.update(transition_count=2, completed=True)
                g.create_dataset("states/objects/position", data=np.zeros((3, 1, 3)))
                g.create_dataset("policy_observation", data=np.zeros((2, 3)))
                g.create_dataset("transitions/done", data=[False, True])
                g.create_dataset("transitions/termination/pose_break", data=[False, True])
                diagnostic = g.create_group("transitions/diagnostics")
                for key, value in {
                    "position_error": [[.05], [.11]], "position_threshold": [.1, .1],
                    "rotation_error_rad": [[0.], [0.]], "rotation_threshold_rad": [1., 1.],
                    "pose_break": [False, True], "reference_frame": [0, 1],
                }.items():
                    diagnostic.create_dataset(key, data=value)
            checker.check(path)
            with h5py.File(path, "r+") as hf:
                hf.attrs["pose_break_mode"] = "log-only"
                diagnostic = hf["episodes/000000/transitions/diagnostics"]
                diagnostic.create_dataset("threshold_exceeded", data=[False, True])
                diagnostic["pose_break"][:] = False
                hf["episodes/000000/transitions/termination/pose_break"][:] = False
            checker.check(path)
            with h5py.File(path, "r+") as hf:
                hf["episodes/000000/transitions/diagnostics/position_threshold"][1] = .2
            with self.assertRaisesRegex(ValueError, "differs from termination"):
                checker.check(path)

    def test_diagnostic_signature_matches_isaac_term_contract(self):
        wrapped = collection.diagnostic_pose_term(lambda env: None)
        parameters = list(inspect.signature(wrapped).parameters.values())
        self.assertEqual([p.name for p in parameters], ["env"])
        self.assertEqual(parameters[0].kind, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertEqual(parameters[0].default, inspect.Parameter.empty)

    def test_diagnostic_captures_evaluation_frame_and_preserves_result(self):
        class Origin(np.ndarray):
            def unsqueeze(self, axis): return np.expand_dims(self, axis)

        class Scene(dict):
            env_origins = np.zeros((1, 3)).view(Origin)

        fake_torch = types.SimpleNamespace(
            clamp=lambda value, min=None, max=None: np.clip(value, min, max),
            norm=lambda value, dim: np.linalg.norm(value, axis=dim),
            abs=np.abs, sum=lambda value, dim: np.sum(value, axis=dim), arccos=np.arccos,
            zeros_like=np.zeros_like)
        for position, angle, expected in [(.05, 0., False), (.11, 0., True), (.01, 1.1, True)]:
            env = types.SimpleNamespace(
                frame_idx=np.array([2]), ref_len=5,
                ref_object_pos=np.zeros((5, 1, 3)),
                ref_object_quat=np.tile([1., 0., 0., 0.], (5, 1, 1)),
                object_pos_threshold=np.array([.1]), object_rot_threshold=np.array([1.]),
                scene=Scene(objects=types.SimpleNamespace(data=types.SimpleNamespace(
                    object_pos_w=np.array([[[position, 0., 0.]]]),
                    object_quat_w=np.array([[[np.cos(angle/2), np.sin(angle/2), 0., 0.]]])))))
            returned = np.array([expected])
            wrapped = collection.diagnostic_pose_term(lambda env: returned)
            with patch.object(collection, "torch", fake_torch), patch.object(collection, "numpy", lambda x: np.array(x).copy()):
                self.assertIs(wrapped(env), returned)
            env.frame_idx[0] = 3
            diagnostic = env._collection_pose_diagnostic
            self.assertEqual(int(diagnostic["reference_frame"]), 2)
            self.assertEqual(bool(diagnostic["pose_break"]), expected)
            np.testing.assert_allclose(diagnostic["position_error"], [position])
            np.testing.assert_allclose(diagnostic["rotation_error_rad"], [angle], atol=1e-7)
            with patch.object(collection, "torch", fake_torch), patch.object(collection, "numpy", lambda x: np.array(x).copy()):
                active = collection.diagnostic_pose_term(lambda env: returned, "log-only")(env)
            self.assertFalse(bool(active[0]))
            self.assertEqual(bool(env._collection_pose_diagnostic["threshold_exceeded"]), expected)

    def test_mapping_uses_actor_order_and_copies(self):
        obs = {"first": Tensor([[1., 2.]]), "second": Tensor([[3.]])}
        result = collection.policy_observation(obs, Actor())
        np.testing.assert_array_equal(result, [3., 1., 2.])
        obs["first"].value[:] = 9
        np.testing.assert_array_equal(result, [3., 1., 2.])

    def test_student_and_flat_tensor(self):
        student = types.SimpleNamespace(get_student_obs=lambda obs: obs["student"])
        np.testing.assert_array_equal(collection.policy_observation(
            {"student": Tensor([[4., 5.]])}, student), [4., 5.])
        np.testing.assert_array_equal(collection.policy_observation(Tensor([[6.]]), object()), [6.])

    def test_object_array_and_unknown_mapping_fail_early(self):
        with self.assertRaises(TypeError):
            collection.policy_observation(Tensor(np.array([[{}]], dtype=object)), object())
        with self.assertRaises(TypeError):
            collection.policy_observation({"policy": Tensor([[1.]])}, object())

    def test_hdf5_round_trip(self):
        state = {"objects/position": np.zeros((1, 3)),
                 "reference/object_position": np.zeros((1, 3)),
                 "objects/quaternion_wxyz": np.array([[1., 0., 0., 0.]]),
                 "reference/object_quaternion_wxyz": np.array([[1., 0., 0., 0.]])}
        obs = collection.policy_observation({"first": Tensor([[1., 2.]]),
                                             "second": Tensor([[3.]])}, Actor())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.h5"
            with h5py.File(path, "w") as hf:
                collection.write_episode(hf.create_group("episode"), [state, state],
                                         [obs], [{"done": np.array(True)}], True)
            with h5py.File(path) as hf:
                dataset = hf["episode/policy_observation"]
                self.assertEqual(dataset.shape, (1, 3))
                self.assertEqual(dataset.dtype.kind, "f")
                np.testing.assert_array_equal(dataset[:], [[3., 1., 2.]])
            with h5py.File(Path(directory) / "bad.h5", "w") as hf:
                with self.assertRaises(TypeError):
                    collection.write_episode(hf, [state, state], [np.array([{}], dtype=object)],
                                             [{"done": np.array(True)}], True)
                self.assertEqual(list(hf.keys()), [])


if __name__ == "__main__":
    unittest.main()
