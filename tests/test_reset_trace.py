"""Offline tracing tests; no Isaac Sim required."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np


def load(name):
    path = Path(__file__).resolve().parents[1] / f"scripts/tools/{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trace_module = load("reset_trace")
comparison = load("compare_reset_traces")


class View:
    def __init__(self):
        self.position = np.zeros((1, 3))
    def get_dof_positions(self): return self.position
    def get_dof_velocities(self): return self.position + 1
    def get_transforms(self): return np.zeros((1, 7))
    def get_velocities(self): return np.zeros((1, 6))
    def get_link_transforms(self):
        value = np.zeros((1, 2, 7))
        value[..., 6] = 1.
        value[0, 1, :3] = self.position[0]
        return value
    def get_link_velocities(self): return np.tile(self.position[:, None, :], (1, 2, 2))


class Sim:
    def __init__(self, view): self.view, self.calls = view, []
    def step(self, render=False):
        self.calls.append(render)
        self.view.position += .1
        return "original return"


def environment():
    data = SimpleNamespace(**{key: np.zeros((1, 3)) for key in (
        "joint_pos", "joint_vel", "root_pos_w", "root_quat_w", "root_lin_vel_w",
        "root_ang_vel_w", "joint_pos_target", "joint_vel_target", "joint_effort_target")})
    view = View()
    robot = SimpleNamespace(data=data, root_physx_view=view, joint_names=["task", "unused1", "unused2"], body_names=["base", "finger"])
    return SimpleNamespace(
        scene={"robot": robot, "objects": SimpleNamespace(root_physx_view=View())},
        ref_hands=[("robot", [0], 0)], sim=Sim(view), frame_idx=np.array([0]),
        action_manager=SimpleNamespace(action=np.zeros((1, 1))),
        cfg=SimpleNamespace(sim=SimpleNamespace(dt=1/120), decimation=2))


class TraceTests(unittest.TestCase):
    def test_contact_readback_is_fresh_without_updating_cache(self):
        env = environment()
        class ContactView:
            def get_net_contact_forces(self, dt):
                return np.array([[env.sim.view.position[0, 0] / dt, 0., 0.]])
            def get_contact_force_matrix(self, dt):
                return self.get_net_contact_forces(dt)[:, None, :]
        sensor = SimpleNamespace(_data=SimpleNamespace(net_forces_w=np.zeros((1, 1, 3)),
                                 force_matrix_w=None), contact_physx_view=ContactView())
        env.hand_contact_sensor_keys = {"right": ["finger_sensor"]}
        env.scene["finger_sensor"] = sensor
        trace = trace_module.ResetTrace(env)
        trace.install()
        try:
            trace.reset_before(); trace.reset_after(); trace.begin_control(0)
            env.sim.step(); env.sim.step(); trace.end_control()
            after = trace.records[0]["substeps"][0]["after"]
            self.assertAlmostEqual(after["contact/right/0/physx/get_net_contact_forces"][0, 0], 12.)
            self.assertEqual(after["contact/right/0/cached/net_forces_w"].max(), 0.)
            self.assertEqual(len(env.sim.calls), 2)
        finally:
            trace.restore()

    def test_link_report_names_sign_invariance_and_velocity(self):
        key = "robots/robot/physx/get_link_transforms"
        velocity_key = "robots/robot/physx/get_link_velocities"
        a = np.zeros((1, 2, 7)); a[..., 6] = 1.
        b = a.copy(); b[0, 1, 0] = .02; b[0, 0, 3:] *= -1
        va = np.zeros((1, 2, 6)); vb = va.copy(); vb[0, 1, 3] = .4
        rows = comparison.link_differences({key:a, velocity_key:va}, {key:b, velocity_key:vb},
                                            {"robot":["base","finger"]})
        self.assertEqual(rows[0]["link"], "base")
        self.assertAlmostEqual(rows[0]["rotation_difference_rad"], 0.)
        self.assertEqual(rows[1]["link"], "finger")
        self.assertAlmostEqual(rows[1]["position_difference_m"], .02)
        self.assertAlmostEqual(rows[1]["angular_velocity_difference_rad_s"], .4)
        with self.assertRaisesRegex(ValueError, "mapping/shape"):
            comparison.link_differences({key:a}, {key:b}, {"robot":["wrong_count"]})

    def test_link_report_names_sign_invariance_and_velocity(self):
        key = "robots/robot/physx/get_link_transforms"
        velocity_key = "robots/robot/physx/get_link_velocities"
        a = np.zeros((1, 2, 7)); a[..., 6] = 1.
        b = a.copy(); b[0, 1, 0] = .02; b[0, 0, 3:] *= -1
        va = np.zeros((1, 2, 6)); vb = va.copy(); vb[0, 1, 3] = .4
        rows = comparison.link_differences({key:a, velocity_key:va}, {key:b, velocity_key:vb},
                                            {"robot":["base","finger"]})
        self.assertEqual(rows[0]["link"], "base")
        self.assertAlmostEqual(rows[0]["rotation_difference_rad"], 0.)
        self.assertEqual(rows[1]["link"], "finger")
        self.assertAlmostEqual(rows[1]["position_difference_m"], .02)
        self.assertAlmostEqual(rows[1]["angular_velocity_difference_rad_s"], .4)
        with self.assertRaisesRegex(ValueError, "mapping/shape"):
            comparison.link_differences({key:a}, {key:b}, {"robot":["wrong_count"]})

    def test_first_control_only_fresh_readback_and_restore(self):
        env = environment()
        original = env.sim.step
        trace = trace_module.ResetTrace(env)
        trace.install()
        try:
            trace.reset_before()
            trace.reset_after()
            trace.before_policy(0)
            trace.begin_control(0)
            self.assertEqual(env.sim.step(render=False), "original return")
            env.sim.step(render=True)
            trace.end_control()
            record = trace.records[0]
            self.assertEqual(len(record["substeps"]), 2)
            np.testing.assert_allclose(record["substeps"][0]["before"]["robots/robot/physx/get_dof_positions"], 0.)
            np.testing.assert_allclose(record["substeps"][0]["after"]["robots/robot/physx/get_dof_positions"], .1)
            np.testing.assert_allclose(record["substeps"][0]["after"]["robots/robot/physx/get_link_transforms"][0, 1, :3], .1)
            np.testing.assert_allclose(record["substeps"][0]["after"]["robots/robot/buffered/joint_pos"], 0.)
            self.assertEqual(record["before_policy"]["robots/robot/buffered/joint_pos"].shape, (1, 3))
            trace.begin_control(0)
            env.sim.step(render=False)
            trace.end_control()
            self.assertEqual(len(record["substeps"]), 2)
            self.assertEqual(env.sim.calls, [False, True, False])
            self.assertIn("robots/robot/physx/get_dof_position_targets", trace.unavailable)
        finally:
            trace.restore()
        self.assertEqual(env.sim.step, original)

    def test_reset_during_control_does_not_replace_previous_record(self):
        env = environment()
        trace = trace_module.ResetTrace(env)
        trace.install()
        try:
            trace.reset_before(); trace.reset_after(); trace.before_policy(0); trace.begin_control(0)
            env.sim.step(); env.sim.step()
            trace.reset_before(); trace.reset_after()
            trace.end_control()
            self.assertIn("first_control_end", trace.records[0])
            self.assertNotIn("first_control_end", trace.records[1])
            self.assertEqual(len(trace.records[0]["substeps"]), 2)
            self.assertEqual(len(trace.records[1]["substeps"]), 0)
        finally:
            trace.restore()

    def test_trace_hdf5_and_comparator(self):
        env = environment()
        trace = trace_module.ResetTrace(env)
        trace.install()
        try:
            trace.reset_before(); trace.reset_after(); trace.before_policy(0); trace.begin_control(0)
            env.sim.step(); env.sim.step(); trace.end_control()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "episodes.h5"
                with h5py.File(path, "w") as hf:
                    hf.attrs["reset_trace_enabled"] = True
                    trace.write(hf.create_group("episodes/000000/reset_trace"), 0)
                result = comparison.compare([path, path])
                self.assertEqual(len(result), 2)
                self.assertTrue(all(not rows for r in result for rows in r["differences"].values()))
                with h5py.File(path, "r+") as hf:
                    hf["episodes/000000/reset_trace/substeps/000/after/robots/robot/physx/get_dof_positions"][0, 0] = 9
                # NaN/missing/shape differences are also explicit, never equal silently.
                self.assertEqual(len(comparison.differences({"x": np.array([1.])}, {"x": np.array([np.nan])})), 1)
                self.assertEqual(len(comparison.differences({"x": np.zeros(1)}, {})), 1)
        finally:
            trace.restore()

    def test_incomplete_substeps_fail_visibly(self):
        env = environment()
        trace = trace_module.ResetTrace(env)
        trace.reset_before(); trace.reset_after()
        with tempfile.TemporaryDirectory() as directory:
            with h5py.File(Path(directory)/"bad.h5", "w") as hf:
                with self.assertRaisesRegex(RuntimeError, "captured 0"):
                    trace.write(hf.create_group("trace"), 0)


if __name__ == "__main__":
    unittest.main()
