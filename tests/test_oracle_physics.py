"""Offline numerical tests of per-environment physical conditions (no simulator)."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np


class Tensor(np.ndarray):
    @property
    def device(self): return "cpu"
    def clone(self): return self.copy()
    def cpu(self): return self
    def to(self, device): return self
    def log(self): return np.log(self).view(Tensor)
    def expand_as(self, other): return np.broadcast_to(self, other.shape).view(Tensor)


def tensor(value, **kwargs):
    return np.asarray(value, dtype=kwargs.get("dtype")).view(Tensor)


fake_torch = SimpleNamespace(
    arange=lambda n, **kw: np.arange(n, dtype=kw.get("dtype")).view(Tensor),
    tensor=tensor, int32=np.int32,
    allclose=np.allclose, cat=lambda x, dim: np.concatenate(x, axis=dim).view(Tensor))
spec = importlib.util.spec_from_file_location("oracle_physics_test", Path(__file__).resolve().parents[1]/"rsl_rl_contrack/env/oracle_physics.py")
physics = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {"torch":fake_torch}):
    spec.loader.exec_module(physics)


class View:
    def __init__(self, n):
        self.mass = tensor(np.full((n, 1), .072))
        self.inertia = tensor(np.ones((n, 9)))
        self.material = tensor(np.tile([2.,2.,0.], (n,8,1)))
    def get_masses(self): return self.mass
    def get_inertias(self): return self.inertia
    def get_material_properties(self): return self.material
    def set_masses(self, value, indices): self.mass[indices] = value
    def set_inertias(self, value, indices): self.inertia[indices] = value
    def set_material_properties(self, value, indices): self.material[indices] = value


class PhysicsTests(unittest.TestCase):
    def test_runner_counter_reset_is_forwarded_to_reward_wrapper(self):
        inner = SimpleNamespace(it_step_count=32)
        proxy = physics.OracleContextWrapper(inner, None)
        proxy.it_step_count = 0
        self.assertEqual(inner.it_step_count, 0)
        inner.it_step_count = 32
        self.assertEqual(proxy.it_step_count, 32)

    def test_condition_validation(self):
        self.assertEqual(physics.parse_conditions("1:2,3:0.7"), [(1.,2.),(3.,.7)])
        for value in ("", "1", "0:2", "1:0", "nan:2", "1:inf", "1:2:3"):
            with self.assertRaises(ValueError): physics.parse_conditions(value)

    def test_mass_inertia_material_assignment_and_readback(self):
        view = View(6)
        base = SimpleNamespace(num_objects=1,num_envs=6,device="cpu",
                               scene={"objects":SimpleNamespace(root_physx_view=view)})
        context, report = physics.configure_physics(base, [(1.,2.),(3.,.7),(6.,.2)])
        np.testing.assert_allclose(view.mass[:,0], [.072,.216,.432,.072,.216,.432])
        np.testing.assert_allclose(view.inertia[:,0], [1,3,6,1,3,6])
        np.testing.assert_allclose(view.material[:,0,0], [2,.7,.2,2,.7,.2])
        np.testing.assert_allclose(view.material[...,0], view.material[...,1])
        np.testing.assert_allclose(np.exp(context), np.c_[view.mass[:,0],view.material[:,0,0]])
        self.assertEqual(report["condition_id"], [0,1,2,0,1,2])

    def test_unsupported_layout_and_uncovered_conditions_rejected(self):
        base = SimpleNamespace(num_objects=2,num_envs=1)
        with self.assertRaises(ValueError): physics.configure_physics(base,[(1,2)])
        base.num_objects=1
        with self.assertRaises(ValueError): physics.configure_physics(base,[(1,2),(3,2)])


if __name__ == '__main__':
    unittest.main()
