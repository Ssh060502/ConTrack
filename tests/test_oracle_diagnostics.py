"""Run with torch/tensordict; checks logging without Isaac or a GPU."""
import importlib.util
from types import SimpleNamespace
import unittest

AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("tensordict") is not None
if AVAILABLE:
    import torch
    from tensordict import TensorDict
    from rsl_rl_contrack.env.oracle_diagnostics import OracleDiagnosticsWrapper


@unittest.skipUnless(AVAILABLE, "Run in env_isaaclab with torch/tensordict")
class DiagnosticTests(unittest.TestCase):
    def test_grouping_clipping_partial_returns_and_terminal_reset_exclusion(self):
        class Env:
            num_envs, device, num_actions, ref_len = 4, "cpu", 2, 10
            def __init__(self):
                self.unwrapped = self
                self.frame_idx = torch.zeros(4, dtype=torch.long)
                self.ref_object_pos = torch.zeros(10, 1, 3)
                self.ref_object_quat = torch.zeros(10, 1, 4)
                self.ref_object_quat[..., 0] = 1
                self.scene = SimpleNamespace(env_origins=torch.zeros(4,3))
                self.data = SimpleNamespace(object_pos_w=torch.zeros(4,1,3),
                                            object_quat_w=self.ref_object_quat[:4].clone())
                self.scene = Scene(self.scene.env_origins, self.data)
                self.steps = 0
            def reset(self): return None
            def get_observations(self): return TensorDict({}, batch_size=[4])
            def step(self, actions):
                self.steps += 1
                done = torch.zeros(4, dtype=torch.bool)
                self.data.object_pos_w.zero_()
                if self.steps == 2:
                    done[0] = True
                    # A reset state deliberately has a huge error; exclude it.
                    self.data.object_pos_w[0] = 100.
                extras = {"rg":torch.tensor([1.,10.,1.,10.]),
                          "rs":torch.ones(4)*2, "rp":torch.ones(4)*-.5}
                return self.get_observations(), torch.ones(4)*3, done, extras
        class Scene:
            def __init__(self, origins, data):
                self.env_origins, self.data = origins, data
            def __getitem__(self, key): return SimpleNamespace(data=self.data)
        env = Env()
        wrapped = OracleDiagnosticsWrapper(env, [(1.,2.),(3.,2.)], [0,1,0,1], [0], [1])
        env.it_step_count = 32
        wrapped.it_step_count = 0
        self.assertEqual(env.it_step_count, 0)
        wrapped.policy = SimpleNamespace(correction_scale=.1, last_action_components={
            "base":torch.zeros(4,2), "correction":torch.ones(4,2)*.05,
            "mean":torch.ones(4,2)*.05})
        action = torch.tensor([[1.2,.0]]).repeat(4,1)
        wrapped.step(action)
        first = wrapped.drain_diagnostics()
        prefix = "Oracle/mass1_friction2/"
        self.assertEqual(first[prefix+"completed_episodes"], 0)
        self.assertNotIn(prefix+"task_return", first)
        self.assertAlmostEqual(first[prefix+"sample_outside_bounds_fraction"], .5)
        self.assertEqual(first[prefix+"mean_outside_bounds_fraction"], 0.)
        torch.testing.assert_close(action, torch.tensor([[1.2,.0]]).repeat(4,1))
        wrapped.step(action)
        second = wrapped.drain_diagnostics()
        self.assertEqual(second[prefix+"completed_episodes"], 1)
        self.assertEqual(second[prefix+"task_return"], 2.)
        self.assertEqual(second[prefix+"environment_return"], 6.)
        self.assertEqual(second[prefix+"task_plus_penalty_return"], 1.)
        self.assertEqual(second[prefix+"position_error_m"], 0.)
        self.assertEqual(second["Oracle/mass3_friction2/completed_episodes"], 0)
        self.assertAlmostEqual(float(wrapped.episode_sums[1,0]), 20.)
        self.assertEqual(len(wrapped.pending), 0)


if __name__ == "__main__":
    unittest.main()
