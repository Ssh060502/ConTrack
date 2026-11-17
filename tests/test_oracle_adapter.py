"""Run inside env_isaaclab: frozen actor, gradients, normalization and checkpoint tests."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

AVAILABLE = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("tensordict") is not None
if AVAILABLE:
    import torch
    from tensordict import TensorDict
    from rsl_rl_contrack.modules.actor_critic import ActorCritic
    from rsl_rl_contrack.modules.oracle_adapter import OracleAdapterActorCritic, frozen_digest


@unittest.skipUnless(AVAILABLE, "Run in env_isaaclab with torch/tensordict")
class AdapterTests(unittest.TestCase):
    def test_zero_identity_frozen_gradients_and_roundtrip(self):
        torch.manual_seed(4)
        obs=TensorDict({"policy":torch.randn(4,5),"critic":torch.randn(4,6),"oracle":torch.randn(4,2)},batch_size=[4])
        groups={"policy":["policy"],"critic":["critic"]}
        base_cfg=dict(actor_hidden_dims=[8],critic_hidden_dims=[8],actor_obs_normalization=True,critic_obs_normalization=True)
        base=ActorCritic(obs,groups,3,**base_cfg)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"base.pt"
            torch.save({"model_state_dict":base.state_dict()},path)
            kwargs=dict(base_checkpoint=str(path),base_policy_cfg=base_cfg,base_obs_groups=groups,
                        actor_hidden_dims=[8],critic_hidden_dims=[8],actor_obs_normalization=True,critic_obs_normalization=True)
            adapter_groups={"policy":["policy","oracle"],"critic":["critic","oracle"]}
            adapter=OracleAdapterActorCritic(obs,adapter_groups,3,**kwargs)
            digest=frozen_digest(adapter.base)
            torch.testing.assert_close(adapter.act_inference(obs),base.act_inference(obs),atol=0,rtol=0)
            adapter.train();self.assertFalse(adapter.base.training)
            adapter.update_normalization(obs)
            actions=adapter.act(obs)
            loss=-adapter.get_actions_log_prob(actions.detach()).mean()+adapter.evaluate(obs,masks=None,hidden_state=None).square().mean()
            loss.backward()
            self.assertTrue(all(p.grad is None for p in adapter.base.parameters()))
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in adapter.actor.parameters()))
            optimizer=torch.optim.Adam([p for p in adapter.parameters() if p.requires_grad],lr=.001)
            optimizer.step()
            self.assertEqual(frozen_digest(adapter.base),digest)
            other=OracleAdapterActorCritic(obs,adapter_groups,3,**kwargs)
            other.load_state_dict(adapter.state_dict())
            torch.testing.assert_close(adapter.act_inference(obs),other.act_inference(obs))


if __name__ == '__main__':
    unittest.main()
