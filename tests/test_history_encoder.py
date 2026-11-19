import importlib.util
import unittest

AVAILABLE=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('tensordict') is not None
if AVAILABLE:
    import torch
    from rsl_rl_contrack.modules.history_encoder import HistoryEncoder


@unittest.skipUnless(AVAILABLE,'Run with torch/tensordict in env_isaaclab')
class HistoryModelTests(unittest.TestCase):
    def test_padding_cannot_change_latent_and_heads_receive_gradients(self):
        torch.manual_seed(3)
        model=HistoryEncoder(3,2,history=4,width=16,layers=1,heads=2,latent_dim=8)
        model.eval()
        h=torch.randn(2,4,8);length=torch.tensor([1,3])
        modified=h.clone();modified[0,1:]=10000;modified[1,3:]=-10000
        with torch.no_grad():
            torch.testing.assert_close(model.encode(h,length),model.encode(modified,length))
        model.train()
        physical,dynamics,z=model(h,length,torch.randn(2,3),torch.randn(2,2))
        self.assertEqual(z.shape,(2,8));self.assertEqual(physical.shape,(2,2));self.assertEqual(dynamics.shape,(2,3))
        (physical.square().mean()+dynamics.square().mean()).backward()
        self.assertTrue(all(p.grad is not None for p in model.parameters()))


if __name__=='__main__':unittest.main()
