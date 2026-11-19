import importlib.util
from types import SimpleNamespace
import unittest
import numpy as np
from scripts.tools.history_evaluation import feature_groups,aggregate_episodes,evaluate_history

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:import torch


class MetricTests(unittest.TestCase):
    def test_feature_groups_and_equal_episode_weighting(self):
        names=['hand0/qpos/0','hand0/qvel/0','contact/right/0/net_force_w/0']
        self.assertEqual(feature_groups(names),{'qpos':[0],'qvel':[1],'contact_force':[2]})
        with self.assertRaises(ValueError):feature_groups(['unknown'])
        episodes=[SimpleNamespace(condition='a'),SimpleNamespace(condition='b')]
        result=aggregate_episodes({0:{'error':1.},1:{'error':3.}},episodes)
        self.assertEqual(result['error'],2.)
        self.assertEqual(result['a/error'],1.)
        self.assertEqual(result['b/episode_count'],1)

    @unittest.skipUnless(HAS_TORCH,'Run with torch in env_isaaclab')
    def test_zero_predictor_matches_component_baselines_in_both_units(self):
        class Zero(torch.nn.Module):
            def forward(self,h,length,state,action):
                return torch.zeros(len(state),2),torch.zeros_like(state),torch.zeros(len(state),1)
        batch=dict(history=torch.zeros(2,2,8),length=torch.ones(2,dtype=torch.long),
                   state=torch.zeros(2,3),action=torch.zeros(2,2),label=torch.zeros(2,2),
                   delta=torch.tensor([[1.,2.,3.],[2.,3.,4.]]),episode=torch.tensor([0,1]))
        episodes=[SimpleNamespace(condition='a'),SimpleNamespace(condition='a')]
        stats=dict(parameter_mean=np.log([.1,1.]).astype(np.float32),parameter_std=np.ones(2,np.float32),
                   state_std=np.array([.1,.2,10.],np.float32))
        metrics,per_episode=evaluate_history(Zero(),[batch],episodes,stats,
            ['hand0/qpos/0','hand0/qvel/0','contact/right/0/net_force_w/0'],'cpu')
        self.assertEqual(len(per_episode),2)
        self.assertEqual(metrics['mass_mae_kg'],0.)
        for group in ('qpos','qvel','contact_force'):
            for unit in ('normalized','physical'):
                prefix=f'dynamics/{group}/'
                self.assertAlmostEqual(metrics[prefix+'mse_'+unit],metrics[prefix+'zero_delta_mse_'+unit])
        self.assertAlmostEqual(metrics['dynamics/qvel/mse_physical'],.26,places=5)
        self.assertAlmostEqual(metrics['dynamics/contact_force/mse_physical'],1250.)


if __name__=='__main__':unittest.main()
