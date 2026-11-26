import unittest
import importlib.util
import numpy as np
from scripts.tools.history_dataset import Episode
from scripts.tools.history_timeline import audit_forces,force_series,predict_timeline


class TimelineTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('torch') is not None,'Needs torch')
    def test_prediction_rows_retain_actual_episode_time_and_do_not_invent_t0(self):
        import torch
        class ConstantModel:
            def eval(self):pass
            def encode(self,history,length):return torch.zeros(len(length),2)
            def parameters_head(self,z):return z
        episode=Episode('e','file','hash',np.zeros((6,3),np.float32),
                        np.zeros((5,2),np.float32),np.log(np.array([.1,2.],np.float32)),'c')
        stats=dict(state_mean=np.zeros(3,np.float32),state_std=np.ones(3,np.float32),
                   action_mean=np.zeros(2,np.float32),action_std=np.ones(2,np.float32),
                   parameter_mean=episode.log_parameters,parameter_std=np.ones(2,np.float32))
        rows=predict_timeline(ConstantModel(),[episode],[0],stats,4,2,'cpu',2)
        self.assertEqual([r['state_index'] for r in rows],[1,3])
        self.assertEqual([r['history_length'] for r in rows],[1,3])
        self.assertAlmostEqual(rows[0]['predicted_mass_kg'],.1,places=6)
        self.assertAlmostEqual(rows[1]['predicted_friction'],2.,places=6)

    def test_initial_spike_and_later_peak_are_reported_without_changing_data(self):
        state=np.array([[0,3000,4000,0],[0,0,0,0],[0,0,0,2]],np.float32)
        episode=Episode('e','file','hash',state,np.zeros((2,1)),np.zeros(2),'condition')
        names=['hand0/qpos/0']+[f'contact/right/0/net_force_w/{i}' for i in range(3)]
        before=state.copy()
        report=audit_forces([episode],names,.1)['episodes'][0]
        self.assertEqual(report['peak_force_N'],5000.)
        self.assertEqual(report['peak_state_index'],0)
        self.assertEqual(report['post_initial_peak_force_N'],2.)
        self.assertEqual(report['samples_above_threshold'],1)
        self.assertEqual(report['peak_sensor'],'contact/right/0/net_force_w')
        np.testing.assert_array_equal(state,before)
        self.assertEqual(force_series(episode,names)[0].shape,(3,1))


if __name__=='__main__':unittest.main()
