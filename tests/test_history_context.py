"""CPU-only closed-loop contract tests, no Isaac Lab required."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np

AVAILABLE=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('tensordict') is not None
if AVAILABLE:
    import torch
    from tensordict import TensorDict
    from rsl_rl_contrack.env.history_context import HistoryContextWrapper,live_state
    from rsl_rl_contrack.modules.history_encoder import HistoryEncoder
    from scripts.tools.history_dataset import Episode,window


@unittest.skipUnless(AVAILABLE,'Needs torch/tensordict; no simulator needed')
class ContextTests(unittest.TestCase):
    def make(self,directory,robot='crx'):
        class Env:
            num_envs=2;num_actions=2;device='cpu';step_dt=1/60
            def __init__(self):
                self.unwrapped=self;self.steps=0
                self.ref_hands=[('crx_sharpa',[0,1],0)]
                self.hand_contact_sensor_keys={'right':['contact0']}
                self.scene={'crx_sharpa':SimpleNamespace(joint_names=['left_J1','left_thumb_IP'],
                           data=SimpleNamespace(joint_pos=torch.zeros(2,2),joint_vel=torch.zeros(2,2),
                                                joint_pos_target=torch.zeros(2,2))),
                            'contact0':SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(2,1,3)))}
                self.action_manager=SimpleNamespace(action=torch.zeros(2,2))
            def reset(self):
                self.steps=0
                for value in vars(self.scene['crx_sharpa'].data).values():value.zero_()
                self.scene['contact0'].data.net_forces_w.zero_()
                return self.get_observations()
            def get_observations(self):
                return TensorDict({'policy':torch.zeros(2,3),'critic':torch.zeros(2,3)},batch_size=[2])
            def step(self,actions):
                self.steps+=1
                self.action_manager.action=actions.clamp(-1,1)
                data=self.scene['crx_sharpa'].data
                data.joint_pos+=self.action_manager.action+1
                data.joint_vel.fill_(self.steps)
                data.joint_pos_target=self.action_manager.action+2
                self.scene['contact0'].data.net_forces_w.fill_(self.steps)
                done=torch.tensor([self.steps==2,False])
                if done[0]:
                    data.joint_pos[0]=0;data.joint_vel[0]=0
                    self.scene['contact0'].data.net_forces_w[0]=0
                return self.get_observations(),torch.zeros(2),done,{}
        env=Env()
        architecture=dict(state_dim=7,action_dim=4,history=3,width=16,layers=1,heads=2,latent_dim=8)
        model=HistoryEncoder(**architecture)
        stats=dict(state_mean=np.zeros(7,np.float32),state_std=np.ones(7,np.float32),
                   action_mean=np.zeros(4,np.float32),action_std=np.ones(4,np.float32),
                   parameter_mean=np.zeros(2,np.float32),parameter_std=np.ones(2,np.float32))
        mapping=[{'hand':0,'asset':'crx_sharpa','joint_names':['left_J1','left_thumb_IP']}]
        saved=dict(architecture=architecture,model_state_dict=model.state_dict(),normalization=stats,
                   feature_names=live_state(env)[1],control_dt=1/60,config={'tool':'hammer','robot':robot},
                   data_schema={'robot':robot,'references':['hammer-crx.h5'],'hand_mapping':mapping,
                                'contact_mapping':env.hand_contact_sensor_keys})
        path=Path(directory)/'encoder.pt';torch.save(saved,path)
        return env,path,stats

    def test_online_tokens_match_offline_and_done_rows_clear_before_next_action(self):
        with tempfile.TemporaryDirectory() as directory:
            env,path,stats=self.make(directory)
            wrapper=HistoryContextWrapper(env,path,'crx','hammer','hammer-crx.h5')
            wrapper.reset()
            torch.testing.assert_close(wrapper.get_observations()['history'],torch.zeros(2,8))
            states=[live_state(env)[0][1].numpy().copy()]
            actions=[]
            for _ in range(3):
                obs,_,_,_=wrapper.step(torch.full((2,2),2.))
                states.append(live_state(env)[0][1].numpy().copy())
                actions.append(np.array([1.,1.,3.,3.],np.float32))
                self.assertNotIn('oracle',obs.keys())
                if env.steps==2:
                    self.assertEqual(wrapper.lengths.tolist(),[0,2])
                    torch.testing.assert_close(obs['history'][0],torch.zeros(8))
            episode=Episode('e','s','d',np.stack(states),np.stack(actions),np.zeros(2),'c')
            # Offline decision t=2 sees only transitions 0 and 1.
            expected=window(episode,2,3,stats)['history'][:2]
            np.testing.assert_allclose(wrapper.tokens[1,:2].numpy(),expected)
            self.assertEqual(wrapper.lengths.tolist(),[1,3])
            # First new token for reset env begins at reset qpos, not old qpos.
            torch.testing.assert_close(wrapper.tokens[0,0,:2],torch.zeros(2))
            wrapper.verify_frozen()
            self.assertTrue(all(not p.requires_grad for p in wrapper.model.parameters()))

    def test_wrong_robot_and_reference_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            env,path,_=self.make(directory)
            with self.assertRaisesRegex(ValueError,'robot mismatch'):
                HistoryContextWrapper(env,path,'rby1a','hammer','hammer-crx.h5')
            with self.assertRaisesRegex(ValueError,'reference/calibration'):
                HistoryContextWrapper(env,path,'crx','hammer','different.h5')

    def test_joint_only_online_features_ignore_force_sensor_values(self):
        with tempfile.TemporaryDirectory() as directory:
            env,path,_=self.make(directory)
            saved=torch.load(path,weights_only=False)
            saved['architecture']['state_dim']=4
            saved['model_state_dict']=HistoryEncoder(**saved['architecture']).state_dict()
            saved['feature_names']=saved['feature_names'][:4]
            saved['data_schema']['state_features']='joints'
            for name in ('state_mean','state_std'):saved['normalization'][name]=saved['normalization'][name][:4]
            torch.save(saved,path)
            wrapper=HistoryContextWrapper(env,path,'crx','hammer','hammer-crx.h5')
            wrapper.reset()
            state,names=live_state(env,'joints')
            env.scene['contact0'].data.net_forces_w.fill_(float('nan'))
            changed,changed_names=live_state(env,'joints')
            torch.testing.assert_close(state,changed)
            self.assertEqual(names,changed_names)
            self.assertEqual(len(names),4)
            wrapper.step(torch.ones(2,2))
            self.assertEqual(wrapper.tokens.shape[-1],12)
            wrapper.verify_frozen()

    def test_residual_optimization_changes_neither_encoder_nor_base(self):
        from rsl_rl_contrack.modules.actor_critic import ActorCritic
        from rsl_rl_contrack.modules.oracle_adapter import OracleAdapterActorCritic,frozen_digest
        with tempfile.TemporaryDirectory() as directory:
            env,path,_=self.make(directory)
            wrapper=HistoryContextWrapper(env,path,'crx','hammer','hammer-crx.h5')
            wrapper.reset();wrapper.step(torch.ones(2,2))
            obs=wrapper.get_observations()
            groups={'policy':['policy'],'critic':['critic']}
            base_cfg=dict(actor_hidden_dims=[8],critic_hidden_dims=[8],activation='elu',
                          actor_obs_normalization=True,critic_obs_normalization=True)
            base=ActorCritic(obs,groups,2,**base_cfg)
            checkpoint=Path(directory)/'base.pt';torch.save({'model_state_dict':base.state_dict()},checkpoint)
            actor=OracleAdapterActorCritic(obs,{'policy':['policy','history'],'critic':['critic','history']},2,
                base_checkpoint=checkpoint,base_policy_cfg=base_cfg,base_obs_groups=groups,
                actor_hidden_dims=[8],critic_hidden_dims=[8],activation='elu')
            actor.eval()
            torch.testing.assert_close(actor.act_inference(obs),actor.base.act_inference(obs),rtol=0,atol=0)
            digest=frozen_digest(actor.base)
            before=[p.clone() for p in actor.actor.parameters()]
            optimizer=torch.optim.Adam([p for p in actor.parameters() if p.requires_grad],lr=.01)
            actor.train();optimizer.zero_grad()
            (actor.mean_action(obs)-torch.ones(2,2)).square().mean().backward();optimizer.step()
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,actor.actor.parameters())))
            self.assertEqual(frozen_digest(actor.base),digest)
            self.assertTrue(all(p.grad is None for p in wrapper.model.parameters()))
            wrapper.verify_frozen()


if __name__=='__main__':unittest.main()
