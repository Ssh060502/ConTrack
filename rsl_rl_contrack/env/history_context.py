"""Frozen history-to-latent context with causal tokens and per-env reset isolation.

No oracle inputs, object states, reference poses or future actions enter E(H).
Base policy observations remain unchanged. This wrapper adds only `history` z.
"""
from pathlib import Path
import math
import torch
from rsl_rl_contrack.modules.history_encoder import HistoryEncoder
from rsl_rl_contrack.modules.oracle_adapter import frozen_digest
from rsl_rl_contrack.env.oracle_physics import OracleContextWrapper
from scripts.tools.adaptation_profiles import history_state_indices


def live_state(base, mode='all'):
    arrays, names = [], []
    for name, ids, hand in sorted(base.ref_hands, key=lambda x:int(x[2])):
        for field, values in (('qpos',base.scene[name].data.joint_pos),
                              ('qvel',base.scene[name].data.joint_vel)):
            value = values[:, ids]
            arrays.append(value)
            names.extend(f'hand{hand}/{field}/{i}' for i in range(value.shape[1]))
    for side in sorted(base.hand_contact_sensor_keys) if mode=='all' else ():
        for sensor,key in enumerate(base.hand_contact_sensor_keys[side]):
            value = base.scene[key].data.net_forces_w.reshape(base.num_envs,-1)
            arrays.append(value)
            names.extend(f'contact/{side}/{sensor}/net_force_w/{i}' for i in range(value.shape[1]))
    result = torch.cat(arrays,dim=-1)
    ids=history_state_indices(names,mode)
    result=result[:,ids];names=[names[i] for i in ids]
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite live history state')
    return result, names


class HistoryContextWrapper(OracleContextWrapper):
    def __init__(self, env, checkpoint, robot, tool, reference, random_seed=None):
        self.env = env
        self.base_env = env.unwrapped
        self.saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
        schema = self.saved['data_schema']
        self.feature_mode=schema.get('state_features',self.saved['config'].get('state_features','all'))
        if schema.get('robot', self.saved.get('config',{}).get('robot','rby1a')) != robot:
            raise ValueError('History checkpoint robot mismatch; do not use RBY1A weights for CRX')
        if self.saved['config']['tool'] != tool:
            raise ValueError('History checkpoint tool mismatch')
        if schema.get('references') and Path(reference).name not in schema['references']:
            raise ValueError('History checkpoint reference/calibration mismatch')
        if not math.isclose(float(self.saved['control_dt']),float(self.base_env.step_dt),abs_tol=1e-9):
            raise ValueError('History checkpoint control period mismatch')
        mapping = [{'hand':int(h),'asset':name,
                    'joint_names':[self.base_env.scene[name].joint_names[int(j)] for j in ids]}
                   for name,ids,h in self.base_env.ref_hands]
        if schema.get('hand_mapping') != mapping or schema.get('contact_mapping') != self.base_env.hand_contact_sensor_keys:
            raise ValueError('History checkpoint joint/contact mapping mismatch')
        self.random_seed = random_seed
        # Isolate initialization from the residual policy/exploration RNG.
        with torch.random.fork_rng(devices=[]):
            if random_seed is not None:
                # Initialize CPU weights without reseeding CUDA exploration RNGs.
                torch.random.default_generator.manual_seed(random_seed)
            self.model = HistoryEncoder(**self.saved['architecture'])
        if random_seed is None:
            self.model.load_state_dict(self.saved['model_state_dict'],strict=True)
        self.model = self.model.to(env.device)
        self.model.requires_grad_(False)
        self.model.eval()
        self.encoder_digest = frozen_digest(self.model)
        self.stats = {key:torch.as_tensor(value,dtype=torch.float32,device=env.device)
                      for key,value in self.saved['normalization'].items()}
        state,names = live_state(self.base_env,self.feature_mode)
        if names != self.saved['feature_names']:
            raise ValueError('History checkpoint feature order mismatch')
        arch = self.saved['architecture']
        if state.shape[1] != arch['state_dim'] or arch['action_dim'] != 2*env.num_actions:
            raise ValueError('History checkpoint state/action dimensions mismatch')
        self.history_steps = arch['history']
        self.tokens = torch.zeros(env.num_envs,self.history_steps,2*arch['state_dim']+arch['action_dim'],device=env.device)
        self.lengths = torch.zeros(env.num_envs,dtype=torch.long,device=env.device)
        self.context = torch.zeros(env.num_envs,arch['latent_dim'],device=env.device)
        self.previous_state = state.clone()

    def _augment(self, obs):
        result = obs.clone(recurse=False)
        if 'oracle' in result.keys():
            raise ValueError('Privileged oracle context must not enter history control')
        result.set('history',self.context.clone())
        return result

    def reset(self):
        result = self.env.reset()
        self.tokens.zero_();self.lengths.zero_();self.context.zero_()
        self.previous_state = live_state(self.base_env,self.feature_mode)[0].clone()
        return result

    @torch.no_grad()
    def step(self, actions):
        obs,rewards,done,extra = self.env.step(actions)
        state,_ = live_state(self.base_env,self.feature_mode)
        base = self.base_env
        target = [base.scene[name].data.joint_pos_target[:,ids]
                  for name,ids,_ in sorted(base.ref_hands,key=lambda x:int(x[2]))]
        # Read the manager action exactly as recorded in offline H5. Targets
        # additionally capture action-term scaling/clipping into joint commands.
        applied = torch.cat([base.action_manager.action,*target],dim=-1)
        x = (self.previous_state-self.stats['state_mean'])/self.stats['state_std']
        u = (applied-self.stats['action_mean'])/self.stats['action_std']
        dx = (state-self.previous_state)/self.stats['state_std']
        token = torch.cat([x,u,dx],dim=-1)
        valid = ~(done.reshape(-1)>0)
        full = valid & (self.lengths == self.history_steps)
        self.tokens[full] = torch.roll(self.tokens[full],shifts=-1,dims=1)
        ids = valid.nonzero(as_tuple=True)[0]
        self.tokens[ids,self.lengths[ids].clamp(max=self.history_steps-1)] = token[ids]
        self.lengths[valid] = (self.lengths[valid]+1).clamp(max=self.history_steps)
        # env.step returns post-reset states on done. Drop that transition
        # entirely rather than pretending reset is a physical state delta.
        self.tokens[~valid] = 0;self.lengths[~valid] = 0;self.context.zero_()
        if len(ids):
            self.context[ids] = self.model.encode(self.tokens[ids],self.lengths[ids])
        self.previous_state = state.clone()
        return self._augment(obs),rewards,done,extra

    def verify_frozen(self):
        if frozen_digest(self.model) != self.encoder_digest:
            raise RuntimeError('Frozen history encoder changed')


class StateOnlyContextWrapper(OracleContextWrapper):
    """Ablation A: constant zero context, with B/C residual input dimensions.

    No history or privileged labels enter actor or critic. Zero padding keeps
    the residual MLP architecture equal; it provides no changing information.
    """
    def __init__(self, env, dimension=64):
        if dimension <= 0:
            raise ValueError('Context dimension must be positive')
        self.env = env
        self.context = torch.zeros(env.num_envs, dimension, device=env.device)

    def _augment(self, obs):
        result = obs.clone(recurse=False)
        if 'oracle' in result.keys():
            raise ValueError('State-only control must not consume oracle labels')
        result.set('history', self.context.clone())
        return result
