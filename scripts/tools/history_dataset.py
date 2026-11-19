"""Simulator-independent, causal history windows from ConTrack episode H5."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import numpy as np
import h5py


@dataclass
class Episode:
    identifier: str
    source: str
    digest: str
    state: np.ndarray
    action: np.ndarray
    log_parameters: np.ndarray
    condition: str


def extract_state(group):
    """No oracle observation tail, reference, phase, or object ground truth."""
    arrays, names = [], []
    for hand in sorted(group['states/hands'], key=int):
        for field in ('qpos', 'qvel'):
            value = group[f'states/hands/{hand}/{field}'][:]
            arrays.append(value)
            names.extend(f'hand{hand}/{field}/{i}' for i in range(value.shape[1]))
    for side in sorted(group['states/contacts']):
        for sensor in sorted(group[f'states/contacts/{side}'], key=int):
            value = group[f'states/contacts/{side}/{sensor}/net_force_w'][:]
            value = value.reshape(len(value), -1)
            arrays.append(value)
            names.extend(f'contact/{side}/{sensor}/net_force_w/{i}' for i in range(value.shape[1]))
    state = np.concatenate(arrays, axis=-1).astype(np.float32)
    if not np.isfinite(state).all():
        raise ValueError('Nonfinite joint/contact state')
    return state, names


def physical_label(physics):
    mass = np.asarray(physics.get('actual_mass_kg', physics.get('mass_kg')), dtype=float).reshape(-1)
    material = np.asarray(physics['material_static_dynamic_restitution'], dtype=float)
    if mass.size != 1 or not np.isfinite(mass).all() or mass[0] <= 0:
        raise ValueError('Expected one positive actual rigid-body mass')
    mu = material[..., 0].reshape(-1)
    if not len(mu) or not np.isfinite(mu).all() or np.any(mu <= 0):
        raise ValueError('History prototype needs positive object friction')
    if not np.allclose(mu, mu[0]) or not np.allclose(material[..., 1], mu[0]):
        raise ValueError('Prototype expects homogeneous, equal static/dynamic object friction')
    return np.log(np.array([mass[0], mu[0]], dtype=np.float32))


def load_episodes(inputs, tool):
    paths = set()
    for item in inputs:
        p = Path(item).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(p)
        paths.update([p] if p.is_file() else p.rglob('episodes.h5'))
    episodes, skipped, seen, schema, reference_paths, control_dt, checkpoints = [], [], set(), None, set(), set(), set()
    for path in sorted(paths):
        with h5py.File(path, 'r') as f:
            if 'episodes' not in f:
                skipped.append({'path':str(path),'reason':'not an episode collection'})
                continue
            metadata = json.loads(f.attrs['metadata_json'])
            reference = metadata['reference_path']
            if tool not in Path(reference).name.lower():
                skipped.append({'path':str(path),'reason':'other tool'})
                continue
            if not f.attrs.get('collection_complete', False):
                raise ValueError(f'Incomplete collection: {path}')
            if metadata.get('diagnostic_no_self_collisions') or metadata.get('robot_self_collisions') is False:
                skipped.append({'path':str(path),'reason':'diagnostic no-self-collisions'})
                continue
            reference_paths.add(Path(reference).name)
            checkpoints.add(metadata.get('checkpoint'))
            control_dt.add(float(f.attrs['control_dt']))
            label = physical_label(json.loads(f.attrs['physics_json']))
            # Readback can differ by a few ulps between collection entry points.
            condition = f'mass_kg={np.exp(label[0]):.6g},friction={np.exp(label[1]):.6g}'
            for name, group in f['episodes'].items():
                if not group.attrs.get('completed', False):
                    raise ValueError(f'Incomplete episode: {path}/{name}')
                state, features = extract_state(group)
                applied = group['transitions/applied_action'][:]
                # The policy action is a residual on a reference joint target.
                # Include the actual actuator targets too; a residual alone
                # does not specify the physical control input.
                targets = [group[f'transitions/hands/{h}/joint_position_target'][:]
                           for h in sorted(group['states/hands'],key=int)]
                action = np.concatenate([applied,*targets],axis=-1).astype(np.float32)
                done = group['transitions/done'][:].reshape(-1)
                if len(action) < 2 or len(state) != len(action)+1 or not np.isfinite(action).all():
                    raise ValueError(f'Invalid state/action alignment: {path}/{name}')
                if not done[-1] or np.any(done[:-1]):
                    raise ValueError(f'Episode crosses a reset: {path}/{name}')
                current = (features, action.shape[1],
                           json.loads(f.attrs.get('hand_mapping_json','null')),
                           json.loads(f.attrs.get('contact_mapping_json','null')))
                if schema is not None and current != schema:
                    raise ValueError('Incompatible feature or action mapping')
                schema = current
                digest = hashlib.sha256(state.tobytes()+action.tobytes()+condition.encode()).hexdigest()
                if digest in seen:
                    skipped.append({'path':f'{path}/{name}','reason':'exact duplicate episode'})
                    continue
                seen.add(digest)
                episodes.append(Episode(f'{path}::{name}',str(path),digest,state,action,label,condition))
    if not episodes:
        raise ValueError(f'No eligible {tool} episode collections')
    if len(reference_paths) != 1 or len(control_dt) != 1:
        raise ValueError('Use one reference clip and control frequency per prototype experiment')
    return episodes, dict(feature_names=schema[0], action_dim=schema[1], control_dt=next(iter(control_dt)),
                          hand_mapping=schema[2],contact_mapping=schema[3],
                          references=sorted(reference_paths), checkpoints=sorted(str(x) for x in checkpoints), skipped=skipped)


def split_episodes(episodes, seed=42, unit='episode'):
    """Stratified by condition; whole episodes/files never cross splits."""
    rng = np.random.default_rng(seed)
    result = {key:[] for key in ('train','validation','test')}
    for condition in sorted({e.condition for e in episodes}):
        ids = [i for i,e in enumerate(episodes) if e.condition==condition]
        units = {}
        for i in ids:
            key = episodes[i].source if unit=='file' else episodes[i].identifier
            units.setdefault(key,[]).append(i)
        keys = sorted(units)
        if len(keys)<3:
            raise ValueError(f'{condition}: need >=3 distinct {unit}s for train/validation/test, got {len(keys)}')
        rng.shuffle(keys)
        held = max(1, int(round(.2*len(keys))))
        for name, chosen in [('test',keys[:held]),('validation',keys[held:2*held]),('train',keys[2*held:])]:
            for key in chosen:
                result[name].extend(units[key])
    if unit=='file':
        sources = {k:{episodes[i].source for i in v} for k,v in result.items()}
        if any(sources[a]&sources[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
            raise ValueError('A multi-condition file crosses splits; use separate collection batches')
    return result


def fit_normalization(episodes, train_ids):
    state = np.concatenate([episodes[i].state for i in train_ids]).astype(np.float64)
    action = np.concatenate([episodes[i].action for i in train_ids]).astype(np.float64)
    labels = np.stack([episodes[i].log_parameters for i in train_ids]).astype(np.float64)
    return dict(state_mean=state.mean(0).astype(np.float32),
                state_std=np.maximum(state.std(0),.01).astype(np.float32),
                action_mean=action.mean(0).astype(np.float32),
                action_std=np.maximum(action.std(0),.01).astype(np.float32),
                parameter_mean=labels.mean(0).astype(np.float32),
                parameter_std=np.maximum(labels.std(0),.1).astype(np.float32))


def restore_manifest_split(episodes, info, manifest):
    """Restore exact episode membership by content hash, even after copying files."""
    for key in ('feature_names','action_dim','control_dt','references','hand_mapping','contact_mapping'):
        if info.get(key) != manifest.get(key):
            raise ValueError(f'Frozen manifest schema mismatch: {key}')
    available = {e.digest:e for e in episodes}
    saved = manifest['episodes']
    digests = [e['digest'] for e in saved]
    if len(set(digests)) != len(digests):
        raise ValueError('Frozen manifest contains duplicate episodes')
    missing = set(digests)-available.keys()
    if missing:
        raise ValueError(f'{len(missing)} frozen episodes missing or changed; supply original collection files')
    ordered = [available[d] for d in digests]
    old_index = {e['id']:i for i,e in enumerate(saved)}
    result = {key:[old_index[x] for x in manifest['splits'][key]] for key in ('train','validation','test')}
    assigned = sum(result.values(),[])
    if sorted(assigned) != list(range(len(ordered))) or any(not v for v in result.values()):
        raise ValueError('Frozen manifest splits must partition all episodes exactly once')
    return ordered, result


def window(episode, time, history, stats, normalized=None):
    """Decision at t: completed transitions <t only; target is delta x[t:t+1]."""
    if not 1 <= time < len(episode.action):
        raise IndexError(time)
    start = max(0, time-history)
    if normalized is None:
        x = (episode.state-stats['state_mean'])/stats['state_std']
        dx = np.diff(episode.state,axis=0)/stats['state_std']
        action = (episode.action-stats['action_mean'])/stats['action_std']
    else:
        x, dx, action = normalized
    token = np.concatenate([x[start:time],action[start:time],dx[start:time]],axis=-1)
    padded = np.zeros((history, token.shape[-1]), dtype=np.float32)
    padded[:len(token)] = token
    return dict(history=padded,length=np.int64(len(token)),state=x[time].copy(),
                action=action[time].copy(),delta=dx[time].copy(),
                label=((episode.log_parameters-stats['parameter_mean'])/stats['parameter_std']).copy())


class HistoryWindows:
    """Map-style dataset usable with torch DataLoader without importing torch."""
    def __init__(self, episodes, ids, stats, history=32, stride=2):
        self.episodes, self.stats, self.history = episodes, stats, history
        self.index = [(i,t) for i in ids for t in range(1,len(episodes[i].action),stride)]
        self.normalized = {i:((episodes[i].state-stats['state_mean'])/stats['state_std'],
                              np.diff(episodes[i].state,axis=0)/stats['state_std'],
                              (episodes[i].action-stats['action_mean'])/stats['action_std']) for i in ids}

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        i, t = self.index[index]
        value = window(self.episodes[i],t,self.history,self.stats,self.normalized[i])
        value['episode'] = np.int64(i)
        return value


def make_manifest(episodes, splits, info, history, stride, unit):
    return dict(schema='contrack_history_v1',history_steps=history,stride=stride,split_unit=unit,
                warning='Episode splits share collection batches; not independent batch or unseen-condition validation.' if unit=='episode' else 'Held-out collection files, same physical conditions.',
                input='joint positions/velocities + simulated net link forces + past applied residual actions, actual actuator joint targets, and observed state deltas; no reference sequence/object ground truth/oracle inputs',
                targets='log(actual_mass_kg), log(object_material_friction); normalized next-state delta',
                **info,episodes=[dict(id=e.identifier,source=e.source,digest=e.digest,condition=e.condition,steps=len(e.action)) for e in episodes],
                splits={key:[episodes[i].identifier for i in ids] for key,ids in splits.items()},
                windows={key:sum(len(range(1,len(episodes[i].action),stride)) for i in ids) for key,ids in splits.items()})
