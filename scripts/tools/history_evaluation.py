"""Shared history metrics, including feature-wise and physical-unit baselines."""
from __future__ import annotations
import numpy as np


def feature_groups(names):
    groups = {key:[] for key in ('qpos','qvel','contact_force')}
    for i,name in enumerate(names):
        if '/qpos/' in name: groups['qpos'].append(i)
        elif '/qvel/' in name: groups['qvel'].append(i)
        elif name.startswith('contact/'): groups['contact_force'].append(i)
        else: raise ValueError(f'Unknown dynamics feature: {name}')
    return {k:v for k,v in groups.items() if v}


def aggregate_episodes(per_episode, episodes):
    """Equal episode weights; each row is already a window mean for one episode."""
    keys = list(next(iter(per_episode.values())))
    metrics = {k:float(np.mean([v[k] for v in per_episode.values()])) for k in keys}
    metrics['episode_count'] = len(per_episode)
    for condition in sorted({episodes[i].condition for i in per_episode}):
        selected = [v for i,v in per_episode.items() if episodes[i].condition==condition]
        metrics.update({f'{condition}/{k}':float(np.mean([v[k] for v in selected])) for k in keys})
        metrics[f'{condition}/episode_count'] = len(selected)
    return metrics


def evaluate_history(model, loader, episodes, stats, names, device, dynamics_weight=1.):
    import torch
    from torch.nn import functional as F
    model.eval()
    mean = torch.as_tensor(stats['parameter_mean'],device=device)
    std = torch.as_tensor(stats['parameter_std'],device=device)
    state_std = torch.as_tensor(stats['state_std'],device=device)
    constant = mean.exp()
    groups = feature_groups(names)
    collected = {}
    total_loss, count = 0., 0
    with torch.no_grad():
        for batch in loader:
            data = {k:v.to(device) for k,v in batch.items() if k!='episode'}
            pred,dynamics,_ = model(data['history'],data['length'],data['state'],data['action'])
            parameter_loss = F.smooth_l1_loss(pred,data['label'])
            dynamics_loss = F.smooth_l1_loss(dynamics,data['delta'])
            loss = parameter_loss+dynamics_weight*dynamics_loss
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite evaluation loss')
            actual = (data['label']*std+mean).exp()
            estimate = (pred*std+mean).clamp(-20,20).exp()
            difference = estimate-actual
            error = dynamics-data['delta']
            values = dict(
                mass_mae_kg=difference[:,0].abs(),friction_mae=difference[:,1].abs(),
                constant_mass_mae_kg=(constant[0]-actual[:,0]).abs(),
                constant_friction_mae=(constant[1]-actual[:,1]).abs(),
                mass_relative_mae=difference[:,0].abs()/actual[:,0],
                friction_relative_mae=difference[:,1].abs()/actual[:,1],
                mass_bias_kg=difference[:,0],friction_bias=difference[:,1],
                mass_prediction_mean_kg=estimate[:,0],friction_prediction_mean=estimate[:,1],
                dynamics_mse_normalized=error.square().mean(-1),
                zero_delta_mse_normalized=data['delta'].square().mean(-1),
            )
            for group,ids in groups.items():
                scale = state_std[ids]
                err, target = error[:,ids], data['delta'][:,ids]
                prefix = f'dynamics/{group}/'
                values.update({prefix+'mse_normalized':err.square().mean(-1),
                               prefix+'zero_delta_mse_normalized':target.square().mean(-1),
                               prefix+'mse_physical':(err*scale).square().mean(-1),
                               prefix+'zero_delta_mse_physical':(target*scale).square().mean(-1),
                               prefix+'smooth_l1_normalized':F.smooth_l1_loss(dynamics[:,ids],target,reduction='none').mean(-1),
                               prefix+'zero_delta_smooth_l1_normalized':F.smooth_l1_loss(torch.zeros_like(target),target,reduction='none').mean(-1)})
            keys = list(values)
            array = torch.stack(list(values.values()),dim=-1).cpu().numpy()
            for eid,row in zip(batch['episode'].tolist(),array):
                collected.setdefault(eid,[]).append(row)
            total_loss += float(loss)*len(pred)
            count += len(pred)
    per_episode = {i:dict(zip(keys,np.mean(rows,axis=0).tolist())) for i,rows in collected.items()}
    metrics = aggregate_episodes(per_episode,episodes)
    # Preserve original checkpoint selection: window-weighted combined SmoothL1.
    metrics['loss'] = total_loss/count
    metrics['window_count'] = count
    return metrics, per_episode
