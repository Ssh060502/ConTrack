"""Force audit and time-resolved parameter predictions; no simulator imports."""
from __future__ import annotations
import numpy as np


def force_series(episode, feature_names):
    ids=[i for i,name in enumerate(feature_names) if name.startswith('contact/')]
    if not ids:
        raise ValueError('No contact force features in this dataset')
    if len(ids)%3:
        raise ValueError('Expected XYZ triplets in contact features')
    vectors=episode.state[:,ids].reshape(len(episode.state),-1,3)
    norms=np.linalg.norm(vectors,axis=-1)
    return norms, [feature_names[ids[i]].rsplit('/',1)[0] for i in range(0,len(ids),3)]


def audit_forces(episodes, names, dt, threshold=100.):
    rows=[]
    for episode in episodes:
        norms,sensors=force_series(episode,names)
        frame,sensor=np.unravel_index(np.argmax(norms),norms.shape)
        rows.append(dict(episode=episode.identifier,source=episode.source,condition=episode.condition,
                         peak_force_N=float(norms[frame,sensor]),peak_state_index=int(frame),
                         peak_time_s=float(frame*dt),peak_sensor=sensors[sensor],
                         initial_peak_force_N=float(norms[0].max()),
                         post_initial_peak_force_N=float(norms[1:].max()),
                         p99_force_N=float(np.quantile(norms,.99)),
                         samples_above_threshold=int((norms>threshold).sum()),
                         vector_samples=int(norms.size)))
    return dict(threshold_N=threshold,
                note='Diagnostic threshold only, not a physical-validity cutoff. '
                     'Net link forces can include self/table contacts, not only hand-object contact. '
                     'State index 0 is the reset snapshot. No samples are removed or clipped.',
                episodes=rows)


def predict_timeline(model, episodes, ids, stats, history, stride, device, batch_size):
    import torch
    from torch.utils.data import DataLoader
    from scripts.tools.history_dataset import HistoryWindows
    dataset=HistoryWindows(episodes,ids,stats,history,stride)
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    mean=torch.as_tensor(stats['parameter_mean'],device=device)
    std=torch.as_tensor(stats['parameter_std'],device=device)
    rows=[];offset=0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            z=model.encode(batch['history'].to(device),batch['length'].to(device))
            prediction=(model.parameters_head(z)*std+mean).clamp(-20,20).exp().cpu().numpy()
            for i,estimate in enumerate(prediction):
                episode_id,time=dataset.index[offset+i]
                if int(batch['episode'][i]) != episode_id:
                    raise RuntimeError('Timeline data ordering changed')
                actual=np.exp(episodes[episode_id].log_parameters)
                rows.append(dict(episode_id=episode_id,state_index=time,
                                 history_length=min(time,history),
                                 actual_mass_kg=float(actual[0]),predicted_mass_kg=float(estimate[0]),
                                 actual_friction=float(actual[1]),predicted_friction=float(estimate[1])))
            offset+=len(prediction)
    return rows
