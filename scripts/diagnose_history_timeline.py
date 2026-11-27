"""Audit force spikes and plot parameter predictions over each held-out episode."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tools.history_dataset import load_episodes,restore_manifest_split,select_state_features
from scripts.tools.history_timeline import audit_forces,force_series,predict_timeline


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--inputs',nargs='+',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--splits',nargs='+',choices=('train','validation','test'),default=['validation','test'])
    parser.add_argument('--plots',action='store_true',help='Write PNG figures (requires matplotlib)')
    parser.add_argument('--force-threshold',type=float,default=100.,help='Reporting threshold only, N')
    parser.add_argument('--stride',type=int,help='Defaults to saved window stride; 1 predicts every control step')
    args=parser.parse_args()
    import math
    if not math.isfinite(args.force_threshold) or args.force_threshold<=0 or (args.stride is not None and args.stride<=0):
        parser.error('Force threshold and stride must be positive and finite')
    import torch
    from rsl_rl_contrack.modules.history_encoder import HistoryEncoder
    run=Path(args.run).expanduser().resolve()
    saved=torch.load(run/'history_best.pt',map_location=args.device,weights_only=False)
    cfg=saved['config'];arch=saved['architecture']
    episodes,info=load_episodes(args.inputs,cfg['tool'],cfg.get('robot','rby1a'))
    manifest=json.loads((run/'manifest.json').read_text())
    episodes,splits=restore_manifest_split(episodes,info,manifest)
    force_episodes,force_names=episodes,info['feature_names']
    episodes,info=select_state_features(episodes,info,cfg.get('state_features','all'))
    if saved['feature_names']!=info['feature_names'] or arch['action_dim']!=info['action_dim']:
        raise ValueError('Checkpoint and collection feature schema differ')
    model=HistoryEncoder(**arch).to(args.device)
    model.load_state_dict(saved['model_state_dict'],strict=True)
    destination=run/'timeline_diagnostics'/f'{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}'
    destination.mkdir(parents=True)
    audit=audit_forces(force_episodes,force_names,info['control_dt'],args.force_threshold)
    (destination/'force_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    ranked=sorted(audit['episodes'],key=lambda r:r['peak_force_N'],reverse=True)
    print('[force] Audit includes all original split episodes; top peaks:',flush=True)
    for row in ranked[:10]:
        print(f"  {row['episode']}: peak={row['peak_force_N']:.3f}N "
              f"at state={row['peak_state_index']} ({row['peak_time_s']:.3f}s), "
              f"initial={row['initial_peak_force_N']:.3f}N, post-initial={row['post_initial_peak_force_N']:.3f}N",flush=True)
    plt=None
    if args.plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        (destination/'plots').mkdir()
    all_rows=[]
    for split in dict.fromkeys(args.splits):
        rows=predict_timeline(model,episodes,splits[split],saved['normalization'],arch['history'],
                              args.stride or cfg['stride'],args.device,cfg['batch_size'])
        for row in rows:
            episode=episodes[row['episode_id']]
            row.update(split=split,episode=episode.identifier,condition=episode.condition,
                       time_s=row['state_index']*info['control_dt'])
        all_rows.extend(rows)
        for episode_id in splits[split]:
            selected=[r for r in rows if r['episode_id']==episode_id]
            if not selected:continue
            if plt is not None:
                episode=episodes[episode_id]
                norms,_=force_series(force_episodes[episode_id],force_names)
                times=[r['time_s'] for r in selected]
                figure,axes=plt.subplots(3,1,figsize=(11,8),sharex=True)
                for ax,key,label in zip(axes[:2],('mass_kg','friction'),('Mass (kg)','Object friction')):
                    ax.plot(times,[r['predicted_'+key] for r in selected],label='predicted')
                    ax.axhline(selected[0]['actual_'+key],color='black',linestyle='--',label='actual label')
                    ax.set_ylabel(label);ax.legend();ax.grid(alpha=.25)
                axes[2].plot([i*info['control_dt'] for i in range(len(norms))],norms.max(axis=1))
                axes[2].set_yscale('symlog',linthresh=1.)
                axes[2].set_ylim(0,max(1.,float(norms.max())*1.1))
                axes[2].set_ylabel('Max link force (N)');axes[2].set_xlabel('Episode time (s)')
                axes[2].grid(alpha=.25)
                figure.suptitle(f"{split} episode {episode_id}: {episode.condition}\n"
                               f"Encoder inputs: {cfg.get('state_features','all')}; force below is diagnostic, not a grasp detector")
                figure.tight_layout()
                figure.savefig(destination/'plots'/f'{split}_episode_{episode_id:04d}.png',dpi=150)
                plt.close(figure)
    with (destination/'predictions_over_time.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
    (destination/'summary.json').write_text(json.dumps(dict(
        checkpoint=str(run/'history_best.pt'),epoch=saved['epoch'],window_count=len(all_rows),
        prediction_splits=args.splits,force_audit_episode_count=len(episodes),
        state_features=cfg.get('state_features','all'),
        control_dt=info['control_dt'],history=arch['history'],
        note='Existing held-out splits re-analyzed; not a new independent test. '
             'No training, force cleaning, or automatic grasp phase labels. '
             'Match time_s to the corresponding episode video for contact/failure timing.'),indent=2),encoding='utf-8')
    print(f'Finished. Timeline diagnostics: {destination}',flush=True)


if __name__=='__main__':main()
