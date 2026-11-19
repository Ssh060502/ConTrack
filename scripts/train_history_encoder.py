"""Offline history encoder pretraining; no Isaac Lab or simulation process."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import random
import sys
import uuid
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tools.history_dataset import load_episodes, split_episodes, fit_normalization, make_manifest, HistoryWindows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',nargs='+',required=True,help='Collection files or directories recursively containing episodes.h5')
    parser.add_argument('--tool',choices=('mug','waterbottle'),default='waterbottle')
    parser.add_argument('--prepare-only',action='store_true',help='Audit/split without importing torch')
    parser.add_argument('--split-unit',choices=('episode','file'),default='episode')
    parser.add_argument('--history',type=int,default=32)
    parser.add_argument('--stride',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--learning-rate',type=float,default=3e-4)
    parser.add_argument('--dynamics-weight',type=float,default=1.)
    parser.add_argument('--logger',choices=('none','wandb'),default='none')
    parser.add_argument('--project',default='ConTrack-history')
    parser.add_argument('--output-root',default='outputs/history_encoder')
    args=parser.parse_args()
    if (min(args.history,args.stride,args.epochs,args.batch_size)<=0
            or not math.isfinite(args.learning_rate) or args.learning_rate<=0
            or not math.isfinite(args.dynamics_weight) or args.dynamics_weight<0):
        parser.error('Counts/learning rate must be positive; dynamics weight nonnegative')
    episodes,info=load_episodes(args.inputs,args.tool)
    if len({e.condition for e in episodes})<2:
        parser.error('Need multiple physical conditions to test parameter inference')
    splits=split_episodes(episodes,args.seed,args.split_unit)
    stats=fit_normalization(episodes,splits['train'])
    manifest=make_manifest(episodes,splits,info,args.history,args.stride,args.split_unit)
    output=Path(args.output_root).expanduser().resolve()/f'{args.tool}_history_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}'
    output.mkdir(parents=True)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (output/'config.json').write_text(json.dumps(vars(args),indent=2))
    np.savez(output/'normalization.npz',**stats)
    print(f'[history] {len(episodes)} unique episodes; windows={manifest["windows"]}',flush=True)
    print(f'[history] {manifest["warning"]}',flush=True)
    print(f'[history] Outputs: {output}',flush=True)
    if args.prepare_only:
        print('[history] Preparation complete; no network training performed.',flush=True)
        return

    import torch
    from torch.utils.data import DataLoader
    from rsl_rl_contrack.modules.history_encoder import HistoryEncoder
    random.seed(args.seed);torch.manual_seed(args.seed);np.random.seed(args.seed)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for a small interface test')
    device=torch.device(args.device)
    state_dim=len(info['feature_names']);action_dim=info['action_dim']
    architecture=dict(state_dim=state_dim,action_dim=action_dim,history=args.history,width=96,layers=2,heads=4,latent_dim=64)

    datasets={name:HistoryWindows(episodes,ids,stats,args.history,args.stride) for name,ids in splits.items()}
    loaders={name:DataLoader(ds,batch_size=args.batch_size,shuffle=name=='train',num_workers=0) for name,ds in datasets.items()}
    model=HistoryEncoder(**architecture).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=1e-4)
    criterion=torch.nn.SmoothL1Loss()
    run=None
    if args.logger=='wandb':
        import wandb
        run=wandb.init(project=args.project,name=output.name,dir=str(output),config={**vars(args),**architecture})
    mean=torch.as_tensor(stats['parameter_mean'],device=device)
    std=torch.as_tensor(stats['parameter_std'],device=device)
    constant=mean.exp()

    def evaluate(loader):
        model.eval();counts=0;loss_sum=0.;metrics={};by_episode={}
        with torch.no_grad():
            for batch in loader:
                data={k:v.to(device) for k,v in batch.items() if k!='episode'}
                pred,dynamics,_=model(data['history'],data['length'],data['state'],data['action'])
                loss=criterion(pred,data['label'])+args.dynamics_weight*criterion(dynamics,data['delta'])
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite evaluation loss')
                actual=(data['label']*std+mean).exp()
                estimate=(pred*std+mean).clamp(-20,20).exp()
                values=torch.stack([(estimate[:,0]-actual[:,0]).abs(),(estimate[:,1]-actual[:,1]).abs(),
                                    (constant[0]-actual[:,0]).abs(),(constant[1]-actual[:,1]).abs(),
                                    (dynamics-data['delta']).square().mean(-1),data['delta'].square().mean(-1)],dim=-1).cpu().numpy()
                for eid,row in zip(batch['episode'].tolist(),values):
                    by_episode.setdefault(eid,[]).append(row)
                counts+=len(actual);loss_sum+=float(loss)*len(actual)
        keys=['mass_mae_kg','friction_mae','constant_mass_mae_kg','constant_friction_mae','dynamics_mse_normalized','zero_delta_mse_normalized']
        # Equal weight per episode, rather than giving long clips more votes.
        per_episode={i:np.mean(v,axis=0) for i,v in by_episode.items()}
        metrics.update(zip(keys,np.mean(list(per_episode.values()),axis=0).tolist()))
        metrics['loss']=loss_sum/counts
        for cond in sorted({episodes[i].condition for i in per_episode}):
            value=np.mean([v for i,v in per_episode.items() if episodes[i].condition==cond],axis=0)
            metrics.update({f'{cond}/{k}':float(x) for k,x in zip(keys,value)})
        return metrics

    best=float('inf')
    try:
        with (output/'metrics.jsonl').open('w') as log:
            for epoch in range(1,args.epochs+1):
                model.train();total=0.;count=0
                for batch in loaders['train']:
                    data={k:v.to(device) for k,v in batch.items() if k!='episode'}
                    pred,dynamics,_=model(data['history'],data['length'],data['state'],data['action'])
                    loss=criterion(pred,data['label'])+args.dynamics_weight*criterion(dynamics,data['delta'])
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                    optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                    total+=float(loss.detach())*len(pred);count+=len(pred)
                val=evaluate(loaders['validation'])
                row={'epoch':epoch,'train/loss':total/count,**{f'validation/{k}':v for k,v in val.items()}}
                log.write(json.dumps(row)+'\n');log.flush()
                if run:run.log(row,step=epoch)
                print(f'[history] epoch={epoch} train={total/count:.4f} val={val["loss"]:.4f} mass_mae={val["mass_mae_kg"]:.4f}kg friction_mae={val["friction_mae"]:.4f}',flush=True)
                payload=dict(model_state_dict=model.state_dict(),architecture=architecture,normalization=stats,
                             feature_names=info['feature_names'],control_dt=info['control_dt'],config=vars(args),epoch=epoch,
                             data_schema=dict(version='contrack_history_v1',hand_mapping=info['hand_mapping'],
                                              contact_mapping=info['contact_mapping'],
                                              action_order='applied_residual_then_joint_targets_in_sorted_hand_order',
                                              parameter_order=['log_mass_kg','log_object_friction']),
                             manifest=str(output/'manifest.json'),validation=val)
                if val['loss']<best:
                    best=val['loss'];torch.save(payload,output/'history_best.pt')
                torch.save(payload,output/'history_last.pt')
        saved=torch.load(output/'history_best.pt',map_location=device,weights_only=False)
        model.load_state_dict(saved['model_state_dict'])
        test=evaluate(loaders['test'])
        (output/'test_metrics.json').write_text(json.dumps(test,indent=2))
        if run:run.summary.update({f'test/{k}':v for k,v in test.items()})
        print('[history] Best validation checkpoint evaluated once on held-out test episodes.',flush=True)
        print(f'Finished. Outputs: {output}',flush=True)
    finally:
        if run:run.finish()


if __name__=='__main__':
    main()
