"""Re-evaluate an existing history checkpoint with component diagnostics; no training."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.tools.history_dataset import load_episodes,restore_manifest_split,HistoryWindows
from scripts.tools.history_evaluation import evaluate_history


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,help='Original history run containing manifest.json and history_best.pt')
    parser.add_argument('--inputs',nargs='+',required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from rsl_rl_contrack.modules.history_encoder import HistoryEncoder
    run=Path(args.run).expanduser().resolve()
    saved=torch.load(run/'history_best.pt',map_location=args.device,weights_only=False)
    config=saved['config'];manifest=json.loads((run/'manifest.json').read_text())
    episodes,info=load_episodes(args.inputs,config['tool'],config.get('robot','rby1a'))
    episodes,splits=restore_manifest_split(episodes,info,manifest)
    architecture=saved['architecture']
    if architecture['state_dim']!=len(info['feature_names']) or architecture['action_dim']!=info['action_dim']:
        raise ValueError('Checkpoint feature dimensions do not match data')
    if saved['feature_names']!=info['feature_names']:
        raise ValueError('Checkpoint state feature order changed')
    model=HistoryEncoder(**architecture).to(args.device)
    model.load_state_dict(saved['model_state_dict'])
    destination=run/'diagnostics'/f'{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}'
    destination.mkdir(parents=True)
    result={'checkpoint':str(run/'history_best.pt'),'epoch':saved['epoch'],
            'note':'Descriptive re-analysis of existing splits, not a fresh independent test.', 'splits':{}}
    rows=[];condition_rows=[]
    for split in ('validation','test'):
        dataset=HistoryWindows(episodes,splits[split],saved['normalization'],architecture['history'],config['stride'])
        loader=DataLoader(dataset,batch_size=config['batch_size'],shuffle=False,num_workers=0)
        metrics,per_episode=evaluate_history(model,loader,episodes,saved['normalization'],info['feature_names'],
                                             args.device,config['dynamics_weight'])
        result['splits'][split]=metrics
        for eid,values in per_episode.items():
            rows.append({'split':split,'episode':episodes[eid].identifier,'condition':episodes[eid].condition,**values})
        print(f'\n[{split}] checkpoint epoch={saved["epoch"]}, history={architecture["history"]}')
        for condition in sorted({episodes[i].condition for i in per_episode}):
            values={key[len(condition)+1:]:value for key,value in metrics.items() if key.startswith(condition+'/')}
            condition_rows.append({'split':split,'condition':condition,**values})
            print(f'  {condition}: mass MAE={values["mass_mae_kg"]:.5f}kg (baseline {values["constant_mass_mae_kg"]:.5f}), '
                  f'friction MAE={values["friction_mae"]:.4f} (baseline {values["constant_friction_mae"]:.4f})')
        for group in ('qpos','qvel','contact_force'):
            prefix=f'dynamics/{group}/'
            if prefix+'mse_normalized' in metrics:
                print(f'  {group}: normalized MSE={metrics[prefix+"mse_normalized"]:.5f}, '
                      f'zero baseline={metrics[prefix+"zero_delta_mse_normalized"]:.5f}; '
                      f'SmoothL1={metrics[prefix+"smooth_l1_normalized"]:.5f}, '
                      f'zero baseline={metrics[prefix+"zero_delta_smooth_l1_normalized"]:.5f}')
    (destination/'diagnostics.json').write_text(json.dumps(result,indent=2))
    for filename,values in [('episodes.csv',rows),('conditions.csv',condition_rows)]:
        with (destination/filename).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    print(f'\nFinished. Diagnostics: {destination}')


if __name__=='__main__':main()
