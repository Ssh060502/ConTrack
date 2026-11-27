"""Train history=1/32 sequentially with one frozen dataset and matched settings."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import uuid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run',required=True)
    parser.add_argument('--inputs',nargs='+',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--epochs',type=int,help='Defaults to original run epochs')
    parser.add_argument('--histories',nargs='+',type=int,default=[1,32])
    parser.add_argument('--logger',choices=('none','wandb'),default='wandb')
    parser.add_argument('--project',default='ConTrack-history')
    parser.add_argument('--output-root',default='outputs/history_comparison')
    args=parser.parse_args()
    reference=Path(args.reference_run).expanduser().resolve()
    config=json.loads((reference/'config.json').read_text())
    manifest=reference/'manifest.json'
    if not manifest.is_file():parser.error('Missing reference manifest.json')
    if min(args.histories)<=0 or len(set(args.histories))!=len(args.histories):
        parser.error('Use distinct positive history lengths')
    epochs=args.epochs if args.epochs is not None else config['epochs']
    if epochs<=0:parser.error('epochs must be positive')
    destination=Path(args.output_root).expanduser().resolve()/f'history_comparison_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}'
    destination.mkdir(parents=True)
    train=Path(__file__).resolve().parent/'train_history_encoder.py'
    commands=[];rows=[]
    for history in args.histories:
        output=destination/f'history_{history}'
        command=[sys.executable,'-u',str(train),'--tool',config['tool'],'--robot',config.get('robot','rby1a'),'--inputs',*args.inputs,
                 '--state-features',config.get('state_features','all'),
                 '--split-manifest',str(manifest),'--history',str(history),'--epochs',str(epochs),
                 '--device',args.device,'--logger',args.logger,'--project',args.project,'--output-dir',str(output)]
        for key in ('seed','stride','batch_size','learning_rate','dynamics_weight'):
            command.extend(['--'+key.replace('_','-'),str(config[key])])
        commands.append(command)
        (destination/'commands.json').write_text(json.dumps(commands,indent=2))
        print(f'[comparison] Starting history={history}, epochs={epochs}; sequential execution.',flush=True)
        subprocess.run(command,check=True)
        metrics=json.loads((output/'test_metrics.json').read_text())
        best_epoch=min((json.loads(line) for line in (output/'metrics.jsonl').read_text().splitlines()),
                       key=lambda x:x['validation/loss'])['epoch']
        row={'history':history,'best_epoch':best_epoch,'run':str(output),**metrics}
        rows.append(row)
        (destination/'comparison.json').write_text(json.dumps(rows,indent=2))
    with (destination/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for row in rows:
        print(f'[comparison] history={row["history"]} best_epoch={row["best_epoch"]} '
              f'mass_mae={row["mass_mae_kg"]:.5f}kg friction_mae={row["friction_mae"]:.4f} '
              f'dynamics_mse={row["dynamics_mse_normalized"]:.5f}',flush=True)
        for group in ('qpos','qvel','contact_force'):
            key=f'dynamics/{group}/'
            if key+'mse_normalized' in row:
                print(f'  {group}: MSE={row[key+"mse_normalized"]:.5f}, '
                      f'zero baseline={row[key+"zero_delta_mse_normalized"]:.5f}',flush=True)
    print(f'Finished. Comparison: {destination}',flush=True)


if __name__=='__main__':main()
