"""Evaluate the four explicitly selected mug/waterbottle adapters, sequentially."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

PROFILES = {
    'waterbottle_mass': ('waterbottle','waterbottle_train_adapter_20261118_195820_40067b','model_1000.pt',[(1.,2.),(3.,2.)]),
    'waterbottle_friction': ('waterbottle','waterbottle_train_adapter_20261118_200151_6ea0de','adapter_final.pt',[(1.,2.),(1.,1.),(1.,.7)]),
    'mug_mass': ('mug','mug_train_adapter_20261118_200610_a23719','adapter_final.pt',[(1.,2.),(3.,2.)]),
    'mug_friction': ('mug','mug_train_adapter_20261118_200320_5741a4','model_1500.pt',[(1.,2.),(1.,1.),(1.,.7)]),
}


def build_plan(repo, experiments, include_base=True):
    jobs=[];baselines=set()
    for experiment in experiments:
        tool,run,model,conditions=PROFILES[experiment]
        checkpoint=str(repo/'logs/oracle_adapter'/run/model)
        for mass,friction in conditions:
            condition=f'{mass:g}:{friction:g}'
            suffix=f'mass{mass:g}_friction{friction:g}'
            if include_base and (tool,condition) not in baselines:
                jobs.append(dict(name=f'{tool}_base_{suffix}',tool=tool,policy='base',condition=condition,checkpoint=None))
                baselines.add((tool,condition))
            label=Path(model).stem.replace('adapter_','')
            jobs.append(dict(name=f'{experiment}_{label}_{suffix}',tool=tool,policy='adapter',
                             condition=condition,checkpoint=checkpoint))
    return jobs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['all',*PROFILES],required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--episodes',type=int,default=10)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--adapter-only',action='store_true',help='Skip original-policy baselines')
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--dry-run',action='store_true',help='Print plan without simulation or server asset checks')
    parser.add_argument('--output-root',default='outputs/adapter_evaluation')
    args=parser.parse_args()
    if args.gpu<0 or args.episodes<=0:parser.error('gpu must be nonnegative and episodes positive')
    repo=Path(__file__).resolve().parents[1]
    experiments=list(PROFILES) if args.experiment=='all' else [args.experiment]
    jobs=build_plan(repo,experiments,not args.adapter_only)
    for job in jobs:
        print(f'{job["name"]}: condition={job["condition"]}; checkpoint={job["checkpoint"] or "original base policy"}',flush=True)
    print(f'{len(jobs)} sequential evaluations, {args.episodes} episodes each; GPU {args.gpu}.',flush=True)
    if args.dry_run:
        print('Dry run only; server checkpoint/data existence not checked.')
        return
    for job in jobs:
        if job['checkpoint'] and not Path(job['checkpoint']).is_file():
            raise FileNotFoundError(job['checkpoint'])
    output=Path(args.output_root).expanduser().resolve()/f'selected_adapters_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}'
    output.mkdir(parents=True)
    for job in jobs:job['output']=str(output/job['name'])
    (output/'plan.json').write_text(json.dumps(dict(jobs=jobs,episodes=args.episodes,seed=args.seed,gpu=args.gpu),indent=2))
    environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import h5py
    for job in jobs:
        command=[sys.executable,'-X','faulthandler','-u',str(repo/'scripts/rsl_rl/oracle_adapter.py'),
                 '--tool',job['tool'],'--mode','evaluate','--policy',job['policy'],'--num_envs','1',
                 '--conditions',job['condition'],'--episodes',str(args.episodes),'--seed',str(args.seed),
                 '--device','cuda:0','--headless','--output-dir',job['output']]
        if job['checkpoint']:command.extend(['--adapter_checkpoint',job['checkpoint']])
        if not args.no_video:command.append('--video')
        print(f'\n[batch] Starting {job["name"]}\nOutputs: {job["output"]}',flush=True)
        with (output/(job['name']+'.log')).open('w') as log:
            with subprocess.Popen(command,cwd=repo,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                  text=True,errors='replace',bufsize=1) as process:
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                code=process.wait()
        if code:raise RuntimeError(f'{job["name"]} exited with code {code}; see batch log')
        # Isaac cleanup can mask Python exceptions with exit code 0. Require
        # completed H5 output rather than treating that exit code as success.
        with h5py.File(Path(job['output'])/'episodes.h5','r') as f:
            if not f.attrs.get('collection_complete') or int(f.attrs['completed_episodes'])!=args.episodes:
                raise RuntimeError(f'{job["name"]}: incomplete collection')
            metadata=json.loads(f.attrs['metadata_json'])
            if metadata['evaluation_policy']!=job['policy']:
                raise RuntimeError('Evaluation policy metadata mismatch')
            if job['checkpoint'] and metadata['checkpoint']!=job['checkpoint']:
                raise RuntimeError('Evaluation checkpoint metadata mismatch')
        job['complete']=True
        (output/'plan.json').write_text(json.dumps(dict(jobs=jobs,episodes=args.episodes,seed=args.seed,gpu=args.gpu),indent=2))
    print(f'Finished. All evaluations: {output}',flush=True)


if __name__=='__main__':main()
