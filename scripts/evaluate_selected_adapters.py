"""Evaluate the four explicitly selected mug/waterbottle adapters, sequentially."""
from __future__ import annotations
import argparse
from datetime import datetime
import json
import math
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


def build_plan(repo, experiments, include_base=True, conditions=None):
    jobs=[];baselines=set()
    for experiment in experiments:
        tool,run,model,defaults=PROFILES[experiment]
        checkpoint=str(repo/'logs/oracle_adapter'/run/model)
        for mass,friction in (conditions if conditions is not None else defaults):
            condition=f'{mass:g}:{friction:g}'
            suffix=f'mass{mass:g}_friction{friction:g}'
            if include_base and (tool,condition) not in baselines:
                jobs.append(dict(name=f'{tool}_base_{suffix}',tool=tool,policy='base',condition=condition,checkpoint=None))
                baselines.add((tool,condition))
            label=Path(model).stem.replace('adapter_','')
            jobs.append(dict(name=f'{experiment}_{label}_{suffix}',tool=tool,policy='adapter',
                             condition=condition,checkpoint=checkpoint))
    return jobs


def parse_conditions(text):
    result=[]
    for item in text.split(','):
        pair=tuple(map(float,item.split(':')))
        if len(pair)!=2 or any(not math.isfinite(x) or x<=0 for x in pair):
            raise ValueError('Use positive finite mass:friction pairs')
        result.append(pair)
    if len(set(result))!=len(result):raise ValueError('Duplicate conditions')
    return result


def expand_plan(jobs,seeds=None,compare_nominal=False):
    result=[]
    for seed in seeds if seeds is not None else [None]:
        for original in jobs:
            job=dict(original)
            if seed is not None:
                job['seed']=seed;job['name']=f'seed{seed}_'+job['name']
            result.append(job)
            mass=float(job['condition'].split(':')[0])
            if compare_nominal and job['policy']=='adapter' and mass!=1.:
                fixed=dict(job)
                fixed['name']+='_oracle_nominal_mass'
                fixed['oracle_mass_scale']=1.
                result.append(fixed)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment',choices=['all',*PROFILES],required=True)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--episodes',type=int,default=10)
    seeds=parser.add_mutually_exclusive_group()
    seeds.add_argument('--seed',type=int,default=42)
    seeds.add_argument('--seeds',type=int,nargs='+',help='Each seed is evaluated in fresh processes')
    parser.add_argument('--conditions',help='Override conditions for one experiment, e.g. 2:2,4:2,6:2')
    parser.add_argument('--compare-nominal-mass-input',action='store_true',help='Add adapter runs with nominal mass INPUT while actual simulator mass stays unchanged')
    parser.add_argument('--adapter-only',action='store_true',help='Skip original-policy baselines')
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--dry-run',action='store_true',help='Print plan without simulation or server asset checks')
    parser.add_argument('--output-root',default='outputs/adapter_evaluation')
    args=parser.parse_args()
    if args.gpu<0 or args.episodes<=0:parser.error('gpu must be nonnegative and episodes positive')
    if args.seed<0 or (args.seeds and (min(args.seeds)<0 or len(set(args.seeds))!=len(args.seeds))):
        parser.error('Use distinct nonnegative seeds')
    if args.conditions and args.experiment=='all':parser.error('--conditions requires one specific experiment')
    try:
        conditions=parse_conditions(args.conditions) if args.conditions else None
    except ValueError as exc:parser.error(str(exc))
    repo=Path(__file__).resolve().parents[1]
    experiments=list(PROFILES) if args.experiment=='all' else [args.experiment]
    jobs=expand_plan(build_plan(repo,experiments,not args.adapter_only,conditions),args.seeds,args.compare_nominal_mass_input)
    for job in jobs:
        print(f'{job["name"]}: actual={job["condition"]}; network mass input={job.get("oracle_mass_scale","actual readback")}; checkpoint={job["checkpoint"] or "original base policy"}',flush=True)
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
    settings=dict(episodes=args.episodes,seed=args.seed,seeds=args.seeds or [args.seed],gpu=args.gpu,
                  compare_nominal_mass_input=args.compare_nominal_mass_input)
    (output/'plan.json').write_text(json.dumps(dict(jobs=jobs,**settings),indent=2))
    environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    import h5py
    import numpy as np
    actual_by_condition={}
    for job in jobs:
        command=[sys.executable,'-X','faulthandler','-u',str(repo/'scripts/rsl_rl/oracle_adapter.py'),
                 '--tool',job['tool'],'--mode','evaluate','--policy',job['policy'],'--num_envs','1',
                 '--conditions',job['condition'],'--episodes',str(args.episodes),'--seed',str(job.get('seed',args.seed)),
                 '--device','cuda:0','--headless','--output-dir',job['output']]
        if job['checkpoint']:command.extend(['--adapter_checkpoint',job['checkpoint']])
        if 'oracle_mass_scale' in job:command.extend(['--oracle-mass-scale',str(job['oracle_mass_scale'])])
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
            physical=json.loads(f.attrs['physics_json'])
            if metadata['evaluation_policy']!=job['policy']:
                raise RuntimeError('Evaluation policy metadata mismatch')
            if job['checkpoint'] and metadata['checkpoint']!=job['checkpoint']:
                raise RuntimeError('Evaluation checkpoint metadata mismatch')
            if metadata.get('oracle_mass_scale_override')!=job.get('oracle_mass_scale'):
                raise RuntimeError('Oracle input override metadata mismatch')
            if metadata['conditions']!=[list(map(float,job['condition'].split(':')))]:
                raise RuntimeError('Actual physical condition metadata mismatch')
            if metadata['seed']!=job.get('seed',args.seed):
                raise RuntimeError('Evaluation seed mismatch')
            expected=actual_by_condition.setdefault((job['tool'],job['condition']),physical)
            for field in ('actual_mass_kg','actual_inertia','material_static_dynamic_restitution'):
                if not np.allclose(physical[field],expected[field],rtol=1e-6,atol=1e-8):
                    raise RuntimeError(f'Physical properties changed between control runs: {field}')
            actual_mass=float(physical['actual_mass_kg'][0][0])
            physical_scale=float(job['condition'].split(':')[0])
            expected_input=actual_mass/physical_scale*job['oracle_mass_scale'] if 'oracle_mass_scale' in job else actual_mass
            if not np.allclose(metadata['oracle_input_mass_kg'],[expected_input],rtol=1e-6,atol=1e-8):
                raise RuntimeError('Network mass input readback mismatch')
        job['complete']=True
        (output/'plan.json').write_text(json.dumps(dict(jobs=jobs,**settings),indent=2))
    print(f'Finished. All evaluations: {output}',flush=True)


if __name__=='__main__':main()
