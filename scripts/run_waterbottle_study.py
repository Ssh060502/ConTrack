"""Auditable RBY1A waterbottle A/B/C/oracle experiments; Isaac only in children."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
METHODS = ('state', 'random-history', 'history', 'oracle')
DEFAULT_INPUTS = [str(REPO/'outputs/physics_validation'/name) for name in (
    'waterbottle_collect_default_20261117_151415_orjsZQ',
    'waterbottle_collect_log-only_mass6.0_frictionoriginal_20261117_192239_SLlWLK',
    'waterbottle_collect_log-only_mass1.0_friction0.5_20261117_200322_KNPxIb',
    'waterbottle_collect_log-only_mass1.0_friction1.0_20261117_192522_dWFdod')]


def parse_conditions(text):
    pairs=[]
    for item in text.split(','):
        values=tuple(map(float,item.split(':')))
        if len(values)!=2 or any(not math.isfinite(x) or x<=0 for x in values):
            raise ValueError('Use positive finite mass_scale:friction pairs')
        pairs.append(values)
    if len(set(pairs))!=len(pairs):raise ValueError('Duplicate conditions')
    return pairs


def context_args(method, checkpoint, latent_dim=64, encoder_seed=123):
    if method not in (*METHODS, 'base'):
        raise ValueError(method)
    source = 'state' if method == 'base' else method
    result = ['--context-source', source]
    if source in ('history', 'random-history'):
        if not checkpoint:
            raise ValueError('B/C require --history-checkpoint')
        result += ['--history-checkpoint', str(checkpoint)]
    if source == 'random-history':
        result += ['--encoder-seed', str(encoder_seed)]
    if source == 'state':
        result += ['--context-dim', str(latent_dim)]
    return result


def execute(command, directory, environment):
    print('COMMAND:', json.dumps(command), flush=True)
    with Path(str(directory)+'.log').open('x', encoding='utf-8') as log:
        with subprocess.Popen(command,cwd=REPO,env=environment,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1) as process:
            finished = False
            for line in process.stdout:
                print(line,end='',flush=True);log.write(line);log.flush()
                finished |= line.startswith('Finished. Outputs:')
            code = process.wait()
    if code or not finished:
        raise RuntimeError(f'Child failed or missing completion marker: {directory} (exit={code})')


def audit(inputs, output):
    from scripts.tools.history_dataset import load_episodes, split_episodes
    episodes, info = load_episodes(inputs,'waterbottle','rby1a')
    counts = Counter(e.condition for e in episodes)
    split = split_episodes(episodes,42)
    result = dict(unique_episodes=len(episodes),conditions=dict(counts),schema=info,
                  sources=dict(Counter(e.source for e in episodes)),
                  episodes=[dict(id=e.identifier,digest=e.digest,condition=e.condition,steps=len(e.action)) for e in episodes],
                  split_counts={k:dict(Counter(episodes[i].condition for i in ids)) for k,ids in split.items()},
                  warnings=['Exact duplicates only: numerically near-identical episodes may remain.',
                            'Episode split shares collection batches; not independent batch validation.',
                            'These previously inspected data are development data, not a new blind test.',
                            'No success labels are inferred from time_out or final pose error.'])
    output=Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as file:json.dump(result,file,indent=2)
    print('Unique episodes:',len(episodes))
    for condition,count in counts.items():print(condition,count)
    print('Skipped:',len(info['skipped']))
    print('Audit:',output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('audit');a.add_argument('--inputs',nargs='+',default=DEFAULT_INPUTS)
    a.add_argument('--output',required=True)
    p=sub.add_parser('prepare',help='Train a joints-only, parameter-loss-only Encoder on explicit old base collections')
    p.add_argument('--inputs',nargs='+',default=DEFAULT_INPUTS)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--history',type=int,default=32)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--split-manifest')
    p.add_argument('--dry-run',action='store_true')
    for mode in ('train','evaluate'):
        p=sub.add_parser(mode)
        p.add_argument('--methods',nargs='+',choices=METHODS if mode=='train' else ('base',*METHODS),
                       default=list(METHODS) if mode=='train' else ['base',*METHODS])
        p.add_argument('--gpu',type=int,default=0)
        p.add_argument('--seed',type=int,default=42)
        p.add_argument('--conditions',default='1:2,3:2,1:0.7')
        p.add_argument('--session');p.add_argument('--data');p.add_argument('--base-checkpoint')
        p.add_argument('--output-root',default='outputs/waterbottle_study')
        p.add_argument('--dry-run',action='store_true')
        if mode=='train':
            p.add_argument('--history-checkpoint')
            p.add_argument('--encoder-seed',type=int,default=123)
            p.add_argument('--latent-dim',type=int,default=64)
            p.add_argument('--num-envs',type=int,default=256)
            p.add_argument('--iterations',type=int,default=1000)
            p.add_argument('--rollout-steps',type=int,default=32)
            p.add_argument('--save-interval',type=int,default=100)
            p.add_argument('--correction-scale',type=float,default=.1)
        else:
            p.add_argument('--study',required=True,help='Train directory containing study.json')
            p.add_argument('--episodes',type=int,default=10)
            p.add_argument('--video',action='store_true')
            p.add_argument('--model',default='adapter_final.pt',help='Same preselected checkpoint filename for all methods')
    args=parser.parse_args()
    if args.mode=='audit':
        audit(args.inputs,args.output);return
    if args.mode=='prepare':
        if min(args.epochs,args.history)<=0 or args.seed<0:parser.error('Invalid epochs/history/seed')
        command=[sys.executable,str(REPO/'scripts/train_history_encoder.py'),'--robot','rby1a',
                 '--tool','waterbottle','--state-features','joints','--dynamics-weight','0',
                 '--inputs',*args.inputs,'--output-dir',args.output_dir,'--device',args.device,
                 '--history',str(args.history),'--epochs',str(args.epochs),'--seed',str(args.seed),'--logger','none']
        if args.prepare_only:command+=['--prepare-only']
        if args.split_manifest:command+=['--split-manifest',args.split_manifest]
        if args.dry_run:print(json.dumps(command,indent=2));return
        subprocess.run(command,cwd=REPO,check=True);return
    try:conditions=parse_conditions(args.conditions)
    except ValueError as exc:parser.error(str(exc))
    if args.gpu<0 or args.seed<0 or len(set(args.methods))!=len(args.methods):
        parser.error('Use nonnegative GPU/seed and distinct methods')
    if args.mode=='train':
        if min(args.num_envs,args.iterations,args.rollout_steps,args.save_interval,args.latent_dim)<=0:
            parser.error('Counts must be positive')
        if args.num_envs<len(conditions):parser.error('Need at least one environment per condition')
        if args.encoder_seed<0:parser.error('Encoder seed must be nonnegative')
        if not math.isfinite(args.correction_scale) or args.correction_scale<=0:
            parser.error('Correction scale must be positive and finite')
        checkpoint=str(Path(args.history_checkpoint).expanduser().resolve()) if args.history_checkpoint else None
        if any(m in args.methods for m in ('history','random-history')) and not checkpoint:
            parser.error('B/C require --history-checkpoint')
        if checkpoint and not args.dry_run:
            import torch
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if saved['architecture']['latent_dim']!=args.latent_dim:
                parser.error('--latent-dim must match Encoder latent_dim; A/B/C dimensions must match')
            if saved['config']['tool']!='waterbottle' or saved['data_schema'].get('robot','rby1a')!='rby1a':
                parser.error('Use an RBY1A waterbottle Encoder')
        settings=dict(history_checkpoint=checkpoint,encoder_seed=args.encoder_seed,latent_dim=args.latent_dim)
        trained=None
    else:
        if args.episodes<=0:parser.error('Episodes must be positive')
        trained=json.loads((Path(args.study)/'study.json').read_text())
        if trained.get('mode')!='train':parser.error('--study must be a training study')
        settings=trained['context_settings']
        for method in args.methods:
            if method!='base' and method not in trained['runs']:parser.error(f'No trained {method}')
        # Keep reference/base identity from training; overrides must match.
        for key in ('session','data','base_checkpoint'):
            old=trained['arguments'].get(key)
            new=getattr(args,key)
            if new and str(Path(new).expanduser().resolve())!=old:
                parser.error(f'Evaluation {key} must match training')
            setattr(args,key,old)
    for key in ('session','data','base_checkpoint'):
        value=getattr(args,key)
        if value:setattr(args,key,str(Path(value).expanduser().resolve()))
    output=Path(args.output_root).expanduser().resolve()/f'{args.mode}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}'
    jobs=[]
    for method in args.methods:
        for condition in ([args.conditions] if args.mode=='train' else [f'{m:g}:{f:g}' for m,f in conditions]):
            name=method if args.mode=='train' else f'{method}_mass{condition.replace(":","_friction")}'
            directory=output/name
            command=[sys.executable,'-X','faulthandler','-u',str(REPO/'scripts/rsl_rl/oracle_adapter.py'),
                     '--robot','rby1a','--tool','waterbottle','--mode',args.mode,'--device','cuda:0','--headless',
                     '--seed',str(args.seed),'--conditions',condition,'--output-dir',str(directory),
                     '--project','ConTrack-waterbottle-study']
            command+=context_args(method,settings['history_checkpoint'],settings['latent_dim'],settings['encoder_seed'])
            for key in ('session','data','base_checkpoint'):
                if getattr(args,key):command+=['--'+key,getattr(args,key)]
            model=None
            if args.mode=='train':
                command+=['--num_envs',str(args.num_envs),'--max_iterations',str(args.iterations),
                          '--rollout_steps',str(args.rollout_steps),'--save_interval',str(args.save_interval),
                          '--correction_scale',str(args.correction_scale),'--objective','task-only']
            else:
                command+=['--num_envs','1','--episodes',str(args.episodes),'--policy','base' if method=='base' else 'adapter']
                if method!='base':
                    model=Path(trained['runs'][method])/args.model
                    if not args.dry_run and not model.is_file():raise FileNotFoundError(model)
                    command+=['--adapter_checkpoint',str(model)]
                if args.video:command+=['--video']
            jobs.append(dict(method=method,condition=condition,output=str(directory),command=command,complete=False))
    if args.dry_run:
        print(json.dumps(jobs,indent=2));return
    output.mkdir(parents=True)
    report=dict(mode=args.mode,arguments=vars(args),context_settings=settings,jobs=jobs,runs={},
                warning='Changing only evaluation seeds may reproduce identical trajectories; not independent trials. Oracle has two context inputs; A/B/C use matched latent dimensions.')
    environment=os.environ.copy();environment['CUDA_VISIBLE_DEVICES']=str(args.gpu)
    physical_by_condition={}
    for job in jobs:
        (output/'study.json').write_text(json.dumps(report,indent=2))
        execute(job['command'],job['output'],environment)
        directory=Path(job['output'])
        if args.mode=='train':
            if not (directory/'adapter_final.pt').is_file():raise RuntimeError('Missing final checkpoint')
            metadata=json.loads((directory/'run_metadata.json').read_text())
            report['arguments']['session']=str(Path(metadata['base_checkpoint']).parent)
            report['arguments']['base_checkpoint']=metadata['base_checkpoint']
            report['arguments']['data']=metadata['reference_path']
            report['runs'][job['method']]=str(directory)
        else:
            import h5py
            import numpy as np
            with h5py.File(directory/'episodes.h5') as f:
                if not f.attrs.get('collection_complete') or int(f.attrs['completed_episodes'])!=args.episodes:
                    raise RuntimeError('Incomplete evaluation')
                metadata=json.loads(f.attrs['metadata_json'])
                expected_source='state' if job['method']=='base' else job['method']
                if metadata['context_source']!=expected_source:raise RuntimeError('Context source mismatch')
                expected_policy='base' if job['method']=='base' else 'adapter'
                if metadata['evaluation_policy']!=expected_policy:raise RuntimeError('Policy mismatch')
                if metadata['conditions']!=[list(map(float,job['condition'].split(':')))]:raise RuntimeError('Condition mismatch')
                physics=json.loads(f.attrs['physics_json'])
                expected=physical_by_condition.setdefault(job['condition'],physics)
                for field in ('actual_mass_kg','actual_inertia','material_static_dynamic_restitution'):
                    if not np.allclose(physics[field],expected[field],rtol=1e-6,atol=1e-8):
                        raise RuntimeError(f'Physics differs across methods: {field}')
                if metadata['seed']!=args.seed:raise RuntimeError('Evaluation seed mismatch')
        job['complete']=True
        (output/'study.json').write_text(json.dumps(report,indent=2))
    print('Finished. Study:',output)


if __name__=='__main__':main()
