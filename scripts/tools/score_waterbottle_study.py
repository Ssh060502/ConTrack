"""Create manual video labels, then summarize them; never infer success from timeout."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path

OUTCOMES=('no_lift','dropped','completed','incomplete','uncertain')


def summarize(rows):
    groups=defaultdict(Counter)
    seen=set()
    for row in rows:
        key=(row['method'],row['condition'],row['episode'])
        if key in seen:raise ValueError(f'Duplicate label: {key}')
        seen.add(key)
        label=row['outcome'].strip()
        if label and label not in OUTCOMES:raise ValueError(f'Invalid outcome: {label}')
        groups[(row['method'],row['condition'])][label or 'unlabeled']+=1
    return [dict(method=method,condition=condition,total=sum(count.values()),counts=dict(count),
                 completion_rate=count['completed']/sum(count.values()) if not count['unlabeled'] and not count['uncertain'] else None)
            for (method,condition),count in sorted(groups.items())]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',required=True,help='Completed evaluate directory')
    parser.add_argument('--create-labels',action='store_true',help='Create labels.csv without overwriting')
    args=parser.parse_args()
    root=Path(args.study).expanduser().resolve()
    study=json.loads((root/'study.json').read_text())
    if study['mode']!='evaluate' or not all(j['complete'] for j in study['jobs']):
        raise ValueError('Use a complete evaluation study')
    path=root/'labels.csv'
    if args.create_labels:
        import h5py
        rows=[]
        for job in study['jobs']:
            directory=Path(job['output'])
            with h5py.File(directory/'episodes.h5') as f:
                for name in sorted(f['episodes']):
                    rows.append(dict(method=job['method'],condition=job['condition'],episode=name,outcome='',
                                     video=str(directory/'videos'/f'episode_{name}.mp4'),notes=''))
        with path.open('x',newline='',encoding='utf-8') as file:
            writer=csv.DictWriter(file,fieldnames=['method','condition','episode','outcome','video','notes'])
            writer.writeheader();writer.writerows(rows)
        print('Label template:',path)
        print('Outcomes:',', '.join(OUTCOMES))
        return
    with path.open(newline='',encoding='utf-8-sig') as file:rows=list(csv.DictReader(file))
    expected={(j['method'],j['condition'],f'{i:06d}') for j in study['jobs'] for i in range(study['arguments']['episodes'])}
    if {(r['method'],r['condition'],r['episode']) for r in rows}!=expected:
        raise ValueError('Label rows must cover every evaluated episode exactly once')
    result=summarize(rows)
    (root/'manual_scores.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
