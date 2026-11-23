#!/usr/bin/env python3
"""Add or update an experiment in train-viz experiments.json, or add a run to one.

  experiment_add.py --id moe-evo --title "MoE expert evolution (300k)" --status running
  experiment_add.py --id moe-evo --run shared/unify-...-spark-300k --label "evolve seed 2" --role treatment
  experiment_add.py --id moe-evo --status done --verdict "..."

--run takes the run id as train-viz names it: <root id>/<folder name>, e.g. shared/<name> or
wsl3090/<name>--<job id>. --role baseline also sets the experiment's baseline. Stdlib only.
"""
import argparse, datetime, json, os, tempfile

FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'experiments.json')

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--id', required=True)
p.add_argument('--title')
p.add_argument('--question')
p.add_argument('--status', choices=['done', 'running', 'planned'])
p.add_argument('--verdict')
p.add_argument('--primary-metric', dest='primary_metric', help='score key the experiment is judged by (METRICS in public/experiments.js)')
p.add_argument('--falsifier')
p.add_argument('--run')
p.add_argument('--label')
p.add_argument('--role', choices=['baseline', 'treatment', 'control'])
p.add_argument('--note')
p.add_argument('--planned', action='append', default=[], help='folder name of a run not started yet (repeatable)')
p.add_argument('--unplan', action='append', default=[],
               help='remove this planned name (a started run whose folder name differs, e.g. a timestamp prefix)')
p.add_argument('--file', default=FILE)
a = p.parse_args()

try:
    doc = json.load(open(a.file))
except FileNotFoundError:
    doc = {'version': 1, 'experiments': []}
exps = doc['experiments']
e = next((x for x in exps if x['id'] == a.id), None)
if e is None:
    e = {'id': a.id, 'title': a.title or a.id, 'created': datetime.date.today().isoformat(),
         'status': a.status or 'running', 'question': None, 'falsifier': None, 'primary_metric': 'fid_ema',
         'score_steps': [], 'baseline': None, 'runs': [], 'verdict': None, 'sources': ['scripts/experiment_add.py']}
    exps.insert(0, e)  # newest first
for k in ('title', 'question', 'status', 'verdict', 'primary_metric', 'falsifier'):
    if getattr(a, k) is not None:
        e[k] = getattr(a, k)
for name in a.planned:
    if name not in e.setdefault('planned_runs', []) and not any(x['run'].split('/', 1)[-1] == name for x in e['runs']):
        e['planned_runs'].append(name)
for name in a.unplan:
    if name in e.get('planned_runs', []):
        e['planned_runs'].remove(name)
if a.run:
    if a.run.split('/', 1)[-1] in e.get('planned_runs', []):  # a planned run that has started
        e['planned_runs'].remove(a.run.split('/', 1)[-1])
    r = next((x for x in e['runs'] if x['run'] == a.run), None)
    if r is None:
        r = {'run': a.run, 'label': a.label or a.run.split('/', 1)[-1].split('--')[0], 'role': a.role or 'treatment', 'note': None}
        e['runs'].append(r)
    for k in ('label', 'role', 'note'):
        if getattr(a, k) is not None:
            r[k] = getattr(a, k)
    if r['role'] == 'baseline':
        e['baseline'] = a.run

fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(a.file)), suffix='.tmp')
with os.fdopen(fd, 'w') as f:
    json.dump(doc, f, indent=1, ensure_ascii=False)
    f.write('\n')
os.replace(tmp, a.file)
print(f"{a.id}: {len(e['runs'])} runs, status {e['status']}")
