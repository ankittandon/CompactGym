"""Does the reward signal point the right way? Joins per-candidate InfoLoss/retention (candidates.jsonl) to live branch outcomes (branches.jsonl).

    uv run python scripts/validate_reward.py [outputs/sft_trial/]
"""
import sys
import json, numpy as np
from scipy.stats import spearmanr
R=sys.argv[1] if len(sys.argv) > 1 else 'outputs/sft_trial/'
b=[json.loads(l) for l in open(R+'branches.jsonl')]
c={}
for l in open(R+'candidates.jsonl'):
    r=json.loads(l)
    if r['candidate'] in('small_prompted','frontier','threshold_pi','cliff','best8_infoloss'): c[(r['episode_id'],r['step'],r['candidate'])]=r
rows=[]
for x in b:
    k=(x['episode_id'],x['step'],x['compactor'])
    if k in c and c[k]['retention'] is not None:
        r=c[k]; rows.append(dict(bp=(x['episode_id'],x['step']),arm=x['compactor'],il=r['infoloss_raw'],ret=r['retention'],comb=r['infoloss_raw']-r['retention'],
            forget=x['n_new_facts_violated'],succ=x['success'],cost=x['cost']))
print(len(rows),'joined rows')
rng=np.random.default_rng(0)
def sp(a,b):
    return spearmanr(a,b)[0] if len(set(a))>1 and len(set(b))>1 else float('nan')
def boot(xs,ys):
    n=len(xs);v=[sp([xs[i] for i in ix],[ys[i] for i in ix]) for ix in (rng.integers(0,n,n) for _ in range(2000))]
    v=[t for t in v if t==t];return np.percentile(v,[2.5,97.5])
for tgt,name in(('forget','newly violated facts (higher=worse)'),('succ','task success')):
    print('\n==',name,' mean',round(np.mean([r[tgt] for r in rows]),3),' sd',round(np.std([r[tgt] for r in rows]),3))
    for m in('il','ret','comb'):
        xs=[r[m] for r in rows];ys=[r[tgt] for r in rows]
        # within-bp (demeaned by bp)
        by={}
        for r in rows: by.setdefault(r['bp'],[]).append(r)
        w=[sp([q[m] for q in g],[q[tgt] for q in g]) for g in by.values()]; w=[t for t in w if t==t]
        lo,hi=boot(xs,ys)
        print(f'  {m:5s} overall rho={sp(xs,ys):+.2f} [95% CI {lo:+.2f},{hi:+.2f}]  within-bp mean rho={np.mean(w):+.2f} (n={len(w)})')
print('\nper arm mean: forget / success / infoloss_raw / retention')
for a in('small_prompted','frontier','threshold_pi','cliff','best8_infoloss'):
    g=[r for r in rows if r['arm']==a]; print(f'  {a:15s} {np.mean([r["forget"] for r in g]):.2f} {np.mean([r["succ"] for r in g]):.3f} {np.mean([r["il"] for r in g]):.3f} {np.mean([r["ret"] for r in g]):.3f}  n={len(g)}')
