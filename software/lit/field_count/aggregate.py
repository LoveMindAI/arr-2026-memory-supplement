import json, glob, collections
S={o['idx']:o for o in json.load(open('screened_with_inclusion.json'))}
rows=[]
for f in sorted(glob.glob('coded_*.jsonl')):
    for l in open(f): rows.append(json.loads(l))
inc=[r for r in rows if r['inclusion'].startswith('confirm')]
exc=[r for r in rows if not r['inclusion'].startswith('confirm')]
c=collections.Counter(x for r in inc for x in set(r['codes']))
print("coded:",len(rows),"| confirmed:",len(inc),"| excluded on reading:",len(exc))
for k in "ABCDEF": print(f"  {k}: {c[k]} papers ({c[k]/len(inc):.0%})")
E=[r for r in inc if 'E' in r['codes'] or r.get('E_candidate')]
print("E:",[ (r['idx'],r['title'][:60]) for r in E])
bench=collections.Counter(b for r in inc for b in r.get('benchmarks',[]) for key in ['LoCoMo','LongMemEval'] if key.lower() in b.lower() and b.lower().startswith(key.lower()[:4]))
lo=sum(1 for r in inc if any('locomo' in b.lower() for b in r.get('benchmarks',[])))
lme=sum(1 for r in inc if any('longmemeval' in b.lower() for b in r.get('benchmarks',[])))
either=sum(1 for r in inc if any(('locomo' in b.lower()) or ('longmemeval' in b.lower()) for b in r.get('benchmarks',[])))
print("uses LoCoMo:",lo,"| LongMemEval:",lme,"| either:",either)
onlyA=sum(1 for r in inc if set(r['codes'])<= {'A','C'} and r['codes'])
print("scored only by fact QA and/or retrieval-to-gold (A/C only):",onlyA)
hum=[r for r in inc if not (r.get('human_eval') or 'none').strip().lower().startswith('none')]
print("any human evaluation (B/D raters, annotators):",len(hum))
json.dump(dict(rows=rows),open('coded_all.json','w'),indent=1,ensure_ascii=False)
