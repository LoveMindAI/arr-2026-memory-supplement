#!/usr/bin/env python3
"""A1's coding page for the public stand-in (A1, Oct 11: "I'm good for 40 scenarios at the 3.5.1 depth"). Same GUI and frozen spec as the
paper's human coders (scripts/coder_gui_v35.html, codebook/spec_v351_FROZEN.py; export button saves JSON). 40 memories, one per person,
seeded, half from each memory type. -> ../human_coding/PUBLIC_STANDIN_A1.html (+ copy on <LOCAL_FILE>), key a1_page_key.json"""
import glob, importlib, json, os, random, shutil, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, ".."); sys.path.insert(0, os.path.join(ROOT, "codebook"))
SPEC = importlib.import_module("spec_v351_FROZEN").SPEC; sv = SPEC.get("storage_version", SPEC["version"])
TPL = open(os.path.join(ROOT, "scripts", "coder_gui_v35.html")).read()
mem = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "memories", "C*.json")))]; mem = [m for m in mem if m.get("ok") and m.get("spec")]   # ordinary memories (not the audio/video subset)
rng = random.Random(4011); rng.shuffle(mem); pick, seen = {"shared_stretch": [], "what_they_were_told": []}, set()
for m in mem:
    if m["person"] in seen or len(pick[m["memory_type"]]) >= 20: continue
    pick[m["memory_type"]].append(m); seen.add(m["person"])
items = pick["shared_stretch"] + pick["what_they_were_told"]; rng.shuffle(items)
key = {f"S{i + 1:03d}": f"{m['call']}_{m['person']}" for i, m in enumerate(items)}
D = dict(spec=SPEC, mode="code", key=f"A1_publicstandin_v{sv}", title="A1 · public stand-in (40)", coder="A1",
         items=[dict(id=f"S{i + 1:03d}", moment=str(m["answers"]["moment"]).strip(), reason=str(m["answers"]["reason"]).strip()) for i, m in enumerate(items)])
html = TPL.replace("/*DATA*/null", json.dumps(D, ensure_ascii=False)).replace("agent1", "Reference")
out = os.path.join(ROOT, "human_coding", "PUBLIC_STANDIN_A1.html"); open(out, "w").write(html)
json.dump(key, open(os.path.join(HERE, "a1_page_key.json"), "w"), indent=1)
shutil.copy(out, os.path.expanduser("<LOCAL_FILE>"))
print(len(items), "items ->", out)
