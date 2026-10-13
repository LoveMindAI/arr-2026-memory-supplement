#!/usr/bin/env python3
"""Build the private human-coding repo (A1 Oct 8): every coder codes the same 50 reports (B001-B050), with an optional second 50 (B051-B100).
A1 also does the transcript part (Part C); A3 gets an optional transcript version in case A1 can't. No agent1 answers anywhere.
Usage: python3 make_coder_repo.py OUTDIR [--spec spec_v34]"""
import argparse, importlib, json, os, shutil, sys
import prompts as P
ROOT = ".."; sys.path.insert(0, os.path.join(ROOT, "codebook"))
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--spec", default="spec_v351_FROZEN")
ap.add_argument("--only", default="", help="e.g. NAME:INITIALS  -> build just this coder's pages"); ap.add_argument("--readme", default="CODER_REPO_README.md"); a = ap.parse_args()
SPEC = importlib.import_module(a.spec).SPEC; v = SPEC["version"]; sv = SPEC.get("storage_version", v)   # autosave key survives point releases
TPL = open("coder_gui_v35.html").read()
key = json.load(open(os.path.join(ROOT, "analysis", "A3_item_key.json"))); rows = {r["perspectiveID"]: r for r in P.rows()}
IDS = sorted(key); FIRST, MORE = IDS[:50], IDS[50:100]
def item(iid, part_c):
    r = rows[key[iid]]; d = dict(id=iid, moment=r["conv_SP_Mem_descr"].strip(), reason=r["conv_SP_Mem_reason"].strip())
    if part_c:
        loc = json.load(open(os.path.join(ROOT, "analysis", "located", key[iid] + ".json"))); j = json.load(open(os.path.join(P.TX, r["dyadID"] + ".json")))
        d.update(tx=[dict(i=k, t=f"{int(t['start'] // 60)}:{int(t['start'] % 60):02d}", who="Reporter" if t["spk"] == r["selfPID"] else "Partner", text=t["text"]) for k, t in enumerate(j["turns"])],
                 loc=sorted(loc.get("turns") or []), lockind=loc.get("kind"), locnote=loc.get("note"))
    return d
def page(fname, name, ini, ids, part_c, setname, outdir=None):
    # the storage key does NOT depend on part_c, so A3's two versions share one autosave
    D = dict(spec=SPEC, mode="code", key=f"{name.upper()}_{setname}_v{sv}", title=f"{name} · {setname}", coder=ini, items=[item(i, part_c) for i in ids])
    html = TPL.replace("/*DATA*/null", json.dumps(D, ensure_ascii=False)).replace("agent1", "Reference")
    assert "agent1" not in html; open(os.path.join(outdir or a.out, fname), "w").write(html)
os.makedirs(a.out, exist_ok=True)
LOCAL = os.path.join(ROOT, "human_coding")  # A1's pages stay on A1's disk, not in the shared repo
CODERS = [("A1", "A1", True, LOCAL), ("A3", "A3", False, None), ("A4", "A4", False, None), ("A5", "A5", False, None)]
if a.only: CODERS = [(a.only.split(":")[0], a.only.split(":")[1], False, None)]      # Oct 9: A2's own private repo
for name, ini, pc, od in CODERS:
    page(f"CODE_{name.upper()}.html", name, ini, FIRST, pc, "first50", od)
    page(f"CODE_{name.upper()}_MORE.html", name, ini, MORE, pc, "more50", od)
if not a.only: page("CODE_A3_WITH_TRANSCRIPT.html", "A3", "A3", FIRST, True, "first50")
shutil.copy(os.path.join(ROOT, "codebook", f"CODEBOOK_v{v}.md"), os.path.join(a.out, f"CODEBOOK_v{v}.md"))
open(os.path.join(a.out, "README.md"), "w").write(open(os.path.join(ROOT, "human_coding", a.readme)).read().replace("{V}", v))
print("built:", sorted(os.listdir(a.out)))
