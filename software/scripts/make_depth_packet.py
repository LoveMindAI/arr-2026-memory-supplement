#!/usr/bin/env python3
"""A1's disclosure-depth coding packet (Oct 10: "find examples with the highest disagreement and give that to me as a coding task").
Unit: one speaker in one call. 30 person-calls with the largest disagreement among the three coders (Haiku 5.5, Luna, Gemini 3 Flash) on
the number of DEEP disclosures, plus 10 drawn at random (so the packet can also say which coder's overall rate is right), mixed and unmarked.
Calls A1 has already seen are excluded (humanlike/seen_dyads.json), one person-call per call.
For each: the whole call (Scribe transcript, the target speaker marked), and every candidate disclosure any coder found for that speaker,
merged across coders (quotes overlapping by >= half their words are one candidate), shuffled, coder names hidden. A1 marks each:
not a disclosure / surface / deep, and the kind (information / thoughts / feelings); plus a box for anything deep that all three missed.
-> human_coding/DEPTH_A1.html (self-contained; autosaves in the browser; Export button) and analysis/depth_packet_key.json (never shown)."""
import collections, glob, json, os, random, re, html
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LV = {"INFO_GENERAL": 2, "THOUGHTS_EVENTS_PLANS": 2, "FEELINGS_MILD": 2, "INFO_REVEALING": 3, "THOUGHTS_INTIMATE": 3, "FEELINGS_DEEP": 3}
C = ["claude-haiku-5.5", "gpt-6-luna", "gemini-3-flash-preview"]
D = collections.defaultdict(dict)
for f in glob.glob(os.path.join(ROOT, "analysis", "disclosure_depth", "*__*.json")):
    j = json.load(open(f)); c = j["coder"].split("/")[1]
    if not j["ok"] or c not in C: continue
    for s in ("A", "B"): D[(j["dyad"], s)][c] = []
    for x in j["disclosures"]: D[(j["dyad"], x["speaker"])][c].append(x)
seen = set(json.load(open(os.path.join(ROOT, "humanlike", "seen_dyads.json"))))
K = [k for k, v in D.items() if all(c in v for c in C) and k[0] not in seen]
deepn = {k: [sum(LV[x["category"]] == 3 for x in D[k][c]) for c in C] for k in K}
rng = random.Random(1010)
order = sorted(K, key=lambda k: (-(max(deepn[k]) - min(deepn[k])), rng.random()))
pick, used = [], set()
for k in order:
    if k[0] in used: continue
    pick.append(("disagree", k)); used.add(k[0])
    if len(pick) == 30: break
rest = [k for k in K if k[0] not in used]; rng.shuffle(rest)
for k in rest[:10]: pick.append(("random", k)); used.add(k[0])
rng.shuffle(pick)
W = lambda s: set(re.findall(r"[a-z']+", s.lower()))
items, key = [], {}
for n, (why, (dyad, spk)) in enumerate(pick, 1):
    cands = []
    for c in C:
        for x in D[(dyad, spk)][c]:
            q = x["quote"].strip(); w = W(q)
            for cd in cands:
                if w and cd["w"] and len(w & cd["w"]) >= 0.5 * min(len(w), len(cd["w"])):
                    cd["by"].append([c, x["category"]]); cd["q"] = q if len(q) > len(cd["q"]) else cd["q"]; cd["w"] |= w; break
            else:
                cands.append(dict(q=q, w=w, by=[[c, x["category"]]]))
    rng.shuffle(cands)
    j = json.load(open(os.path.join(ROOT, "transcripts_v3_scribe", dyad + ".json")))
    first = []
    for t in j["turns"]:
        if t["spk"] not in first: first.append(t["spk"])
    lab = {s: "AB"[i] for i, s in enumerate(first[:2])}
    turns = [(lab.get(t["spk"], "?"), t["text"]) for t in j["turns"] if not t.get("bc")]
    iid = f"D{n:02d}"
    items.append(dict(id=iid, spk=spk, turns=turns, cands=[dict(cid=f"{iid}c{i+1}", q=cd["q"]) for i, cd in enumerate(cands)]))
    key[iid] = dict(dyad=dyad, speaker=spk, pid=next((s for s, l in lab.items() if l == spk), None), why=why, deep_counts=dict(zip(C, deepn[(dyad, spk)])),
                    cands={f"{iid}c{i+1}": dict(quote=cd["q"], by=cd["by"]) for i, cd in enumerate(cands)})
json.dump(key, open(os.path.join(ROOT, "analysis", "depth_packet_key.json"), "w"), indent=1)
DATA = json.dumps(items)
PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Depth coding (A1)</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#faf8f4;--fg:#222;--mut:#666;--card:#fff;--line:#e3ded4;--hi:#fff4d6;--acc:#2f6f5e}
@media (prefers-color-scheme:dark){:root{--bg:#1b1a18;--fg:#eee;--mut:#aaa;--card:#252421;--line:#3a3833;--hi:#3d3520;--acc:#7fc4ae}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 Georgia,serif;margin:0;padding:16px;max-width:1100px;margin:auto}
h1{font-size:20px}.mut{color:var(--mut)}.wrap{display:grid;grid-template-columns:1.1fr 1fr;gap:16px}@media(max-width:800px){.wrap{grid-template-columns:1fr}}
.tx{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;max-height:70vh;overflow:auto;font:14px/1.45 system-ui,sans-serif}
.tx p{margin:2px 0}.me{background:var(--hi)}.cand{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 10px;margin:0 0 8px}
.q{font-style:italic;margin-bottom:6px}.row{display:flex;flex-wrap:wrap;gap:6px;font:13px system-ui,sans-serif}
.row label{border:1px solid var(--line);border-radius:14px;padding:2px 9px;cursor:pointer}.row input{margin-right:4px}
textarea{width:100%;min-height:60px;font:13px system-ui;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px}
.nav{display:flex;gap:8px;align-items:center;margin:12px 0;font:14px system-ui}button{font:14px system-ui;padding:6px 12px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.done{color:var(--acc)}
</style></head><body>
<h1>Disclosure depth: one speaker per call</h1>
<p class="mut">Each screen is one real call. The highlighted speaker is the one to code. On the right is every moment where an AI coder thought this speaker disclosed something about <b>themselves</b>, merged and shuffled (coder names hidden). For each, mark whether it is a disclosure at all, and if so, how deep and what kind. <b>Surface</b> = ordinary facts (job, where they live, hobbies), thoughts about plans or past events, mild feelings. <b>Deep</b> = revealing personal information (their traits, a personal experience, problems in their own or their family's behaviour), thoughts about their own character, looks, health or private wishes, or deep feelings (humiliation, anxiety, fears, pain). Answers save in this browser; press Export when done.</p>
<div class="nav"><button id="prev">&larr; Prev</button><span id="pos"></span><button id="next">Next &rarr;</button><span style="flex:1"></span><button id="exp">Export answers</button></div>
<div class="wrap"><div class="tx" id="tx"></div><div><div id="cands"></div>
<p class="row" style="margin-top:6px">Anything deep from this speaker that isn't listed? Quote it here:</p><textarea id="missed"></textarea>
<p class="row">Note (optional):</p><textarea id="note"></textarea></div></div>
<script>
const ITEMS = __DATA__; const KEY = "depth_a1_v1"; let A = {}; try { A = JSON.parse(localStorage.getItem(KEY) || "{}") } catch (e) {}
let i = 0; try { i = +(localStorage.getItem(KEY + "_pos") || 0) } catch (e) {}
const save = () => { try { localStorage.setItem(KEY, JSON.stringify(A)); localStorage.setItem(KEY + "_pos", i) } catch (e) {} };
const OPTS = [["no", "Not a disclosure"], ["surface", "Surface"], ["deep", "Deep"]], KIND = [["info", "Information"], ["thoughts", "Thoughts"], ["feelings", "Feelings"]];
function el(t, a = {}, k = []) { const e = document.createElement(t); for (const x in a) x === "text" ? e.textContent = a[x] : e.setAttribute(x, a[x]); k.forEach(c => e.appendChild(c)); return e }
function radios(name, opts, cur, on) { return el("div", { class: "row" }, opts.map(([v, l]) => { const r = el("input", { type: "radio", name }); r.checked = cur === v; r.onchange = () => on(v); return el("label", {}, [r, document.createTextNode(l)]) })) }
function render() {
  const it = ITEMS[i]; const a = A[it.id] = A[it.id] || { c: {}, missed: "", note: "" };
  const tx = document.getElementById("tx"); tx.innerHTML = "";
  it.turns.forEach(([s, t]) => tx.appendChild(el("p", { class: s === it.spk ? "me" : "", text: s + ": " + t })));
  const cs = document.getElementById("cands"); cs.innerHTML = "";
  it.cands.forEach(c => {
    const v = a.c[c.cid] = a.c[c.cid] || {};
    const box = el("div", { class: "cand" }, [el("div", { class: "q", text: "“" + c.q + "”" })]);
    box.appendChild(radios(c.cid + "_l", OPTS, v.level, x => { v.level = x; save(); render() }));
    if (v.level && v.level !== "no") box.appendChild(radios(c.cid + "_k", KIND, v.kind, x => { v.kind = x; save() }));
    cs.appendChild(box) });
  const m = document.getElementById("missed"), n = document.getElementById("note"); m.value = a.missed; n.value = a.note;
  m.oninput = () => { a.missed = m.value; save() }; n.oninput = () => { a.note = n.value; save() };
  const done = ITEMS.filter(x => A[x.id] && x.cands.every(c => A[x.id].c[c.cid] && A[x.id].c[c.cid].level)).length;
  document.getElementById("pos").innerHTML = `Call ${i + 1} of ${ITEMS.length} &middot; speaker <b>${it.spk}</b> &middot; <span class="done">${done} complete</span>`;
  save(); window.scrollTo(0, 0) }
document.getElementById("prev").onclick = () => { if (i > 0) { i--; render() } };
document.getElementById("next").onclick = () => { if (i < ITEMS.length - 1) { i++; render() } };
document.getElementById("exp").onclick = () => { const b = new Blob([JSON.stringify({ coder: "A1", packet: "depth_a1_v1", answers: A }, null, 1)], { type: "application/json" });
  const u = URL.createObjectURL(b); const x = el("a", { href: u, download: "DEPTH_A1_answers.json" }); document.body.appendChild(x); x.click(); x.remove() };
render();
</script></body></html>"""
open(os.path.join(ROOT, "human_coding", "DEPTH_A1.html"), "w").write(PAGE.replace("__DATA__", DATA))
print("items", len(items), "candidates", sum(len(x["cands"]) for x in items), "disagree/random", collections.Counter(w for w, _ in pick))
print("deep-count spread in the disagree set:", sorted([max(deepn[k]) - min(deepn[k]) for w, k in pick if w == "disagree"], reverse=True)[:12])
