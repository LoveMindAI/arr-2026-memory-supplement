#!/usr/bin/env python3
"""A1's locator packet, round 3 (Oct 11: "I'd rather do it where there's both ambiguity AND locator disagreement"). People's memories where
the three locators (Sonnet 5.5, Gemini 3 Flash, GPT-6 Luna; original prompt) disagree, or where the revised prompt (v2, Sonnet) places the
memory differently; none A1 has judged before. Each distinct placement is a lettered candidate (sources hidden, order by position);
'general' and 'not in the words' are candidates too. A1 marks every candidate that is right, or none / can't tell / needs video-audio.
A toggle shows the Scribe transcript (marks laughter and fillers). -> human_coding/LOCATOR_DISAGREE_A1.html, key analysis/locator_disagree_packet_key.json"""
import glob, json, os, random
import prompts as P
ROOT = ".."; A = os.path.join(ROOT, "analysis"); rng = random.Random(3131)
def load(d): return {json.load(open(f))["perspectiveID"]: json.load(open(f)) for f in glob.glob(os.path.join(A, d, "PID_*.json"))}
SRC = {"sonnet_v1": load("located"), "gemini_v1": load("located_g3f"), "luna_v1": load("located_luna"), "sonnet_v2": load("located_v2_sonnet")}
R = {r["perspectiveID"]: r for r in P.rows()}
seen = set()
for k in ("locator_a1_packet_key.json", "locator_a1_packet2_key.json", "locator_models_packet_key.json"): seen |= {v["pid"] for v in json.load(open(os.path.join(A, k))).values()}
def turns_of(src, j):
    t = j.get("turns") or []
    if j.get("kind") != "in_conversation" or not t: return None
    return tuple(range(min(t), max(t) + 1)) if src.endswith("_v1") else tuple(sorted(set(t)))
dis = set(json.load(open(os.path.join(A, "locator_disagree_pids.json"))))
v12 = {p for p in SRC["sonnet_v2"] if SRC["sonnet_v2"][p]["kind"] != SRC["sonnet_v1"][p]["kind"] or
       (turns_of("sonnet_v1", SRC["sonnet_v1"][p]) and turns_of("sonnet_v2", SRC["sonnet_v2"][p]) and not set(turns_of("sonnet_v1", SRC["sonnet_v1"][p])) & set(turns_of("sonnet_v2", SRC["sonnet_v2"][p])))}
pids = sorted((dis | v12) - seen); rng.shuffle(pids)
items, key = [], {}
for n, pid in enumerate(pids, 1):
    r = R[pid]; tx = json.load(open(os.path.join(P.TX, r["dyadID"] + ".json")))["turns"]
    sc = json.load(open(os.path.join(ROOT, "transcripts_v3_scribe", r["dyadID"] + ".json")))["turns"]
    cands = {}
    for src, D in SRC.items():
        j = D.get(pid)
        if not j: continue
        t = turns_of(src, j); c = ("turns", t) if t else (j.get("kind") or "none", None)
        cands.setdefault(c, []).append(src)
    order = sorted(cands, key=lambda c: (c[0] != "turns", c[1][0] if c[1] else 0, c[0]))
    letters = "ABCDEFG"; cl = []
    for i, c in enumerate(order):
        cl.append(dict(L=letters[i], kind=c[0], turns=list(c[1]) if c[1] else []))
    iid = f"D{n:03d}"
    items.append(dict(id=iid, mem=r["conv_SP_Mem_descr"].strip(), why=r["conv_SP_Mem_reason"].strip(),
                      turns=[["You" if t["spk"] == r["selfPID"] else "Partner", t["text"]] for t in tx],
                      scribe=[["You" if t["spk"] == r["selfPID"] else "Partner", t["text"]] for t in sc], cands=cl))
    key[iid] = dict(pid=pid, cands={cl[i]["L"]: dict(kind=order[i][0], turns=cl[i]["turns"], sources=cands[order[i]]) for i in range(len(order))})
PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><title>Where Is the Memory</title><meta name="viewport" content="width=device-width,initial-scale=1">
<style>:root{--bg:#faf8f4;--fg:#222;--mut:#666;--card:#fff;--line:#e3ded4;--hi:#fff1b8;--acc:#2f6f5e}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1b1a18;--fg:#eee;--mut:#aaa;--card:#252421;--line:#3a3833;--hi:#4a3f18;--acc:#7fc4ae}}
:root[data-theme="dark"]{--bg:#1b1a18;--fg:#eee;--mut:#aaa;--card:#252421;--line:#3a3833;--hi:#4a3f18;--acc:#7fc4ae}
body{background:var(--bg);color:var(--fg);font:16px/1.5 Georgia,serif;margin:0 auto;padding:12px 16px;max-width:1100px}h1{font-size:19px;margin:6px 0}.mut{color:var(--mut);font-size:14px}
.wrap{display:grid;grid-template-columns:1fr 1.3fr;gap:16px}@media(max-width:800px){.wrap{grid-template-columns:1fr}}
.mem{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;position:sticky;top:8px;align-self:start}
.tx{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px;max-height:78vh;overflow:auto;font:14px/1.45 system-ui}.tx p{margin:2px 0;padding-left:46px;text-indent:-46px}.tx p.h{background:var(--hi);border-radius:3px}
.tag{display:inline-block;width:42px;font:600 11px system-ui;color:var(--acc);text-indent:0}
.q{font:15px system-ui;margin:12px 0 6px}.c{display:flex;gap:6px;align-items:center;margin:5px 0;font:14px system-ui}
.c button,.v button{padding:7px 10px;border-radius:7px;border:1px solid var(--line);background:var(--card);color:var(--fg);font:14px system-ui;cursor:pointer}
.c button.on,.v button.on{outline:2px solid var(--acc)}.c .show{flex:1;text-align:left}.c .show.cur{background:var(--hi)}
.v{display:grid;grid-template-columns:1fr 1fr 1fr;gap:6px;margin-top:8px}.nav{display:flex;gap:8px;align-items:center;font:14px system-ui;margin:8px 0}
.nav button{font:14px system-ui;padding:5px 10px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}</style></head><body>
<h1>Where is the memory?</h1><p class="mut">Left: what the participant wrote right after the call, and the places different readers put it (lettered; who suggested which is hidden). Click a letter to highlight its lines. Mark <b>every</b> placement that is right; if none is, say so. The Scribe button shows a second transcript that marks laughter and fillers (no highlights there).</p>
<div class="nav"><button id="prev">&larr; Back</button><button id="next">Next &rarr;</button><span id="pos"></span><span style="flex:1"></span><button id="sc">Scribe transcript</button><button id="exp">Export answers</button></div>
<div class="wrap"><div class="mem" id="mem"></div><div class="tx" id="tx"></div></div><script>
const ITEMS=__DATA__,KEY="locator_disagree_A1";let A={};try{A=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}let i=0,cur=null,scribe=false;try{i=+(localStorage.getItem(KEY+"_pos")||0)}catch(e){}
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(A));localStorage.setItem(KEY+"_pos",i)}catch(e){}};
function el(t,a={},k=[]){const e=document.createElement(t);for(const x in a)x==="text"?e.textContent=a[x]:e.setAttribute(x,a[x]);k.forEach(c=>e.appendChild(c));return e}
const label=c=>c.kind==="turns"?(c.turns.length===1?"line "+(c.turns[0]+1):"lines "+(c.turns[0]+1)+"–"+(c.turns[c.turns.length-1]+1)+(c.turns.length<c.turns[c.turns.length-1]-c.turns[0]+1?" (some)":"")):c.kind==="general"?"the call in general, no single moment":c.kind==="not_in_transcript"?"not in the words (looks, voice, a pause…)":c.kind;
function ans(){return A[ITEMS[i].id]||(A[ITEMS[i].id]={right:[],verdict:null})}
function render(){const it=ITEMS[i],a=ans();if(cur===null){const f=it.cands.find(c=>c.kind==="turns");cur=f?f.L:null}
const m=document.getElementById("mem");m.innerHTML="";m.appendChild(el("div",{text:"Moment: "+it.mem}));m.appendChild(el("div",{style:"margin-top:6px",text:"Why: "+it.why}));
m.appendChild(el("div",{class:"q",text:"Which placements are right? (mark all that are)"}));
it.cands.forEach(c=>{const s=el("button",{class:"show"+(cur===c.L?" cur":""),text:c.L+": "+label(c)});s.onclick=()=>{cur=c.L;scribe=false;render()};
 const ok=el("button",{class:a.right.includes(c.L)?"on":"",text:"right"});ok.onclick=()=>{a.right=a.right.includes(c.L)?a.right.filter(x=>x!==c.L):a.right.concat([c.L]);if(a.right.length)a.verdict=null;save();render()};
 m.appendChild(el("div",{class:"c"},[s,ok]))});
const v=el("div",{class:"v"});[["none","None is right"],["cant","Can't tell"],["av","Needs video/audio"]].forEach(([k,t])=>{const b=el("button",{class:a.verdict===k?"on":"",text:t});b.onclick=()=>{a.verdict=a.verdict===k?null:k;if(a.verdict)a.right=[];save();render()};v.appendChild(b)});m.appendChild(v);
const t=document.getElementById("tx");t.innerHTML="";const C=it.cands.find(c=>c.L===cur);let first=null;
const rows=scribe?it.scribe:it.turns;rows.forEach(([w,s],k)=>{const marks=scribe?"":it.cands.filter(c=>c.kind==="turns"&&c.turns.includes(k)).map(c=>c.L).join("");
 const h=!scribe&&C&&C.kind==="turns"&&C.turns.includes(k);const p=el("p",{class:h?"h":""},[el("span",{class:"tag",text:marks}),document.createTextNode(w+": "+s)]);if(h&&!first)first=p;t.appendChild(p)});
if(first)setTimeout(()=>first.scrollIntoView({block:"center"}),30);else t.scrollTop=0;
document.getElementById("sc").textContent=scribe?"Back to main transcript":"Scribe transcript";
const done=Object.values(A).filter(x=>x.right.length||x.verdict).length;document.getElementById("pos").innerHTML=`${i+1} of ${ITEMS.length} &middot; ${done} answered`;save()}
document.getElementById("prev").onclick=()=>{if(i>0){i--;cur=null;scribe=false;render()}};document.getElementById("next").onclick=()=>{if(i<ITEMS.length-1){i++;cur=null;scribe=false;render()}};
document.getElementById("sc").onclick=()=>{scribe=!scribe;render()};
document.getElementById("exp").onclick=()=>{const b=new Blob([JSON.stringify({coder:"A1",packet:KEY,answers:A},null,1)],{type:"application/json"});const u=URL.createObjectURL(b);const x=el("a",{href:u,download:"LOCATOR_DISAGREE_A1_answers.json"});document.body.appendChild(x);x.click();x.remove()};
render();</script></body></html>'''
open(os.path.join(ROOT, "human_coding", "LOCATOR_DISAGREE_A1.html"), "w").write(PAGE.replace("__DATA__", json.dumps(items)))
json.dump(key, open(os.path.join(A, "locator_disagree_packet_key.json"), "w"), indent=1)
import collections; print(len(items), "items; candidates per item", collections.Counter(len(x["cands"]) for x in items))
