import json, re, sys, glob
def norm(s): return re.sub(r"\s+"," ",s.replace("ﬁ","fi").replace("ﬂ","fl").replace("ﬀ","ff").replace("’","'").replace("“",'"').replace("”",'"').replace("-\n","")).lower()
bad=0; tot=0
for f in sorted(glob.glob("coded_*.jsonl")):
    for l in open(f):
        o=json.loads(l); 
        try: src=norm(open(f"plain/{o['idx']}.txt").read()); src2=re.sub(r"-\s+","",src); src3=src.replace("- ","")
        except Exception: print("NO TEXT", o['idx']); continue
        for ev in o.get("evidence",[]):
            q=norm(ev["quote"]).strip(' ."'); tot+=1
            if q not in src and q not in src2 and q.replace("-","") not in src.replace("-",""):
                # try without layout whitespace damage
                if re.sub(r"[^a-z0-9]","",q) not in re.sub(r"[^a-z0-9]","",src):
                    bad+=1; print("MISSING", f, o['idx'], ev.get("page"), q[:120])
print(f"{tot-bad}/{tot} evidence quotes found in the extracted text")
