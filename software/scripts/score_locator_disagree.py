"""Score A1's LOCATOR_DISAGREE_A1 answers (Oct 11) by locator source.
A source is right if A1 ticked its candidate. A1's verdicts: 'none' -> every source wrong; 'cant' -> item dropped;
'av' (needs audio/video) -> a source that said not_in_transcript is right (same meaning); 'general' counted right only in the lenient score."""
import json, collections
K = json.load(open("analysis/locator_disagree_packet_key.json"))
A = json.load(open("human_coding/LOCATOR_DISAGREE_A1_answers.json"))["answers"]
SRC = ["sonnet_v1", "gemini_v1", "luna_v1", "sonnet_v2"]
out = {}
for mode in ("strict", "lenient"):
    right = collections.Counter(); n = collections.Counter(); kinds = collections.Counter()
    for iid, it in K.items():
        a = A.get(iid)
        if not a or a["verdict"] == "cant": continue
        for L, c in it["cands"].items():
            for s in c["sources"]:
                n[s] += 1
                ok = L in a["right"]
                if a["verdict"] == "av":
                    ok = c["kind"] == "not_in_transcript" or (mode == "lenient" and c["kind"] == "general")
                    kinds[(s, c["kind"])] += 1
                right[s] += ok
    out[mode] = {s: [right[s], n[s]] for s in SRC}
verd = collections.Counter(a["verdict"] or "ticked" for a in A.values())
# v1 vs v2 head to head (sonnet)
h = collections.Counter()
for iid, it in K.items():
    a = A[iid]
    if a["verdict"] == "cant": continue
    def ok(s):
        for L, c in it["cands"].items():
            if s in c["sources"]:
                if a["verdict"] == "av": return c["kind"] == "not_in_transcript"
                return L in a["right"]
    h[(ok("sonnet_v1"), ok("sonnet_v2"))] += 1
# breadth: among items with both ticked and unticked turn candidates, are ticked ones wider?
wider = collections.Counter()
for iid, it in K.items():
    a = A[iid]
    T = [(L, c) for L, c in it["cands"].items() if c["kind"] == "turns"]
    t = [len(c["turns"]) for L, c in T if L in a["right"]]; u = [len(c["turns"]) for L, c in T if L not in a["right"]]
    if t and u: wider["ticked wider" if max(t) > max(u) else "unticked wider" if max(u) > max(t) else "tie"] += 1
# av items: what did each source say
av = {iid: {s: c["kind"] for L, c in it["cands"].items() for s in c["sources"]} for iid, it in K.items() if A[iid]["verdict"] == "av"}
res = dict(verdicts=verd, scores=out, sonnet_v1_vs_v2={f"v1 {k[0]} / v2 {k[1]}": v for k, v in h.items()}, breadth=wider, av_items=av)
json.dump(res, open("analysis/locator_disagree_scored.json", "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
