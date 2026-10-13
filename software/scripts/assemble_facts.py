#!/usr/bin/env python3
"""Per-person fact lists for the fact conditions (A1, Oct 7): all calls, kept facts only (drop = procedure / session_feeling / partner /
name_only), private names removed (fact_no_names), repeats merged. Merging is done by GPT-6-Luna per person, and every fact it returns
must be an exact copy of an input fact (checked; non-verbatim outputs are rejected and the unmerged list is used).
Writes facts/person_facts.json {pid: [facts]} and facts/REDACTIONS.tsv (every name removal) and facts/ASSEMBLY_STATS.json."""
import json, os, sys, glob, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(__file__))
import orc
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
D = os.path.join(ROOT, "facts")
MERGE = """Below is a list of facts about one person, collected from several conversations. Some facts repeat or say the same thing in different words.

Return the list with repeats removed. When two or more facts say the same thing, keep only the most specific one, copied exactly. Keep every fact that adds anything new. Do not rewrite, combine, shorten or add facts: every fact you return must be copied exactly from the list.

FACTS:
{facts}

Return JSON only: {{"facts": ["...", "..."]}}"""


def merge(item):
    pid, facts = item
    p = os.path.join(D, "merged", pid + ".json")
    if os.path.exists(p):
        return pid, json.load(open(p)), 0.0
    if len(facts) <= 1:
        return pid, dict(facts=facts, verbatim=True), 0.0
    r = orc.call("openai/gpt-6-luna", [{"role": "user", "content": MERGE.format(facts="\n".join(f"- {x}" for x in facts))}],
                 os.path.join(D, "merged", "calls.jsonl"), tag=pid, max_tokens=8000, temperature=0, response_format={"type": "json_object"},
                 reasoning={"effort": "low"})
    t = r.get("text") or ""
    try:
        out = json.loads(t[t.find("{"): t.rfind("}") + 1])["facts"]
        ok = all(x in facts for x in out) and len(out) > 0
    except Exception:
        out, ok = facts, False
    res = dict(facts=out if ok else facts, verbatim=ok, n_in=len(facts))
    json.dump(res, open(p, "w"))
    return pid, res, orc.cost(r)


def main():
    os.makedirs(os.path.join(D, "merged"), exist_ok=True)
    items = [json.load(open(f)) for f in glob.glob(os.path.join(D, "coded_v2", "items", "*.json"))]
    stats = collections.Counter(x.get("drop", "?") for x in items)
    per = collections.defaultdict(list); red = []
    for x in sorted(items, key=lambda x: (x["pid"], x["dyad"])):
        if x.get("drop") != "keep":
            continue
        f = x["fact"]
        if int(x.get("has_name", 0)) and x.get("fact_no_names") and x["fact_no_names"] != f:
            red.append((x["pid"], f, x["fact_no_names"])); f = x["fact_no_names"]
        if f not in per[x["pid"]]:
            per[x["pid"]].append(f)
    with open(os.path.join(D, "REDACTIONS.tsv"), "w") as fh:
        fh.write("pid\toriginal\tredacted\n")
        for r in red:
            fh.write("\t".join(r) + "\n")
    res = list(ThreadPoolExecutor(32).map(merge, sorted(per.items())))
    out = {pid: r["facts"] for pid, r, _ in res}
    json.dump(out, open(os.path.join(D, "person_facts.json"), "w"), indent=1, ensure_ascii=False)
    n_in = [len(per[p]) for p in out]; n_out = [len(v) for v in out.values()]
    st = dict(coded=len(items), drop_counts=dict(stats), redactions=len(red), people=len(out),
              facts_before_merge=sum(n_in), facts_after_merge=sum(n_out), merge_rejected=[p for p, r, _ in res if not r["verbatim"]],
              per_person_median=sorted(n_out)[len(n_out) // 2], cost=round(sum(c for *_, c in res), 3))
    json.dump(st, open(os.path.join(D, "ASSEMBLY_STATS.json"), "w"), indent=1); print(json.dumps(st, indent=1))


if __name__ == "__main__":
    main()
