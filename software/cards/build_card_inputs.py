#!/usr/bin/env python3
"""Build all card inputs.
- facets.json: {PID: {kit facet name: 1-5 facet mean}} from canonical HEXACO-60 facet scores (24 facets; no Altruism).
- card_prompts.jsonl: narrative card prompt per person (kit card logic on the no-Altruism variant template; A1 approved Oct 7).
- thin_cards/PID.txt: the "thin trait" rung = the numbers themselves, no writer (A1, Oct 7: "No writer").
- person_jobs.jsonl: one job per memory (perspective): the person's facts from their OTHER calls (leave-one-call-out) + trait block.
- frontier50.json: 50 seeded people for the frontier-writer comparison (Opus 5.5 + Sonnet 5.5 vs lcw150)."""
import csv, glob, json, os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import prompts as P
CANON = "<LOCAL>/Documents/Master_Prompt_Workshop/LSI_StoriesOfYourLife_Canonical_PARSEL_Data/PARSEL_GroundTruth/V6_Canonical_290_Private_Human_Data/canonical_scale_scores_290.tsv"
VAR = json.load(open(os.path.join(HERE, "card_hexaco60_noaltruism.json")))
OUT = os.path.join(HERE, "inputs"); os.makedirs(os.path.join(OUT, "thin_cards"), exist_ok=True)


def kitname(col):  # HEXACO_HH_Greed_Avoidance -> Greed Avoidance ; Social_Self_Esteem -> Social Self-Esteem
    n = col.split("_", 2)[2].replace("_", " ")
    return {"Social Self Esteem": "Social Self-Esteem"}.get(n, n)


def card_messages(fs):
    s = VAR["template"]
    for n in VAR["facets"]:
        s = s.replace(f"{{F:{n}}}", f"{float(fs[n]):.4f}")
    for d, xs in VAR["broad_from_facets"].items():
        s = s.replace(f"{{B:{d}}}", f"{sum(float(fs[x]) for x in xs) / len(xs):.4f}")
    assert "{F:" not in s and "{B:" not in s
    return [{"role": "user", "content": s}]


def trait_lines(fs):
    # the same facet lines (with meanings) the card prompt shows the writer
    lines = [l for l in VAR["template"].split("\n") if any(l.startswith(n + ":") for n in VAR["facets"])]
    out = []
    for l in lines:
        n = l.split(":")[0]
        out.append(l.replace(f"{{F:{n}}}", f"{float(fs[n]):.2f}"))
    return "\n".join(out)


def thin_card(fs):
    doms = {d: sum(float(fs[x]) for x in xs) / len(xs) for d, xs in VAR["broad_from_facets"].items()}
    lines = ["Personality questionnaire results (HEXACO-60). Each score is a mean on a 1-5 scale; higher means more of the named tendency.", "",
             "Broad domains: " + "; ".join(f"{d} {v:.2f}" for d, v in doms.items()), "", "Facets:"]
    lines += [f"- {n}: {float(fs[n]):.2f}" for n in VAR["facets"]]
    return "\n".join(lines)


def main():
    R = list(csv.DictReader(open(CANON), delimiter="\t"))
    cols = [c for c in R[0] if c.startswith("HEXACO_") and not c.endswith("_mean")]
    F = {r["globalPID"]: {kitname(c): float(r[c]) for c in cols} for r in R}
    assert all(set(v) == set(VAR["facets"]) for v in F.values()), "facet names must match the variant template"
    json.dump(F, open(os.path.join(OUT, "facets.json"), "w"), indent=1)
    with open(os.path.join(OUT, "card_prompts.jsonl"), "w") as f:
        for pid, fs in sorted(F.items()):
            f.write(json.dumps(dict(pid=pid, messages=card_messages(fs))) + "\n")
            open(os.path.join(OUT, "thin_cards", pid + ".txt"), "w").write(thin_card(fs))
    # facts per call -> leave-one-call-out per perspective
    facts = {}
    for fp in glob.glob(os.path.join(ROOT, "analysis", "facts_per_call", "PID_*.json")):
        d = json.load(open(fp)); facts[d["dyadID"]] = d["facts"]
    rows = [r for r in P.rows() if r["dyadID"] != "PID_n+PID_n"]
    calls_of = {}
    for r in rows:
        calls_of.setdefault(r["selfPID"], set()).add(r["dyadID"])
    n_jobs = 0; nofacts = 0
    with open(os.path.join(OUT, "person_jobs.jsonl"), "w") as f:
        for r in rows:
            pid = r["selfPID"]
            if pid not in F:
                continue
            fl, seen = [], set()
            for dy in sorted(calls_of[pid] - {r["dyadID"]}):
                for x in (facts.get(dy, {}).get(pid) or []):
                    k = x.strip().lower()
                    if k and k not in seen:
                        seen.add(k); fl.append(x.strip())
            nofacts += not fl
            f.write(json.dumps(dict(perspectiveID=r["perspectiveID"], pid=pid, heldout_dyad=r["dyadID"], facts=fl, traits=trait_lines(F[pid]))) + "\n")
            n_jobs += 1
    pids = sorted(F); rng = random.Random(50)
    json.dump(sorted(rng.sample(pids, 50)), open(os.path.join(OUT, "frontier50.json"), "w"))
    print("people", len(F), "card prompts", len(F), "person jobs", n_jobs, "jobs with no facts", nofacts)


if __name__ == "__main__":
    main()
