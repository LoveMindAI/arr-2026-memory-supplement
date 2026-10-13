#!/usr/bin/env python3
"""Agreement for codebook v3.5 codes, per codebook/AGREEMENT_PLAN_v33.md: no pooled headline number; per question, in four families.
single nominal -> nominal alpha; ordered (details, intensity) -> ordinal alpha (Krippendorff 2011 ordinal metric); multi -> set alpha (Jaccard)
+ per-option binary alpha with prevalence; appraisal -> per statement ordinal alpha + mean pairwise RMSE (crowd-enVENT's measure) + Can't-tell rate;
reason -> first-choice nominal alpha, set alpha on 'about', set alpha on details, set alpha on MeMo labels rebuilt from the details.
Works for any coders (models or humans): reads <dir>/*__<coder>.json (models) or --human files {coder: path}.
Usage: code_v35_agree.py analysis/code_v35/pilot40 [--coders a,b,c]"""
import argparse, collections, glob, itertools, json, os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "codebook")); import importlib; SPEC = importlib.import_module(os.environ.get("SPEC", "spec_v351_FROZEN")).SPEC
Q = [q for p in SPEC["parts"] if p.get("models") is not False for q in p["questions"]]
FAMILY = {"topic": 1, "remembered": 1, "new_info": 1, "extent": 1, "details": 1, "moment_text": 1, "shared_by": 1, "personal_kinds": 1,
          "feelings": 2, "intensity": 2, "appraisal": 3, "reason": 4, "matters_for": 4, "bigger_meaning": 4}
MEMO = {s["code"]: s.get("memo") for q in Q if q["type"] == "nested_ranked" for o in q["options"] for s in o.get("sub", [])}
MEMO.update({o["code"]: o.get("memo") for q in Q if q["type"] == "nested_ranked" for o in q["options"] if not o.get("sub")})


def alpha(units, dist):
    pairs, n, allv = [], 0, []
    for u in units:
        v = [x for x in u if x is not None]
        if len(v) < 2: continue
        m = len(v); n += m; allv += v
        pairs += [dist(v[a], v[b]) / (m - 1) for a, b in itertools.permutations(range(m), 2)]
    if n < 2: return None
    Do = sum(pairs) / n; De = sum(dist(a, b) for a, b in itertools.permutations(allv, 2)) / (n * (n - 1))
    return None if De == 0 else round(1 - Do / De, 3)


def ordinal_dist(units):
    vals = [x for u in units for x in u if x is not None]; cnt = collections.Counter(vals); levels = sorted(cnt)
    def d(a, b):
        lo, hi = sorted((levels.index(a), levels.index(b)))
        return (sum(cnt[levels[g]] for g in range(lo, hi + 1)) - (cnt[levels[lo]] + cnt[levels[hi]]) / 2) ** 2
    return d


nominal = lambda a, b: float(a != b)
jacc = lambda a, b: 1 - (len(a & b) / len(a | b) if (a | b) else 1.0)
up = lambda x: str(x).strip().upper() if x is not None else None


def load(d, coders):
    C = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(d, "*__*.json")):
        j = json.load(open(f)); c = j["coder"].split("/")[-1]
        if j.get("ok") and (not coders or c in coders): C[j["key"]][c] = j["codes"]
    return C


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--coders", default=""); a = ap.parse_args()
    coders_f = [c for c in a.coders.split(",") if c]
    C = load(os.path.join(ROOT, a.dir) if not os.path.isabs(a.dir) else a.dir, coders_f)
    coders = sorted({c for v in C.values() for c in v}); keys = sorted(k for k in C if len(C[k]) >= 2)
    R = {"n_items": len(keys), "coders": coders, "families": {1: {}, 2: {}, 3: {}, 4: {}}, "schema": collections.Counter()}
    LAB = {}
    for q in Q:
        for o in q.get("options", []) + q.get("scale", []): LAB[(q["id"], up(o["label"]))] = o["code"]
    def fix(qid, v):                                           # a coder sometimes writes the label ('Slightly agree') instead of the code
        if isinstance(v, str) and (qid, up(v)) in LAB:
            code = LAB[(qid, up(v))]
            if up(code) != up(v): R["schema"][f"{qid}:label_not_code"] += 1      # only genuine slips ('Slightly agree'), not 'Yes' for YES
            return code
        return v
    def get(k, c, qid):
        v = (C[k].get(c) or {}).get(qid)
        if isinstance(v, list): return [fix(qid, x) for x in v]
        if isinstance(v, dict) and qid == "appraisal": return {kk: fix(qid, vv) for kk, vv in v.items()}
        return fix(qid, v)
    for q in Q:
        fam = R["families"][FAMILY[q["id"]]]; codes = {up(o["code"]) for o in q.get("options", [])}
        if q["type"] == "single":
            ordered = all(isinstance(o["code"], int) for o in q["options"] if o["code"] != "CANT_TELL")
            units = []
            for k in keys:
                u = []
                for c in coders:
                    v = get(k, c, q["id"]); v = v[0] if isinstance(v, list) and len(v) == 1 else v; v = up(v)
                    if v is not None and v not in codes: R["schema"][f"{q['id']}={v}"] += 1; v = None
                    if ordered: v = None if v in (None, "CANT_TELL") else int(v)
                    u.append(v)
                units.append(u)
            fam[q["id"]] = {"ordinal_alpha" if ordered else "nominal_alpha": alpha(units, ordinal_dist(units) if ordered else nominal)}
            if ordered: fam[q["id"]]["cant_tell_rate"] = round(float(np.mean([up(get(k, c, q["id"])) == "CANT_TELL" for k in keys for c in coders])), 3)
        elif q["type"] == "multi":
            units, per = [], collections.defaultdict(list)
            for k in keys:
                u = []
                for c in coders:
                    v = get(k, c, q["id"]); v = [v] if isinstance(v, str) else (v or None)
                    if v is not None:
                        v = {up(x) for x in v}; bad = v - codes
                        for b in bad: R["schema"][f"{q['id']}={b}"] += 1
                        v = v & codes or None
                    u.append(frozenset(v) if v else None)
                units.append(u)
                for o in codes: per[o].append([None if x is None else int(o in x) for x in u])
            prev = {o: round(float(np.mean([x for u in per[o] for x in u if x is not None])), 3) for o in codes}
            fam[q["id"]] = {"set_alpha_jaccard": alpha(units, jacc),
                            "per_option": {o: {"alpha": alpha(per[o], nominal), "chosen_rate": prev[o], "unstable": bool(prev[o] < .05)} for o in sorted(codes)}}
        elif q["type"] == "statements":
            out = {}
            for s in q["statements"]:
                def num(x):
                    if up(x) in (None, "CANT_TELL"): return None
                    try: return int(x)
                    except Exception: R["schema"][f"appraisal.{s['code']}={x}"] += 1; return None
                units = [[num((get(k, c, q["id"]) or {}).get(s["code"])) for c in coders] for k in keys]
                rm = [((a - b) ** 2) for u in units for a, b in itertools.combinations([x for x in u if x is not None], 2)]
                ct = np.mean([up((get(k, c, q["id"]) or {}).get(s["code"])) == "CANT_TELL" for k in keys for c in coders])
                out[s["code"]] = {"ordinal_alpha": alpha(units, ordinal_dist(units)), "mean_pairwise_rmse": round(float(np.sqrt(np.mean(rm))), 3) if rm else None, "cant_tell_rate": round(float(ct), 3)}
            fam[q["id"]] = out
        elif q["type"] == "nested_ranked":
            first, about, det, memo = [], [], [], []
            for k in keys:
                f_, a_, d_, m_ = [], [], [], []
                for c in coders:
                    v = get(k, c, q["id"])
                    if not isinstance(v, dict) or not v.get("about"): f_.append(None); a_.append(None); d_.append(None); m_.append(None); continue
                    ab = [up(x) for x in v["about"]]; dd = {up(x) for x in (v.get("details") or [])}
                    f_.append(ab[0]); a_.append(frozenset(ab)); d_.append(frozenset(dd) if dd else frozenset({"(NONE)"}))
                    mm = {MEMO.get(x) or "other" for x in dd} | {MEMO.get(x) or "other" for x in ab if x in ("NO_REASON", "OTHER")}
                    m_.append(frozenset(mm) if mm else None)
                first.append(f_); about.append(a_); det.append(d_); memo.append(m_)
            fam[q["id"]] = {"first_choice_nominal_alpha": alpha(first, nominal), "about_set_alpha": alpha(about, jacc),
                            "details_set_alpha": alpha(det, jacc), "memo_rebuilt_set_alpha": alpha(memo, jacc)}
    R["schema"] = dict(R["schema"])
    print(json.dumps(R, indent=1))
    json.dump(R, open(os.path.join(ROOT, a.dir, "AGREEMENT.json") if not os.path.isabs(a.dir) else os.path.join(a.dir, "AGREEMENT.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
