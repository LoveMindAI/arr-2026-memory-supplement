#!/usr/bin/env python3
"""Prolific twin validation (Oct 11 overnight; A1 approved the five tasks; agent2 ran them natively). For each task, from agent2's
de-identified exports (PROLIFIC_OVERNIGHT/exports/<task>/set*/ANNOTATIONS_DEIDENTIFIED_PRIVATE.json) and the private keys
(twin/prolific/*_key_PRIVATE.json): rater-rater agreement against each AI's agreement with the raters, on the same items.
Agreement: mean pairwise Cohen's kappa (nominal), Krippendorff's alpha (nominal) among raters, mean pairwise Jaccard for multi-answer
questions, Spearman for the 0-10 closeness. 'Skip' / missing answers are dropped pairwise. No API calls.
Usage: prolific_twin_analysis.py [TASK ...]   -> analysis/prolific_twin/<task>.json and a printed summary"""
import collections, glob, itertools, json, os, sys
import numpy as np
from sklearn.metrics import cohen_kappa_score
from scipy.stats import spearmanr
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); EXP = os.path.join(ROOT, "PROLIFIC_OVERNIGHT", "exports")
KEYS = os.path.join(ROOT, "twin", "prolific"); OUT = os.path.join(ROOT, "analysis", "prolific_twin"); os.makedirs(OUT, exist_ok=True)

def load(task):
    R = collections.defaultdict(dict); sets = []
    for f in sorted(glob.glob(os.path.join(EXP, task, "set*", "ANNOTATIONS_DEIDENTIFIED_PRIVATE.json"))):
        sets.append(os.path.basename(os.path.dirname(f)))
        for r in json.load(open(f))["reader_item_rows"]:
            ans = {}
            for q, a in (r.get("annotations") or {}).items():
                v = [str(x) for x in (a.get("raw_values") or []) if str(x).lower() not in ("skip", "__skipped", "", "none")]
                if v: ans[q] = v
            R[r["item_id"]][r["reader_code"]] = ans
    return R, sets

def kappa_pairs(pairs):
    """pairs: list of (a_labels, b_labels) per rater pair over shared items -> mean kappa (skip degenerate)."""
    ks = []
    for a, b in pairs:
        if len(a) >= 5 and len(set(a) | set(b)) > 1:
            try: ks.append(cohen_kappa_score(a, b))
            except Exception: pass
    return round(float(np.mean(ks)), 3) if ks else None

def alpha_nominal(units):
    """units: list of lists of labels (one list per item, raters' labels). Krippendorff's alpha, nominal."""
    units = [u for u in units if len(u) >= 2]
    if not units: return None
    cats = sorted({x for u in units for x in u}); ix = {c: i for i, c in enumerate(cats)}; o = np.zeros((len(cats), len(cats)))
    for u in units:
        m = len(u)
        for i, j in itertools.permutations(range(m), 2): o[ix[u[i]], ix[u[j]]] += 1 / (m - 1)
    n = o.sum(); nc = o.sum(1)
    if n <= 1: return None
    do = (n - np.trace(o)); de = (n * n - (nc ** 2).sum()) / (n - 1)
    return round(float(1 - do / de), 3) if de > 0 else None

def nominal(R, q, ai):
    """R: item -> reader -> answers; ai: dict name -> item -> label. Single-answer question q."""
    lab = {it: {rd: a[q][0] for rd, a in rs.items() if q in a} for it, rs in R.items()}
    readers = sorted({rd for v in lab.values() for rd in v})
    rr = []
    for a, b in itertools.combinations(readers, 2):
        its = [it for it in lab if a in lab[it] and b in lab[it]]
        rr.append(([lab[it][a] for it in its], [lab[it][b] for it in its]))
    out = {"items": len(lab), "rater_rater_kappa": kappa_pairs(rr), "rater_alpha": alpha_nominal([list(v.values()) for v in lab.values()])}
    for name, A in ai.items():
        pr = []
        for rd in readers:
            its = [it for it in lab if rd in lab[it] and A.get(it) is not None]
            pr.append(([lab[it][rd] for it in its], [A[it] for it in its]))
        out[f"{name}_rater_kappa"] = kappa_pairs(pr)
        maj = [(collections.Counter(lab[it].values()).most_common(1)[0][0], A[it]) for it in lab if A.get(it) is not None and len(lab[it]) >= 2]
        out[f"{name}_matches_majority"] = round(float(np.mean([m == a for m, a in maj])), 3) if maj else None
    return out

def jacc(a, b):
    a, b = set(a), set(b); return len(a & b) / len(a | b) if a | b else 1.0

def multi(R, q, ai):
    lab = {it: {rd: a[q] for rd, a in rs.items() if q in a} for it, rs in R.items()}
    rr = [jacc(x, y) for v in lab.values() for x, y in itertools.combinations(list(v.values()), 2)]
    out = {"items": len(lab), "rater_rater_jaccard": round(float(np.mean(rr)), 3) if rr else None}
    for name, A in ai.items():
        s = [jacc(x, A[it]) for it, v in lab.items() if A.get(it) is not None for x in v.values()]
        out[f"{name}_rater_jaccard"] = round(float(np.mean(s)), 3) if s else None
    return out

def norm_code(x): return str(x).strip().upper()

# ---------------- tasks
def task_F():
    R, sets = load("F_form"); K = json.load(open(os.path.join(KEYS, "form_key_PRIVATE.json")))
    ai = {}
    for coder in ("claude-haiku-5.5", "gpt-6-luna"):
        A = collections.defaultdict(dict)
        for it, k in K.items():
            f = os.path.join(ROOT, "analysis", "code_v351", "twin_final", f"{k['codebook_key']}__{coder}.json")
            if os.path.exists(f):
                c = json.load(open(f)).get("codes") or {}
                for q in ("extent", "new_info"):
                    if c.get(q) is not None: A[q][it] = norm_code(c[q])
                if c.get("remembered"): A["remembered"][it] = [norm_code(x) for x in c["remembered"]]
        ai[coder] = A
    for it in R:
        for rd in R[it]:
            for q in ("extent", "new_info", "remembered"):
                if q in R[it][rd]: R[it][rd][q] = [norm_code(x) for x in R[it][rd][q]]
    res = {"sets": sets}
    for q in ("extent", "new_info"): res[q] = nominal(R, q, {c: ai[c][q] for c in ai})
    res["remembered"] = multi(R, "remembered", {c: ai[c]["remembered"] for c in ai})
    return res

def task_D():
    R, sets = load("D_depth"); K = json.load(open(os.path.join(KEYS, "depth_key_PRIVATE.json")))
    cand = {cid: c for v in K.values() for cid, c in v["cands"].items()}
    q = next((qq for it in R.values() for a in it.values() for qq in a), "depth")
    ai = {}
    for coder in ("claude-haiku-5.5", "gpt-6-luna", "gemini-3-flash-preview"):
        A = {}
        for cid, c in cand.items():
            ds = [d for co, kind, d in c["by"] if co == coder]
            A[cid] = "deep" if "DEEP" in ds else "surface" if ds else "no"
        ai[coder] = A
    for it in R:
        for rd in R[it]:
            if q in R[it][rd]: R[it][rd][q] = [str(R[it][rd][q][0]).lower()]
    res = {"sets": sets, "question": q, "agreement": nominal(R, q, ai)}
    lab = [a[q][0] for it in R.values() for a in it.values() if q in a]
    res["rates"] = {"raters": {k: round(lab.count(k) / len(lab), 3) for k in ("no", "surface", "deep")} if lab else None}
    for coder, A in ai.items():
        v = [A[it] for it in R if it in A]; res["rates"][coder] = {k: round(v.count(k) / len(v), 3) for k in ("no", "surface", "deep")} if v else None
    return res

def task_J():
    R, sets = load("J_judge"); K = json.load(open(os.path.join(KEYS, "judge_key_PRIVATE.json")))
    QJ = {"topic": "topic_match", "event": "event_match", "focus": "same_focus", "reason": "same_reason"}
    res = {"sets": sets}
    for kind in ("model", "partner", "all"):
        Rk = {it: v for it, v in R.items() if kind == "all" or K[it]["kind"] == kind}; res[kind] = {}
        for q, jq in QJ.items():
            ai = {}
            for j in ("openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash", "anthropic/claude-haiku-5.5"):
                ai[j.split("/")[1]] = {it: str(int(float(K[it]["judges"][j][jq]))) for it in Rk if K[it]["judges"].get(j) and K[it]["judges"][j].get(jq) is not None}
            res[kind][q] = nominal(Rk, q, ai)
        # closeness 0-10 vs judges' similarity 0-1
        pairs = [(float(x["close"][0]), float(y["close"][0])) for v in Rk.values() for x, y in itertools.combinations(list(v.values()), 2) if "close" in x and "close" in y]
        cl = {"rater_rater_spearman": round(float(spearmanr(*zip(*pairs)).correlation), 3) if len(pairs) > 5 else None}
        for j in ("openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash", "anthropic/claude-haiku-5.5"):
            pj = [(float(a["close"][0]), float(K[it]["judges"][j]["similarity"])) for it, v in Rk.items() for a in v.values() if "close" in a and K[it]["judges"].get(j)]
            cl[f"{j.split('/')[1]}_rater_spearman"] = round(float(spearmanr(*zip(*pj)).correlation), 3) if len(pj) > 5 else None
        res[kind]["close"] = cl
    return res

def task_K():
    R, sets = load("K_keenan"); K = json.load(open(os.path.join(KEYS, "keenan_key_PRIVATE.json")))
    res = {"sets": sets}
    for stratum in ("random", "all"):
        Rk = {it: v for it, v in R.items() if stratum == "all" or K[it]["stratum"] == stratum}; res[stratum] = {}
        for q in ("compliment", "surprise", "humor"):
            ai = {c: {it: str(int(K[it]["ai"][c][q])) for it in Rk if K[it]["ai"].get(c) and q in K[it]["ai"][c]} for c in ("gpt-6-luna", "deepseek-v4.1-flash")}
            res[stratum][q] = nominal(Rk, q, ai)
            lab = [a[q][0] for v in Rk.values() for a in v.values() if q in a]
            res[stratum][q]["rater_yes_rate"] = round(lab.count("1") / len(lab), 3) if lab else None
    return res

def task_Y():
    R, sets = load("Y_types"); K = json.load(open(os.path.join(KEYS, "types_key_PRIVATE.json")))
    T = {"a": "stretch_we_shared", "b": "what_they_told_me"}
    q = next((qq for it in R.values() for a in it.values() for qq in a), "type")
    picks = [(K[it]["coded_type"], T.get(a[q][0], "unsure")) for it, v in R.items() for a in v.values() if q in a]
    res = {"sets": sets, "items": len(R), "answers": len(picks),
           "match_coded_type": round(float(np.mean([c == p for c, p in picks])), 3) if picks else None,
           "unsure_share": round(float(np.mean([p == "unsure" for _, p in picks])), 3) if picks else None,
           "by_coded_type": {t: round(float(np.mean([p == t for c, p in picks if c == t])), 3) for t in T.values() if any(c == t for c, _ in picks)}}
    maj = []
    for it, v in R.items():
        ps = [T.get(a[q][0], "unsure") for a in v.values() if q in a]
        if len(ps) >= 2:
            top, n = collections.Counter(ps).most_common(1)[0]
            if n * 2 > len(ps): maj.append(top == K[it]["coded_type"])   # strict majority only; ties have no majority
    res["majority_matches_coded_type"] = round(float(np.mean(maj)), 3) if maj else None
    res["items_with_strict_majority"] = len(maj); res["majority_agreeing_items"] = int(sum(maj))
    lab = {it: {rd: a[q][0] for rd, a in v.items() if q in a} for it, v in R.items()}
    res["rater_alpha"] = alpha_nominal([list(v.values()) for v in lab.values()])
    return res

if __name__ == "__main__":
    want = sys.argv[1:] or ["F_form", "D_depth", "J_judge", "K_keenan", "Y_types"]
    fn = {"F_form": task_F, "D_depth": task_D, "J_judge": task_J, "K_keenan": task_K, "Y_types": task_Y}
    for t in want:
        if not glob.glob(os.path.join(EXP, t, "set*", "ANNOTATIONS_DEIDENTIFIED_PRIVATE.json")): print(t, "no export yet"); continue
        r = fn[t](); r["complete"] = os.path.exists(os.path.join(EXP, f"DONE_{t}.md"))
        json.dump(r, open(os.path.join(OUT, f"{t}.json"), "w"), indent=1); print("==", t, json.dumps(r)[:1500])
