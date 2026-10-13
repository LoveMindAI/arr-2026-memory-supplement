#!/usr/bin/env python3
"""Steps 1 and 2 of codebook/SCORING_PARADIGM_v351.md: human-human agreement against each source's yardstick,
then whether each AI coder can stand in for a person. Nothing is pooled across questions.

Usage: human_agreement.py [DIR] [--boot 1000]     (DIR defaults to analysis/code_v351/humans, built by load_human_coders.py)

For every question slot it reports, with bootstrap intervals over items:
  HH    mean pairwise agreement between human coders (and the all-humans alpha)
  AI-H  for each AI coder, mean pairwise agreement with each human
  AI-AI mean pairwise agreement between AI coders (stability only; never decides a status)
Statistics by kind (as in the paradigm): nominal alpha; ordinal alpha (details, intensity, appraisal); set alpha with
Jaccard distance (multi-select); appraisal also pairwise RMSE (crowd-enVENT's measure); feelings also crowd-enVENT-style
pairwise accuracy (exact same set) and any-overlap rate; reason first choice (nominal) and MeMo labels rebuilt (set).
Status per AI coder (only with 2+ humans), on alpha, as stated in the paper (A1, Oct 12; source yardsticks are context only):
  trusted          humans agree above chance (HH interval above 0) AND AI-H is inside or above the HH interval
  descriptive      humans agree above chance, AI-H below the HH interval
  not interpreted  humans do not agree above chance
Writes DIR/HUMAN_AGREEMENT.json and DIR/HUMAN_AGREEMENT.md.
"""
import argparse, collections, glob, itertools, json, os, sys, importlib
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "codebook"))
import scope_merge as SM
SPEC = SM.spec(importlib.import_module(os.environ.get("SPEC", "spec_v351_FROZEN")).SPEC)
Q = {q["id"]: q for p in SPEC["parts"] if p.get("models") is not False for q in p["questions"]}
MEMO = {s["code"]: s.get("memo") for q in Q.values() if q["type"] == "nested_ranked" for o in q["options"] for s in o.get("sub", [])}
MEMO.update({o["code"]: o.get("memo") for q in Q.values() if q["type"] == "nested_ranked" for o in q["options"] if not o.get("sub")})
AIS = ["claude-haiku-5.5", "gpt-6-luna", "deepseek-v4.1-flash"]
up = lambda x: str(x).strip().upper() if x is not None else None

# Yardsticks from SCORING_PARADIGM_v351.md (the source study's own human agreement). (value, statistic it was reported in, source)
YARD = {"new_info": (.845, "K", "Nissim et al. 2004"), "details": (.88, "ICC", "Levine et al. 2002 p. 680 (internal details; A1 OK Oct 9 to use the primary)"),
        "extent": (.83, "kappa", "Singer & Blagov specificity"), "moment_text": (.545, "kappa", "Brown-Schmidt et al. 2024"),
        "remembered": (.63, "kappa", "De Kuthy et al. 2018 (.63-.67)"), "shared_by": (.64, "kappa", "Ravichander & Black"),
        "personal_kinds": (.71, "kappa", "Barak & Gluck-Ofri (.71-.77)"), "feelings": (.52, "accuracy", "crowd-enVENT V-V accuracy (Table 5)"),
        "reason.first": (.60, "kappa", "Tsfasman MeMo reasons"), "reason.memo": (.60, "kappa", "Tsfasman MeMo reasons"),
        "bigger_meaning": (.70, "kappa", "Singer & Blagov meaning (.70-.72)"),
        **{f"appraisal.{s}": (1.48, "RMSE", "crowd-enVENT V-V RMSE (Table 5)") for s in ("pleasant", "unpleasant", "own_behavior", "other_behavior")}}


def label_map(q):
    m = {}
    for o in q.get("options", []) + q.get("scale", []): m[up(o["label"])] = o["code"]; m[up(o["code"])] = o["code"]
    return m


def slots():
    """(slot name, kind, extractor(codes)->value or None)."""
    out = []
    for qid, q in Q.items():
        L = label_map(q); codes = {up(o["code"]) for o in q.get("options", [])}
        fx = lambda v, L=L: up(L.get(up(v), v)) if v is not None else None
        if q["type"] == "single":
            ordered = all(isinstance(o["code"], int) for o in q["options"] if o["code"] != "CANT_TELL")
            def ex(c, qid=qid, fx=fx, codes=codes, ordered=ordered):
                v = c.get(qid); v = v[0] if isinstance(v, list) and len(v) == 1 else v; v = fx(v)
                if v is None or v not in codes: return None
                if ordered: return None if v == "CANT_TELL" else int(v)
                return v
            out.append((qid, "ordinal" if ordered else "nominal", ex))
        elif q["type"] == "multi":
            def ex(c, qid=qid, fx=fx, codes=codes):
                v = c.get(qid); v = [v] if isinstance(v, str) else v
                if not v: return None
                s = frozenset(fx(x) for x in v) & codes
                return s or None
            out.append((qid, "set", ex))
        elif q["type"] == "statements":
            for s in q["statements"]:
                def ex(c, qid=qid, s=s["code"], fx=fx):
                    v = fx((c.get(qid) or {}).get(s))
                    try: return None if v in (None, "CANT_TELL") else int(v)
                    except ValueError: return None
                out.append((f"{qid}.{s['code']}", "ordinal", ex))
        elif q["type"] == "nested_ranked":
            def first(c, qid=qid):
                v = c.get(qid)
                return up(v["about"][0]) if isinstance(v, dict) and v.get("about") else None
            def memo(c, qid=qid):
                v = c.get(qid)
                if not isinstance(v, dict) or not v.get("about"): return None
                dd = {up(x) for x in (v.get("details") or [])}; ab = [up(x) for x in v["about"]]
                mm = {MEMO.get(x) or "other" for x in dd} | {MEMO.get(x) or "other" for x in ab if x in ("NO_REASON", "OTHER")}
                return frozenset(mm) or None
            out += [(f"{qid}.first", "nominal", first), (f"{qid}.memo", "set", memo)]
    return out


class Alpha:
    """Krippendorff's alpha via a coincidence matrix, so item-bootstrap resamples are a weighted sum."""
    def __init__(self, units, kind):
        vals = sorted({v for u in units for v in u if v is not None}, key=lambda x: (str(type(x)), sorted(x) if isinstance(x, frozenset) else x))
        self.ix = {v: i for i, v in enumerate(vals)}; K = len(vals); self.kind = kind; self.vals = vals
        self.O = np.zeros((len(units), K, K))
        for n, u in enumerate(units):
            v = [self.ix[x] for x in u if x is not None]; m = len(v)
            if m < 2: continue
            for a, b in itertools.permutations(range(m), 2): self.O[n, v[a], v[b]] += 1 / (m - 1)
        if kind == "set":
            self.D = np.array([[1 - len(a & b) / len(a | b) for b in vals] for a in vals])
        elif kind == "nominal":
            self.D = 1 - np.eye(K)

    def __call__(self, w):
        o = np.tensordot(w, self.O, axes=1); nc = o.sum(1); n = nc.sum()
        if n < 2 or len(self.vals) < 2: return np.nan
        if self.kind == "ordinal":
            K = len(nc); cum = np.concatenate([[0], np.cumsum(nc)])
            D = np.array([[0.0 if g == h else ((cum[max(g, h) + 1] - cum[min(g, h)]) - (nc[g] + nc[h]) / 2) ** 2 for h in range(K)] for g in range(K)])
        else:
            D = self.D
        Do = (o * D).sum() / n; De = (np.outer(nc, nc) * D).sum() / (n * (n - 1))
        return np.nan if De == 0 else 1 - Do / De


def pair_metric(units, kind, slot):
    """Per-unit extra pairwise measures for a 2-coder unit list: RMSE (appraisal), exact-set accuracy and any-overlap (feelings)."""
    sq = np.array([np.nan if None in u else (u[0] - u[1]) ** 2 for u in units]) if slot.startswith("appraisal.") else None
    acc = np.array([np.nan if None in u else float(u[0] == u[1]) for u in units]) if slot == "feelings" else None
    ovl = np.array([np.nan if None in u else float(bool(u[0] & u[1])) for u in units]) if slot == "feelings" else None
    return sq, acc, ovl


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir", nargs="?", default="analysis/code_v351/humans"); ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args(); d = a.dir if os.path.isabs(a.dir) else os.path.join(ROOT, a.dir)
    C = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(d, "*__*.json")):
        j = json.load(open(f)); C[j["key"]][j["coder"].split("/")[-1]] = SM.codes(j["codes"])
    humans = sorted({c for v in C.values() for c in v if c.startswith("human-")})
    keys = sorted(k for k in C if any(c.startswith("human-") for c in C[k]))
    ais = [m for m in AIS if any(m in C[k] for k in keys)]
    rng = np.random.default_rng(20261009); N = len(keys)
    W = np.vstack([np.ones(N)] + [rng.multinomial(N, np.ones(N) / N) for _ in range(a.boot)])
    ci = lambda x: [round(float(np.nanpercentile(x[1:], 2.5)), 3), round(float(np.nanpercentile(x[1:], 97.5)), 3)]
    R = {"n_items": N, "humans": humans, "ais": ais, "boot": a.boot, "slots": {}}
    for slot, kind, ex in slots():
        val = {(k, c): ex(C[k][c]) if c in C[k] else None for k in keys for c in humans + ais}
        def pairstats(pairs):
            """Mean over coder pairs of pairwise alpha (and extras), as a bootstrap vector over items."""
            al, rm, ac, ov = [], [], [], []
            for x, y in pairs:
                units = [[val[(k, x)], val[(k, y)]] for k in keys]
                if sum(None not in u for u in units) < 5: continue
                A = Alpha(units, kind); al.append(np.array([A(w) for w in W]))
                sq, acc, ovl = pair_metric(units, kind, slot)
                msk = lambda arr: np.array([np.nansum(w * arr) / max(np.sum(w * ~np.isnan(arr)), 1e-9) for w in W])
                if sq is not None: rm.append(np.sqrt(msk(sq)))
                if acc is not None: ac.append(msk(acc)); ov.append(msk(ovl))
            if not al: return None
            out = {"alpha": np.nanmean(al, 0), "n_pairs": len(al)}
            if rm: out["rmse"] = np.nanmean(rm, 0)
            if ac: out["accuracy"] = np.nanmean(ac, 0); out["any_overlap"] = np.nanmean(ov, 0)
            return out
        def pack(s):
            if s is None: return None
            return {m: ({"est": round(float(v[0]), 3), "ci": ci(v)} if m != "n_pairs" else v) for m, v in s.items()}
        r = {"kind": kind, "n_coded_by_humans": int(sum(val[(k, h)] is not None for k in keys for h in humans))}
        hh = pairstats(list(itertools.combinations(humans, 2))); r["HH"] = pack(hh)
        if len(humans) >= 2:
            A = Alpha([[val[(k, h)] for h in humans] for k in keys], kind); v = np.array([A(w) for w in W])
            r["HH_all_alpha"] = {"est": round(float(v[0]), 3), "ci": ci(v)}
        r["AI_H"] = {m: pack(pairstats([(m, h) for h in humans])) for m in ais}
        r["AI_AI"] = pack(pairstats(list(itertools.combinations(ais, 2))))
        y = YARD.get(slot); r["yardstick"] = {"value": y[0], "reported_as": y[1], "source": y[2]} if y else None
        if hh is not None:
            # Status rule in the paper (A1, Oct 12): from our own coders, on Krippendorff's alpha. Source agreement is context only.
            #   trusted          people agree above chance (HH alpha interval above 0) and the AI coder is inside or above the HH interval
            #   descriptive      people agree above chance, AI coder below the HH interval
            #   not interpreted  people do not agree above chance
            metric = "alpha"; lo, hi = ci(hh[metric])
            r["humans_above_chance"] = bool(lo > 0); r["status_metric"] = metric; r["status"] = {}
            for m in ais:
                s = r["AI_H"].get(m)
                if not s: continue
                r["status"][m] = "not interpreted" if lo <= 0 else ("trusted" if s[metric]["est"] >= lo else "descriptive")
        R["slots"][slot] = r
    json.dump(R, open(os.path.join(d, "HUMAN_AGREEMENT.json"), "w"), indent=1)
    # readable table
    f = lambda s, m="alpha": "" if not s or m not in s else f"{s[m]['est']:.2f} [{s[m]['ci'][0]:.2f}, {s[m]['ci'][1]:.2f}]"
    L = [f"# Human coder agreement ({N} reports; humans: {', '.join(h[6:] for h in humans)})", "",
         "Pairwise means with 95% item-bootstrap intervals. HH = between humans; AI-H = that AI with each human; AI-AI = stability only.", "",
         "| Question | Stat | HH | " + " | ".join(f"{m.split('-')[0]}-H" for m in ais) + " | AI-AI | Yardstick | Status (" + "/".join(m.split('-')[0] for m in ais) + ") |",
         "|" + "---|" * (5 + len(ais) + 1)]
    for slot, r in R["slots"].items():
        for m in (["rmse"] if slot.startswith("appraisal.") else []) + (["accuracy"] if slot == "feelings" else []) + ["alpha"]:
            y = r["yardstick"]; ys = f"{y['value']} ({y['reported_as']})" if y and m == r.get("status_metric", m) else ""
            st = "/".join(r.get("status", {}).get(x, "") for x in ais) if m == r.get("status_metric") else ""
            L.append(f"| {slot} | {m} | {f(r['HH'], m)} | " + " | ".join(f(r['AI_H'].get(x), m) for x in ais) + f" | {f(r['AI_AI'], m)} | {ys} | {st} |")
    open(os.path.join(d, "HUMAN_AGREEMENT.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
