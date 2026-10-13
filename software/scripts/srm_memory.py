#!/usr/bin/env python3
"""Social Relations Model variance partition for PARSEL memory measures (after Bond, Dorsky & Kenny 1992; Kenny & LaVoie 1984).
Round robin within batch (≈6 people, 5 rounds). One memory per directed pair (rememberer -> partner), so relationship variance
is confounded with error (as in Bond et al.); we also report dyadic reciprocity (corr of A-on-B and B-on-A residuals).
Estimator: linear mixed model, crossed random effects for rememberer and partner nested in batch
(statsmodels MixedLM, groups=batch, variance components). Binary codes analysed as linear probability.
Measures: ratings (timing slider, pleasure/arousal/dominance; conv_TP_Mem_sc2/sc3/ent1) and blind codes (analysis/coded)."""
import glob, json, os, sys, warnings
from collections import defaultdict
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
sys.path.insert(0, os.path.dirname(__file__))
import style_ceiling as sc
warnings.filterwarnings("ignore")


def main():
    import prompts as P
    raw = P.rows()
    CODES = {json.load(open(f))["i"]: json.load(open(f))["code"] for f in glob.glob(os.path.join(sc.OUT, "coded", "*.json"))}
    rec = []
    for i, r in enumerate(raw):
        d = dict(batch=r["batchID"], rem=r["selfPID"], par=r["otherPID"])
        for k, col in (("timing", "conv_SP_Mem"), ("pleasure", "conv_SP_Mem_p"), ("arousal", "conv_SP_Mem_a"), ("dominance", "conv_SP_Mem_d"),
                       ("getting_along", "conv_TP_Mem_sc2"), ("min_conflict", "conv_TP_Mem_sc3"), ("unit", "conv_TP_Mem_ent1")):
            try: d[k] = float(r[col])
            except: d[k] = np.nan
        c = CODES[i]
        g = lambda f: str(c.get(f, "")).split(" ")[0].lower()
        d["focus_partner"] = float(g("focus") == "partner"); d["focus_both"] = float(g("focus") == "both")
        d["specific_event"] = float(g("specificity") == "specific_event")
        d["reason_connection"] = float(g("reason_type") == "social_connection"); d["reason_selfrel"] = float(g("reason_type") == "self_relevance")
        d["reason_novelty"] = float(g("reason_type") == "novelty")
        d["kind_fact"] = float(g("kind") == "personal_fact"); d["kind_common"] = float(g("kind") == "commonality"); d["kind_task"] = float(g("kind") == "task_or_study")
        d["mem_words"] = float(len(r["conv_SP_Mem_descr"].split()))
        rec.append(d)
    df = pd.DataFrame(rec)
    measures = ["timing", "pleasure", "arousal", "dominance", "getting_along", "min_conflict", "unit", "focus_partner", "focus_both", "specific_event",
                "reason_connection", "reason_selfrel", "reason_novelty", "kind_fact", "kind_common", "kind_task", "mem_words"]
    out = {}
    print(f"{'measure':18s} {'rememberer':>10s} {'partner':>8s} {'resid':>6s}   reciprocity")
    for m in measures:
        d = df.dropna(subset=[m]).copy()
        d["y"] = (d[m] - d[m].mean()) / (d[m].std() or 1)
        try:
            fit = smf.mixedlm("y ~ 1", d, groups="batch", re_formula="0", vc_formula={"rem": "0 + C(rem)", "par": "0 + C(par)"}).fit(reml=True, method="lbfgs")
            vc = dict(zip(fit.model.exog_vc.names, fit.vcomp)); res = fit.scale
            tot = vc["rem"] + vc["par"] + res
            # reciprocity: residual of A->B vs B->A
            d["resid"] = fit.resid
            key = {(a, b): v for a, b, v in zip(d["rem"], d["par"], d["resid"])}
            pairs = [(key[(a, b)], key[(b, a)]) for (a, b) in key if (b, a) in key and a < b]
            rho = float(np.corrcoef(np.array(pairs).T)[0, 1]) if len(pairs) > 10 else float("nan")
            out[m] = dict(rememberer=vc["rem"] / tot, partner=vc["par"] / tot, residual=res / tot, reciprocity=rho, n=len(d), n_dyads=len(pairs))
            print(f"{m:18s} {vc['rem']/tot:10.2f} {vc['par']/tot:8.2f} {res/tot:6.2f}   {rho:+.2f} (dyads {len(pairs)})")
        except Exception as e:
            print(m, "fit failed", e)
    json.dump(out, open(os.path.join(sc.OUT, "srm_memory.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
