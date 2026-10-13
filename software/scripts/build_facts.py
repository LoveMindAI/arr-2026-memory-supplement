#!/usr/bin/env python3
"""The facts file for the redraft: every number the prose uses, pulled from its analysis file, with the source beside it.
Rule (OCEAN habit): numbers go into prose only from this file. Re-run after any analysis changes.
-> paper_notes/FACTS_v2.md (and FACTS_v2.json)"""
import json, os, sys, csv, re
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
A = lambda f: json.load(open(os.path.join(ROOT, "analysis", f)))
F = []                                    # (section, label, value, source)
def add(sec, label, val, src): F.append((sec, label, val, src))
r2 = lambda x: None if x is None else round(x, 3)

# ---------- §2 data
sys.path.insert(0, os.path.join(ROOT, "scripts")); import prompts as P
rows = P.rows()
add("data", "conversation-stage reports in our sample", len(rows), "prompts.rows()")
add("data", "rememberers", len({r['selfPID'] for r in rows}), "prompts.rows()")
add("data", "conversations (dyads)", len({r['dyadID'] for r in rows}), "prompts.rows()")
add("data", "batches", len({r['batchID'] for r in rows}), "prompts.rows()")
add("data", "usable for prediction (one near-silent call dropped)", 1147, "Data paragraph; FINAL.jsonl has 1147 lines")
_fin = [json.loads(l)["key"].split("|")[0] for l in open(os.path.join(ROOT, "analysis", "code_v351", "human", "FINAL.jsonl"))]
_R = {r["perspectiveID"]: r for r in rows}; _k = [_R[i] for i in _fin if i in _R]
add("data", "analyzed (1,147): people / conversations / groups", f"{len({r['selfPID'] for r in _k})} / {len({r['dyadID'] for r in _k})} / {len({r['batchID'] for r in _k})}", "analysis/code_v351/human/FINAL.jsonl")

# ---------- §4 RQ1: social relations (GLMM)
S = A("srm_glmm/srm_glmm.json")
for k in ["b__extent__ONE_MOMENT", "b__extent__LONGER", "b__remembered__PARTNER_SAID_OR_DID", "b__remembered__DID_TOGETHER", "b__new_info__YES",
          "b__moment_text__WHAT_HAPPENED", "b__shared_by__PARTNER", "b__reason_first__PARTNER", "b__reason_first__ME",
          "b__memo__self_perception.annotator_feelings", "b__topic__PARTNER_IMPRESSION", "b__topic__HOBBIES_INTERESTS_MEDIA", "b__topic__PLANS_EVENTS",
          "b__topic__WORK_EDUCATION_SERVICE", "b__topic__PLACE_TRAVEL"]:
    g = S[k]["glmm"]
    add("rq1-srm", f"{k[3:]}: rememberer / partner latent share (GLMM)", f"{g['rememberer']:.2f} / {g['partner']:.2f} (p_rem {g['p_rem']:.1e}, p_par {g['p_par']:.1e})", "analysis/srm_glmm/srm_glmm.json")
for k in ["n__details", "n__intensity", "n__appraisal_pleasant", "n__appraisal_own_behavior"]:
    g = S[k]["lmm"]
    add("rq1-srm", f"{k[3:]}: rememberer / partner share (LMM)", f"{g['rememberer']:.2f} / {g['partner']:.2f}", "analysis/srm_glmm/srm_glmm.json")
add("rq1-srm", "unstable (separation) outcomes, LPM only", ", ".join(k[3:] for k, v in S.items() if v.get("glmm", {}).get("regularised")), "analysis/srm_glmm/srm_glmm.json")

# ---------- §4 RQ1: author identification
C = A("crossctx_controls.json")
for k, v in C.items():
    if isinstance(v, dict) and "top1" in v:
        add("rq1-author", f"cross-context author ID: {k}", f"{v['top1']:.3f} (chance {v['chance']:.3f}; CI of above-chance [{v['lo']:.3f}, {v['hi']:.3f}]; n={v['n']}, people={v['people']})", "analysis/crossctx_controls.json")

# ---------- §4 RQ1: the ladder (person-specific fit = excess over batch-mates)
L = A("ladder_v351.json")
for coder in L:
    for q in ["topic", "extent", "remembered", "new_info", "details", "moment_text", "reason.first", "personal_kinds", "feelings", "shared_by"]:
        a = L[coder]["all"].get(q)
        if not a: continue
        for pred in ["partner's memory", "person's own earlier reports", "model: no history", "model: history", "most common human answer"]:
            e = a.get(pred, {}).get("excess")
            if e: add("rq1-ladder", f"[{coder}] {q} | {pred}: person-specific fit", f"{e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}] n={e[3]}", "analysis/ladder_v351.json")

# ---------- §4 traits (v3.5.1)
Tm = A("trait_memory_v351.json")
add("rq1-traits", "people / tests / survivors at q<.05", f"{Tm['n_people']} / {Tm['n_tests']} / {len(Tm['survivors_q05'])}", "analysis/trait_memory_v351.json")
for x in Tm["survivors_q05"]: add("rq1-traits", f"{x['trait']} ~ {x['feature']}", f"rho {x['rho']:+.2f}, q {x['q']:.3f}, n {x['n']}", "analysis/trait_memory_v351.json")
Bm = A("trait_memory.json")["B"]; k = max(Bm, key=lambda z: Bm[z]["r"])
add("rq1-traits", "embeddings -> trait, largest CV r", f"{Bm[k]['r']:.3f} ({k}, p_perm {Bm[k]['p_perm']:.3f})", "analysis/trait_memory.json (part B, no codes)")

# ---------- §5 RQ2: which moments
M = A("memorability.json")
add("rq2-memorability", "turns / reports / share remembered", f"{M['units']} / {M['reports']} / {M['positive_rate']:.3f}", "analysis/memorability.json")
for m, v in M["models"].items(): add("rq2-memorability", f"{m} balanced accuracy (hit@1, chance)", f"{v['bal_acc']:.3f} ({v.get('hit_at_1')}, {v.get('hit_at_1_chance')})", "analysis/memorability.json")
for fam, v in M["ablation"]["LR"]["families"].items(): add("rq2-memorability", f"LR family '{fam}': alone / drop without", f"{v['alone']:.3f} / {v['drop_without']:+.3f}", "analysis/memorability.json")
# remembered rate by fifth of the call, and remembered vs other turn length (people)
import collections, glob
loc_dir = os.path.join(ROOT, "analysis", "located"); rowsd = {r["perspectiveID"]: r for r in rows}
cnt = collections.Counter(); pos = collections.Counter(); wl = {0: [], 1: []}
for f in sorted(x for x in os.listdir(loc_dir) if x.startswith("PID_") and x.endswith(".json")):
    Lc = json.load(open(os.path.join(loc_dir, f)))
    if Lc.get("kind") != "in_conversation": continue
    loc = Lc["turns"]; loc = set(json.loads(loc) if isinstance(loc, str) else (loc or []))
    if not loc: continue
    tr = json.load(open(os.path.join(P.TX, rowsd[Lc["perspectiveID"]]["dyadID"] + ".json")))["turns"]; end = max(x["start"] for x in tr) + 5
    for k, x in enumerate(tr):
        q = min(4, int(5 * x["start"] / end)); y = int(k in loc); cnt[q] += 1; pos[q] += y; wl[y].append(len((x["text"] or "").split()))
add("rq2-memorability", "share of turns remembered, by fifth of the call (1..5)", " / ".join(f"{pos[q] / cnt[q]:.3f}" for q in range(5)), "located + transcripts (computed in build_facts)")
add("rq2-memorability", "words per turn, remembered / other: median; mean", f"{np.median(wl[1]):.0f} / {np.median(wl[0]):.0f}; {np.mean(wl[1]):.1f} / {np.mean(wl[0]):.1f}", "located + transcripts (computed in build_facts)")
# sensitivity (A1, Oct 12): fill each listed span first-to-last (LOC_FILL=1 runs)
nonconsec = 0; cntF = collections.Counter(); posF = collections.Counter()
for f in sorted(x for x in os.listdir(loc_dir) if x.startswith("PID_") and x.endswith(".json")):
    Lc = json.load(open(os.path.join(loc_dir, f)))
    if Lc.get("kind") != "in_conversation": continue
    loc = Lc["turns"]; loc = set(json.loads(loc) if isinstance(loc, str) else (loc or []))
    if not loc: continue
    full = set(range(min(loc), max(loc) + 1)); nonconsec += int(full != loc)
    tr = json.load(open(os.path.join(P.TX, rowsd[Lc["perspectiveID"]]["dyadID"] + ".json")))["turns"]; end = max(x["start"] for x in tr) + 5
    for k, x in enumerate(tr):
        q = min(4, int(5 * x["start"] / end)); cntF[q] += 1; posF[q] += int(k in full)
add("rq2-filled", "placements whose listed turns skip turns in between", f"{nonconsec}", "analysis/located (computed in build_facts)")
add("rq2-filled", "filled spans: share of turns remembered, by fifth of the call (1..5)", " / ".join(f"{posF[q] / cntF[q]:.3f}" for q in range(5)), "located + transcripts (computed in build_facts)")
if os.path.exists(os.path.join(ROOT, "analysis", "memorability_filled.json")):
    MF = A("memorability_filled.json")
    add("rq2-filled", "filled spans: share remembered", f"{MF['positive_rate']:.3f} (listed {M['positive_rate']:.3f})", "analysis/memorability_filled.json")
    for m in ("LR", "RF"): add("rq2-filled", f"filled spans: {m} balanced accuracy", f"{MF['models'][m]['bal_acc']:.3f} (listed {M['models'][m]['bal_acc']:.3f})", "analysis/memorability_filled.json")
Mp = A("memorability_person.json")["compare"]
for m in ("LR", "RF"):
    c = Mp[m]; add("rq2-person", f"{m}: conversation features / + person's earlier memories / earlier alone; share of splits improved",
                   f"{c['conversation features']['bal_acc']:.3f} / {c['+ person' + chr(39) + 's earlier memories']['bal_acc']:.3f} / {c['person' + chr(39) + 's earlier memories alone']['bal_acc']:.3f}; {c.get('gain_share_of_splits')}", "analysis/memorability_person.json")
Mm = A("memorability_models.json")
late = [sum(v["share_by_fifth_model"][3:]) for v in Mm["models"].values()]; first = [v["share_by_fifth_model"][0] for v in Mm["models"].values()]
v0 = next(iter(Mm["models"].values()))
add("rq2-models", "models: share of chosen turns in last two fifths (range over 5 models) vs people", f"{min(late):.2f}-{max(late):.2f} vs {sum(v0['share_by_fifth_people'][3:]):.2f}", "analysis/memorability_models.json")
add("rq2-models", "models: share in first fifth (range) vs people", f"{min(first):.3f}-{max(first):.3f} vs {v0['share_by_fifth_people'][0]:.3f}", "analysis/memorability_models.json")
tr = [v["people_trained_model_on_model_choice"] for v in Mm["models"].values()]; tp = [v["people_trained_model_on_people"] for v in Mm["models"].values()]
add("rq2-models", "people-trained classifier on model choices (range) vs on people (range)", f"{min(tr):.3f}-{max(tr):.3f} vs {min(tp):.3f}-{max(tp):.3f}", "analysis/memorability_models.json")
pf = [v["family_alone"]["position"]["model_choice"] for v in Mm["models"].values()]
add("rq2-models", "position family alone: model choices (range) vs people", f"{min(pf):.3f}-{max(pf):.3f} vs {v0['family_alone']['position']['people']:.3f}", "analysis/memorability_models.json")
jj = [v["jaccard_with_person"] for v in Mm["models"].values()]
add("rq2-models", "turn overlap (Jaccard) with the person's stretch: models (range) vs partner's own memory", f"{min(jj):.2f}-{max(jj):.2f} vs {Mm['partner_jaccard_with_person']:.3f}", "analysis/memorability_models.json")
Ps = A("position_swap_result.json")
for k, v in Ps["by_version"].items(): add("rq2-swap", k, f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}] n={v[3]}", "analysis/position_swap_result.json")
for k, v in Ps["judges_swap_minus_control"].items(): add("rq2-swap", f"judges swap-control {k}", f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]", "analysis/position_swap_result.json")
Lc2 = A("late_content.json")
for who in ("models", "people"):
    for q, v in Lc2[who].items(): add("rq2-late", f"{who} late-vs-earlier shift, {q} (n late {v['late_n']})", "; ".join(f"{a} {b:+.3f}" for a, b in v["top"]), "analysis/late_content.json")
K = A("keenan_test.json"); Kn = A("keenan_report_negative.json")
for c in ("compliment", "humor", "self_disclosure", "teasing", "disagreement", "surprise"):
    if c in K: add("rq2-keenan", f"{c}: remembered stretch vs control (Luna), sign-test p", f"{K[c]['remembered_rate']:.3f} vs {K[c]['control_rate']:.3f}, p {K[c]['p_sign']:.2g}", "analysis/keenan_test.json")
for k, v in Kn.items(): add("rq2-keenan", f"report does not mention it: {k}", f"{v['remembered_rate']:.3f} vs {v['control_rate']:.3f}, p {v['p_sign']:.2g}", "analysis/keenan_report_negative.json")
G = A("gaze_memory.json"); V = A("voice_memory.json"); Gt = A("gaze_memory_train.json"); Vt = A("voice_memory_train.json")
add("rq2-gaze", "mean away-from-screen rate (self / partner)", str(G["mean_away_rate"]), "analysis/gaze_memory.json")
for k in ("partner_speaking_lenmatched:partner_away", "partner_speaking_lenmatched:self_away", "rememberer_speaking_lenmatched:self_away", "partner_speaking:self_away", "partner_speaking:partner_away"):
    v = G["features"][k]; add("rq2-gaze", k, f"{v['inside_minus_outside']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] n={v.get('n_people')}", "analysis/gaze_memory.json")
    if k in Gt["features"]: v = Gt["features"][k]; add("rq2-gaze", "TRAIN HALF " + k, f"{v['inside_minus_outside']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]", "analysis/gaze_memory_train.json")
for k, v in V["features"].items():
    if v["ci95"][0] > 0 or v["ci95"][1] < 0: add("rq2-voice", k, f"{v['inside_minus_outside']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] ({v['in_sd_units']:+.3f} SD)", "analysis/voice_memory.json")
for k, v in Vt["features"].items():
    if v["ci95"][0] > 0 or v["ci95"][1] < 0: add("rq2-voice", "TRAIN HALF " + k, f"{v['inside_minus_outside']:+.4f} ({v['in_sd_units']:+.3f} SD)", "analysis/voice_memory_train.json")
for nm, f in (("gaze", "memorability_gaze.json"), ("voice", "memorability_voice.json")):
    Mx = A(f)
    for m in ("LR", "RF"):
        fa = Mx["ablation"][m]["families"][nm]
        add("rq2-multimodal", f"{nm} family, {m}: all-features bal.acc / alone / drop without", f"{Mx['ablation'][m]['all_families']:.3f} / {fa['alone']:.3f} / {fa['drop_without']:+.3f}", "analysis/" + f)
Mo = A("modality.json")["summary"]
for k in ("audio|none|similarity", "audio|none|same_focus", "text_audio|none|similarity", "text_audio|none|same_focus", "text_audio_video|none|similarity", "text_audio_video|none|same_focus"):
    v = Mo[k]; add("rq2-multimodal", f"prediction: {k} vs transcript (mean over {v['models']} models; up/down)", f"{v['mean']:+.4f} ({v['up']} up, {v['down']} down)", "analysis/modality.json")
Fn = A("feeling_nocue_scan.json")["gain"]
for k in ("audio|similarity|feeling_vibe_no_cue", "audio|similarity|other"):
    v = Fn[k]; add("rq2-multimodal", f"audio gain {k}", f"{v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] people={v['people']}", "analysis/feeling_nocue_scan.json")

# ---------- §6 RQ3: models vs references (judges), history, controls
Mr = A("model_results.json")
add("rq3-judges", "judges", ", ".join(Mr["judges"]), "analysis/model_results.json")
add("rq3-judges", "partner's memory as a prediction: similarity / same focus / same reason / event / topic", f"{Mr['ref_partner']['similarity']:.3f} / {Mr['ref_partner']['same_focus']:.3f} / {Mr['ref_partner']['same_reason']:.3f} / {Mr['ref_partner']['event_match']:.3f} / {Mr['ref_partner']['topic_match']:.3f}", "analysis/model_results.json")
add("rq3-judges", "random same-batch memory: similarity / same focus / same reason / event / topic", f"{Mr['ref_random']['similarity']:.3f} / {Mr['ref_random']['same_focus']:.3f} / {Mr['ref_random']['same_reason']:.3f} / {Mr['ref_random']['event_match']:.3f} / {Mr['ref_random']['topic_match']:.3f}", "analysis/model_results.json")
sims = [(x["model"], x["n"], x["levels"]["none"]["similarity"]) for x in Mr["text"]]
add("rq3-judges", "models (no info): similarity range; n models; n above partner", f"{min(s for _, _, s in sims):.3f}-{max(s for _, _, s in sims):.3f}; {len(sims)}; {sum(s > Mr['ref_partner']['similarity'] for _, _, s in sims)}", "analysis/model_results.json")
add("rq3-judges", "models on all reports vs 300-sample", f"{sum(n > 1000 for _, n, _ in sims)} full, {sum(n <= 400 for _, n, _ in sims)} on ~300", "analysis/model_results.json")
for m in ("similarity", "same_focus", "same_reason", "event_match"):
    up = sum(1 for x in Mr["text"] if x.get("paired", {}).get(m, {}).get("lo", -1) > 0); dn = sum(1 for x in Mr["text"] if x.get("paired", {}).get(m, {}).get("hi", 1) < 0)
    add("rq3-history", f"history minus none, {m}: models with interval above / below zero (of {len(Mr['text'])})", f"{up} / {dn}", "analysis/model_results.json")
Ct = A("controls.json")
add("rq3-controls", "reports with an earlier call (controls set)", Ct["n_with_prior"], "analysis/controls.json")
for comp, d in Ct["comparisons"].items():
    for m in ("same_focus", "same_reason", "similarity"):
        if m in d: add("rq3-controls", f"{comp} | {m} (5 models: mean; up/down)", f"{d[m]['mean']:+.3f}; {d[m]['up']}/{d[m]['down']}", "analysis/controls.json")
for coder in L:
    for q in ["topic", "reason.first", "personal_kinds", "extent", "remembered"]:
        a = L[coder]["held_out_half"].get(q, {})
        for pred in ["model: norms note", "model: norms + history", "model: history", "model: no history"]:
            e = a.get(pred, {}).get("excess")
            if e: add("rq3-norms", f"[{coder}] held-out half, {q} | {pred}", f"{e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}]", "analysis/ladder_v351.json")
Nr = A("norms_result.json")
for coder, d in Nr["population"].items():
    for q, v in d.items(): add("rq3-norms", f"[{coder}] population gap (TVD people vs models) {q}: none -> norms note", f"{v['tvd_none']:.2f} -> {v['tvd_norms']:.2f}", "analysis/norms_result.json")
for k, v in Nr["judges"].items(): add("rq3-norms", f"judges, norms note vs none: {k}", json.dumps(v)[:120], "analysis/norms_result.json")

# ---------- §7 identity: trait-to-rating coupling (trait_steering), content coupling, fine-tuning
Ts = A("trait_steering.json"); DOMS = ["H", "E", "X", "A", "C", "O"]
for rating in ("pleasure", "arousal", "dominance", "getting_along", "unit", "timing", "words"):
    h = Ts["human"].get(rating)
    row = {"people": h["cv_r2"] if h else None}
    for cond in ("none", "card_thin", "card_narr"):
        vals = [v[rating]["cv_r2"] for k, v in Ts["models"].items() if k.endswith("|" + cond) and rating in v]
        row[cond] = float(np.median(vals)) if vals else None; row[cond + "_n"] = len(vals)
    add("rq4-coupling", f"{rating}: CV R^2 from the 6 domains (people / median over models: none, scores card, narrative card)",
        f"{row['people']:.3f} / {row['none']:.3f}, {row['card_thin']:.3f}, {row['card_narr']:.3f} (n models {row['card_narr_n']})", "analysis/trait_steering.json")
    for d in ("X", "E", "A"):
        j = DOMS.index(d); vals = [v[rating]["r"][j] for k, v in Ts["models"].items() if k.endswith("|card_narr") and rating in v]
        add("rq4-coupling", f"{rating} ~ {d}: r in people / median r in models with narrative card", f"{h['r'][j]:+.2f} / {np.median(vals):+.2f}" if h else f"- / {np.median(vals):+.2f}", "analysis/trait_steering.json")
Cc = A("card_content_coupling.json")
for coder, d in Cc.items():
    for cond in ("people", "no_card", "scores_card", "narrative_card"):
        g = [v["auc_gain"] for v in d[cond].values()]; cont = [v["auc_gain"] for k, v in d[cond].items() if not k.startswith(("feelings", "matters_for"))]
        add("rq4-content", f"[{coder}] {cond}: codes with AUC gain > .05 / median gain (all) / median gain (content only)", f"{sum(x > .05 for x in g)} of {len(g)} / {np.median(g):+.3f} / {np.median(cont):+.3f}", "analysis/card_content_coupling.json")
    for k in ("feelings=JOY", "feelings=FEAR", "matters_for=CONNECTION", "personal_kinds=THOUGHTS_INTIMATE", "topic=HEALTH_WELLBEING", "topic=HOBBIES_INTERESTS_MEDIA", "remembered=REPORTER_THOUGHT"):
        v = d["narrative_card"].get(k); n0 = d["no_card"].get(k)
        if v: add("rq4-content", f"[{coder}] narrative card AUC gain {k} (no card)", f"{v['auc_gain']:+.3f} ({n0['auc_gain']:+.3f})" if n0 else f"{v['auc_gain']:+.3f} (-)", "analysis/card_content_coupling.json")
Dp = A("dpo_results.json")
for k, v in Dp["paired"].items():
    if "|test|" in k: add("rq4-dpo", k, json.dumps(v)[:140], "analysis/dpo_results.json")
for k, v in Dp["stats"].items(): add("rq4-dpo", "stats " + k, json.dumps(v)[:160], "analysis/dpo_results.json")

# ---------- §6 RQ3: stability vs history
St = A("stability_vs_history.json")
for coder, v in St.items():
    add("rq3-history", f"[{coder}] rho(stability, right-minus-wrong history gain)", str(v["rho_stability_vs_history_minus_wrong"]), "analysis/stability_vs_history.json")

# ---------- Oct 9 night additions (A1 OK): character stability, shared memory and connection, who is predictable, human card readers
Cs = A("character_stability.json")
for coder, v in Cs.items():
    for k in ("feelings", "reason.first", "extent", "remembered", "new_info"):
        d = v[k]; f = lambda x: f"{x[0]:+.3f} [{x[1]:+.3f}, {x[2]:+.3f}]" if x else "-"
        add("rq4-stability", f"[{coder}] {k}: people / no profile / scores card / narrative profile (own earlier calls, same calls)",
            f"{f(d['people'])} / {f(d['none']['pooled'])} / {f(d['card_thin']['pooled'])} / {f(d['card_narr']['pooled'])}; per model narr {d['card_narr']['per_model']}", "analysis/character_stability.json")
_pm = {m: sum(Cs[c]["feelings"]["card_narr"]["per_model"][m] for c in Cs) / len(Cs) for m in next(iter(Cs.values()))["feelings"]["card_narr"]["per_model"]}
add("rq4-stability", "feelings, narrative profile: per-model coder mean (range); people coder mean", f"{min(_pm.values()):.4f} to {max(_pm.values()):.4f} {{{', '.join(f'{m}: {v:.4f}' for m, v in _pm.items())}}}; people {sum(Cs[c]['feelings']['people'][0] for c in Cs) / len(Cs):.4f}", "analysis/character_stability.json")
# sensitivity (A1, Oct 12): codes over the codebook's limit of three answers
O3 = A("over_three.json")
for coder, v in O3.items():
    for q, x in v.items():
        if x[0]: add("codebook-over3", f"[{coder}] {q}: codes with more than three answers / coded", f"{x[0]:,} / {x[1]:,} ({100 * x[2]:.1f}%)", "analysis/over_three.json")
_L, _Lc = A("ladder_v351.json"), A("ladder_v351_comply.json"); _d = []
for c in _L:
    for p in _L[c]:
        for k in _L[c][p]:
            for nme, v in _L[c][p][k].items():
                w = _Lc[c][p].get(k, {}).get(nme)
                for s in ("raw", "excess"):
                    if w and v.get(s) and w.get(s): _d.append(abs(v[s][0] - w[s][0]))
add("codebook-over3", "over-three codes set to missing: largest change in any ladder value / values compared", f"{max(_d):.3f} / {len(_d):,}", "analysis/ladder_v351.json vs analysis/ladder_v351_comply.json")
# other descriptions (round 2, five models): judges' whose-moment change vs no information; trait coupling
_R2 = A("round2.json")
for _c, _n in (("card_thin", "scores"), ("card_narr", "written profile"), ("card_facts", "facts list"), ("card_factnarr", "facts narrative"), ("card_weave", "woven profile (written profile + facts from all their calls, one per person)")):
    _fo = _R2["fidelity"][_c]["same_focus"]; _cp = _R2["coupling"][_c]
    add("profiles-other", f"{_n}: whose-moment change vs no information / models lower / dominance R2 / pleasure R2", f"{_fo['mean']:+.3f} / {_fo['down']} of {_fo['models']} / {_cp['dominance']:.2f} / {_cp['pleasure']:.2f}", "analysis/round2.json")
    for _m, _lab in (("same_reason", "kind of reason"), ("similarity", "similarity"), ("event_match", "exact moment")):   # Oct 12: woven-profile claim on every judged measure
        _x = _R2["fidelity"][_c][_m]; add("profiles-other", f"{_n}: {_lab} change vs no information / models lower", f"{_x['mean']:+.3f} / {_x['down']} of {_x['models']}", "analysis/round2.json")
# fine-tuning / preference-training data sizes (app_dpo), counted from the training files
_FT = os.path.join(ROOT, "finetune")
def _rows(p): return [json.loads(l) for l in open(os.path.join(_FT, p))]
_sp = {s: _rows(f"data_v1/memory_none_{s}.jsonl") for s in ("train", "val", "test")}
add("finetune-data", "reports / people per split (train; val; test)", "; ".join(f"{len(v)} / {len({r['person_key'] for r in v})}" for v in _sp.values()), "finetune/data_v1/memory_none_*.jsonl")
_ph = _rows("data_v3_dpo/pairs_history_train.jsonl"); _pc = _rows("data_v3_dpo/pairs_card_train.jsonl"); _arm = collections.Counter(r["arm"] for r in _ph)
add("finetune-data", "DPO pairs: history run / profile run; rejected = partner's answer / own other call", f"{len(_ph)} / {len(_pc)}; {_arm['A_partner']} / {_arm['B_other_call']}", "finetune/data_v3_dpo/pairs_*_train.jsonl")
# appendix counts (Oct 12): profile lengths, speaker tracks, media sample split
_w = [len(json.loads(l)["text"].split()) for l in open(os.path.join(ROOT, "cards", "run_cards1", "cards_out", "cards_lcw150.jsonl"))]
add("appendix-counts", "written profiles: n / min / max / median words", f"{len(_w)} / {min(_w)} / {max(_w)} / {int(np.median(_w))}", "cards/run_cards1/cards_out/cards_lcw150.jsonl")
_tk = collections.Counter(m.get("kind") for f in glob.glob(os.path.join(ROOT, "transcripts_v2", "*.json")) for m in json.load(open(f))["meta"].values())
add("appendix-counts", "speaker tracks: own microphone / partner's copy (remote) / total", f"{_tk['self_local']} / {_tk['remote_copy']} / {sum(_tk.values())}", "transcripts_v2/*.json meta")
_ms = A("feeling_nocue_scan.json")["counts_media_sample"]
add("appendix-counts", "media sample: feeling, no cue / feeling, cued / other", f"{_ms['feeling_vibe_no_cue']} / {_ms['feeling_vibe_cued']} / {_ms['other']}", "analysis/feeling_nocue_scan.json")
# human coders' agreement (Oct 12): scope in three classes (scripts/scope_merge.py); four-way kept for the appendix line
_HA = os.path.join(ROOT, "analysis", "code_v351", "humans", "HUMAN_AGREEMENT.json")
if os.path.exists(_HA):
    _H = json.load(open(_HA)); _sl = _H["slots"]
    add("agreement", "human coders / reports", f"{len(_H['humans'])} / {_H['n_items']}", "analysis/code_v351/humans/HUMAN_AGREEMENT.json")
    _st = {k: sorted(set(v.get("status", {}).values())) for k, v in _sl.items()}
    add("agreement", "scored answers trusted for all three AI coders / total", f"{sum(s == ['trusted'] for s in _st.values())} / {len(_st)}", "same")
    add("agreement", "not interpreted (people not above chance)", ", ".join(k for k, s in _st.items() if s == ['not interpreted']) or "none", "same")
    _ge = sum(all(v["AI_H"].get(m) and v["AI_H"][m]["alpha"]["est"] >= v["HH"]["alpha"]["est"] for m in v["AI_H"]) for v in _sl.values() if v.get("HH"))
    add("agreement", "answers where every AI coder agrees with people at least as well as people agree with each other (point estimates)", f"{_ge} of {len(_sl)}", "same")
    for k, v in _sl.items():
        if v.get("HH"):
            add("agreement", f"{k}: people alpha [CI] / AI-people alpha (Haiku, Luna, DeepSeek) / AI-AI / status",
                f"{v['HH']['alpha']['est']:.2f} [{v['HH']['alpha']['ci'][0]:.2f}, {v['HH']['alpha']['ci'][1]:.2f}] / " + ", ".join(f"{v['AI_H'][m]['alpha']['est']:.2f}" for m in v["AI_H"] if v["AI_H"].get(m)) + f" / {v['AI_AI']['alpha']['est']:.2f} / {', '.join(_st[k])}", "same")
    _H4 = os.path.join(ROOT, "analysis", "code_v351", "humans_scope4", "HUMAN_AGREEMENT.json")
    if os.path.exists(_H4):
        _e = json.load(open(_H4))["slots"]["extent"]["HH"]["alpha"]
        add("agreement", "scope as the original four answers: people alpha [CI]", f"{_e['est']:.2f} [{_e['ci'][0]:.2f}, {_e['ci'][1]:.2f}]", "analysis/code_v351/humans_scope4/HUMAN_AGREEMENT.json (SCOPE4=1)")
Sm = A("shared_memory_rapport.json")
add("disc-connection", "calls / base rates (judges)", f"{Sm['n_calls']} / {Sm['base_rates']}", "analysis/shared_memory_rapport.json")
for sec in ("within_person", "across_calls"):
    for k, v in Sm[sec].items():
        add("disc-connection", f"{sec}: {k}", f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] n={v[3]}", "analysis/shared_memory_rapport.json")
Wp = A("who_is_predictable.json")
add("rq3-who", "rho(person consistency, history gain right-minus-wrong) [CI], p", str(Wp["rho_consistency_vs_history_gain"]) + f" n={Wp['n_people']}", "analysis/who_is_predictable.json")
add("rq3-who", "history gain by consistency tercile (low/mid/high)", str(Wp["gain_by_consistency_tercile"]), "analysis/who_is_predictable.json")
add("rq3-who", "smallest q among consistency~measure tests", str(min(t[3] for t in Wp["consistency_traits"])) + " (" + min(Wp["consistency_traits"], key=lambda t: t[3])[0] + ")", "analysis/who_is_predictable.json")
Hc = A("human_card_readers.json")
add("rq3-cards", "human 24-facet readers: pooled CCC / median reader CCC / reads / profiles", f"{Hc['facet_pooled_ccc']} / {Hc['facet_median_reader_ccc']} / {len(Hc['facet_readers'])} / {len(Hc['profiles_facet'])}", "analysis/human_card_readers.json")
add("rq3-cards", "human six-domain readers: pooled CCC / reads / profiles", f"{Hc['domain_pooled_ccc']} / {len(Hc['domain_readers'])} / {len(Hc['profiles_domain'])}", "analysis/human_card_readers.json")
add("rq3-cards", "AI readers on the same people (pooled CCC)", str(Hc["ai_pooled_ccc_same_people"]), "analysis/human_card_readers.json")

# ---------- Oct 10: agent2's six-domain wave (human readers) and AI readers on the same measures
import os as _os
_cw = json.load(open(_os.path.join(ROOT, "prolific", "six_trait_wave2_20261009", "analysis_private", "EXPANDED_READBACK_RESULTS_PRIVATE.json")))
for _set in ("primary_random_sample", "clean24"):
    _d = _cw[_set]
    add("rq3-cards", f"human six-domain readers, {_set}: profiles / macro Spearman / mean gap / pop-mean baseline gap",
        f"{_d['profiles']} / {_d['macro_domain_spearman']:.3f} / {_d['mean_cell_numerical_gap']:.3f} / {np.mean([v['leave_target_out_population_mean_baseline_mae'] for v in _d['domains'].values()]):.3f}",
        "prolific/six_trait_wave2_20261009/analysis_private/EXPANDED_READBACK_RESULTS_PRIVATE.json")
    add("rq3-cards", f"human six-domain readers, {_set}: Spearman per domain", str({k: round(v['canonical']['spearman'], 2) for k, v in _d['domains'].items()}),
        "prolific/six_trait_wave2_20261009/analysis_private/EXPANDED_READBACK_RESULTS_PRIVATE.json")
_ar = A("ai_readers_rank.json")
for k, v in _ar.items():
    if isinstance(v, dict): add("rq3-cards", f"AI reader {k}: n / mean Spearman / gap / pop-mean gap", f"{v['n']} / {v['mean_rho']} / {v['gap']} / {v['pop_mean_gap']}", "analysis/ai_readers_rank.json")

# ---------- transcript ablation (MemoryMultimodal, Oct 10): recognizer agreement, listening gate, laughter
if os.path.exists(os.path.join(ROOT, "analysis", "asr_laughter.json")):
    _al = A("asr_laughter.json"); _ag = _al["agreement"]
    for k in ("scribe_vs_mai", "whisper_vs_scribe", "whisper_vs_mai", "nofill_scribe_vs_mai", "nofill_whisper_vs_scribe", "nofill_whisper_vs_mai"):
        v = _ag[k]; add("tx-ablation", f"WER {k}: median [IQR] / mean / tracks", f"{v['median']:.3f} [{v['p25']:.3f}, {v['p75']:.3f}] / {v['mean']:.3f} / {v['n']}", "analysis/asr_laughter.json")
    _p = _ag["pooled_nofill_whisper_vs_scribe"]
    add("tx-ablation", "Whisper vs Scribe, fillers removed, pooled: reference words / deleted / substituted / inserted (share of words)",
        f"{_p['ref_words']} / {_p['delete']:.4f} / {_p['replace']:.4f} / {_p['insert']:.4f}", "analysis/asr_laughter.json")
    add("tx-ablation", "fillers per 100 words, median (Whisper / Scribe / MAI)",
        " / ".join(f"{_al['fillers'][k]['median']:.2f}" for k in ("whisper", "scribe", "mai")), "analysis/asr_laughter.json")
    _l = _al["laughter"]
    add("tx-ablation", "laughter events per call (Scribe): median [IQR] / share of calls with any", f"{_l['per_call']['median']} [{_l['per_call']['p25']}, {_l['per_call']['p75']}] / {_l['calls_with_any']:.3f}", "analysis/asr_laughter.json")
    _r = _al["remembered_turns"]
    for k in ("speaker_laughs", "listener_laughs"):
        add("tx-ablation", f"remembered turn ~ {k}: OR [95% CI] (turns {_r['turns']}, memories {_r['perspectives']})", f"{_r[k]['odds_ratio']:.2f} [{_r[k]['ci'][0]:.2f}, {_r[k]['ci'][1]:.2f}]", "analysis/asr_laughter.json")
    _h = _al["humor_reports"]
    add("tx-ablation", "report mentions laughing/joke ~ log(1 + laughs in call): OR [95% CI] / report rate", f"{_h['log1p_laughs_odds_ratio']:.2f} [{_h['ci'][0]:.2f}, {_h['ci'][1]:.2f}] / {_h['rate']:.3f}", "analysis/asr_laughter.json")
if os.path.exists(os.path.join(ROOT, "analysis", "transcript_gate.json")):
    _g = A("transcript_gate.json")
    for jm, v in _g["by_judge"].items():
        add("tx-ablation", f"listening gate, {jm}: Scribe / Whisper / equal (p, decisive)", f"{v['scribe']} / {v['whisper']} / {v['equal']} (p={v['p']:.1e})", "analysis/transcript_gate.json")
    for k, v in _g["agreement"].items():
        add("tx-ablation", f"listening gate agreement {k}", f"{v['same']:.3f} of {v['n']}", "analysis/transcript_gate.json")
    for k, v in _g["by_judge_stratum"].items():
        add("tx-ablation", f"listening gate {k}: Scribe / Whisper / equal", f"{v['scribe']} / {v['whisper']} / {v['equal']}", "analysis/transcript_gate.json")
# transcriber as a variable: same models, prompts and settings, Scribe minus Whisper transcript, three judges (MemoryMultimodal, Oct 10)
if os.path.exists(os.path.join(ROOT, "analysis", "tx_v3_all.json")):
    _ta = A("tx_v3_all.json")
    for k, v in sorted(_ta["pooled"].items()):
        add("tx-fidelity", f"all 13 models, {k}: Scribe minus Whisper [95% CI] (pairs, people)", f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}] ({v[3]}, {v[4]})", "analysis/tx_v3_all.json")
    _cells = [(k, m, r["diff"]) for k, row in _ta["rows"].items() for m, r in row.items()]
    _sig = [c for c in _cells if c[2][1] > 0 or c[2][2] < 0]
    add("tx-fidelity", "model x condition x measure cells whose 95% CI excludes zero / all cells (up / down)",
        f"{len(_sig)} / {len(_cells)} ({sum(c[2][0] > 0 for c in _sig)} / {sum(c[2][0] < 0 for c in _sig)})", "analysis/tx_v3_all.json")
    _big = max(_cells, key=lambda c: abs(c[2][0]))
    add("tx-fidelity", "largest single-cell change (model|condition, measure): Scribe minus Whisper [95% CI]", f"{_big[0]}, {_big[1]}: {_big[2][0]:+.4f} [{_big[2][1]:+.4f}, {_big[2][2]:+.4f}]", "analysis/tx_v3_all.json")
for _s in ("five", "cheap2", "pro"):
    if os.path.exists(os.path.join(ROOT, "analysis", f"tx_v3_compare_{_s}.json")):
        for k, v in sorted(A(f"tx_v3_compare_{_s}.json")["pooled"].items()):
            add("tx-fidelity", f"model set {_s}, {k}: Scribe minus Whisper [95% CI] (pairs, people)", f"{v[0]:+.4f} [{v[1]:+.4f}, {v[2]:+.4f}] ({v[3]}, {v[4]})", f"analysis/tx_v3_compare_{_s}.json")
if os.path.exists(os.path.join(ROOT, "analysis", "ladder_tx_v3_summary.json")):
    for k, v in A("ladder_tx_v3_summary.json").items():
        add("tx-fidelity", f"ladder, coder|condition {k}: mean over questions of (Scribe - Whisper) excess fit / max abs / questions over .01 (of n)",
            f"{v['mean_diff']:+.4f} / {v['max_abs']:.3f} / {v['n_over_01']} (of {v['n_questions']})", "analysis/ladder_tx_v3.json")

if os.path.exists(os.path.join(ROOT, "analysis", "norms_result_tx_v3.json")):
    _nt = A("norms_result_tx_v3.json")
    for coder, d in _nt["population"].items():
        for q, v in d.items(): add("tx-fidelity", f"Scribe rerun, [{coder}] population gap (TVD people vs models) {q}: none -> norms note", f"{v['tvd_none']:.2f} -> {v['tvd_norms']:.2f}", "analysis/norms_result_tx_v3.json")
    for coder, d in _nt["person"].items():
        _b = sorted(k for k, v in d.items() if v["diff"][1] > 0); _w = sorted(k for k, v in d.items() if v["diff"][2] < 0)
        add("tx-fidelity", f"Scribe rerun, [{coder}] person-level questions better / worse with the norms note (95% CI excludes 0)", f"{len(_b)} / {len(_w)} (worse: {', '.join(_w)})", "analysis/norms_result_tx_v3.json")
    for k, v in _nt["judges"].items(): add("tx-fidelity", f"Scribe rerun, judges, norms note vs none: {k}", json.dumps(v)[:120], "analysis/norms_result_tx_v3.json")

if os.path.exists(os.path.join(ROOT, "analysis", "controls_tx_v3.json")):
    _ct = A("controls_tx_v3.json")
    for key, d in _ct["comparisons"].items():
        for m, v in d.items():
            if v and v.get("mean") is not None: add("tx-fidelity", f"Scribe rerun, controls {key}, {m}: mean over 5 models (models up / down, 95% CI)", f"{v['mean']:+.3f} ({v['up']} / {v['down']} of {v['models']})", "analysis/controls_tx_v3.json")
    for c, d in _ct["coupling_wrong"].items():
        add("tx-fidelity", f"Scribe rerun, coupling with a stranger's {c}: to the card's traits / to the person's own traits", f"{json.dumps({k: round(v, 2) for k, v in d['card_traits'].items() if v is not None})} / {json.dumps({k: round(v, 2) for k, v in d['own_traits'].items() if v is not None})}", "analysis/controls_tx_v3.json")
    for info, D in _ct["transcriber_delta"].items():
        for m, v in D.items(): add("tx-fidelity", f"transcriber effect inside condition {info}, {m}: Scribe minus Whisper, mean of 3 judges [95% CI] (n)", f"{v['mean']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] ({v['n']})", "analysis/controls_tx_v3.json")

# ---------- voice v2: own-microphone tracks, two prosody instruments, two emotion models (MemoryMultimodal voice lane, Oct 10)
if os.path.exists(os.path.join(ROOT, "analysis", "voice_memory_v2.json")):
    _v = A("voice_memory_v2.json")
    for part in ("contrast_all", "contrast_own_machine_only"):
        for k, v in _v[part].items():
            if isinstance(v, dict) and "in_sd_units" in v:
                add("rq2-voice-v2", f"{part} {k}: inside-outside [95% CI] (SD units; n people)",
                    f"{v['inside_minus_outside']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] ({v['in_sd_units']:+.3f} SD; {v['n_people']})", "analysis/voice_memory_v2.json")
    for k, v in _v["affect_button"].items():
        if isinstance(v, dict):
            add("rq2-voice-v2", f"vocal ~ affect button, {k}: within-person r [95% CI] (people / reports)", f"{v['within_person_r']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] ({v['n_people']} / {v['n_reports']})", "analysis/voice_memory_v2.json")
    for k, v in _v["agreement"].items():
        if not (isinstance(v, dict) and "raw_r_n" in v): continue
        add("rq2-voice-v2", f"instrument agreement {k}: raw r / within-speaker r (turns)", f"{v['raw_r_n'][0]:.3f} / {v['within_speaker_r_n'][0]:.3f} ({v['within_speaker_r_n'][1]})", "analysis/voice_memory_v2.json")

# ---------- gaze v2: own full-resolution camera, MediaPipe + UniGaze + PureGaze (MemoryMultimodal gaze lane, Oct 10)
if os.path.exists(os.path.join(ROOT, "analysis", "gaze_memory_v2.json")):
    _gz = A("gaze_memory_v2.json")
    for k, v in _gz["features"].items():
        if isinstance(v, dict) and "inside_minus_outside" in v:
            add("rq2-gaze-v2", f"{k}: inside-outside away rate [95% CI] (n people)", f"{v['inside_minus_outside']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] ({v.get('n_people')})", "analysis/gaze_memory_v2.json")
    add("rq2-gaze-v2", "instrument agreement (as reported)", json.dumps(_gz.get("agreement"))[:600], "analysis/gaze_memory_v2.json")

# ---------- time course around the remembered stretch, length-adjusted (descriptive; vs Tsfasman ICMI 2022 p. 102)
if os.path.exists(os.path.join(ROOT, "analysis", "timecourse_v2_lenadj.json")):
    for m, D in A("timecourse_v2_lenadj.json").items():
        for lab, v in D.items():
            add("rq2-timecourse", f"{m} at {lab} (length-adjusted): mean [95% CI] (n people)", f"{v['mean']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] ({v['n_people']})", "analysis/timecourse_v2_lenadj.json")

# ---------- classifier with v2 gaze / own-mic voice families, and the voice exclusion check (MemoryMultimodal, Oct 10)
for _f, _fam, _lab in (("memorability_gaze_v2nets.json", "gaze", "gaze v2 (UniGaze, PureGaze, both, head pose)"),
                       ("memorability_voice_ownmic.json", "voice", "voice own-mic (eGeMAPS)"),
                       ("memorability_voice_ownmic_emo.json", "voice", "voice own-mic + vocal emotion (audEERING A/D/V)")):
    if os.path.exists(os.path.join(ROOT, "analysis", _f)):
        _m = A(_f)
        for _mod in ("LR", "RF"):
            _a = _m["ablation"][_mod]; _fv = _a["families"][_fam]
            add("rq2-classifier-v2", f"{_lab}, {_mod}: alone / text+family (all) / drop when removed",
                f"{_fv['alone']:.3f} / {_a['all_families']:.3f} / {_fv['drop_without']:+.3f}", f"analysis/{_f}")
_mt = A("memorability.json")
for _mod in ("LR", "RF"):
    add("rq2-classifier-v2", f"text families only, {_mod}: all", f"{_mt['ablation'][_mod]['all_families']:.3f}", "analysis/memorability.json")
if os.path.exists(os.path.join(ROOT, "analysis", "voice_contrast_exclusions.json")):
    _vx = A("voice_contrast_exclusions.json")
    for _tag in ("none", "after3", "before3", "both3"):
        for _k, _v in _vx[_tag].items():
            if _k.split(":")[1] in ("loud", "praat_int_db", "loud_var", "praat_int_sd_db", "voiced_rate", "arousal", "dominance"):
                add("rq2-voice-exclusions", f"exclude {_tag}: {_k} inside-outside (SD units) [95% CI] (n)", f"{_v['in_sd_units']:+.3f} [{_v['ci95'][0]:+.4f}, {_v['ci95'][1]:+.4f}] ({_v['n_people']})", "analysis/voice_contrast_exclusions.json")

# ---------- write
os.makedirs(os.path.join(ROOT, "paper_notes"), exist_ok=True)
# ---- public stand-in dataset for the reproducibility packet (Oct 11; A1)
_SI = os.path.join(ROOT, "twin_public")
if os.path.exists(os.path.join(_SI, "validation.json")):
    _v = json.load(open(os.path.join(_SI, "validation.json"))); _pp = json.load(open(os.path.join(_SI, "people.json")))
    _ss = json.load(open(os.path.join(_SI, "sessions.json")))
    _ns = len(_ss["sessions"]) if isinstance(_ss, dict) and "sessions" in _ss else len(_ss)
    add("standin", "people / sessions / calls / memories", f"{len(_pp)} / {_ns} / {_v['n_calls']} / {_v['n_memories']}", "twin_public/validation.json; twin_public/people.json; twin_public/sessions.json")
    _av = [x for x in json.load(open(os.path.join(_SI, "av_check.json"))) if not x.get("leak")]
    add("standin", "AV subset kept / text-only locator says not in transcript", f"{len(_av)} / {sum(((x.get('locator') or {}).get('kind') == 'not_in_transcript') for x in _av)}", "twin_public/av_check.json")
    add("standin", "moment / reason median words (target)", f"{_v['moment_words']['standin']['p50']} ({_v['moment_words']['target']['p50']}) / {_v['reason_words']['standin']['p50']} ({_v['reason_words']['target']['p50']})", "twin_public/validation.json")
out = ["# FACTS_v2: every number the redraft uses, with its source", "Generated by scripts/build_facts.py. Do not edit by hand; re-run it.", ""]
sec = None
# ---- memory types (Oct 10 night; A1: "the memory types should go in there")
_ms = json.load(open(os.path.join(ROOT, "analysis", "memory_styles.json"))); _md = json.load(open(os.path.join(ROOT, "analysis", "memory_styles_describe.json")))
_k2 = _ms["k"].get("2") or _ms["k"].get(2)
add("memory-types", "people profiled (>= 3 reports)", _ms["n_profiled"], "analysis/memory_styles.json")
add("memory-types", "k=2 sizes", _k2["sizes"], "analysis/memory_styles.json k.2")
add("memory-types", "k=2 stability ARI / GMM agreement", f"{_k2['stability_ari']} / {_k2['gmm_agree_with_kmeans_ari']}", "analysis/memory_styles.json k.2")
add("memory-types", "k=2 silhouette / shuffled-null mean / null 95th pct", f"{_k2['silhouette']} / {_k2['null_silhouette_mean']} / {_k2['null_silhouette_95']}", "analysis/memory_styles.json k.2")   # A2 cluster question (Oct 12)
add("memory-types", "k>=3 stability ARI range", f"{min(v['stability_ari'] for k, v in _ms['k'].items() if int(k) >= 3)}-{max(v['stability_ari'] for k, v in _ms['k'].items() if int(k) >= 3)}", "analysis/memory_styles.json")
add("memory-types", "k>=3 GMM agreement max", max(v['gmm_agree_with_kmeans_ari'] for k, v in _ms['k'].items() if int(k) >= 3), "analysis/memory_styles.json")
_t = {int(k): v for k, v in _md["k2_means"].items()}; _A = max(_t, key=lambda k: _t[k]["what:did_together"]); _B = 1 - _A
for lab, i in (("stretch we shared", _A), ("what they were told", _B)):
    for f in ("what:did_together", "shared:nobody", "new_fact", "scope:one_moment", "what:partner_said_or_did"):
        add("memory-types", f"{lab}: {f}", round(_t[i][f], 2), "analysis/memory_styles_describe.json k2_means")
add("memory-types", "k=2 without word counts vs with (ARI)", _md["k2_nowords_vs_k2_ari"], "analysis/memory_styles_describe.json")
add("memory-types", "types vs traits: smallest q", min(t[4] for t in _md["traits"]), "analysis/memory_styles_describe.json")
for n in ("words_moment", "words_why", "disclosed_deep"):
    add("memory-types", f"carry ICC {n} (null 95th)", f"{_ms['carry'][n]['icc']} ({_ms['carry'][n]['null_95']})", "analysis/memory_styles.json carry")
_mm = json.load(open(os.path.join(ROOT, "analysis", "memory_styles_models.json")))
for c in ("none", "card_thin", "card_narr", "history", "norms", "norms_history"):
    add("memory-types", f"models {c}: share 'what they were told' (5 models pooled)", _mm["models"][f"{c}|ALL5"]["type_share"]["what_they_told_me"], "analysis/memory_styles_models.json")
_rl = json.load(open(os.path.join(ROOT, "analysis", "memory_styles_reportlevel.json")))
import numpy as _np
for c in ("none", "card_narr", "card_thin", "norms", "history_prior", "history", "norms_history", "history_wrong"):
    add("memory-types", f"type match rho, {c} (mean of 5 models)", round(float(_np.mean([v["own"][0] for v in _rl[c].values() if v["own"]])), 2), "analysis/memory_styles_reportlevel.json")
add("memory-types", "type match rho, history_wrong toward DONOR", round(float(_np.mean([v["donor"][0] for v in _rl["history_wrong"].values() if v.get("donor")])), 2), "analysis/memory_styles_reportlevel.json")
add("memory-types", "people ceiling (one report vs own others)", _rl["people_ceiling"][0], "analysis/memory_styles_reportlevel.json")

# ---- locator check, 150 placements (Oct 11; A1: "Of 150 placements we checked")
import collections as _cc
_lt = _cc.Counter(); _lc = _cc.defaultdict(_cc.Counter)
for _a, _k in (("human_coding/answers/LOCATOR_A1_answers_20261010.json", "analysis/locator_a1_packet_key.json"), ("human_coding/answers/LOCATOR_A1_2_answers_20261011.json", "analysis/locator_a1_packet2_key.json")):
    _A = json.load(open(os.path.join(ROOT, _a)))["answers"]; _K = json.load(open(os.path.join(ROOT, _k)))
    for _i, _v in _K.items():
        _lt[(_v["kind"], _A.get(_i))] += 1; _lc[_v["confidence"] if _v["kind"] == "in_conversation" else "absent"][_A.get(_i)] += 1
_bc = collections.Counter(v.get("locator") for v in json.load(open(os.path.join(ROOT, "human_coding", "answers", "A1_first50_v3.5_20261009.json")))["codes"].values())
add("locator-check", f"earlier author check ({_bc['RIGHT'] + _bc['PARTLY_RIGHT'] + _bc['NOT_FINDABLE'] + _bc['WRONG']}): right / partly / no match / wrong turns (blank)", f"{_bc['RIGHT']} / {_bc['PARTLY_RIGHT']} / {_bc['NOT_FINDABLE']} / {_bc['WRONG']} ({_bc[None]})", "human_coding/answers/A1_first50_v3.5_20261009.json (field 'locator', per report)")
add("locator-check", "new placements (102): yes / no / can't tell / needs a/v", " / ".join(str(_lt[("in_conversation", x)]) for x in ("yes", "no", "cant", "av")), "human_coding/answers/LOCATOR_A1*_answers")
add("locator-check", "high-confidence right", f"{_lc['high']['yes']} of {sum(_lc['high'].values())}", "same")
add("locator-check", "low-confidence right", f"{_lc['low']['yes']} of {sum(_lc['low'].values())}", "same")
add("locator-check", "judged absent (10): found in words / needs a/v / not there", f"{_lc['absent']['yes']} / {_lc['absent']['av']} / {_lc['absent']['no']}", "same")
# combined, as the paper states it (A1, Oct 12: "we checked 150 placements by hand")
_K50 = json.load(open(os.path.join(ROOT, "analysis", "A3_item_key.json"))); _B50 = json.load(open(os.path.join(ROOT, "human_coding", "answers", "A1_first50_v3.5_20261009.json")))["codes"]
_hc50 = [v.get("locator") for b, v in _B50.items() if json.load(open(os.path.join(ROOT, "analysis", "located", _K50[b] + ".json"))).get("confidence") == "high" and json.load(open(os.path.join(ROOT, "analysis", "located", _K50[b] + ".json"))).get("kind") == "in_conversation"]
_n150 = sum(_bc[x] for x in ("RIGHT", "PARTLY_RIGHT", "NOT_FINDABLE", "WRONG")) + sum(_lt.values()) - sum(_lc["absent"].values())
add("locator-check", "all hand checks: placements / right / wrong turns", f"{_n150} / {_bc['RIGHT'] + _lt[('in_conversation', 'yes')]} / {_bc['WRONG'] + _lt[('in_conversation', 'no')]}", "first-50 coding + LOCATOR_A1 packets (packet 1's judged-absent items not counted)")
add("locator-check", "all hand checks: high-confidence right", f"{sum(x == 'RIGHT' for x in _hc50) + _lc['high']['yes']} of {len(_hc50) + sum(_lc['high'].values())}", "same + analysis/located confidence")

# ---- checks without a language model (Oct 11; A1: "anything ... with just the data rather than LLMs")
_PS = os.path.join(ROOT, "analysis", "plainstats")
if os.path.exists(os.path.join(_PS, "plainstats.json")):
    _P = json.load(open(os.path.join(_PS, "plainstats.json")))["results"]; _C = json.load(open(os.path.join(_PS, "contrasts.json"))); _W = json.load(open(os.path.join(_PS, "within.json")))
    _f = lambda v: f"{v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}] n={v[3]}" if v else "--"
    for _r in ("partner's memory", "person's own earlier reports", "model: none", "model: history", "model: history_prior", "model: history_wrong", "model: card_narr", "model: card_thin", "model: norms", "model: norms_history"):
        add("check-plain", f"FORM (ten counted features), beats session-mates: {_r}", _f(_P["form (all ten)"][_r]["beats"]), "analysis/plainstats/plainstats.json")
    for _k in ("history minus history_wrong", "history minus none", "history_prior minus none", "card_narr minus none", "blind model minus partner"):
        add("check-plain", f"FORM paired: {_k}", _f(_C["form (all ten)"].get(_k)), "analysis/plainstats/contrasts.json")
    for _m in [k for k in _P if k.startswith("subject")]:
        for _r in ("partner's memory", "person's own earlier reports", "model: none", "model: history", "model: history_wrong"):
            add("check-plain", f"{_m}, beats session-mates: {_r}", _f(_P[_m][_r]["beats"]), "analysis/plainstats/plainstats.json")
        add("check-plain", f"{_m}, paired blind model minus partner", _f(_C[_m].get("blind model minus partner")), "analysis/plainstats/contrasts.json")
        if _m in _W:
            for _r in ("partner's memory", "model: none", "model: history", "history minus history_wrong (paired)", "history minus none (paired)"):
                add("check-plain", f"{_m}, WITHIN-PERSON (vs own other memories): {_r}", _f(_W[_m].get(_r)), "analysis/plainstats/within.json")
    _Pos = json.load(open(os.path.join(_PS, "position.json")))
    _lv = _Pos["locator_vs_self_report"]
    add("check-position", "locator position vs person's own timing answer: Spearman (n); high / medium / low confidence", f"{_lv['spearman']} (n={_lv['n']}); " + " / ".join(f"{_lv['by_confidence'][c][0]} (n={_lv['by_confidence'][c][1]})" for c in ("high", "medium", "low")) + f"; median gap {_lv['median_abs_diff']}", "analysis/plainstats/position.json")
    for _ms, _R in _Pos["located"].items():
        for _r, _v in _R.items(): add("check-position", f"{_ms}: {_r}", _f(_v), "analysis/plainstats/position.json")
    add("check-position", "'general' memories: people / blind models", f"{_Pos['general_share']['people']} / {_Pos['general_share']['models_none']}", "analysis/plainstats/position.json")
    _ST = json.load(open(os.path.join(_PS, "span_stranger_test.json")))
    for _g in ("history guess", "stranger-history guess"):
        add("check-position", f"span regression, {_g}: slope on person's habit / on stranger's habit (blind guess held fixed)", str(_ST[_g]["on person's habit"]) + " / " + str(_ST[_g]["on stranger's habit"]) + f" (rows {_ST['n_rows']}, people {_ST['n_people']})", "analysis/plainstats/span_stranger_test.json")
    _S = json.load(open(os.path.join(_PS, "span_person_test.json")))
    add("check-position", "span habit test: history guess on person's habit (controlling call) / blind guess / difference; actual span on habit", f"{_S['history_guess_on_habit_controlling_call']} / {_S['blind_guess_on_habit']} / {_S['difference']}; {_S['person_actual_span_on_habit']} (pairs {_S['n_pairs']}, people {_S['n_people']})", "analysis/plainstats/span_person_test.json")
# ---- three locators (Oct 11)
_LA = os.path.join(ROOT, "analysis", "locator_agreement.json")
if os.path.exists(_LA):
    _la = json.load(open(_LA))
    for _nm, _v in _la.items():
        if _nm == "vs_a1_accepted":
            for _l, _c in _v.items(): add("check-locators", f"{_l} vs A1-accepted placements", ", ".join(f"{k} {n}" for k, n in _c.items()), "analysis/locator_agreement.json")
            continue
        add("check-locators", f"{_nm}: n / all three same kind", f"{_v['n']} / {_v['all_three_same_kind']}", "analysis/locator_agreement.json")
        for _p, _x in _v["pairs"].items(): add("check-locators", f"{_nm}: {_p} kind agree / any shared turn / Jaccard / median midpoint gap", f"{_x['kind_agree']} / {_x['overlap_any']} / {_x['jaccard']} / {_x['median_mid_gap']} (both placed {_x['both_placed']})", "analysis/locator_agreement.json")
        for _l, _x in _v.get("vs_self_timing", {}).items(): add("check-locators", f"{_nm}: {_l}", str(_x), "analysis/locator_agreement.json")
# A1's disputed-placement packet (Oct 11): 79 memories where the three locators disagree; scored by scripts/score_locator_disagree.py logic
_dz = os.path.join(ROOT, 'analysis', 'locator_disagree_scored_v1only.json')
if os.path.exists(_dz):
    _d = json.load(open(_dz))
    add('check-locators', 'disputed: items / judged (cant dropped) / any of three right', f"{_d['n_items']} / {_d['judged']} / {_d['any']}", 'analysis/locator_disagree_scored_v1only.json')
    add('check-locators', 'disputed: right of judged (Sonnet / Gemini / Luna)', f"{_d['right']['sonnet_v1']} / {_d['right']['gemini_v1']} / {_d['right']['luna_v1']}", 'analysis/locator_disagree_scored_v1only.json')
_SR = os.path.join(ROOT, "analysis", "scribe_recheck.json")
if os.path.exists(_SR):
    for _nm, _v in json.load(open(_SR)).items(): add("check-locators", f"Scribe re-check, {_nm}: n / all three same kind / fully agree", f"{_v['n']} / {_v['all_three_same_kind']} / {_v['all_three_agree_fully']}", "analysis/scribe_recheck.json")
# ---- lay raters on the synthetic twin (Prolific, Oct 11)
_PT = os.path.join(ROOT, "analysis", "prolific_twin")
if os.path.exists(os.path.join(_PT, "F_form.json")):
    _F = json.load(open(os.path.join(_PT, "F_form.json"))); _D = json.load(open(os.path.join(_PT, "D_depth.json"))); _J = json.load(open(os.path.join(_PT, "J_judge.json"))); _Kn = json.load(open(os.path.join(_PT, "K_keenan.json"))); _Y = json.load(open(os.path.join(_PT, "Y_types.json")))
    for _q in ("extent", "new_info"):
        add("twin-prolific", f"F {_q}: rater-rater kappa / Haiku-rater / Luna-rater", f"{_F[_q]['rater_rater_kappa']} / {_F[_q]['claude-haiku-5.5_rater_kappa']} / {_F[_q]['gpt-6-luna_rater_kappa']} (items {_F[_q]['items']})", "analysis/prolific_twin/F_form.json")
    add("twin-prolific", "F remembered (multi): rater-rater Jaccard / Haiku / Luna", f"{_F['remembered']['rater_rater_jaccard']} / {_F['remembered']['claude-haiku-5.5_rater_jaccard']} / {_F['remembered']['gpt-6-luna_rater_jaccard']}", "analysis/prolific_twin/F_form.json")
    _da = _D["agreement"]; add("twin-prolific", "D depth: rater-rater kappa / Haiku / Luna / Gemini", f"{_da['rater_rater_kappa']} / {_da['claude-haiku-5.5_rater_kappa']} / {_da['gpt-6-luna_rater_kappa']} / {_da['gemini-3-flash-preview_rater_kappa']} (items {_da['items']})", "analysis/prolific_twin/D_depth.json")
    add("twin-prolific", "D deep rate: raters / Haiku / Luna / Gemini", " / ".join(str(_D["rates"][k]["deep"]) for k in ("raters", "claude-haiku-5.5", "gpt-6-luna", "gemini-3-flash-preview")), "analysis/prolific_twin/D_depth.json")
    for _q in ("topic", "event", "focus", "reason"):
        _v = _J["model"][_q]; add("twin-prolific", f"J {_q} (model pairs): rater-rater kappa / Luna / DeepSeek / Haiku", f"{_v['rater_rater_kappa']} / {_v['gpt-6-luna_rater_kappa']} / {_v['deepseek-v4.1-flash_rater_kappa']} / {_v['claude-haiku-5.5_rater_kappa']} (items {_v['items']})", "analysis/prolific_twin/J_judge.json")
    _v = _J["model"]["close"]; add("twin-prolific", "J closeness Spearman: rater-rater / Luna / DeepSeek / Haiku", f"{_v['rater_rater_spearman']} / {_v['gpt-6-luna_rater_spearman']} / {_v['deepseek-v4.1-flash_rater_spearman']} / {_v['claude-haiku-5.5_rater_spearman']}", "analysis/prolific_twin/J_judge.json")
    for _q in ("compliment", "surprise", "humor"):
        _v = _Kn["random"][_q]; add("twin-prolific", f"K {_q} (random stratum): rater-rater kappa / Luna / DeepSeek", f"{_v['rater_rater_kappa']} / {_v['gpt-6-luna_rater_kappa']} / {_v['deepseek-v4.1-flash_rater_kappa']} (items {_v['items']})", "analysis/prolific_twin/K_keenan.json")
    add("twin-prolific", "Y types: picks match coded type / unsure / majority matches / alpha", f"{_Y['match_coded_type']} / {_Y['unsure_share']} / {_Y['majority_matches_coded_type']} / {_Y['rater_alpha']} (items {_Y['items']})", "analysis/prolific_twin/Y_types.json")
    _TV = json.load(open(os.path.join(ROOT, "analysis", "twin_validate_twin_final_fixed2.json")))
    _pts = [abs(v["gap"]) for v in _TV["features"].values() if v["unit"] == "points"]
    add("twin-prolific", "twin: invented people / reports / sessions", f"{_TV['n_twins']} / {_TV['n_twin_reports']} / {len(json.load(open(os.path.join(ROOT, 'twin', 'sessions.json')))['sessions'])}", "analysis/twin_validate_twin_final_fixed2.json; twin/sessions.json")
    add("twin-prolific", "twin vs people: mean |gap| over share features (n)", f"{sum(_pts) / len(_pts):.3f} ({len(_pts)})", "analysis/twin_validate_twin_final_fixed2.json")
    add("twin-prolific", "twin vs people: largest gaps", "; ".join(f"{k} {v['gap']:+.3f} {v['unit']}" for k, v in sorted(_TV["features"].items(), key=lambda kv: -abs(kv[1]["gap"]))[:6]), "analysis/twin_validate_twin_final_fixed2.json")
    add("twin-prolific", "twin types (stretch / told) vs people; twin matches its template's type", f"{_TV['types']['twin_share']['stretch_we_shared']} / {_TV['types']['twin_share']['what_they_told_me']} vs {_TV['types']['people_share']['stretch_we_shared']} / {_TV['types']['people_share']['what_they_told_me']}; {_TV['types']['twin_matches_template_type']['acc']} (n={_TV['types']['twin_matches_template_type']['n']})", "analysis/twin_validate_twin_final_fixed2.json")
# ---- disclosure depth in the conversation (scheme v2, Oct 10)
_DD = os.path.join(ROOT, "analysis", "disclosure_depth_v2.json")
if os.path.exists(_DD):
    _d = json.load(open(_DD))
    add("disclosure-depth", "person-calls / calls", f"{_d['n_person_calls']} / {_d['n_calls']}", "analysis/disclosure_depth_v2.json")
    add("disclosure-depth", "coder agreement: deep Spearman / any-deep agree", f"{_d['coder_agreement']['deep_spearman']} / {_d['coder_agreement']['any_deep_agree']}", "analysis/disclosure_depth_v2.json")
    add("disclosure-depth", "deep disclosures per person-call; share of calls where someone goes deep (both coders)", f"{_d['describe']['deep_per_person_call']}; {_d['describe']['share_calls_someone_deep_both_coders']}", "analysis/disclosure_depth_v2.json")
    add("disclosure-depth", "carry (deep): ICC vs shuffled 95th pct", f"{_d['carry']['deep']['icc']} vs {_d['carry']['deep']['null_95']}", "analysis/disclosure_depth_v2.json")
    add("disclosure-depth", "own other calls rho / partner's other calls rho [CI]", f"{_d['whose']['own_other_calls_rho']} / {_d['whose']['partner_other_calls_rho']} {_d['whose']['partner_ci']}", "analysis/disclosure_depth_v2.json")
    add("disclosure-depth", "reciprocity in the same call rho vs null 95th", f"{_d['reciprocity']['rho_same_call']} vs {_d['reciprocity']['null_95']}", "analysis/disclosure_depth_v2.json")
    for _t in _d["who"]["tests"]:
        if _t[4] < .1: add("disclosure-depth", f"trait link {_t[0]} ~ {_t[1]}: standardized OLS beta (11 predictors), p, BH q over 22", f"{_t[2]}, {_t[3]}, {_t[4]}", "analysis/disclosure_depth_v2.json")
    for _k, _v in _d["connection_link"].items(): add("disclosure-depth", f"deep disclosure ~ {_k}: rho (p)", f"{_v['rho']} ({_v['p']:.2g})", "analysis/disclosure_depth_v2.json")
    for _k, _v in _d["memory_link"].items(): add("disclosure-depth", f"memory link {_k}: rho (p)", f"{_v['rho']} ({_v['p']:.2g})", "analysis/disclosure_depth_v2.json")
    for _c in ("claude-haiku-5.5", "gpt-6-luna", "gemini-3-flash-preview"):
        _x = json.load(open(os.path.join(ROOT, "analysis", f"disclosure_depth_v2_{_c}.json")))
        add("disclosure-depth", f"[{_c} alone] carry ICC / reciprocity / social anxiety rho", f"{_x['carry']['deep']['icc']} / {_x['reciprocity']['rho_same_call']} / " + str(next((t[2] for t in _x['who']['tests'] if t[0] == 'deep' and t[1] == 'social_anxiety'), None)), f"analysis/disclosure_depth_v2_{_c}.json")

# twin realism check by two authors (Oct 12): who wrote it, real vs synthetic (twin v2c), fresh items per author
def _ww(af, kf):
    a = json.load(open(os.path.join(ROOT, "human_coding", "answers", af))); a = a.get("answers", a)
    k = json.load(open(os.path.join(ROOT, "analysis", kf)))
    t = {i: ("person" if v["src"] == "real" else "ai") for i, v in k.items()}
    d = [i for i in t if a.get(i) in ("person", "ai")]; c = sum(a[i] == t[i] for i in d)
    return len(t), len(d), c
for _lab, _af, _kf in [("author A (team packet)", "WHO_WROTE_TEAM_A3_20261012.json", "who_wrote_team_packet_key.json"),
                       ("author B (fresh packet)", "WHO_WROTE_A13_20261012.json", "who_wrote_A13_packet_key.json"),
                       ("author C (team packet)", "WHO_WROTE_TEAM_A4_20261012.json", "who_wrote_team_packet_key.json")] + [
                       (f"author {chr(68 + _i)} (team packet)", os.path.basename(_f), "who_wrote_team_packet_key.json") for _i, _f in enumerate(sorted(
                        f for f in glob.glob(os.path.join(ROOT, "human_coding", "answers", "WHO_WROTE_TEAM_*.json")) if not re.search(r"_(A3|A4)_", f)))]:   # later coders (e.g. A5) drop in
    _n, _d, _c = _ww(_af, _kf)
    add("twin-authors", f"{_lab}: items / classified / correct / share correct", f"{_n} / {_d} / {_c} / {_c/_d:.3f}", f"human_coding/answers/{_af}")

# human check of the LLM judges on real pairs (Oct 12): two researchers, 25-pair packet; R15 struck for everyone (one researcher knew its source)
def _judge_humans():
    import itertools
    from scipy.stats import spearmanr
    from sklearn.metrics import cohen_kappa_score as _K
    H = {"A": json.load(open(os.path.join(ROOT, "human_coding", "answers", "JUDGE_TEAM_A3_20261012.json")))["answers"],
         "B": json.load(open(os.path.join(ROOT, "human_coding", "answers", "JUDGE_A1_v2_20261012.json")))["answers"],
         "C": json.load(open(os.path.join(ROOT, "human_coding", "answers", "JUDGE_TEAM_A2_20261012.json")))["answers"],
         "D": json.load(open(os.path.join(ROOT, "human_coding", "answers", "JUDGE_TEAM_A4_20261012.json")))["answers"]}
    for _i, _f in enumerate(sorted(f for f in glob.glob(os.path.join(ROOT, "human_coding", "answers", "JUDGE_TEAM_*.json")) if not re.search(r"_(A3|A2|A4)_", f))):
        H[chr(69 + _i)] = json.load(open(_f))["answers"]   # later coders (e.g. A5) drop in
    rk = json.load(open(os.path.join(ROOT, "analysis", "judge_real_packet_key.json"))); JJ = json.load(open(os.path.join(ROOT, "analysis", "judge_packet_key.json")))
    js = ["openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash", "anthropic/claude-haiku-5.5"]; items = [r for r in sorted(rk) if r != "R15"]
    rng = lambda xs: f"{min(xs):.2f}..{max(xs):.2f}"
    for hq, jq in [("topic", "topic_match"), ("event", "event_match"), ("focus", "same_focus"), ("reason", "same_reason")]:
        hv = {h: [int(H[h][r][hq]) for r in items] for h in H}; jv = {j: [int(JJ[rk[r]]["judges"][j][jq]) for r in items] for j in js}
        hj = [_K(hv[h], jv[j]) for h in H for j in js]; jj = [_K(jv[a], jv[b]) for a, b in itertools.combinations(js, 2)]
        add("judge-humans", f"{hq}: kappa researcher-judge / researcher-researcher / judge-judge (n pairs; {len(H)} researchers)", f"{rng(hj)} / {rng([_K(hv[a], hv[b]) for a, b in itertools.combinations(H, 2)])} / {rng(jj)} ({len(items)})", "human_coding/answers/JUDGE_*_20261012.json; analysis/judge_packet_key.json")
    hs = {h: [(H[h][r]["close"] / 10 if H[h][r].get("close") not in (None, "") else None) for r in items] for h in H}; jv = {j: [JJ[rk[r]]["judges"][j]["similarity"] for r in items] for j in js}
    def _sp(x, y):   # pairwise complete (one closeness rating was left blank)
        k = [i for i in range(len(x)) if x[i] is not None and y[i] is not None]; return spearmanr([x[i] for i in k], [y[i] for i in k])[0]
    hj = [_sp(hs[h], jv[j]) for h in H for j in js]; jj = [_sp(jv[a], jv[b]) for a, b in itertools.combinations(js, 2)]
    nb = sum(v is None for h in H for v in hs[h])
    add("judge-humans", f"similarity: Spearman researcher-judge / researcher-researcher / judge-judge (n pairs; {len(H)} researchers; blank ratings skipped pairwise)", f"{rng(hj)} / {rng([_sp(hs[a], hs[b]) for a, b in itertools.combinations(H, 2)])} / {rng(jj)} ({len(items)}; {nb} blank)", "human_coding/answers/JUDGE_*_20261012.json")
_judge_humans()

def _people_predict():   # Oct 12: two authors predicting 12 reports each (people_predict_v351.py; judge.py --run human_predict)
    import json as _j, collections as _c
    d = _j.load(open(os.path.join(ROOT, "analysis", "people_predict_v351.json"))); C = list(d)
    Q = ["topic", "extent", "remembered", "new_info", "details", "moment_text", "reason.first", "personal_kinds", "feelings"]
    def m(name, q): return sum(d[c]["q"][q][name]["excess"][0] for c in C) / len(C)
    for q in Q:
        add("people-predict", f"{q}: person-specific fit, people / LLMs no info / LLMs history (same 18 reports; mean of 2 annotators; both CIs exclude 0?)",
            f"{m('people', q):+.3f} / {m('model: no history', q):+.3f} / {m('model: history', q):+.3f} ({all(d[c]['q'][q]['people']['excess'][1] > 0 or d[c]['q'][q]['people']['excess'][2] < 0 for c in C)})", "analysis/people_predict_v351.json")
    add("people-predict", "questions where people beat LLMs with no info", f"{sum(m('people', q) > m('model: no history', q) for q in Q)} of {len(Q)}", "analysis/people_predict_v351.json")
    J = ["openai/gpt-6-luna", "deepseek/deepseek-v4.1-flash", "anthropic/claude-haiku-5.5"]
    H = {}
    for l in open(os.path.join(ROOT, "runs", "human_predict", "judged.jsonl")):
        x = _j.loads(l)
        if x["scores"] and x["judge"] in J: H[(x["id"], x["judge"])] = x
    pids = {k[0].split("|")[0] for k in H}; g = _c.defaultdict(list)
    for (i, j), x in H.items():
        if x["kind"] == "model": g[i.split("|")[1]].append(float(x["scores"]["event_match"]))
    for l in open(os.path.join(ROOT, "runs", "main_text_cheap", "judged.jsonl")):
        x = _j.loads(l)
        if x["scores"] and x["judge"] in J and x["kind"] == "model":
            p = x["id"].split("|")
            if p[0] in pids and p[3] == "none": g["LLMs, no info"].append(float(x["scores"]["event_match"]))
    _pool = [x for k, v in g.items() if k.startswith("human/") for x in v]
    # Oct 13 (agent2): the per-report judgment files stay private; this aggregate is what ships
    _agg = {"what": "same-moment rate (mean of 3 judges' event_match) on the 18 reports the authors predicted", "authors pooled": {"mean": round(sum(_pool) / len(_pool), 4), "n_judgments": len(_pool)}}
    for k, v in sorted(g.items()):
        if not k.startswith("human/"): _agg[k] = {"mean": round(sum(v) / len(v), 4), "n_judgments": len(v)}   # per-author rates stay private (A1, Oct 13)
    _j.dump(_agg, open(os.path.join(ROOT, "analysis", "people_predict_judged.json"), "w"), indent=1)
    add("people-predict", "same moment (3 judges), authors pooled, 18 predicted reports", f"{sum(_pool) / len(_pool):.2f} (n judgments {len(_pool)})", "analysis/people_predict_judged.json")
    for k, v in sorted(g.items()):
        add("people-predict", f"same moment (3 judges), {k}, 18 predicted reports", f"{sum(v) / len(v):.2f} (n judgments {len(v)})", "withheld: per-author rate" if k.startswith("human/") else "analysis/people_predict_judged.json")
    n = sum(1 for _ in open(os.path.join(ROOT, "runs", "human_predict", "preds.jsonl")))
    add("people-predict", "predictions / reports / per person / shared", f"{n} / {len(pids)} / 12 / 6", "runs/human_predict/preds.jsonl; analysis/human_predict_key.json")
_people_predict()

def _A3_notes():   # Oct 13 (A3's review): numbers added to the paper
    import json as _j, collections as _c
    mr = _j.load(open(os.path.join(ROOT, "analysis", "model_results.json")))
    add("A3-notes", "partners share the topic (3 judges): partner / random group member", f"{mr['ref_partner']['topic_match']:.3f} / {mr['ref_random']['topic_match']:.3f}", "analysis/model_results.json")
    sb = _j.load(open(os.path.join(ROOT, "analysis", "session_bootstrap.json")))
    for c, d in sb.items():
        for m, r in d.items():
            add("A3-notes", f"session vs person bootstrap: {c}, {m}: up/down of n (person; session)", f"{r['person']['up']}/{r['person']['down']} of {r['person']['models']}; {r['session']['up']}/{r['session']['down']} of {r['session']['models']} (mean width {r['person']['mean_width']}; {r['session']['mean_width']})", "analysis/session_bootstrap.json")
    k = _j.load(open(os.path.join(ROOT, "analysis", "judge_real_packet_key.json"))); J = _j.load(open(os.path.join(ROOT, "analysis", "judge_packet_key.json")))
    it = [J[v] for r, v in sorted(k.items()) if r != "R15"]
    add("A3-notes", "judge-check pairs: n / people / models / none / note on answers", f"{len(it)} / {len({x['pid'].split('+')[0] for x in it})} / {len({x['model'] for x in it})} / {sum(x['cond'] == 'none' for x in it)} / {sum(x['cond'] == 'norms' for x in it)}", "analysis/judge_*packet_key.json")
    add("A3-notes", "locator hand checks, the 16 neither right nor wrong: partly right / unsettled (no match + can't tell + needs a/v)", "3 / 13 (4 + 6 + 3)", "locator-check rows above")
_A3_notes()

for s, l, v, src in F:
    if s != sec: out += ["", f"## {s}", "| fact | value | source |", "|---|---|---|"]; sec = s
    out.append(f"| {l} | {v} | `{src}` |")
open(os.path.join(ROOT, "paper_notes", "FACTS_v2.md"), "w").write("\n".join(out) + "\n")
json.dump(F, open(os.path.join(ROOT, "paper_notes", "FACTS_v2.json"), "w"), indent=1)
print(len(F), "facts ->", "paper_notes/FACTS_v2.md")

