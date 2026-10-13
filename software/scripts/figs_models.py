#!/usr/bin/env python3
"""Model-side figures + LaTeX table. Primary judges: GPT-6-Luna + DeepSeek v4.1 Flash + Claude Haiku 5.5 (GLM fills DeepSeek's slot for DeepSeek predictions).
Writes paper/figs/fig_history.pdf, fig_gradient.pdf, fig_timing.pdf and paper/tab_main.tex; numbers to analysis/model_results.json."""
import collections, json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "paper", "figs")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
JUDGES = ["gpt-6-luna", "deepseek-v4.1-flash", "claude-haiku-5.5"]   # Oct 10 (A1): Haiku joins so judging matches coding
MET = ["topic_match", "event_match", "similarity", "same_focus", "same_reason"]
NICE = {"deepseek-v4.1-flash": "DeepSeek v4.1 Flash", "gemini-3.8-flash": "Gemini 3.8 Flash", "gemma-4-31b-it": "Gemma 4 31B", "glm-5.3-flashx": "GLM-5.3 FlashX",
        "kimi-k2-0905": "Kimi K2", "mimo-v2.5": "MiMo v2.5", "mimo-v2.5-pro": "MiMo v2.5 Pro", "mimo-v2.6-flash": "MiMo v2.6 Flash", "muse-glimmer-30b": "Muse Glimmer 30B",
        "qwen3.8-omni-flash": "Qwen3.8 Omni Flash", "gemini-3.1-pro-preview": "Gemini 3.1 Pro$^\\dagger$", "gpt-5.6-sol": "GPT-5.6 Sol$^\\dagger$",
        "mimo-v2.6-pro-ultraspeed": "MiMo v2.6 Pro$^\\dagger$"}


def load(run):
    S = collections.defaultdict(dict); refs = collections.defaultdict(list)
    p = os.path.join(ROOT, "runs", run, "judged.jsonl")
    if not os.path.exists(p):
        return S, refs
    for l in open(p):
        j = json.loads(l)
        if not j["scores"]:
            continue
        jd = j["judge"].split("/")[1]
        if jd not in JUDGES:
            continue
        if j["kind"] != "model":
            refs[(j["kind"], jd)].append(j["scores"]); continue
        pid, model, mod, info, frame, rep = j["id"].split("|")
        S[(model.split("/")[1], mod, info, pid)][jd] = j["scores"]
    return S, refs


def num(x):
    try: return float(x)
    except Exception: return None


def avg2(d, m):  # mean of the three judges for one item, if all present and valid (name kept for callers)
    v = [num(d[j].get(m)) for j in JUDGES if j in d]
    v = [x for x in v if x is not None]
    return np.mean(v) if len(v) == len(JUDGES) else None


def cboot(d, pids, rng, B=2000):
    """Oct 8 (A2/agent2 review; A1: "Reanalysis is easy"): bootstrap over PEOPLE (the rememberer, selfPID = the part of the
    perspective ID before '+'), not over reports, since most people contribute four or five reports. Returns (lo, hi) of the mean."""
    import collections
    g = collections.defaultdict(list)
    for i, p in enumerate(pids):
        g[str(p).split("+")[0]].append(i)
    G = list(g.values()); d = np.asarray(d, float)
    gs = np.array([d[ix].sum() for ix in G]); gn = np.array([len(ix) for ix in G])
    bs = []
    for _ in range(B):
        k = rng.integers(0, len(G), len(G)); bs.append(gs[k].sum() / gn[k].sum())
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired(S, model, mod, a, b, m, rng):
    pids = sorted({k[3] for k in S if k[0] == model and k[1] == mod})
    d = []; who = []
    for pid in pids:
        x, y = S.get((model, mod, a, pid)), S.get((model, mod, b, pid))
        if x and y:
            u, v = avg2(x, m), avg2(y, m)
            if u is not None and v is not None:
                d.append(v - u); who.append(pid)
    d = np.array(d)
    if len(d) < 20:
        return None
    lo, hi = cboot(d, who, rng)
    return dict(mean=float(d.mean()), lo=lo, hi=hi, n=int(len(d)))


SAMPLE = ("gemini-3.1-pro-preview", "gpt-5.6-sol", "mimo-v2.6-pro-ultraspeed")   # 300-report sample models


def level(S, model, mod, info, m):
    # Oct 13 (lit room): for the sample models, 'none' also holds the 59 feeling memories run for the media check (no history);
    # restrict both columns to the reports that have a history prediction, so none and +history share the same 300
    keep = {k[3] for k in S if k[0] == model and k[1] == mod and k[2] == "history"} if model in SAMPLE else None
    v = [avg2(S[k], m) for k in S if k[0] == model and k[1] == mod and k[2] == info and (keep is None or k[3] in keep)]
    v = [x for x in v if x is not None]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def main():
    rng = np.random.default_rng(0); res = {"judges": JUDGES}
    S = {}; R = collections.defaultdict(list)
    for run in ["main_text_cheap", "main_text_pro"]:
        s, r = load(run); S.update(s)
        for k, v in r.items(): R[k] += v
    models = sorted({k[0] for k in S}, key=lambda m: (m in ("gemini-3.1-pro-preview", "gpt-5.6-sol", "mimo-v2.6-pro-ultraspeed"), m))
    def refmean(kind, m):
        per = []
        for j in JUDGES:
            v = [num(x.get(m)) for x in R[(kind, j)]]; v = [x for x in v if x is not None]
            if v: per.append(np.mean(v))
        return float(np.mean(per)) if len(per) == len(JUDGES) else None
    ref = {m: refmean("ref_partner", m) for m in MET}; rnd = {m: refmean("ref_random", m) for m in MET}
    res["ref_partner"] = ref; res["ref_random"] = rnd
    rows = []
    for mdl in models:
        lv = {info: {m: level(S, mdl, "text", info, m)[0] for m in MET} for info in ("none", "history")}
        pr = {m: paired(S, mdl, "text", "none", "history", m, rng) for m in ("similarity", "same_focus", "same_reason", "event_match")}
        n = level(S, mdl, "text", "none", "similarity")[1]
        rows.append(dict(model=mdl, n=n, levels=lv, paired=pr))
    res["text"] = rows
    # table
    def f(x): return "--" if x is None else f"{x:.2f}"
    def star(p): return "" if (p is None or (p["lo"] <= 0 <= p["hi"])) else "$^*$"
    L = ["\\begin{table*}[t]\\centering\\small", "\\begin{tabular}{lrcccccc}", "\\toprule",
         "& & \\multicolumn{2}{c}{Similarity} & \\multicolumn{2}{c}{Same focus} & \\multicolumn{2}{c}{Same reason}\\\\",
         "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}", "Model & $n$ & none & +history & none & +history & none & +history\\\\", "\\midrule"]
    for r in rows:
        p = r["paired"]
        L.append(f"{NICE.get(r['model'], r['model'])} & {r['n']} & {f(r['levels']['none']['similarity'])} & {f(r['levels']['history']['similarity'])}{star(p['similarity'])} & "
                 f"{f(r['levels']['none']['same_focus'])} & {f(r['levels']['history']['same_focus'])}{star(p['same_focus'])} & "
                 f"{f(r['levels']['none']['same_reason'])} & {f(r['levels']['history']['same_reason'])}{star(p['same_reason'])}\\\\")
    L += ["\\midrule", f"Partner's own memory & & \\multicolumn{{2}}{{c}}{{{f(ref.get('similarity'))}}} & \\multicolumn{{2}}{{c}}{{{f(ref.get('same_focus'))}}} & \\multicolumn{{2}}{{c}}{{{f(ref.get('same_reason'))}}}\\\\",
          f"Random same-batch memory & & \\multicolumn{{2}}{{c}}{{{f(rnd.get('similarity'))}}} & \\multicolumn{{2}}{{c}}{{{f(rnd.get('same_focus'))}}} & \\multicolumn{{2}}{{c}}{{{f(rnd.get('same_reason'))}}}\\\\",
          "\\bottomrule", "\\end{tabular}",
          "\\caption{Predicting each person's memory from the transcript, with no information about the person (none) or with their memories from their other calls (+history). Scores are the mean of three judges (GPT-6-Luna, DeepSeek v4.1 Flash, Claude Haiku 5.5; GLM-5.3 FlashX fills DeepSeek's place on DeepSeek's own predictions). $^*$: paired history$-$none 95\\% bootstrap interval (resampling people) excludes zero. $^\\dagger$: 300-memory stratified sample.}",
          "\\label{tab:main}", "\\end{table*}"]
    open(os.path.join(ROOT, "paper", "tab_main.tex"), "w").write("\n".join(L))
    # history effect figure
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    y = np.arange(len(rows))[::-1]
    for off, m, c, lab in ((0.15, "same_focus", "#1b6e7a", "Same focus"), (-0.15, "similarity", "#d98e32", "Similarity")):
        mu = [r["paired"][m]["mean"] if r["paired"][m] else np.nan for r in rows]
        lo = [r["paired"][m]["lo"] if r["paired"][m] else np.nan for r in rows]; hi = [r["paired"][m]["hi"] if r["paired"][m] else np.nan for r in rows]
        ax.errorbar(mu, y + off, xerr=[np.array(mu) - np.array(lo), np.array(hi) - np.array(mu)], fmt="o", ms=3, color=c, lw=0.8, label=lab)
    ax.axvline(0, color="#888", lw=0.6); ax.set_yticks(y); ax.set_yticklabels([NICE.get(r["model"], r["model"]).replace("$^\\dagger$", "†") for r in rows])
    ax.set_xlabel("Gain from history (paired, 95% CI)"); ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_history.pdf")); plt.close(fig)
    # gradient
    G, _ = load("history_gradient")
    five = set(json.load(open(os.path.join(ROOT, "runs", "MAIN_SETS.json")))["five"])
    grad = {}
    fig, ax = plt.subplots(figsize=(3.2, 2.0))
    for mdl in ["gemini-3.8-flash", "deepseek-v4.1-flash", "gemma-4-31b-it", "muse-glimmer-30b"]:
        pts = []
        for k, info in [(0, "none"), (1, "history_k1"), (2, "history_k2"), (3, "history_k3"), (4, "history")]:
            src = S if info in ("none", "history") else G
            v = [avg2(src[key], "same_focus") for key in src if key[0] == mdl and key[2] == info and key[3] in five]
            v = [x for x in v if x is not None]
            pts.append((k, float(np.mean(v)) if v else np.nan, len(v)))
        grad[mdl] = pts
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, lw=1, label=NICE[mdl])
    ax.set_xlabel("Other memories in context ($k$)"); ax.set_ylabel("Same focus"); ax.set_xticks(range(5)); ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_gradient.pdf")); plt.close(fig)
    res["gradient_same_focus"] = grad
    # timing
    tb = json.load(open(os.path.join(ROOT, "analysis", "timing_bias.json")))
    fig, ax = plt.subplots(figsize=(3.2, 1.6))
    meds = sorted([(v["model_median"], k.split("|")[0]) for k, v in tb["models"].items() if k.endswith("|none")])
    ax.scatter([m for m, _ in meds], range(len(meds)), s=8, color="#1b6e7a", label="models (no history)")
    ax.axvline(tb["human_median"], color="#d98e32", lw=1.2, label="people")
    ax.set_yticks([]); ax.set_xlim(0, 100); ax.set_xlabel("Median timing of the remembered moment (0 = start, 100 = end)")
    ax.legend(frameon=False, fontsize=7, loc="upper left"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig_timing.pdf")); plt.close(fig)
    json.dump(res, open(os.path.join(ROOT, "analysis", "model_results.json"), "w"), indent=1)
    print("ok")


if __name__ == "__main__":
    main()
