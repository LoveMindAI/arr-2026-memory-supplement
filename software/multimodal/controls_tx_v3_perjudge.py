import json, os, sys
SCRIPTS = os.getcwd(); sys.path.insert(0, SCRIPTS)
SP = sys.argv[1]; RUNSET = sys.argv[2]
src = open("controls_analysis.py").read()
reps = [('    for info in ("card_thin", "card_narr", "card_thin_wrong", "card_narr_wrong"):\n        rung', '    for info in ():\n        rung'),
        ('if __name__ == "__main__":\n    main()', '')]
if RUNSET == "scribe":
    reps.append(('RUNS = ["main_text_cheap", "cards_text_cheap", "cards_narr_cheap"] + [f"r3c_{t}" for t in ("deepseek", "gemini", "gemma", "mimo", "qwen3.8")]', 'RUNS = ["tx_v3_cheap", "tx_v3_r3c"]'))
for a, b in reps:
    assert src.count(a) == 1, a[:50]; src = src.replace(a, b)
out = {}
for J in (["gpt-6-luna"], ["deepseek-v4.1-flash"], ["claude-haiku-5.5"], ["gpt-6-luna", "deepseek-v4.1-flash", "claude-haiku-5.5"]):
    s = src.replace('"controls.json"', repr(os.path.join(SP, "ctrl_tmp.json")))
    g = {"__file__": os.path.join(SCRIPTS, "controls_analysis.py"), "__name__": "x"}
    exec(compile(s, "x", "exec"), g)
    g["FM"].JUDGES[:] = J
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()): g["main"]()
    r = json.load(open(os.path.join(SP, "ctrl_tmp.json")))
    out["+".join(j.split("-")[0] for j in J)] = r["comparisons"]
    g["FM"].JUDGES[:] = ["gpt-6-luna", "deepseek-v4.1-flash", "claude-haiku-5.5"]
json.dump(out, open(os.path.join(SP, f"perjudge_{RUNSET}.json"), "w"), indent=1)
ROWS = [("right_vs_wrong_history", "same_reason"), ("right_vs_wrong_history", "same_focus"), ("wrong_narr_vs_none", "same_focus"),
        ("wrong_narr_vs_none", "same_reason"), ("prior_history_vs_none", "same_focus"), ("prior_history_vs_none", "same_reason")]
print(RUNSET)
for c, m in ROWS:
    print(f"  {c:24s} {m:12s} " + "  ".join(f"{k:>18s} {v[c][m]['mean']:+.3f} {v[c][m]['up']}u/{v[c][m]['down']}d n={v[c][m]['models']}" for k, v in out.items()))
