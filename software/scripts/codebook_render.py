#!/usr/bin/env python3
"""Render codebook v3.1 SPEC into (1) the model prompt and (2) the readable codebook. Same words as the GUI (scripts/make_gui_v31.py)."""
import json, os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "codebook")); from spec_v351 import SPEC
READ = {"MOMENT": "Read: MOMENT", "REASON": "Read: REASON", "BOTH": "Read: MOMENT and REASON", "TRANSCRIPT": "Read: the transcript"}


def how(q):
    t = q["type"]
    return {"single": "Choose one.", "multi": f"Choose {'up to ' + str(q['max']) if q.get('max') else 'all that apply'}.", "statements": "Rate every statement.",
            "nested_ranked": "Rank the main targets; then choose all options that apply under each.", "text": "Write a short answer."}[t]


def opt_line(o, codes=True):
    return f"- {o['label']}" + (f" ({o['help']})" if o.get("help") else "") + (" (only on its own)" if o.get("exclusive") else "") + (" (write what it is in the matching _other field)" if o.get("text") else "") + (f" [{o['code']}]" if codes else "")


def render(models=True, codes=True):
    L = ["HOW THIS WORKS"] + [f"- {x}" for x in SPEC["intro"]]
    for p in SPEC["parts"]:
        if models and p.get("models") is False:
            continue
        L += ["", f"PART {p['id']}: {p['title'].upper()}"]
        for i, q in enumerate(p["questions"], 1):
            L += ["", f"{p['id']}{i}. {q['q']} ({READ[q['read']]}. {how(q)})"]
            if q.get("help"): L.append(q["help"])
            if q["type"] in ("single", "multi"):
                L += [opt_line(o, codes) for o in q["options"]]
            elif q["type"] == "statements":
                L.append("Answer scale: " + " · ".join(f"{s['label']}" + (f" [{s['code']}]" if codes else "") for s in q["scale"]))
                L += [f"- {s['text']}" + (f" [{s['code']}]" if codes else "") for s in q["statements"]]
            elif q["type"] == "nested_ranked":
                for o in q["options"]:
                    L.append(opt_line(o, codes))
                    L += ["    " + opt_line(s, codes) for s in o.get("sub", [])]
    return "\n".join(L)


def output_spec():
    keys = []
    for p in SPEC["parts"]:
        if p.get("models") is False: continue
        for q in p["questions"]:
            if q["type"] == "single": keys.append(f'"{q["id"]}": one code')
            elif q["type"] == "multi": keys.append(f'"{q["id"]}": [codes]')
            elif q["type"] == "statements": keys.append(f'"{q["id"]}": {{' + ", ".join(f'"{s["code"]}": scale code' for s in q["statements"]) + "}")
            elif q["type"] == "nested_ranked": keys.append(f'"{q["id"]}": {{"about": [main target codes, most important first], "details": [codes of every option chosen under them]}}')
            if any(o.get("text") for o in q.get("options", [])): keys.append(f'"{q["id"]}_other": "a few words if you chose the write-in option, otherwise empty"')
    return "OUTPUT: return JSON only, with exactly these keys, using the codes in square brackets:\n{" + ", ".join(keys) + "}"


def prompt():
    return render(models=True, codes=True) + "\n\n" + output_spec()


if __name__ == "__main__":
    md = [f"# Codebook v{SPEC['version']} (generated from codebook/spec_v351.py; do not edit by hand)", "", "Grounding, adaptations and procedure: see the header of codebook/CODEBOOK_v3.md.",
          "Model coders see the text below with the codes in square brackets, Parts A and B only, plus the output format. People see the same words in the coding page.", "", "```", render(models=False, codes=True), "```"]
    open(os.path.join(ROOT, "codebook", f"CODEBOOK_v{SPEC['version']}.md"), "w").write("\n".join(md) + "\n")
    print(prompt()[:1500]); print("...\n", output_spec()[:600])
