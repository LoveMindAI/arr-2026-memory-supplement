#!/usr/bin/env python3
"""Generate paper/app_prompts.tex from the prompt strings the scripts actually use (no retyping)."""
import os, re, sys
sys.path.insert(0, os.path.dirname(__file__))
import prompts as P, judge as J, locate_moments as L, code_v1 as C, keenan_test as K
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "app_prompts.tex")
SPEC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}", "<": r"\textless{}", ">": r"\textgreater{}"}


def tex(s):
    s = "".join(SPEC.get(ch, ch) for ch in s)
    s = s.replace("{", r"\{").replace("}", r"\}")
    s = re.sub(r"\\\{\\\{", r"\\{", s); s = re.sub(r"\\\}\\\}", r"\\}", s)          # format-string doubled braces -> single
    s = re.sub(r"\\\{(tx|mem|why|am|ar|bm|br|seg|m|w)\\\}", lambda m: r"\textit{[" + {"tx": "numbered transcript", "mem": "memory", "why": "reason",
               "am": "moment A", "ar": "reason A", "bm": "moment B", "br": "reason B", "seg": "stretch of transcript", "m": "moment", "w": "reason"}[m.group(1)] + "]}", s)
    s = s.replace(r"\_", r"\_\allowbreak{}").replace(",", r",\allowbreak{}").replace("|", r"|\allowbreak{}")
    s = s.replace('"', r'\textquotedbl{}').replace("\n", "\\\\\n")
    s = re.sub(r"(\\\\\n){2,}", r"\\\\[0.5ex]\n", s)
    return s


def box(title, body):
    return f"\\paragraph{{{title}}}\n\\begin{{quote}}\\raggedright\\footnotesize\\ttfamily\n{tex(body)}\n\\end{{quote}}\n"


import orc
P.transcript = lambda *a, **k: "[transcript]"
orc.audio_part = lambda *a, **k: {"type": "text", "text": "[audio]"}
orc.video_part = lambda *a, **k: {"type": "text", "text": "[video]"}
_row = P.rows()[0]
_m = P.build(_row, "text_audio_video", "none", "seat")
assert len(_m) == 1 and _m[0]["role"] == "user"
USER = "\n\n".join(x["text"] for x in _m[0]["content"])
hist_lead = ("Earlier in the same study you also had calls with other people. After each one you were asked what moment you "
             "remembered most clearly, and why. Here is what you wrote:\n\nAnother call:\nMoment: {m}\nWhy: {w}\n\n[one block per other call]")
assert hist_lead.split("\n")[0] in open(P.__file__).read().replace('"\n                "', "")  # lead text appears in prompts.py
parts = [
    "\\section{Prompts}\n\\label{app:prompts}\nAll prompts are reproduced from the scripts that ran them. Bracketed italics mark where data is inserted.\n",
    box("Prediction: system message (history condition)", hist_lead),
    box("Prediction: system message (profile conditions)", "Here are your scores on a personality questionnaire you completed:\n\n[facet scores]\n\n"
        "or\n\nHere is a description of you:\n\n[card]\n\nThat is who you are."),
    box("Prediction: user message (text, audio and video; text-only and audio-only conditions omit the corresponding parts)", USER),
    box("Judge", J.JUDGE),
    box("Locator", L.PROMPT),
    box("Memory coding (codebook)", C.PROMPT),
    box("Interactional content", K.PROMPT),
]
open(OUT, "w").write("\n".join(parts)); print("wrote", OUT)
