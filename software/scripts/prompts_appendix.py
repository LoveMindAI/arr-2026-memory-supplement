#!/usr/bin/env python3
"""Verbatim prompt blocks for the appendix, generated from the source strings (scripts/prompts.py, scripts/judge.py,
scripts/NORMS_NOTE_v2_APPROVED.txt, cards/build_card_inputs.py thin-card format). No participant text: slots are shown in angle
brackets. -> paper_v2/sections/app_parts/prompts_verbatim.tex"""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
import prompts as P
JUDGE = open(os.path.join(HERE, "judge.py")).read().split('JUDGE = """')[1].split('"""')[0]
NORMS = open(os.path.join(HERE, "NORMS_NOTE_v2_APPROVED.txt")).read().strip()
VAR = json.load(open(os.path.join(ROOT, "cards", "card_hexaco60_noaltruism.json")))
import re
SRC = re.sub(r'"\s*\n\s*(?:\+\s*)?"', "", open(os.path.join(HERE, "prompts.py")).read())   # join implicitly concatenated literals

def grab(start, end):
    i = SRC.index(start); return SRC[i:SRC.index(end, i)]
# literal strings from prompts.py, checked against the source text so nothing is retyped
HIST_LEAD = ("Earlier in the same study you also had calls with other people. After each one you were asked what moment you "
             "remembered most clearly, and why. Here is what you wrote:")
TX_HEAD = ("Here is a transcript of the call, transcribed automatically (expect some errors). "
           "Lines marked You are what you said; lines marked Partner are what the other person said. Times are minutes:seconds from the start.")
for s in (HIST_LEAD, "Here are your scores on a personality questionnaire you completed:", "Here is a description of you:", "That is who you are.",
          "Lines marked You are what you said; lines marked Partner are what the other person said. ", "transcribed automatically (expect some errors). ",
          "Now please answer the questions you were given right after the call:", "Recording 1 is ", "The video shows the call side by side"):
    assert s in SRC, s
doms = list(VAR["broad_from_facets"])
THIN = ("Personality questionnaire results (HEXACO-60). Each score is a mean on a 1-5 scale; higher means more of the named tendency.\n\n"
        "Broad domains: " + "; ".join(f"{d} <score>" for d in doms) + "\n\nFacets:\n" + "\n".join(f"- {n}: <score>" for n in VAR["facets"]))
assert "Personality questionnaire results (HEXACO-60). Each score is a mean on a 1-5 scale; higher means more of the named tendency." in open(os.path.join(ROOT, "cards", "build_card_inputs.py")).read()

def esc(t):
    rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#", "_": r"\_", "%": r"\%", "~": r"\textasciitilde{}",
           "^": r"\textasciicircum{}", "<": r"\textless{}", ">": r"\textgreater{}", '"': r"\textquotedbl{}", "[": "{[}", "]": "{]}"}
    out = "".join(rep.get(c, c) for c in t)
    paras = [p.strip("\n") for p in out.split("\n\n")]
    return "\n\n\\medskip\\noindent ".join("\\\\\n".join(p.split("\n")) for p in paras if p)

def block(title, text):
    return (f"\\paragraph{{{title}}}\n\\begin{{quote}}\\footnotesize\\ttfamily\\raggedright\\noindent " + esc(text) + "\n\\end{quote}\n")

Q = P.questions(False)
USER_TEXT = (P.STUDY + "\n\n" + TX_HEAD + "\n\n[0:05] You: <what the rememberer said>\n[0:09] Partner: <what the partner said>\n...\n\n" + Q)
blocks = [
    block("System message: history (the person's memories of their other calls)",
          HIST_LEAD + "\n\nAnother call:\nMoment: <their description>\nWhy: <their reason>\n\nAnother call:\n..."),
    block("System message: score profile", "Here are your scores on a personality questionnaire you completed:\n\n" + THIN),
    block("System message: written profile", "Here is a description of you:\n\n<written profile, 900 to 1,250 words, second person>\n\nThat is who you are."),
    block("System message: note on how people answer (held-out half only)", NORMS),
    block("User message (transcript condition)", USER_TEXT),
    block("Judge prompt (GPT-6 Luna and DeepSeek v4.1 Flash; reports A and B in random order)",
          JUDGE.replace("{{", "{").replace("}}", "}").replace("{am}", "<moment>").replace("{ar}", "<reason>").replace("{bm}", "<moment>").replace("{br}", "<reason>")),
]
media = ("Recording 1 is your own microphone (only your voice):\n<audio>\nRecording 2 is the other person as you heard them (only their voice). "
         "Both recordings start at the same moment:\n<audio>\nThe video shows the call side by side: the left half is the other person as you saw them, "
         "the right half is you. The video has no sound; the sound is in the recordings above.\n<video>")
for frag in ("Recording 1 is {yours} microphone (only {'their' if observer else 'your'} voice):", "Both recordings start at",
             "the same moment:", "The video has no sound; the sound is in the recordings above."):
    assert frag in SRC, frag
blocks.insert(5, block("Media parts (audio and video conditions; the transcript part is dropped in the audio-only condition)", media))
out = "% generated by scripts/prompts_appendix.py from the source strings; do not edit by hand\n" + "\n".join(blocks)
open(os.path.join(ROOT, "paper_v2", "sections", "app_parts", "prompts_verbatim.tex"), "w").write(out)
print("wrote prompts_verbatim.tex", len(out), "chars")
