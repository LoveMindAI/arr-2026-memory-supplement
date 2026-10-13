#!/usr/bin/env python3
"""The paper's norms-note analysis (scripts/norms_analysis.py, run unchanged otherwise) on the Scribe-transcript reruns: the no-history
baseline from runs/tx_v3_cheap and the norms note from runs/tx_v3_norms_note, coded by the same two coders, judged by the same three
judges, test half only. -> analysis/norms_result_tx_v3.json"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = os.path.join(HERE, "..", "scripts")
src = open(os.path.join(SCRIPTS, "norms_analysis.py")).read()
for a, b in [('load("main_text_cheap__none", coder)', 'load("tx_v3_cheap__none", coder)'),
             ('load("norms_note_v2__norms", coder)', 'load("tx_v3_norms_note__norms", coder)'),
             ('["main_text_cheap", "norms_note_v2"]', '["tx_v3_cheap", "tx_v3_norms_note"]'),
             ('"norms_result.json"', '"norms_result_tx_v3.json"')]:
    assert src.count(a) == 1, a; src = src.replace(a, b)
os.chdir(SCRIPTS); sys.path.insert(0, SCRIPTS)
exec(compile(src, "norms_tx_v3", "exec"), {"__name__": "norms_tx_v3"})
