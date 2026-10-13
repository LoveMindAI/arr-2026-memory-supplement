#!/usr/bin/env python3
"""The paper's person-specific-fit ladder (scripts/ladder_v351.py, run unchanged) with two extra rungs: the same five models
predicting from the Scribe transcripts (runs/tx_v3_cheap, coded by the same two coders). Run with PYTHONHASHSEED=0.
-> analysis/ladder_tx_v3.json"""
import os
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = os.path.join(HERE, "..", "scripts")
src = open(os.path.join(SCRIPTS, "ladder_v351.py")).read()
# ladder_v351.py on disk has its docstring opener glued to the line-2 comment (does not parse); split it in memory only
glued = 'order)."""Item 1'
if glued in src: src = src.replace(glued, 'order).\n"""Item 1', 1)
old = 'SETS = {"model: no history": ["main_text_cheap__none"], "model: history": ["main_text_cheap__history"],'
assert old in src
src = src.replace(old, old[:-1] + ', "model (Scribe transcript): no history": ["tx_v3_cheap__none"], "model (Scribe transcript): history": ["tx_v3_cheap__history"], "model (Scribe transcript): earlier-calls history": ["tx_v3_r3c__history_prior"], "model (Scribe transcript): norms note": ["tx_v3_norms_note__norms"], "model (Scribe transcript): norms + history": ["tx_v3_norms_history__norms_history"],')
assert '"ladder_v351" + ' in src   # output name is built with an optional suffix since Oct 12
src = src.replace('"ladder_v351" + ', '"ladder_tx_v3" + ')
os.chdir(SCRIPTS); import sys; sys.path.insert(0, SCRIPTS)
exec(compile(src, "ladder_tx_v3", "exec"))
