#!/usr/bin/env python3
"""Transcript-fidelity test (A1, Oct 10): the paper's own prediction script (scripts/run_predict.py), unchanged, with only the
transcript source swapped from transcripts_v2 (Whisper large-v3) to transcripts_v3_scribe (Scribe v2: fewer dropped phrases,
fillers kept, laughter and other sounds tagged). Same prompts, models, temperature, conditions. Usage = run_predict.py's."""
import os, sys
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import prompts
prompts.TX = os.path.join(ROOT, "transcripts_v3_scribe")
import run_predict
assert run_predict.P is prompts and run_predict.P.TX.endswith("transcripts_v3_scribe")
run_predict.main()
