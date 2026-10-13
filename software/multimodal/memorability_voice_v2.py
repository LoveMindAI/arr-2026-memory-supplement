#!/usr/bin/env python3
"""scripts/memorability.py, run unchanged, with the VOICE=1 family read from the own-mic features
(VFILE = voice_turn_features_v2.npy or voice_turn_features_v2emo.npy). Output: analysis/memorability_voice_<tag>.json."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = os.path.join(HERE, "..", "scripts")
vfile, tag = sys.argv[1], sys.argv[2]
src = open(os.path.join(SCRIPTS, "memorability.py")).read()
old_load = '"analysis", "voice_turn_features.npy"'
assert old_load in src and '("_voice" if os.environ.get("VOICE") == "1" else "")' in src
src = src.replace(old_load, f'"analysis", "{vfile}"')
src = src.replace('("_voice" if os.environ.get("VOICE") == "1" else "")', f'("_voice_{tag}" if os.environ.get("VOICE") == "1" else "")')
os.environ["VOICE"] = "1"; os.chdir(SCRIPTS); sys.path.insert(0, os.path.abspath(SCRIPTS)); sys.argv = [os.path.join(SCRIPTS, "memorability.py")]
exec(compile(src, "memorability_voice_v2", "exec"), {"__name__": "__main__", "__file__": os.path.join(SCRIPTS, "memorability.py")})
