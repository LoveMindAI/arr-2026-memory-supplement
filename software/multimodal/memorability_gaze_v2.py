#!/usr/bin/env python3
"""scripts/memorability.py, run unchanged, with the GAZE=1 family read from the v2 gaze instruments
(GFILE = gaze_turn_features_v2_nets.npy or gaze_turn_features_v2.npy). Output: analysis/memorability_gaze_<tag>.json."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); SCRIPTS = os.path.join(HERE, "..", "scripts")
gfile, tag = sys.argv[1], sys.argv[2]
src = open(os.path.join(SCRIPTS, "memorability.py")).read()
old_load = '"analysis", "gaze_turn_features.npy"'
assert old_load in src and '("_gaze" if os.environ.get("GAZE") == "1" else "")' in src
src = src.replace(old_load, f'"analysis", "{gfile}"')
src = src.replace('("_gaze" if os.environ.get("GAZE") == "1" else "")', f'("_gaze_{tag}" if os.environ.get("GAZE") == "1" else "")')
os.environ["GAZE"] = "1"; os.environ.pop("VOICE", None); os.chdir(SCRIPTS); sys.path.insert(0, os.path.abspath(SCRIPTS)); sys.argv = [os.path.join(SCRIPTS, "memorability.py")]
exec(compile(src, "memorability_gaze_v2", "exec"), {"__name__": "__main__", "__file__": os.path.join(SCRIPTS, "memorability.py")})
