"""Scope scored in three classes (A1, Oct 12): the codebook's line between 'a stretch of the call' and 'the whole call, or a pattern across
it' proved hard to draw (two human coders did not agree above chance on the four-way answer), so both count as LONGER:
ONE_MOMENT | LONGER | NOT_AN_EVENT. The frozen codebook and every stored code are unchanged; this mapping is applied when codes are read.
SCOPE4=1 restores the original four answers (used only for the appendix agreement table)."""
import copy, os
ON = os.environ.get("SCOPE4") != "1"
MAP = {"STRETCH": "LONGER", "WHOLE_CALL_OR_PATTERN": "LONGER"}
LABEL = "A longer part of the call, or the whole call"
OPTIONS = ["ONE_MOMENT", "LONGER", "NOT_AN_EVENT"] if ON else ["ONE_MOMENT", "STRETCH", "WHOLE_CALL_OR_PATTERN", "NOT_AN_EVENT"]


def value(v):
    if not ON or v is None: return v
    if isinstance(v, list): return [MAP.get(str(x).upper(), x) if isinstance(x, str) else x for x in v]
    return MAP.get(str(v).upper(), v) if isinstance(v, str) else v


def codes(c):
    """A copy of one coder's code dict with scope merged (key 'extent')."""
    if not ON or not isinstance(c, dict) or "extent" not in c: return c
    c = dict(c); c["extent"] = value(c["extent"]); return c


def spec(S):
    """A copy of the codebook spec whose scope question offers the three scored answers."""
    if not ON: return S
    S = copy.deepcopy(S)
    for p in S["parts"]:
        for q in p["questions"]:
            if q["id"] == "extent":
                keep = [o for o in q["options"] if o["code"] not in MAP]
                i = next((k for k, o in enumerate(keep) if o["code"] == "ONE_MOMENT"), 0) + 1
                keep.insert(i, {"code": "LONGER", "label": LABEL})
                q["options"] = keep
    return S
