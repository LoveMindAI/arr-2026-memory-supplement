"""Neutral tie rule for a person's (or model's) 'typical answer' (Oct 12, A1 OK).
- Several labels allowed: keep labels on at least half of the reports; if none reaches half, keep every label tied for most frequent.
- One label: the most frequent; a tie is scored as the average over the tied labels (no arbitrary pick).
- Ordinal scales: the median, kept as x.5 when it falls between two points (scores linearly, so this equals averaging the two).
Order-free: the result no longer depends on list order or PYTHONHASHSEED."""
import collections
import numpy as np


class Tie(tuple):
    """Tied single-label prediction; scored as the mean over its members."""


def typical(vals, k, ORD):
    vals = [v for v in vals if v is not None]
    if not vals: return None
    if isinstance(vals[0], frozenset):
        c = collections.Counter(x for v in vals for x in v)
        keep = frozenset(x for x, n in c.items() if n * 2 >= len(vals))
        if keep: return keep
        top = max(c.values()); return frozenset(x for x, n in c.items() if n == top)
    if k in ORD or k.startswith("appraisal."):
        m = float(np.median(vals)); return int(m) if m == int(m) else m
    c = collections.Counter(vals); top = max(c.values()); tied = sorted(x for x, n in c.items() if n == top)
    return tied[0] if len(tied) == 1 else Tie(tied)


def wrap_score(score):
    def s(k, a, b):
        if isinstance(b, Tie):
            v = [score(k, a, x) for x in b]; v = [x for x in v if x is not None]
            return float(np.mean(v)) if v else None
        if isinstance(a, Tie):
            return s(k, b, a)
        return score(k, a, b)
    return s
