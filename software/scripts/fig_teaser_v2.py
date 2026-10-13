#!/usr/bin/env python3
"""Figure 1 (composite, not a real call): one call's record, two people's memories, a model's memory with and without an identity.
Actor colors used in every v2 figure: person A (she) teal, person B (he) amber, model purple. -> paper_v2/figs/fig_teaser.pdf"""
import os, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
C = {"A": "#1f7a72", "B": "#b7791f", "M": "#6b4fa0", "rec": "#4a5560"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "font.size": 7.2})
fig = plt.figure(figsize=(3.15, 3.0)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 103); ax.axis("off")

def box(x, y, w, h, col, title, body, wrap, fill=0.07, tsize=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2.2", fc=matplotlib.colors.to_rgba(col, fill), ec=col, lw=0.9))
    ax.text(x + 2, y + h - 2.2, title, color=col, weight="bold", va="top", fontsize=tsize)
    ax.text(x + 2, y + h - 8.2, "\n".join(textwrap.wrap(body, wrap)), va="top", fontsize=7.0, linespacing=1.18)

rec = ("B: So what made you take up bees?\n"
       "A: Honestly, I wanted something to look after.\n"
       "B: Have you been stung yet?\n"
       "A: Twice! Both on the first day.\n"
       "B: I live on a narrowboat, and one morning I woke up\n"
       "     to find it had drifted halfway across the canal.\n"
       "A: [laughs] No way!")
ax.add_patch(FancyBboxPatch((3, 64), 94, 36, boxstyle="round,pad=0.6,rounding_size=2.2", fc="#f3f4f5", ec=C["rec"], lw=0.9))
ax.text(5, 98, "[1] The record: three minutes on a video call", color=C["rec"], weight="bold", va="top", fontsize=7.4)
ax.text(5, 92, rec, va="top", fontsize=7.0, linespacing=1.22)

box(3, 33, 45.5, 26, C["A"], "[2a] What she remembers", "When he asked whether I'd been stung yet. I laughed because I'd been stung twice on day one.", 30)
box(51.5, 33, 45.5, 26, C["B"], "[2b] What he remembers", "How anxious I was telling her about the boat drifting across the canal, and the relief when she laughed.", 30)
box(3, 2, 45.5, 26, C["M"], "[3a] A model, as him", "The part where I told her my boat had drifted across the canal. It got a laugh and broke the ice.", 30)
box(51.5, 2, 45.5, 26, C["M"], "[3b] ...with an outgoing profile", "Cracking her up with the drifting-boat story! I felt energized and completely at ease, like we really clicked.", 30)
out = os.environ.get("OUT") or os.path.join(ROOT, "paper_v2", "figs", "fig_teaser.pdf"); fig.savefig(out); print(out)
