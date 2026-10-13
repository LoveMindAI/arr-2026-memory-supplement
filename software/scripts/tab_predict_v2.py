#!/usr/bin/env python3
"""Table 3: person-specific fit by codebook question for every predictor (mean of the two AI coders; bold when both coders'
person-clustered 95% intervals exclude zero). Rows from analysis/ladder_v351.json ('all' panel; the norms note exists only for the
held-out half and is marked). -> paper_v2/tab_predict.tex"""
import json, os
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
L = json.load(open(os.path.join(ROOT, "analysis", "ladder_v351.json")))
Q = [("topic", "Topic"), ("extent", "Scope"), ("remembered", "What"), ("new_info", "New fact"), ("details", "Details"),
     ("moment_text", "Commentary"), ("reason.first", "Reason"), ("personal_kinds", "Disclosed"), ("feelings", "Feelings")]
R = [("most common human answer", "Most common answer"), ("partner's memory", "Partner's memory"), ("person's own earlier reports", "Own earlier memories"),
     ("model: no history", "Model, nothing"), ("model: history", "Model, history"), ("model: earlier-calls history", "Model, earlier history"),
     ("model: norms note", "Model, note on answers$^\\dagger$")]
lines = []
for key, lab in R:
    cells = []
    for q, _ in Q:
        es = [L[c]["all"].get(q, {}).get(key, {}).get("excess") for c in L]; es = [e for e in es if e]
        if not es: cells.append("--"); continue
        m = sum(e[0] for e in es) / len(es); sig = all(e[1] > 0 or e[2] < 0 for e in es)
        s = ".00" if abs(m) < 0.005 else f"{m:+.2f}".replace("+0.", "+.").replace("-0.", "$-$.")
        cells.append(f"\\textbf{{{s}}}" if sig else s)
    lines.append(lab + " & " + " & ".join(cells) + r" \\")
tex = r"""\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{4.5pt}
\begin{tabular}{l""" + "c" * len(Q) + r"""}
\toprule
 & """ + " & ".join(l for _, l in Q) + r""" \\
\midrule
""" + "\n".join(lines[:3]) + "\n\\midrule\n" + "\n".join(lines[3:]) + r"""
\bottomrule
\end{tabular}
\caption{Person-specific fit by codebook question (\S\ref{sec:design}; 0 = no closer to this person's report than to other people's in the same session). Mean of two LLM annotators; bold where both annotators' 95\% intervals exclude zero. $^\dagger$Held-out half only. Scope: how much of the conversation is reported; What: whose words or acts are retained.}
\label{tab:predict}
\end{table*}
"""
_o = os.environ.get("OUT") or os.path.join(ROOT, "paper_v2", "tab_predict.tex"); os.makedirs(os.path.dirname(_o), exist_ok=True); open(_o, "w").write(tex); print(tex)
