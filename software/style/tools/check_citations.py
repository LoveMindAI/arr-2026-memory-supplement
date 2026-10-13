#!/usr/bin/env python3
"""Citation vetting for the OCEAN paper. A bad citation can kill a paper, so run this after every edit session.

WHY THIS EXISTS: after reviewers called our earlier writing "AI slop", a single fabricated, misattributed, or
misdescribed citation would confirm the accusation. Every cited work must be (1) REAL: its record resolves and
the title matches; (2) READ: the lit room (or the main room) read it in full, with a note on disk;
(3) AUDITED: every sentence that cites it has been checked against that note; (4) not WEIRD, or weird with a
written reason to keep it.

Usage:
    python3 check_citations.py                 # offline: ledger + bib + .tex cross-checks
    python3 check_citations.py --online        # also resolve every DOI / arXiv ID and compare titles
Exit status is 1 if anything is RED.

Inputs (paths relative to ICLR_2027_Prep/):
    OCEAN/overleaf/ocean.bib, OCEAN/overleaf/**/*.tex, OCEAN/CITATION_LEDGER.md
"""
import re, sys, glob, json, difflib, urllib.request, urllib.parse, urllib.error, os, html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIB = os.environ.get("CIT_BIB") or os.path.join(ROOT, "OCEAN/overleaf/ocean.bib")   # env overrides for other papers (Oct 7, memory paper)
TEX = glob.glob(os.environ.get("CIT_TEX") or os.path.join(ROOT, "OCEAN/overleaf/**/*.tex"), recursive=True)
LEDGER = os.environ.get("CIT_LEDGER") or os.path.join(ROOT, "OCEAN/CITATION_LEDGER.md")
ONLINE = "--online" in sys.argv

RED, YEL, GRN, END = "\033[31m", "\033[33m", "\033[32m", "\033[0m"

def parse_bib(text):
    entries, pending_comments = {}, []
    for line_block in re.split(r"\n(?=@)", text):
        m = re.match(r"@(\w+)\{([^,]+),(.*)", line_block, re.S)
        if not m:
            continue
        typ, key, body = m.group(1).lower(), m.group(2).strip(), m.group(3)
        def f(name):
            mm = re.search(r"\b" + name + r"\s*=\s*\{(.*?)\}\s*,?\s*\n", body + "\n", re.S | re.I)
            return re.sub(r"[{}]", "", mm.group(1)).strip() if mm else ""
        # comments that follow this entry up to the next @ belong to the NEXT entry in our file layout;
        # CHECK lines are gathered separately below.
        entries[key] = dict(type=typ, title=f("title"), doi=f("DOI") or f("doi"), author=f("author"),
                            year=f("year"), eprint=f("eprint"), note=f("note"), howpublished=f("howpublished"),
                            booktitle=f("booktitle"), journal=f("journal"))
    # CHECK comments: the '% CHECK' lines sitting directly above an entry
    checks = {}
    lines = text.splitlines()
    for i, l in enumerate(lines):
        m = re.match(r"@\w+\{([^,]+),", l)
        if m:
            j, acc = i - 1, []
            while j >= 0 and lines[j].startswith("%"):
                if "CHECK" in lines[j]:
                    acc.append(lines[j].strip())
                j -= 1
            if acc:
                checks[m.group(1).strip()] = acc
    return entries, checks

def cited_keys():
    keys = {}
    for p in TEX:
        t = open(p).read()
        t = re.sub(r"(?<!\\)%.*", "", t)  # ignore comments
        for m in re.finditer(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}", t):
            for k in m.group(1).split(","):
                keys.setdefault(k.strip(), set()).add(os.path.relpath(p, ROOT))
    return keys

def parse_ledger():
    rows = {}
    if not os.path.exists(LEDGER):
        return rows
    for l in open(LEDGER):
        if l.startswith("|") and not re.match(r"\|\s*(key|-)", l):
            c = [x.strip() for x in l.strip().strip("|").split("|")]
            if len(c) >= 5:
                rows[c[0].strip("`")] = dict(read=c[1], audited=c[2], weird=c[3], decision=c[4])
    return rows

def weird_flags(e):
    w = []
    if not e["doi"] and not e["eprint"] and "arXiv" not in e["note"] and e["type"] not in ("book",):
        w.append("no DOI or arXiv ID to resolve")
    if e["type"] == "misc" and e["eprint"]:
        w.append("preprint only (no peer-reviewed venue)")
    if "SSRN" in e["howpublished"]:
        w.append("SSRN / industry report")
    if "press" in e["year"].lower():
        w.append("in press (permission/version?)")
    if e["author"] and any(len(a.strip().split()) == 1 for a in e["author"].split(" and ") if "," not in a):
        w.append("author(s) missing first names")
    return w

def title_sim(a, b):
    n = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower())
    return difflib.SequenceMatcher(None, n(a), n(b)).ratio()

def resolve(e):
    try:
        if e["doi"]:
            try:
                r = urllib.request.Request("https://api.crossref.org/works/" + urllib.parse.quote(e["doi"]))
                j = json.load(urllib.request.urlopen(r, timeout=30))["message"]
                return (j.get("title") or [""])[0]
            except urllib.error.HTTPError as he:
                if he.code != 404:
                    raise
                # Not a Crossref DOI (e.g. DataCite: Dataverse, PsychArchives, OSF). Ask the DOI resolver for CSL-JSON.
                r = urllib.request.Request("https://doi.org/" + e["doi"], headers={"Accept": "application/vnd.citationstyles.csl+json"})
                j = json.load(urllib.request.urlopen(r, timeout=30))
                t = j.get("title") or ""
                return html.unescape(t[0] if isinstance(t, list) else t)
        aid = e["eprint"] or (re.search(r"\d{4}\.\d{4,5}", e["note"]) or [None])[0]
        if aid:
            x = urllib.request.urlopen("http://export.arxiv.org/api/query?id_list=" + aid, timeout=30).read().decode()
            m = re.findall(r"<title>(.*?)</title>", x, re.S)
            return " ".join(m[1].split()) if len(m) > 1 else ""
    except Exception as ex:
        return "ERROR: %s" % ex
    return None

def main():
    bib, checks = parse_bib(open(BIB).read())
    cited = cited_keys()
    ledger = parse_ledger()
    red = 0
    print("=" * 78)
    print("CITATION VETTING. A weird or fabricated citation can sink the paper. Every RED line is a blocker.")
    print("=" * 78)
    for p in TEX:
        for n, line in enumerate(open(p), 1):
            code = re.sub(r"(?<!\\)%.*", "", line)
            for m in re.finditer(r"\\tbd\{cite[^}]*\}", code):
                print(f"{RED}RED  pending citation, not yet vetted: {os.path.relpath(p, ROOT)}:{n}  {m.group(0)}{END}"); red += 1
    for k in sorted(set(cited) - set(bib)):
        print(f"{RED}RED  {k}: cited in {sorted(cited[k])} but missing from ocean.bib{END}"); red += 1
    for k in sorted(set(bib) - set(cited)):
        print(f"{YEL}NOTE {k}: in ocean.bib but not cited (fine while drafting; prune before submission){END}")
    for k in sorted(set(cited) & set(bib)):
        e, row, issues, warns = bib[k], ledger.get(k), [], []
        warns_pre = None
        if not row:
            issues.append("no row in CITATION_LEDGER.md")
        else:
            if row["read"].upper().startswith("UNREAD"):
                # Unread is allowed only as an explicit, written decision: 'unread' named in the weird column AND a keep reason.
                if not ("unread" in row["weird"].lower() and row["decision"].lower().startswith("keep")):
                    issues.append("not read in full by anyone, and no written keep decision (" + row["read"] + ")")
                else:
                    warns_pre = "unread by decision: " + row["decision"]
            if row["audited"].lower() in ("no", "", "pending"):
                issues.append("claims in the .tex not yet audited against the note")
            if row["weird"] and row["weird"] != "-" and not row["decision"].lower().startswith("keep"):
                issues.append("weird (" + row["weird"] + ") with no keep decision")
        for c in checks.get(k, []):
            issues.append("open bib " + c.lstrip("% ").strip())
        warns += weird_flags(e)
        if warns_pre: warns.append(warns_pre)
        if ONLINE:
            t = resolve(e)
            if t is None:
                warns.append("nothing to resolve online (book/SSRN): verify by hand")
            elif t.startswith("ERROR"):
                issues.append("resolution failed: " + t)
            else:
                s = title_sim(t, e["title"])
                if s < 0.85:
                    issues.append(f"TITLE MISMATCH (sim {s:.2f}): record says '{t[:80]}'")
        tag = f"{RED}RED  " if issues else (f"{YEL}WEIRD" if warns else f"{GRN}OK   ")
        red += bool(issues)
        print(f"{tag} {k}{END}" + ("" if not (issues or warns) else ""))
        for i in issues:
            print(f"       - {i}")
        for w in warns:
            print(f"       ~ {w}")
    print("-" * 78)
    print(f"{len(cited)} cited keys; {red} RED. Weird-but-kept entries need a reason in the ledger's 'decision' column.")
    print("Reminder: a sentence can misdescribe a REAL paper. OK here means the citation was vetted, not that every claim is right;")
    print("the lit room's claim audit (ledger column 'audited') is the check for that.")
    sys.exit(1 if red else 0)

if __name__ == "__main__":
    main()
