"""Keyword screen exactly as written in FIELD_COUNT_RULE.md (Population 1-3)."""
import gzip, re, json, collections
txt = gzip.open("anthology+abstracts.bib.gz", "rt", encoding="utf-8").read()
entries = re.split(r"\n(?=@)", txt)
MEM = ["long-term memory","long term memory","conversational memory","dialogue memory","dialog memory","episodic memory","personal memory","user memory","memory-augmented","memory augmented","memory bank","memory module","memory mechanism","memory management","memory system"]
CONV = ["dialog","dialogue","conversation","chat","chatbot","assistant","companion","session"]
LONG = ["long-term conversation","long-term dialogue","long-term dialog","multi-session","multi session","long-horizon conversation","lifelong dialogue"]
def field(e, name):
    m = re.search(r"\b" + name + r"\s*=\s*(\"(.*?)\"|\{(.*?)\})\s*,?\s*\n", e, re.S)
    if not m: return ""
    v = m.group(2) if m.group(2) is not None else m.group(3)
    return " ".join(v.replace("{", "").replace("}", "").split())
out = []; n_years = collections.Counter()
for e in entries:
    m = re.match(r"@(\w+)\{([^,]+),", e)
    if not m: continue
    y = field(e, "year")
    if y not in ("2024", "2025", "2026"): continue
    n_years[y] += 1
    title, abstract = field(e, "title"), field(e, "abstract")
    s = (title + " " + abstract).lower()
    hit_mem = [k for k in MEM if k in s]; hit_conv = [k for k in CONV if k in s]; hit_long = [k for k in LONG if k in s]
    if (hit_mem and hit_conv) or hit_long:
        out.append(dict(key=m.group(2).strip(), type=m.group(1), year=y, title=title, abstract=abstract,
                        venue=field(e, "booktitle") or field(e, "journal"), url=field(e, "url"), doi=field(e, "doi"),
                        hits=hit_mem + hit_long, conv=hit_conv))
json.dump(out, open("screened.json", "w"), indent=1, ensure_ascii=False)
print("papers per year in the export:", dict(n_years)); print("screened:", len(out))
print(collections.Counter(o["year"] for o in out))
