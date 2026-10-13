# Leak audit (leak_audit.py)

Texts checked: 2042 (calls, memories, persona cards).

1. Distinctive 10-word runs from real transcripts: **4** in 4 texts.
2. Real reports of 40+ characters: **0**.
3. Capitalized words that are rare in the real transcripts (at most two calls) and appear in the stand-in: 502 distinct; the list itself is withheld, since it was drawn from the private corpus (leak_audit.json keeps the counts).

## Human read (Oct 11 2026, 20:44 EDT, builder)
- **Runs.** The four "distinctive" runs are stock openers that a generator writes on its own: "i don't really know what we're meant to talk about" (three calls) and "hello yeah i can hear you can you hear me". The generators never saw a real transcript (gen.py reads no study text), so these count as coincidence, not copying.
- **Capitalized words.** The rare capitalized words are ordinary words at sentence starts, common UK first names (Darren, Craig, Karen), UK towns and areas (Luton, Stockport, Ilford), places abroad, and brands (Tesco, Deliveroo). None is a distinctive personal detail.
- **Verdict:** no leak found.
