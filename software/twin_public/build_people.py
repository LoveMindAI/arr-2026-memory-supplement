#!/usr/bin/env python3
"""Public stand-in, step 1 (local, no API): people, sessions, schedule, memory types and length targets.
Inputs: public_data/ons2021_teaching_sample.csv (ONS, Open Government Licence v3.0) and targets_from_parsel.json (numbers only).
Never imports PARSEL code or data.

Demographics: no census row becomes a person. Each person is assembled from three to four different census records ("donors"):
  core (age band, sex, region) from one adult record; birth country, ethnic group and religion from a second record with the same core;
  partnership and household from a third with the same age band and sex; work (occupation, hours, social grade, full-time education,
  health) from a fourth with the same core. Age and sex follow the published PARSEL summary (A1, Oct 11); the core donor matches the person's census age band and sex.
Sessions: 55 groups with the study's size mix, round robin; calls dropped at random (never leaving anyone without one) to the study's total.
Types: each person is assigned one of the two memory types by the study's shares. Length targets: a person-level log-length mean
drawn from the between-person spread, and per-memory targets from the within-person spread (moment and reason separately).
-> people.json, sessions.json (with calls)"""
import csv, json, os, random, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(20261011); nr = np.random.default_rng(20261011)
T = json.load(open(os.path.join(HERE, "targets_from_parsel.json")))
REGION = {"E12000001": "North East", "E12000002": "North West", "E12000003": "Yorkshire and The Humber", "E12000004": "East Midlands",
          "E12000005": "West Midlands", "E12000006": "East of England", "E12000007": "London", "E12000008": "South East", "E12000009": "South West",
          "W92000004": "Wales"}
LAB = json.load(open(os.path.join(HERE, "public_data", "ons2021_labels.json")))
OCC = {"1": "Managers, directors and senior officials", "2": "Professional occupations", "3": "Associate professional and technical occupations",
       "4": "Administrative and secretarial occupations", "5": "Skilled trades occupations", "6": "Caring, leisure and other service occupations",
       "7": "Sales and customer service occupations", "8": "Process, plant and machine operatives", "9": "Elementary occupations", "-8": "no current occupation"}
GRADE = {"1": "AB (higher and intermediate managerial, administrative, professional)", "2": "C1 (supervisory, clerical, junior managerial)",
         "3": "C2 (skilled manual)", "4": "DE (semi-skilled and unskilled manual; unemployed and lowest grade)", "-8": "not applicable"}
AGE = {"2": (18, 24), "3": (25, 34), "4": (35, 44), "5": (45, 54), "6": (55, 64), "7": (65, 80)}
lab = lambda v, c: (LAB.get(v) or {}).get(c, c)

rows = [r for r in csv.DictReader(open(os.path.join(HERE, "public_data", "ons2021_teaching_sample.csv"))) if r["resident_age_7d"] in AGE and r["residence_type"] != "2"]
by_core = collections.defaultdict(list); by_as = collections.defaultdict(list)
for r in rows:
    by_core[(r["resident_age_7d"], r["sex"], r["region"])].append(r); by_as[(r["resident_age_7d"], r["sex"])].append(r)

# A1, Oct 11: age and sex follow PARSEL's PUBLISHED demographic summary (Hrkalovic et al. 2025, p. 3491: 297 participants, 166 women,
# M age 37.39, SD 11.42; quoted in SOURCES.md 1E). Age ~ normal(37.39, 11.42), redrawn below 18; woman with p = 166/297.
DEMO = dict(age_mean=37.39, age_sd=11.42, p_female=166 / 297)
def band(age): return next(k for k, (lo, hi) in AGE.items() if lo <= age <= hi) if age <= 80 else "7"
def person(i):
    age = 0
    while age < 18: age = int(round(nr.normal(DEMO["age_mean"], DEMO["age_sd"])))
    sex = "1" if rng.random() < DEMO["p_female"] else "2"
    a = rng.choice(by_as[(band(age), sex)]); core = (a["resident_age_7d"], a["sex"], a["region"])   # region etc. conditional on the census
    b, d = rng.choice(by_core[core]), rng.choice(by_core[core]); c = rng.choice(by_as[core[:2]])
    return dict(id=f"P{i:03d}", age=age, sex=lab("sex", core[1]), region=REGION.get(core[2], core[2]),
                born=lab("country_of_birth_3a", b["country_of_birth_3a"]), ethnic_group=lab("ethnic_group_tb_6a", b["ethnic_group_tb_6a"]),
                religion=lab("religion_tb", b["religion_tb"]), partnership=lab("legal_partnership_status_6a", c["legal_partnership_status_6a"]),
                household=lab("hh_families_type_6a", c["hh_families_type_6a"]), occupation=OCC.get(d["occupation_10a"], d["occupation_10a"]),
                hours=lab("hours_per_week_worked", d["hours_per_week_worked"]), social_grade=GRADE.get(d["approx_social_grade"], d["approx_social_grade"]),
                full_time_student=(d["in_full_time_education"] == "1"), health=lab("health_in_general", d["health_in_general"]))

# sessions with the study's size mix
mix = [int(k) for k, v in T["groups"]["size_mix"].items() for _ in range(v)]; rng.shuffle(mix)
people, sessions, k = {}, {}, 0
for s, n in enumerate(mix):
    sid = f"S{s + 1:02d}"; members = []
    for _ in range(n): p = person(k); people[p["id"]] = p; members.append(p["id"]); k += 1
    sessions[sid] = members
def rounds(m):   # circle-method round robin; with an odd number one sits out each round
    m = list(m) + ([None] if len(m) % 2 else []); n = len(m); out = []
    for r in range(n - 1):
        out.append([(m[i], m[n - 1 - i]) for i in range(n // 2) if m[i] and m[n - 1 - i]]); m = [m[0]] + [m[-1]] + m[1:-1]
    return out
calls = []
for sid, mem in sessions.items():
    for r, pairs in enumerate(rounds(mem)):
        for A, B in pairs: calls.append(dict(session=sid, round=r, A=A, B=B))
target = T["groups"]["n_calls"]
while len(calls) > target:
    i = rng.randrange(len(calls)); c = calls[i]
    left = collections.Counter(x for cc in calls for x in (cc["A"], cc["B"]))
    if left[c["A"]] > 1 and left[c["B"]] > 1: calls.pop(i)
calls.sort(key=lambda c: (c["session"], c["round"])); [c.update(id=f"C{j + 1:04d}") for j, c in enumerate(calls)]

# memory type and length targets
mt = T["memory_types"]; tm, tr = T["memory_moment_words"], T["memory_reason_words"]
ids = sorted(people); rng.shuffle(ids); n_stretch = round(len(ids) * mt["shared_stretch"]["share"])   # exact shares, random people
for j, pid in enumerate(ids): people[pid]["memory_type"] = "shared_stretch" if j < n_stretch else "what_they_were_told"
for p in people.values():
    p["len_mu"] = {"moment": float(nr.normal(tm["log1p_mean"], tm["between_person_sd"])), "reason": float(nr.normal(tr["log1p_mean"], tr["between_person_sd"]))}
for c in calls:
    for side in ("A", "B"):
        p = people[c[side]]
        c[f"len_{side}"] = {"moment": max(2, int(round(np.expm1(nr.normal(p["len_mu"]["moment"], tm["within_person_sd"]))))),
                            "reason": max(2, int(round(np.expm1(nr.normal(p["len_mu"]["reason"], tr["within_person_sd"])))))}
        prof = mt[p["memory_type"]]["profile"]
        u = rng.random(); scope = "one_moment" if u < prof["scope:one_moment"] else "stretch" if u < prof["scope:one_moment"] + prof["scope:stretch"] else "free"
        v = rng.random(); what = "did_together" if v < prof["what:did_together"] else "partner_said_or_did" if v < prof["what:did_together"] + prof["what:partner_said_or_did"] else "free"
        c[f"spec_{side}"] = dict(scope=scope, what=what, new_fact=rng.random() < prof["new_fact"])
json.dump(people, open(os.path.join(HERE, "people.json"), "w"), indent=1)
json.dump(dict(sessions=sessions, calls=calls), open(os.path.join(HERE, "sessions.json"), "w"), indent=1)
print(len(people), "people;", len(sessions), "sessions;", len(calls), "calls;", collections.Counter(p["memory_type"] for p in people.values()))
