"""Local orchestrator for the card pipeline (1x H200/B200). Pattern from finetune/orchestrate_memft.py."""
import glob, json, os, subprocess, sys, time
LANE = "<LOCAL>/Documents/Codex/LAB/research/reader_night_20260928"
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "run_cards1"); os.makedirs(OUT, exist_ok=True)
KIT = "<LOCAL>/Documents/Codex/LAB/research/personality_models_kit"
AD = {"writer_lcw150": f"{LANE}/adapters_pending_8tb/release_20261006/writer_lcw150", "reader_gemma_rvB": "<LOCAL>/Documents/Codex/LAB/research/merge_fix_20261003/adapters/reader_gemma_rvB"}
sys.path.insert(0, "<LOCAL>/Documents/Codex/LAB/research/ocean_bigmodel_agent1_20260924")
say = lambda *a: print(time.strftime("%H:%M"), *a, flush=True)
name = "agent1-reader-0928-memcards1"; F = f"{LANE}/fleet/{name}"
t0 = time.time(); got = False
while time.time() - t0 < 2 * 3600 and not got:
    for gpus in ("NVIDIA H200", "NVIDIA B200", "NVIDIA H100 NVL"):
        subprocess.run(["rm", "-rf", F])
        env = dict(os.environ, AGENT1_GPU_COUNT="1", AGENT1_MIN_CUDA="13.0", AGENT1_HOURS="2.5", AGENT1_GPUS=gpus)
        subprocess.run(["python3", "launch_train.py", "memcards1", "cards_unused", f"{OUT}/NO_SUCH_FILE.jsonl"], cwd=LANE, env=env,
                       stdout=open(f"{OUT}/launch.log", "a"), stderr=subprocess.STDOUT)
        if os.path.exists(F + "/SSH.json"): got = gpus; break
    if not got: say("no capacity; retry in 5 min"); time.sleep(300)
if not got: say("NO CAPACITY"); sys.exit(1)
s = json.load(open(F + "/SSH.json")); json.dump(s, open(f"{OUT}/POD.json", "w")); say("pod up", got, s["id"])
base = ["ssh", "-i", "<LOCAL>/.runpod/ssh/runpodctl-ssh-key", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", f"UserKnownHostsFile={F}/KNOWN_HOSTS", "-p", str(s["port"])]
m = base + [f"root@{s['host']}"]; e = " ".join(base)
sh = lambda c, t=120: subprocess.run(m + [c], capture_output=True, text=True, timeout=t).stdout
rs = lambda src, dst, t=3600: subprocess.run(["rsync", "-rtz", "-e", e, src, f"root@{s['host']}:{dst}"], capture_output=True, text=True, timeout=t).returncode
sh("mkdir -p /workspace/adapters /workspace/cards_in /workspace/kit")
say("push inputs", rs(f"{HERE}/inputs/", "/workspace/cards_in/"), "kit", rs(f"{KIT}/", "/workspace/kit/"), "scripts", rs(f"{HERE}/pod_cards.py", "/workspace/"), rs(f"{HERE}/run_cards.sh", "/workspace/"))
say(sh("cd /workspace && (nohup bash run_cards.sh > run_cards.log 2>&1 < /dev/null &); sleep 2; pgrep -f '[r]un_cards.sh' >/dev/null && echo driver_started"))
for n, p in AD.items():   # pushed while the base model downloads
    say("push adapter", n, rs(p + "/", f"/workspace/adapters/{n}/"))
say(sh("touch /workspace/ADAPTERS_READY && echo adapters_ready"))
t0 = time.time(); last = ""
while time.time() - t0 < 2.4 * 3600:
    try: o = sh("cd /workspace; ls CARDS_EXIT 2>/dev/null; grep -h STAGE run_cards.log | tail -1; tail -c 300 cards.log 2>/dev/null | tr '\\n' ' '")
    except Exception as ex: o = f"(poll error {ex})"
    if o != last: say("poll:", o.strip()[:400]); last = o
    if "CARDS_EXIT" in o: break
    time.sleep(120)
else: say("timeout; pod left up"); sys.exit(1)
subprocess.run(["rsync", "-rtz", "-e", e, f"root@{s['host']}:/workspace/cards_out/", f"{OUT}/cards_out/"], timeout=3600)
subprocess.run(["rsync", "-tz", "-e", e, f"root@{s['host']}:/workspace/cards.log", f"root@{s['host']}:/workspace/run_cards.log", f"{OUT}/"], timeout=600)
need = {"cards_lcw150.jsonl": 290, "cards_person.jsonl": sum(1 for _ in open(f"{HERE}/inputs/person_jobs.jsonl"))}
cnt = {k: sum(1 for _ in open(f"{OUT}/cards_out/{k}")) if os.path.exists(f"{OUT}/cards_out/{k}") else 0 for k in need}
reads = sum(1 for _ in open(f"{OUT}/cards_out/reads_readerB.jsonl")) if os.path.exists(f"{OUT}/cards_out/reads_readerB.jsonl") else 0
say("counts", cnt, "need", need, "reads", reads)
if all(cnt[k] == v for k, v in need.items()) and reads > 0 and os.path.exists(f"{OUT}/cards_out/CARDS_DONE"):
    from rp import cli
    say("delete pod", cli(["pod", "delete", s["id"]]).returncode); json.dump({"deleted_at": time.strftime("%c")}, open(F + "/DELETED.json", "w"))
else:
    say("NOT deleting — counts short or not done; pod left up"); sys.exit(1)
say("ALL DONE")
