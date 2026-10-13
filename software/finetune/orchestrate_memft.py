"""Local orchestrator for the memory SFT (Gemma 4 31B, text, info=none). Pattern copied from reader_night orchestrate_v2h.py.
1) launch a 4x H200/B200 pod via the lane's launch_train.py (it installs the pinned stack + weights; its own training waits on a
   .READY file we never create), 2) push data + our driver + evaluator, 3) run run_memft.sh, 4) poll every 3 min (heartbeat),
5) pull eval outputs, training records and adapters (hash-checked), 6) delete the pod only if every eval file has full counts.
Failures leave the pod up and say so; the launcher's --stop-after (AGENT1_HOURS) is the backstop."""
import glob, hashlib, json, os, subprocess, sys, time
LANE = "<LOCAL>/Documents/Codex/LAB/research/reader_night_20260928"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data_v1")
OUT = os.path.join(HERE, "run_memft1")
sys.path.insert(0, "<LOCAL>/Documents/Codex/LAB/research/ocean_bigmodel_agent1_20260924")
say = lambda *a: print(time.strftime("%H:%M"), *a, flush=True)
name = "agent1-reader-0928-memft1"; F = f"{LANE}/fleet/{name}"
os.makedirs(OUT, exist_ok=True)
t0 = time.time(); got = int(os.environ["ATTACH_GPUS"]) if os.environ.get("ATTACH_GPUS") else False
while time.time() - t0 < 3 * 3600 and not got:
    for n in (4, 2):
        if os.environ.get("ATTACH_GPUS"): break
        subprocess.run(["rm", "-rf", F])
        env = dict(os.environ, AGENT1_GPU_COUNT=str(n), AGENT1_MIN_CUDA="13.0", AGENT1_HOURS=os.environ.get("AGENT1_HOURS", "4"),
                   AGENT1_GPUS="NVIDIA H200,NVIDIA B200")
        subprocess.run(["python3", "launch_train.py", "memft1", "memft_unused", f"{OUT}/NO_SUCH_FILE.jsonl"], cwd=LANE, env=env,
                       stdout=open(f"{OUT}/launch.log", "a"), stderr=subprocess.STDOUT)
        if os.path.exists(F + "/SSH.json"):
            got = n; break
    if not got:
        say("no capacity; retry in 5 min"); time.sleep(300)
if not got:
    say("NO CAPACITY in 3 h"); sys.exit(1)
say("pod up with", got, "GPUs")
s = json.load(open(F + "/SSH.json")); json.dump(s, open(f"{OUT}/POD.json", "w"))
base = ["ssh", "-i", "<LOCAL>/.runpod/ssh/runpodctl-ssh-key", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", f"UserKnownHostsFile={F}/KNOWN_HOSTS", "-p", str(s["port"])]
m = base + [f"root@{s['host']}"]; e = " ".join(base)
sh = lambda c, t=120: subprocess.run(m + [c], capture_output=True, text=True, timeout=t).stdout
for src in [f"{DATA}/memory_none_train.jsonl", f"{DATA}/memory_none_val.jsonl", f"{DATA}/memory_none_test.jsonl"]:
    r = subprocess.run(["rsync", "-tz", "-e", e, src, f"root@{s['host']}:/workspace/g31b/data/"], capture_output=True, text=True, timeout=1800); say("push", os.path.basename(src), r.returncode)
for src in [f"{HERE}/run_memft.sh", f"{HERE}/eval_memory_vllm.py"]:
    r = subprocess.run(["rsync", "-tz", "-e", e, src, f"root@{s['host']}:/workspace/g31b/"], capture_output=True, text=True, timeout=600); say("push", os.path.basename(src), r.returncode)
say(sh(f"cd /workspace/g31b && (nohup env NPROC={got} bash run_memft.sh > run_memft.log 2>&1 < /dev/null &) ; sleep 2; pgrep -f '[r]un_memft.sh' >/dev/null && echo driver_started"))
t0 = time.time(); last = ""
while time.time() - t0 < 4 * 3600:
    try:
        o = sh("cd /workspace/g31b; ls MEMFT_ALL_DONE 2>/dev/null; grep -l Traceback train_memft.log eval_*.log 2>/dev/null; grep -h STAGE run_memft.log | tail -1; tail -c 300 train_memft.log 2>/dev/null | tr '\\n' ' '")
    except Exception as ex:
        o = f"(poll error {ex})"
    if o != last:
        say("poll:", o.strip()[:400]); last = o
    if "Traceback" in o or "train_failed" in o:
        say("POD ERROR — pod left up for inspection"); sys.exit(1)
    if "MEMFT_ALL_DONE" in o:
        break
    time.sleep(180)
else:
    say("timeout; pod left up"); sys.exit(1)
say("pod work done; pulling")
subprocess.run(["rsync", "-rtz", "-e", e, f"root@{s['host']}:/workspace/g31b/eval_out/", f"{OUT}/eval_out/"], timeout=3600)
for f in ["run_memft.log", "train_memft.log", "train_dir_listing.txt", "eval_base.log", "eval_epoch1.log", "eval_epoch2.log", "eval_epoch4.log"]:
    subprocess.run(["rsync", "-tz", "-e", e, f"root@{s['host']}:/workspace/g31b/{f}", f"{OUT}/logs/"], timeout=600)
TD = "/workspace/g31b/results/training/memft_none_lr5e-05"
subprocess.run(["rsync", "-rtz", "--exclude=*.safetensors", "-e", e, f"root@{s['host']}:{TD}/", f"{OUT}/training_records/"], timeout=1800)
bad = []
for ep in ("epoch1", "epoch2", "epoch4"):
    dst = f"{OUT}/adapters/{ep}/"; os.makedirs(dst, exist_ok=True)
    subprocess.run(["rsync", "-rt", "-e", e, f"root@{s['host']}:{TD}/{ep}/", dst], timeout=3600)
    remote = sh(f"cd {TD}/{ep} && sha256sum adapter_model.safetensors").split()
    local = hashlib.sha256(open(dst + "adapter_model.safetensors", "rb").read()).hexdigest() if os.path.exists(dst + "adapter_model.safetensors") else None
    if not remote or remote[0] != local:
        bad.append(ep)
    say("adapter", ep, "hash ok" if ep not in bad else "HASH MISMATCH")
need = sum(1 for f in ("val", "test") for _ in open(f"{DATA}/memory_none_{f}.jsonl"))
cnt = {os.path.basename(p): sum(1 for _ in open(p)) for p in glob.glob(f"{OUT}/eval_out/*.jsonl")}
say("eval counts", cnt, "need", need)
if len(cnt) == 4 and all(v == need for v in cnt.values()) and not bad:
    from rp import cli
    say("delete pod", cli(["pod", "delete", s["id"]]).returncode)
    json.dump({"deleted_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(F + "/DELETED.json", "w"))
else:
    say("NOT deleting (counts short or adapter mismatch) — pod left up"); sys.exit(1)
say("ALL DONE")
