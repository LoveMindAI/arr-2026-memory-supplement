"""Local orchestrator for the two memory DPO runs (A1, Oct 7). Pattern: finetune/orchestrate_memft.py, with the Oct 7 launcher fix:
launch_train.py runs in the BACKGROUND; we watch for SSH.json + the box's BOOTSTRAP_DONE ourselves and then stop the launcher (it hung today
waiting for training data an inference/DPO job never sends). Pushes data + dpo_worker.py + eval script + driver; polls; pulls eval outputs,
training records and adapters (hash-checked); deletes the pod only if every eval file is complete and every adapter hash matches."""
import glob, hashlib, json, os, subprocess, sys, time
LANE = "<LOCAL>/Documents/Codex/LAB/research/reader_night_20260928"
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data_v3_dpo"); OUT = os.path.join(HERE, "run_dpo1")
sys.path.insert(0, "<LOCAL>/Documents/Codex/LAB/research/ocean_bigmodel_agent1_20260924")
say = lambda *a: print(time.strftime("%H:%M"), *a, flush=True)
name = "agent1-reader-0928-memdpo1"; F = f"{LANE}/fleet/{name}"; os.makedirs(OUT, exist_ok=True)
TAGS = ["dpo_history_lr2e-05", "dpo_history_lr5e-05", "dpo_card_lr2e-05", "dpo_card_lr5e-05"]
t0 = time.time(); got = False
while time.time() - t0 < 3 * 3600 and not got:
    for gpus in ("NVIDIA H200", "NVIDIA B200"):
        subprocess.run(["rm", "-rf", F])
        env = dict(os.environ, AGENT1_GPU_COUNT="4", AGENT1_MIN_CUDA="13.0", AGENT1_HOURS="5", AGENT1_GPUS=gpus)
        L = subprocess.Popen(["python3", "launch_train.py", "memdpo1", "dpo_unused", f"{OUT}/NO_SUCH_FILE.jsonl"], cwd=LANE, env=env,
                             stdout=open(f"{OUT}/launch.log", "a"), stderr=subprocess.STDOUT)
        for _ in range(240):                     # up to 2 h for SSH + bootstrap
            time.sleep(30)
            if L.poll() is not None and not os.path.exists(F + "/SSH.json"):
                break                            # launcher gave up (no capacity)
            if os.path.exists(F + "/SSH.json"):
                s = json.load(open(F + "/SSH.json"))
                base = ["ssh", "-i", "<LOCAL>/.runpod/ssh/runpodctl-ssh-key", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", f"UserKnownHostsFile={F}/KNOWN_HOSTS", "-p", str(s["port"])]
                try:
                    r = subprocess.run(base + [f"root@{s['host']}", "ls /workspace/g31b/results/BOOTSTRAP_DONE && nvidia-smi -L | wc -l"], capture_output=True, text=True, timeout=60).stdout
                except Exception:
                    r = ""
                if "BOOTSTRAP_DONE" in r:
                    got = gpus; say("bootstrap done;", r.split()[-1], "GPUs"); break
        if L.poll() is None:
            L.terminate(); say("launcher stopped (pod kept)")
        if got:
            break
    if not got:
        say("no capacity; retry in 5 min"); time.sleep(300)
if not got:
    say("NO CAPACITY"); sys.exit(1)
s = json.load(open(F + "/SSH.json")); json.dump(s, open(f"{OUT}/POD.json", "w")); say("pod up", got, s["id"])
base = ["ssh", "-i", "<LOCAL>/.runpod/ssh/runpodctl-ssh-key", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", f"UserKnownHostsFile={F}/KNOWN_HOSTS", "-p", str(s["port"])]
m = base + [f"root@{s['host']}"]; e = " ".join(base)
sh = lambda c, t=120: subprocess.run(m + [c], capture_output=True, text=True, timeout=t).stdout
sh("mkdir -p /workspace/g31b/data")
for src in sorted(glob.glob(f"{DATA}/*.jsonl")):
    say("push", os.path.basename(src), subprocess.run(["rsync", "-t", "-e", e, src, f"root@{s['host']}:/workspace/g31b/data/"], capture_output=True, timeout=3600).returncode)
for src in ("run_dpo.sh", "eval_memory_vllm.py", "dpo_worker.py"):
    say("push", src, subprocess.run(["rsync", "-t", "-e", e, f"{HERE}/{src}", f"root@{s['host']}:/workspace/g31b/"], capture_output=True, timeout=600).returncode)
say(sh("cd /workspace/g31b && (nohup bash run_dpo.sh > run_dpo.log 2>&1 < /dev/null &); sleep 2; pgrep -f '[r]un_dpo.sh' >/dev/null && echo driver_started"))
t0 = time.time(); last = ""
while time.time() - t0 < 5 * 3600:
    try:
        o = sh("cd /workspace/g31b; ls DPO_ALL_DONE 2>/dev/null; grep -l Traceback train_*.log eval_*.log 2>/dev/null; grep -h -E 'STAGE|TRAIN_ERROR' run_dpo.log | tail -2; "
               "for d in results/training/dpo_*; do [ -f $d/steps.jsonl ] && tail -1 $d/steps.jsonl | python3 -c \"import sys,json;x=json.loads(sys.stdin.read());print('$(basename $d)', 'step', x['step'], 'loss %.3f acc %.2f' % (x['loss'], x['acc']))\"; done")
    except Exception as ex:
        o = f"(poll error {ex})"
    if o != last:
        say("poll:", " ".join(o.split())[:600]); last = o
    if "DPO_ALL_DONE" in o:
        break
    time.sleep(180)
else:
    say("timeout; pod left up"); sys.exit(1)
say("pod work done; pulling")
subprocess.run(["rsync", "-rt", "-e", e, f"root@{s['host']}:/workspace/g31b/eval_out/", f"{OUT}/eval_out/"], timeout=3600)
os.makedirs(f"{OUT}/logs", exist_ok=True)
for f in sh("cd /workspace/g31b; ls run_dpo.log train_*.log eval_*.log").split():
    subprocess.run(["rsync", "-t", "-e", e, f"root@{s['host']}:/workspace/g31b/{f}", f"{OUT}/logs/"], timeout=600)
bad = []
for tg in TAGS:
    TD = f"/workspace/g31b/results/training/{tg}"
    subprocess.run(["rsync", "-rt", "--exclude=*.safetensors", "-e", e, f"root@{s['host']}:{TD}/", f"{OUT}/training_records/{tg}/"], timeout=1800)
    dst = f"{OUT}/adapters/{tg}/"; os.makedirs(dst, exist_ok=True)
    subprocess.run(["rsync", "-rt", "-e", e, f"root@{s['host']}:{TD}/epoch1/", dst], timeout=3600)
    remote = sh(f"cd {TD}/epoch1 && sha256sum adapter_model.safetensors").split()
    local = hashlib.sha256(open(dst + "adapter_model.safetensors", "rb").read()).hexdigest() if os.path.exists(dst + "adapter_model.safetensors") else None
    if not remote or remote[0] != local:
        bad.append(tg)
    say("adapter", tg, "hash ok" if tg not in bad else "HASH MISMATCH / missing")
need_h = sum(1 for f in ("val", "test") for _ in open(f"{DATA}/eval_history_{f}.jsonl")); need_c = sum(1 for f in ("val", "test") for _ in open(f"{DATA}/eval_card_{f}.jsonl"))
need = {"base_history.jsonl": need_h, "base_card.jsonl": need_c, **{f"{t}.jsonl": (need_h if "history" in t else need_c) for t in TAGS}}
cnt = {k: (sum(1 for _ in open(f"{OUT}/eval_out/{k}")) if os.path.exists(f"{OUT}/eval_out/{k}") else 0) for k in need}
say("eval counts", cnt, "need", need)
if all(cnt[k] == v for k, v in need.items()) and not bad:
    from rp import cli
    say("delete pod", cli(["pod", "delete", s["id"]]).returncode); json.dump({"deleted_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(F + "/DELETED.json", "w"))
else:
    say("NOT deleting (counts short or adapter mismatch) — pod left up"); sys.exit(1)
say("ALL DONE")
