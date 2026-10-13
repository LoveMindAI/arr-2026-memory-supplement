"""Overnight ROUND-2 memory DPO (A1, Oct 8: counterfactuals from the round-1 tuned model, which already writes human-length answers).
Attach mode: if the fleet dir already has SSH.json (a pod from an earlier start), reuse that pod. Unattended chain, each step checked before the next:
 1 pod (4x H200/B200, launcher pattern of orchestrate_dpo.py) -> push gen prompts + scripts (sha256-verified)
 2 phase 1 on pod: base model writes 4 guesses per training memory (run_onpol_gen.sh) -> pull, count-check
 3 local: judge guesses vs the person's real memory, build pairs (build_onpol.py pairs)
 4 push pairs (sha256-verified) -> phase 2 on pod: 4 trainings + evals (run_onpol_train.sh) -> pull evals + training records
 5 adapters -> B2 straight from the pod (size-checked putdir; credentials removed after) -> delete pod only if every check passed
 6 local: score (score_onpol.py: judges + paired deltas + lengths)."""
import glob, hashlib, json, os, subprocess, sys, time
LANE = "<LOCAL>/Documents/Codex/LAB/research/reader_night_20260928"
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data_v4_onpol"); OUT = os.path.join(HERE, "run_onpol")
sys.path.insert(0, "<LOCAL>/Documents/Codex/LAB/research/ocean_bigmodel_agent1_20260924")
B2PY = f"{LANE}/trainer/m3/b2.py"; B2C = os.path.expanduser("~/.config/LAB/credentials/b2-research-archive.json")
say = lambda *a: print(time.strftime("%H:%M"), *a, flush=True)
name = "agent1-reader-0928-memdpo2"; F = f"{LANE}/fleet/{name}"; os.makedirs(OUT, exist_ok=True)
TAGS = ["dpo_onpol_lr2e-05", "dpo_onpol_lr5e-05", "dpo_mixed_lr2e-05", "dpo_mixed_lr5e-05"]
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()

# ---------------------------------------------------------------- 1 pod
t0 = time.time(); got = False
if os.path.exists(F + "/SSH.json"):          # attach to the pod an earlier start created
    s = json.load(open(F + "/SSH.json"))
    base = ["ssh", "-i", "<LOCAL>/.runpod/ssh/runpodctl-ssh-key", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", f"UserKnownHostsFile={F}/KNOWN_HOSTS", "-p", str(s["port"])]
    for _ in range(240):
        try:
            r = subprocess.run(base + [f"root@{s['host']}", "ls /workspace/g31b/results/BOOTSTRAP_DONE && nvidia-smi -L | wc -l"], capture_output=True, text=True, timeout=60).stdout
        except Exception:
            r = ""
        if "BOOTSTRAP_DONE" in r:
            got = "attached"; say("attached; bootstrap done;", r.split()[-1], "GPUs"); break
        time.sleep(30)
    subprocess.run(["pkill", "-f", "[l]aunch_train.py memdpo2"]); say("old launcher stopped (pod kept)")
    if not got:
        say("ATTACH FAILED"); sys.exit(1)
while time.time() - t0 < 3 * 3600 and not got:
    for gpus in ("NVIDIA H200", "NVIDIA B200"):
        subprocess.run(["rm", "-rf", F])
        env = dict(os.environ, AGENT1_GPU_COUNT="4", AGENT1_MIN_CUDA="13.0", AGENT1_HOURS="11", AGENT1_GPUS=gpus)
        L = subprocess.Popen(["python3", "launch_train.py", "memdpo2", "dpo_unused", f"{OUT}/NO_SUCH_FILE.jsonl"], cwd=LANE, env=env,
                             stdout=open(f"{OUT}/launch.log", "a"), stderr=subprocess.STDOUT)
        for _ in range(240):
            time.sleep(30)
            if L.poll() is not None and not os.path.exists(F + "/SSH.json"):
                break
            if os.path.exists(F + "/SSH.json"):
                s = json.load(open(F + "/SSH.json"))
                base = ["ssh", "-i", "<LOCAL>/.runpod/ssh/runpodctl-ssh-key", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", f"UserKnownHostsFile={F}/KNOWN_HOSTS", "-p", str(s["port"])]
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
base = ["ssh", "-i", "<LOCAL>/.runpod/ssh/runpodctl-ssh-key", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30", "-o", f"UserKnownHostsFile={F}/KNOWN_HOSTS", "-p", str(s["port"])]
m = base + [f"root@{s['host']}"]; e = " ".join(base)
sh = lambda c, t=120: subprocess.run(m + [c], capture_output=True, text=True, timeout=t).stdout


def push(src, dst_dir):
    for attempt in range(3):
        subprocess.run(["rsync", "-t", "-e", e, src, f"root@{s['host']}:{dst_dir}/"], capture_output=True, timeout=3600)
        r = sh(f"sha256sum {dst_dir}/{os.path.basename(src)}").split()
        if r and r[0] == sha(src):
            say("push ok", os.path.basename(src)); return
    say("PUSH FAILED", src); sys.exit(1)


def poll(flag, errs, hours):
    t0 = time.time(); last = ""
    while time.time() - t0 < hours * 3600:
        try:
            o = sh(f"cd /workspace/g31b; ls {flag} 2>/dev/null; grep -h -E 'STAGE|ERROR' run_onpol_*.log | tail -2; grep -l Traceback {errs} 2>/dev/null | sed 's/^/tb:/'; "
                   "for d in results/training/dpo_*; do [ -f $d/steps.jsonl ] && tail -1 $d/steps.jsonl | python3 -c \"import sys,json;x=json.loads(sys.stdin.read());print('$(basename $d)', 'step', x['step'], 'loss %.3f acc %.2f' % (x['loss'], x['acc']))\"; done")
        except Exception as ex:
            o = f"(poll error {ex})"
        if o != last:
            say("poll:", " ".join(o.split())[:700]); last = o
        if flag in o.split():
            return True
        time.sleep(180)
    say("timeout; pod left up"); return False


sh("mkdir -p /workspace/g31b/data")
for src in sorted(glob.glob(f"{DATA}/gen_train_s*.jsonl")) + [f"{DATA}/eval_history_val.jsonl", f"{DATA}/eval_history_test.jsonl"]:
    push(src, "/workspace/g31b/data")
for src in ("run_onpol_gen.sh", "run_onpol_train.sh", "eval_memory_vllm.py", "dpo_worker.py"):
    push(f"{HERE}/{src}", "/workspace/g31b")
sh("mkdir -p /root/agent1_secrets")   # B2 key for fetching the round-1 adapter; run_onpol_gen.sh deletes it right after the fetch
push(B2C, "/root/agent1_secrets"); sh(f"mv /root/agent1_secrets/{os.path.basename(B2C)} /root/agent1_secrets/b2.json; chmod 600 /root/agent1_secrets/b2.json")
push(B2PY, "/workspace/g31b")

# ---------------------------------------------------------------- 2 phase 1: guesses
say(sh("cd /workspace/g31b && (nohup bash run_onpol_gen.sh > run_onpol_gen.log 2>&1 < /dev/null &); sleep 2; pgrep -f '[r]un_onpol_gen.sh' >/dev/null && echo gen_started"))
if not poll("GEN_DONE", "eval_gen_s*.log merge_r1.log", 3):
    sys.exit(1)
os.makedirs(f"{OUT}/eval_out", exist_ok=True)
for k in range(4):
    subprocess.run(["rsync", "-t", "-e", e, f"root@{s['host']}:/workspace/g31b/eval_out/gen_s{k}.jsonl", f"{OUT}/eval_out/"], timeout=1800)
cnt = {k: sum(1 for _ in open(f"{OUT}/eval_out/gen_s{k}.jsonl")) if os.path.exists(f"{OUT}/eval_out/gen_s{k}.jsonl") else 0 for k in range(4)}
need = sum(1 for _ in open(f"{DATA}/gen_train_s0.jsonl")); say("guesses", cnt, "need", need)
if any(v != need for v in cnt.values()):
    say("GUESS COUNTS SHORT; pod left up"); sys.exit(1)

# ---------------------------------------------------------------- 3 local: judge + pairs
r = subprocess.run(["python3", f"{HERE}/build_onpol.py", "pairs"], capture_output=True, text=True, env=dict(os.environ, JUDGE_MAX_TOKENS="12000"))
say("pairs:", r.stdout[-600:], r.stderr[-400:])
st = json.load(open(f"{OUT}/PAIRS_STATS.json"))
if st["onpol_pairs"] < 400:
    say("TOO FEW ON-POLICY PAIRS; pod left up"); sys.exit(1)
for arm in ("onpol", "mixed"):
    push(f"{DATA}/pairs_{arm}_train.jsonl", "/workspace/g31b/data")

# ---------------------------------------------------------------- 4 phase 2: train + eval
say(sh("cd /workspace/g31b && (nohup bash run_onpol_train.sh > run_onpol_train.log 2>&1 < /dev/null &); sleep 2; pgrep -f '[r]un_onpol_train.sh' >/dev/null && echo train_started"))
if not poll("DPO_ALL_DONE", "train_*.log eval_dpo_*.log eval_base_history.log", 8):
    sys.exit(1)
say("pod work done; pulling")
subprocess.run(["rsync", "-rt", "-e", e, f"root@{s['host']}:/workspace/g31b/eval_out/", f"{OUT}/eval_out/"], timeout=3600)
os.makedirs(f"{OUT}/logs", exist_ok=True)
for f in sh("cd /workspace/g31b; ls run_onpol_*.log train_*.log eval_*.log").split():
    subprocess.run(["rsync", "-t", "-e", e, f"root@{s['host']}:/workspace/g31b/{f}", f"{OUT}/logs/"], timeout=600)
for tg in TAGS:
    subprocess.run(["rsync", "-rt", "--exclude=*.safetensors", "-e", e, f"root@{s['host']}:/workspace/g31b/results/training/{tg}/", f"{OUT}/training_records/{tg}/"], timeout=1800)

# ---------------------------------------------------------------- 5 adapters -> B2 from the pod
sh("mkdir -p /root/agent1_secrets")
push(B2C, "/root/agent1_secrets"); sh(f"mv /root/agent1_secrets/{os.path.basename(B2C)} /root/agent1_secrets/b2.json; chmod 600 /root/agent1_secrets/b2.json")
push(B2PY, "/workspace/g31b")
need = sum(1 for f in ("val", "test") for _ in open(f"{DATA}/eval_history_{f}.jsonl"))
cnt = {t: (sum(1 for _ in open(f"{OUT}/eval_out/{t}.jsonl")) if os.path.exists(f"{OUT}/eval_out/{t}.jsonl") else 0) for t in TAGS + ["base_history", "r1_history"]}
OKT = [t for t in TAGS if cnt[t] == need]; say("eval counts", cnt, "need", need, "complete", OKT)
json.dump({"complete_tags": OKT, "counts": cnt}, open(f"{OUT}/COMPLETE_TAGS.json", "w"))
o = sh("cd /workspace/g31b; venv/bin/pip install -q boto3 2>&1 | tail -1; for t in " + " ".join(OKT) +
       "; do venv/bin/python b2.py putdir results/training/$t/epoch1 memory_dpo_r2_20261008/$t 2>&1 | tail -1; sha256sum results/training/$t/epoch1/adapter_model.safetensors; done", t=7200)
sh("rm -f /root/agent1_secrets/b2.json")
say("b2:", " ".join(o.split())[:900]); json.dump({"b2": o}, open(f"{OUT}/B2_UPLOAD.json", "w"), indent=1)
ok_b2 = o.count("B2_PUTDIR_OK") == len(OKT)
if cnt["base_history"] == need and cnt["r1_history"] == need and OKT and ok_b2:
    from rp import cli
    say("delete pod", cli(["pod", "delete", s["id"]]).returncode); json.dump({"deleted_at": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, open(F + "/DELETED.json", "w"))
else:
    say("NOT deleting (base eval short, no complete arm, or B2 incomplete) — pod left up"); sys.exit(1)

# ---------------------------------------------------------------- 6 score
r = subprocess.run(["python3", f"{HERE}/score_onpol.py"], capture_output=True, text=True, env=dict(os.environ, JUDGE_MAX_TOKENS="12000"))
say("score:", r.stdout[-3000:], r.stderr[-500:])
say("ALL DONE")
