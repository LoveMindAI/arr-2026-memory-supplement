#!/bin/bash
# Round-2 memory DPO, phase 1 (pod). A1, Oct 8: write the counterfactuals with the round-1 TUNED model (already human-length), not the untuned one.
#  a) fetch the round-1 adapter dpo_history_lr5e-05 from B2, merge it into Gemma 4 31B-it (dpo_worker.py merge: self-checking) -> /workspace/r1_merged
#  b) the merged round-1 model writes 4 guesses (seeds s0..s3) per history training prompt, same eval script and sampling settings as every eval.
set -uo pipefail
cd /workspace/g31b
export HF_HOME=/workspace/hf OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
echo "STAGE:wait_bootstrap $(date +%s)"
until [ -f results/BOOTSTRAP_DONE ]; do sleep 20; done
echo "STAGE:fetch_r1 $(date +%s)"
mkdir -p /workspace/r1_adapter
venv/bin/pip install -q boto3 2>&1 | tail -1
for f in COMPLETE.json adapter_config.json adapter_model.safetensors; do
  venv/bin/python b2.py get memory_dpo_20261007/dpo_history_lr5e-05/$f /workspace/r1_adapter/$f || { echo "GEN_ERROR fetch $f"; rm -f /root/agent1_secrets/b2.json; exit 1; }
done
rm -f /root/agent1_secrets/b2.json
echo "STAGE:merge $(date +%s)"
CUDA_VISIBLE_DEVICES=0 RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 venv/bin/python dpo_worker.py merge --m03 /workspace/r1_adapter --out /workspace/r1_merged > merge_r1.log 2>&1
[ -f /workspace/r1_merged/MERGE_INFO.json ] || { echo "GEN_ERROR merge"; exit 1; }
echo "STAGE:gen $(date +%s)"
for k in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$k venv/bin/python eval_memory_vllm.py --tag gen_s$k --model /workspace/r1_merged --files /workspace/g31b/data/gen_train_s$k.jsonl > eval_gen_s$k.log 2>&1 &
done
wait
for k in 0 1 2 3; do grep -q Traceback eval_gen_s$k.log && echo "GEN_ERROR s$k"; done
echo "STAGE:gen_done $(date +%s)"
touch GEN_DONE
