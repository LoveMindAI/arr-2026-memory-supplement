#!/bin/bash
# Driver v3 (Oct 7): history runs already training on GPUs 0-1 (started by v2). Start the card runs on GPUs 2-3 with the fixed pair file,
# wait for all four adapters (epoch1/COMPLETE.json), then evaluate all four DPO adapters (base evals already done), then DPO_ALL_DONE.
set -uo pipefail
cd /workspace/g31b
export HF_HOME=/workspace/hf OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(< /root/agent1_secrets/hf.token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
REV=$(python3 -c "import json;print(json.load(open('MANIFEST.json'))['revision'])"); BASE=$(ls -d /workspace/hf/hub/models--google--gemma-4-31B-it/snapshots/$REV)
echo "STAGE:train_card $(date +%s)"
i=2
for lr in 2e-05 5e-05; do
  CUDA_VISIBLE_DEVICES=$i RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 /workspace/g31b/venv/bin/python dpo_worker.py train --tag dpo_card_lr${lr} \
    --pairs /workspace/g31b/data/pairs_card_train.jsonl --lr $lr --beta 150 --len-norm --lr-floor 0.3 --nll-alpha 0.2 --base "$BASE" > train_card_${lr}.log 2>&1 &
  i=$((i+1))
done
TAGS="dpo_history_lr2e-05 dpo_history_lr5e-05 dpo_card_lr2e-05 dpo_card_lr5e-05"
while true; do
  n=0; for t in $TAGS; do [ -f results/training/$t/epoch1/COMPLETE.json ] && n=$((n+1)); done
  [ $n -eq 4 ] && break
  if ! pgrep -f "[d]po_worker.py train" >/dev/null; then echo "TRAIN_ERROR: trainers gone with $n of 4 adapters"; break; fi
  sleep 60
done
echo "STAGE:eval $(date +%s)"
H="/workspace/g31b/data/eval_history_val.jsonl /workspace/g31b/data/eval_history_test.jsonl"
C="/workspace/g31b/data/eval_card_val.jsonl /workspace/g31b/data/eval_card_test.jsonl"
g=0
for t in $TAGS; do
  case $t in *history*) FL=$H;; *) FL=$C;; esac
  [ -d results/training/$t/epoch1 ] && CUDA_VISIBLE_DEVICES=$g /workspace/g31b/venv/bin/python eval_memory_vllm.py --tag $t --adapter results/training/$t/epoch1 --files $FL > eval_$t.log 2>&1 &
  g=$((g+1))
done
wait
echo "STAGE:done $(date +%s)"
touch DPO_ALL_DONE
