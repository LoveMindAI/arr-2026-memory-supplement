#!/bin/bash
# Pod-side driver for the two memory DPO runs (A1, Oct 7). Four trainings in parallel, one GPU each:
#   {history, card} x LR {2e-5, 5e-5}; beta 150, --len-norm, --lr-floor 0.3, --nll-alpha 0.2 (all pairs flagged nll), 1 epoch,
#   policy start = reference = plain Gemma 4 31B-it (the HF snapshot), reader-lane dpo_worker.py unchanged.
# Then evaluation on held-out people (val + test): base with history prompts, base with card prompts, and each DPO adapter on its own prompts.
set -uo pipefail
cd /workspace/g31b
export HF_HOME=/workspace/hf OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_TOKEN="$(< /root/agent1_secrets/hf.token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
echo "STAGE:wait_base_evals $(date +%s)"; until [ -f eval_out/base_history.DONE ] && [ -f eval_out/base_card.DONE ]; do sleep 20; done
REV=$(python3 -c "import json;print(json.load(open('MANIFEST.json'))['revision'])")
BASE=$(ls -d /workspace/hf/hub/models--google--gemma-4-31B-it/snapshots/$REV)
[ -f "$BASE/tokenizer.json" ] || { echo "STAGE:no_base"; exit 1; }
echo "STAGE:train $(date +%s) base=$BASE"
i=0
for cond in history card; do for lr in 2e-05 5e-05; do
  CUDA_VISIBLE_DEVICES=$i RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 /workspace/g31b/venv/bin/python dpo_worker.py train --tag dpo_${cond}_lr${lr} \
    --pairs /workspace/g31b/data/pairs_${cond}_train.jsonl --lr $lr --beta 150 --len-norm --lr-floor 0.3 --nll-alpha 0.2 --base "$BASE" > train_${cond}_${lr}.log 2>&1 &
  i=$((i+1))
done; done
wait
for f in train_*.log; do grep -q Traceback $f && echo "TRAIN_ERROR $f"; done
echo "STAGE:eval $(date +%s)"
H="/workspace/g31b/data/eval_history_val.jsonl /workspace/g31b/data/eval_history_test.jsonl"
C="/workspace/g31b/data/eval_card_val.jsonl /workspace/g31b/data/eval_card_test.jsonl"
CUDA_VISIBLE_DEVICES=0 /workspace/g31b/venv/bin/python eval_memory_vllm.py --tag dpo_history_lr2e-05 --adapter results/training/dpo_history_lr2e-05/epoch1 --files $H > eval_dh2.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 /workspace/g31b/venv/bin/python eval_memory_vllm.py --tag dpo_history_lr5e-05 --adapter results/training/dpo_history_lr5e-05/epoch1 --files $H > eval_dh5.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 /workspace/g31b/venv/bin/python eval_memory_vllm.py --tag dpo_card_lr2e-05 --adapter results/training/dpo_card_lr2e-05/epoch1 --files $C > eval_dc2.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 /workspace/g31b/venv/bin/python eval_memory_vllm.py --tag dpo_card_lr5e-05 --adapter results/training/dpo_card_lr5e-05/epoch1 --files $C > eval_dc5.log 2>&1 &
wait
echo "STAGE:done $(date +%s)"
touch DPO_ALL_DONE
