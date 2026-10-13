#!/bin/bash
# Round-2 memory DPO, phase 3 (pod; Oct 8, added after the onpol arms saturated on a human-vs-model style cue by step ~16).
# Arm "self": chosen = a round-1 guess BOTH judges call the person's moment, rejected = a round-1 guess both call a different moment,
# same memory (same writer, style and length; only the moment differs). 460 pairs, 241 memories. No NLL anchor (both sides on-policy).
# Same recipe otherwise (beta 150, --len-norm, --lr-floor 0.3), start = reference = round-1 merged. Waits for phase 2 to finish.
set -uo pipefail
cd /workspace/g31b
export HF_HOME=/workspace/hf OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
R1=/workspace/r1_merged
until [ -f DPO_ALL_DONE ]; do sleep 30; done
echo "STAGE:p3_train $(date +%s)"
i=0
for lr in 2e-05 5e-05; do
  CUDA_VISIBLE_DEVICES=$i RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 venv/bin/python dpo_worker.py train --tag dpo_self_lr${lr} \
    --pairs /workspace/g31b/data/pairs_self_train.jsonl --lr $lr --beta 150 --len-norm --lr-floor 0.3 --base "$R1" > train_self_${lr}.log 2>&1 &
  i=$((i+1))
done
wait
for f in train_self_*.log; do grep -q Traceback $f && echo "TRAIN_ERROR $f"; done
echo "STAGE:p3_eval $(date +%s)"
H="/workspace/g31b/data/eval_history_val.jsonl /workspace/g31b/data/eval_history_test.jsonl"
i=0
for tg in dpo_self_lr2e-05 dpo_self_lr5e-05; do
  CUDA_VISIBLE_DEVICES=$i venv/bin/python eval_memory_vllm.py --tag $tg --model $R1 --adapter results/training/$tg/epoch1 --files $H > eval_$tg.log 2>&1 &
  i=$((i+1))
done
wait
echo "STAGE:p3_done $(date +%s)"
touch P3_DONE
