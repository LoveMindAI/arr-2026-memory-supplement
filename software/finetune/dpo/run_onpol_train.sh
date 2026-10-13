#!/bin/bash
# Round-2 memory DPO, phase 2 (pod). Policy start = reference = the merged round-1 model (/workspace/r1_merged). Same recipe as round 1
# (dpo_worker.py unchanged; beta 150, --len-norm, --lr-floor 0.3, --nll-alpha 0.2, 1 epoch), history prompts only.
# Arms: onpol (chosen = real answer, rejected = round-1 guess judged a different moment) and mixed (1 such pair + 1 round-1 human pair), x LR {2e-5, 5e-5}.
# Eval on held-out people (val + test): untuned base, round-1 merged, and the 4 round-2 adapters.
set -uo pipefail
cd /workspace/g31b
export HF_HOME=/workspace/hf OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
R1=/workspace/r1_merged
echo "STAGE:train $(date +%s) base=$R1"
i=0
for arm in onpol mixed; do for lr in 2e-05 5e-05; do
  CUDA_VISIBLE_DEVICES=$i RANK=0 WORLD_SIZE=1 LOCAL_RANK=0 venv/bin/python dpo_worker.py train --tag dpo_${arm}_lr${lr} \
    --pairs /workspace/g31b/data/pairs_${arm}_train.jsonl --lr $lr --beta 150 --len-norm --lr-floor 0.3 --nll-alpha 0.2 --base "$R1" > train_${arm}_${lr}.log 2>&1 &
  i=$((i+1))
done; done
wait
for f in train_*.log; do grep -q Traceback $f && echo "TRAIN_ERROR $f"; done
echo "STAGE:eval $(date +%s)"
H="/workspace/g31b/data/eval_history_val.jsonl /workspace/g31b/data/eval_history_test.jsonl"
i=0
for tg in dpo_onpol_lr2e-05 dpo_onpol_lr5e-05 dpo_mixed_lr2e-05 dpo_mixed_lr5e-05; do
  CUDA_VISIBLE_DEVICES=$i venv/bin/python eval_memory_vllm.py --tag $tg --model $R1 --adapter results/training/$tg/epoch1 --files $H > eval_$tg.log 2>&1 &
  i=$((i+1))
done
wait
CUDA_VISIBLE_DEVICES=0 venv/bin/python eval_memory_vllm.py --tag base_history --files $H > eval_base_history.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 venv/bin/python eval_memory_vllm.py --tag r1_history --model $R1 --files $H > eval_r1_history.log 2>&1 &
wait
echo "STAGE:done $(date +%s)"
touch DPO_ALL_DONE
