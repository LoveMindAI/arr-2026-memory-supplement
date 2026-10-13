#!/bin/bash
# Driver v4 (Oct 7 speed-up): history adapters are done -> evaluate them now on idle GPUs 0-1; wait for the card adapters, evaluate them on GPUs 2-3; then DPO_ALL_DONE.
cd /workspace/g31b
export HF_HOME=/workspace/hf OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
H="/workspace/g31b/data/eval_history_val.jsonl /workspace/g31b/data/eval_history_test.jsonl"
C="/workspace/g31b/data/eval_card_val.jsonl /workspace/g31b/data/eval_card_test.jsonl"
echo "STAGE:eval_history $(date +%s)"
CUDA_VISIBLE_DEVICES=0 /workspace/g31b/venv/bin/python eval_memory_vllm.py --tag dpo_history_lr2e-05 --adapter results/training/dpo_history_lr2e-05/epoch1 --files $H > eval_dpo_history_lr2e-05.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 /workspace/g31b/venv/bin/python eval_memory_vllm.py --tag dpo_history_lr5e-05 --adapter results/training/dpo_history_lr5e-05/epoch1 --files $H > eval_dpo_history_lr5e-05.log 2>&1 &
until [ -f results/training/dpo_card_lr2e-05/epoch1/COMPLETE.json ] && [ -f results/training/dpo_card_lr5e-05/epoch1/COMPLETE.json ]; do
  pgrep -f "[d]po_worker.py train" >/dev/null || { echo "TRAIN_ERROR: card trainers gone"; break; }; sleep 60; done
echo "STAGE:eval_card $(date +%s)"
CUDA_VISIBLE_DEVICES=2 /workspace/g31b/venv/bin/python eval_memory_vllm.py --tag dpo_card_lr2e-05 --adapter results/training/dpo_card_lr2e-05/epoch1 --files $C > eval_dpo_card_lr2e-05.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 /workspace/g31b/venv/bin/python eval_memory_vllm.py --tag dpo_card_lr5e-05 --adapter results/training/dpo_card_lr5e-05/epoch1 --files $C > eval_dpo_card_lr5e-05.log 2>&1 &
wait
echo "STAGE:done $(date +%s)"
touch DPO_ALL_DONE
