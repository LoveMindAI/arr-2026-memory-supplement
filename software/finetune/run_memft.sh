#!/bin/bash
# Pod-side driver: wait for the lane bootstrap (pinned vllm/transformers/peft + weights), train SFT LoRA (r16/a32, completion-only,
# recipe unchanged from the reader lane), then evaluate base + epoch1/2/4 on val+test in parallel (one GPU each).
set -uo pipefail
cd /workspace/g31b
export HF_HOME=/workspace/hf OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false NCCL_NVLS_ENABLE=0
export HF_TOKEN="$(< /root/agent1_secrets/hf.token)"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
LR=${LR:-5e-05}; EPOCHS=${EPOCHS:-4}; TASK=memft_none
echo "STAGE:wait_bootstrap $(date +%s)"
until [ -f results/BOOTSTRAP_DONE ]; do sleep 20; done
echo "STAGE:train $(date +%s)"
/workspace/g31b/venv/bin/torchrun --standalone --nproc_per_node ${NPROC:-4} gpu_worker_ddp.py train --train-task $TASK \
   --train-file /workspace/g31b/data/memory_none_train.jsonl --lr $LR --epochs $EPOCHS --diagnostic-deterministic > train_memft.log 2>&1 || { echo "STAGE:train_failed"; exit 1; }
TD=results/training/${TASK}_lr${LR}
ls $TD > train_dir_listing.txt
echo "STAGE:eval $(date +%s)"
F="/workspace/g31b/data/memory_none_val.jsonl /workspace/g31b/data/memory_none_test.jsonl"
NG=${NPROC:-4}
TAGS="base epoch1 epoch2 epoch4"; i=0
for t in $TAGS; do
  if [ "$t" = base ]; then AD=""; else AD="--adapter $TD/$t"; [ -d "$TD/$t" ] || continue; fi
  CUDA_VISIBLE_DEVICES=$((i % NG)) /workspace/g31b/venv/bin/python eval_memory_vllm.py --tag $t $AD --files $F > eval_$t.log 2>&1 &
  i=$((i+1)); if [ $((i % NG)) -eq 0 ]; then wait; fi
done
wait
echo "STAGE:done $(date +%s)"
touch MEMFT_ALL_DONE
