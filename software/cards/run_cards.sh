#!/bin/bash
cd /workspace
export HF_HOME=/workspace/hf HF_TOKEN="$(< /root/agent1_secrets/hf.token)" HUGGING_FACE_HUB_TOKEN="$(< /root/agent1_secrets/hf.token)"
until [ -f /workspace/g31b/results/BOOTSTRAP_DONE ] && [ -f /workspace/ADAPTERS_READY ]; do sleep 20; done
echo "STAGE:cards $(date +%s)"
/workspace/g31b/venv/bin/python /workspace/pod_cards.py > /workspace/cards.log 2>&1 && echo "STAGE:done" || echo "STAGE:failed"
touch /workspace/CARDS_EXIT
