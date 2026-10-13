#!/bin/zsh
# Norms-note experiment (A1 approved Oct 9): test half only; 5 main models; no history; then judge, then code (Haiku + Luna).
cd "$(dirname "$0")"
PIDS=($(python3 -c "import json;print(' '.join(json.load(open('../analysis/norms_test_pids.json'))))"))
L=../runs/norms_note_v2.log
python3 run_predict.py --run norms_note_v2 --pids $PIDS --models deepseek/deepseek-v4.1-flash google/gemini-3.8-flash google/gemma-4-31b-it qwen/qwen3.8-omni-flash xiaomi/mimo-v2.6-flash --modalities text --infos norms --conc 32 --budget 8 >> $L 2>&1
python3 run_predict.py --run norms_note_v2 --pids $PIDS --models deepseek/deepseek-v4.1-flash google/gemini-3.8-flash google/gemma-4-31b-it qwen/qwen3.8-omni-flash xiaomi/mimo-v2.6-flash --modalities text --infos norms --conc 8 --budget 2 >> $L 2>&1
python3 judge.py --run norms_note_v2 --conc 32 >> $L 2>&1
python3 code_v35.py --set norms_note_v2:norms --coders claude-haiku-5.5,gpt-6-luna --conc 96 >> $L 2>&1
python3 code_v35.py --set norms_note_v2:norms --coders claude-haiku-5.5,gpt-6-luna --conc 16 >> $L 2>&1
echo DONE >> $L
