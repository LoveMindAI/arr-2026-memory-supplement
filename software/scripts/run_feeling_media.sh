#!/bin/zsh
# A1 (Oct 9): media predictions for the 59 feeling/vibe no-cue memories not already in the 300 sample. No-history condition only
# (the scan compares media vs text with nothing about the person). Appends to the existing runs so all loaders see one dataset.
cd "$(dirname "$0")"
PIDS=($(python3 -c "import json;print(' '.join(json.load(open('../analysis/feeling_nocue_pids.json'))['new_for_media']))"))
L=../runs/feeling_media_20261009.log
python3 run_predict.py --run main_media_cheap --pids $PIDS --models google/gemini-3.8-flash qwen/qwen3.8-omni-flash xiaomi/mimo-v2.5 xiaomi/mimo-v2.6-flash --modalities audio text_audio text_audio_video --infos none --conc 16 --budget 8 >> $L 2>&1
python3 run_predict.py --run main_media_pro --pids $PIDS --models google/gemini-3.1-pro-preview xiaomi/mimo-v2.6-pro-ultraspeed --modalities audio text_audio text_audio_video --infos none --conc 12 --budget 20 >> $L 2>&1
python3 run_predict.py --run main_text_pro --pids $PIDS --models google/gemini-3.1-pro-preview xiaomi/mimo-v2.6-pro-ultraspeed --modalities text --infos none --conc 12 --budget 4 >> $L 2>&1
for r in main_media_cheap main_media_pro main_text_pro; do python3 judge.py --run $r --conc 24 >> $L 2>&1; done
echo DONE >> $L
