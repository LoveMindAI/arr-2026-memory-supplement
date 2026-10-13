#!/bin/zsh
# One retry pass for every prediction run (run_predict resumes: only non-ok calls are redone), then judge each run once more.
cd <LOCAL>/Documents/PROJECT/NAACL/MemoryProject/scripts
ALL=$(python3 -c "import json;print(' '.join(json.load(open('../runs/MAIN_SETS.json'))['all']))")
PRO=$(python3 -c "import json;print(' '.join(json.load(open('../runs/MAIN_SETS.json'))['pro300']))")
CHEAP=(deepseek/deepseek-v4.1-flash z-ai/glm-5.3-flashx moonshotai/kimi-k2-0905 xiaomi/mimo-v2.5-pro xiaomi/mimo-v2.6-flash qwen/qwen3.8-omni-flash google/gemini-3.8-flash xiaomi/mimo-v2.5 google/gemma-4-31b-it meta/muse-glimmer-30b)
PROM=(google/gemini-3.1-pro-preview openai/gpt-5.6-sol xiaomi/mimo-v2.6-pro-ultraspeed)
python3 run_predict.py --run cards_narr_cheap --pids ${=ALL} --models $CHEAP --modalities text --infos card_narr card_narr_history --budget 5 --conc 16 > ../runs/retry_narr_cheap.log 2>&1
python3 run_predict.py --run cards_person_cheap --pids ${=ALL} --models $CHEAP --modalities text --infos card_person card_person_history --budget 5 --conc 16 > ../runs/retry_person_cheap.log 2>&1
python3 run_predict.py --run main_text_cheap --pids ${=ALL} --models $CHEAP --modalities text --infos none history --budget 5 --conc 16 > ../runs/retry_main_text_cheap.log 2>&1
python3 run_predict.py --run main_media_cheap --pids ${=PRO} --models xiaomi/mimo-v2.5 xiaomi/mimo-v2.6-flash qwen/qwen3.8-omni-flash google/gemini-3.8-flash --modalities audio text_audio text_audio_video --infos none history --budget 15 --conc 8 > ../runs/retry_media_cheap.log 2>&1
python3 run_predict.py --run main_media_pro --pids ${=PRO} --models google/gemini-3.1-pro-preview xiaomi/mimo-v2.6-pro-ultraspeed --modalities audio text_audio text_audio_video --infos none history --budget 20 --conc 6 > ../runs/retry_media_pro.log 2>&1
export JUDGE_MAX_TOKENS=12000
for r in cards_narr_cheap cards_person_cheap main_text_cheap main_media_cheap main_media_pro cards_text_cheap cards_text_pro cards_narr_pro cards_person_pro main_text_pro; do python3 -u judge.py --run $r --conc 64 2>&1 | grep -E "to make|cost"; done
echo "RETRY_ALL DONE $(date '+%H:%M')"
