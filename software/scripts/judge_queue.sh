#!/bin/zsh
# Sweep-judge all runs until prediction jobs finish, then one final pass. Never runs two judges at once.
cd <LOCAL>/Documents/PROJECT/NAACL/MemoryProject/scripts
RUNS=(cards_text_pro cards_text_cheap cards_narr_pro cards_narr_cheap cards_person_pro cards_person_cheap main_media_pro main_media_cheap)
export JUDGE_MAX_TOKENS=12000
while pgrep -f "python3 -u judge.py|python3 judge.py" >/dev/null; do sleep 30; done
final=0
while true; do
  alive=$(pgrep -f run_predict.py | wc -l | tr -d ' ')
  for r in $RUNS; do
    [ -f ../runs/$r/preds.jsonl ] || continue
    echo "$(date '+%H:%M') judge $r (predictors alive: $alive)"
    python3 -u judge.py --run $r --conc 96 2>&1 | grep -E "judgements to make|judge cost|Traceback|Error"
  done
  [ $final = 1 ] && break
  [ "$alive" = "0" ] && final=1
  sleep 120
done
echo "$(date '+%H:%M') JUDGE QUEUE DONE"
