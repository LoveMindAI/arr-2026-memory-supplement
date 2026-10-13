#!/bin/zsh
# judge_loop.sh RUN CONC: judge RUN repeatedly until no predictor for RUN is alive, then one final pass.
cd <LOCAL>/Documents/PROJECT/NAACL/MemoryProject/scripts
R=$1; C=${2:-64}; export JUDGE_MAX_TOKENS=12000
while true; do
  alive=$(pgrep -f "run_predict.py --run $R " | wc -l | tr -d ' ')
  echo "$(date '+%H:%M') $R alive=$alive"
  python3 -u judge.py --run $R --conc $C 2>&1 | grep -E "judgements to make|judge cost|Traceback|Error"
  [ "$alive" = "0" ] && break
  sleep 60
done
echo "$(date '+%H:%M') LOOP DONE $R"
