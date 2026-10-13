#!/bin/zsh
cd "$(dirname "$0")"
until grep -q "^DONE" ../runs/norms_history_v2.log; do sleep 30; done
python3 norms_history_analysis.py > ../analysis/norms_history_analysis.log 2>&1
python3 norms_history_specificity.py >> ../analysis/norms_history_analysis.log 2>&1
echo FINISHED >> ../analysis/norms_history_analysis.log
