#!/bin/zsh
cd "$(dirname "$0")"
L=../runs/position_swap.log
python3 position_swap.py --stage predict >> $L 2>&1
python3 position_swap.py --stage predict >> $L 2>&1
python3 position_swap.py --stage locate >> $L 2>&1
python3 judge.py --run position_swap --conc 24 >> $L 2>&1
echo DONE >> $L
