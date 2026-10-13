#!/bin/zsh
# Continues run_v351_full.sh (A1, Oct 8): waits for the human-memory coder (3 coders + tiers) to finish, retries its failed parses,
# then codes all prediction sets with Haiku + Luna only at higher parallelism (no DeepSeek tying up slots).
cd "$(dirname "$0")"
L=../analysis/code_v351_full.log
while pgrep -f "code_v35.py --set human" >/dev/null; do sleep 20; done
python3 code_v35.py --set human --conc 16 >> $L 2>&1
SETS=(main_text_cheap:none main_text_cheap:history cards_text_cheap:card_thin cards_narr_cheap:card_narr)
for m in deepseek gemini gemma mimo qwen3.8; do for i in history_wrong history_prior card_thin_wrong card_narr_wrong; do SETS+=(r3c_$m:$i); done; done
for s in $SETS; do
  echo "== $s (haiku+luna)" >> $L
  python3 code_v35.py --set $s --coders claude-haiku-5.5,gpt-6-luna --conc 200 >> $L 2>&1
  python3 code_v35.py --set $s --coders claude-haiku-5.5,gpt-6-luna --conc 16 >> $L 2>&1
done
echo DONE >> $L
