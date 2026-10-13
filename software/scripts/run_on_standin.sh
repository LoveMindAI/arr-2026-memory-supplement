#!/usr/bin/env bash
# Run the paper's method end to end on the public synthetic stand-in, with no access to PARSEL:
# export -> predict (no information, and with the person's other memories) -> three judges -> codebook annotation by two LLMs
# -> person-specific fit (analysis/ladder_v351.json, the measure behind Table 1). The numbers are synthetic and are not the paper's.
# Needs OPENROUTER_API_KEY. Two sessions cost well under $1 with the models below. Every step is resumable.
# Usage: bash scripts/run_on_standin.sh PATH/TO/data/twin_public [S01,S02]
set -euo pipefail
TP="$1"; SESS="${2:-S01,S02}"
HERE="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(cd "$HERE/.." && pwd)"; OUT="$ROOT/standin_data"
# Outputs (runs/, analysis/, standin_data/) are not namespaced by selection: one selection per copy of software/.
MARK="$ROOT/.standin_selection"; TAG="$SESS"
if [ -f "$MARK" ] && [ "$(cat "$MARK")" != "$TAG" ]; then
  echo "This copy already holds outputs for sessions $(cat "$MARK"). For a different selection, use a fresh copy of software/." >&2; exit 2
fi
python3 "$ROOT/twin_public/export_for_pipeline.py" --src "$TP" --sessions "$SESS" --out "$OUT"
echo "$TAG" > "$MARK"
echo "Cost note: --budget caps the prediction step only; judging and annotation are uncapped (two sessions cost about \$0.65 in total in our October 2026 run)."
export MEMORY_DATA="$OUT"; cd "$HERE"
PIDS=$(python3 -c "import csv,sys; print(' '.join(r['perspectiveID'] for r in csv.DictReader(open(sys.argv[1]))))" "$OUT/conv.csv")
python3 run_predict.py --run main_text_cheap --pids $PIDS --models deepseek/deepseek-v4.1-flash --infos none history --budget 1.0
python3 judge.py --run main_text_cheap --judges openai/gpt-6-luna deepseek/deepseek-v4.1-flash anthropic/claude-haiku-5.5
python3 code_v35.py --set human --coders claude-haiku-5.5,gpt-6-luna
python3 code_v35.py --set main_text_cheap:none --coders claude-haiku-5.5,gpt-6-luna
python3 code_v35.py --set main_text_cheap:history --coders claude-haiku-5.5,gpt-6-luna
python3 ladder_v351.py
echo "done: predictions in $ROOT/runs/main_text_cheap, person-specific fit in $ROOT/analysis/ladder_v351.json"
