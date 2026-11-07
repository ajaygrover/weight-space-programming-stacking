#!/bin/bash
# usage: lane.sh MODEL PORT NP BENCHES  -- one model on its own llama-server (own CPU sampling thread)
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
B=${4:-mmlupro200,gsm8k,math500rest}
echo "=== $(date +%T) START lane $1 ($3 slots): $B"
$PY -I -u scripts/run_llama_eval.py $1 --benches $B --np $3 --port $2 --full-penalty 2>&1 | grep -a --line-buffered -E "serving|done,|_fullpen [a-z0-9]+ \{|refusing|Error|error|skipped" || echo "!!! FAILED lane $1"
echo "=== $(date +%T) LANE DONE $1 ($B)"
