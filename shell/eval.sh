export CUDA_VISIBLE_DEVICES=0
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libgomp.so.1

CONFIG_PATH=${1:-eval_config.json}

LOG_BASENAME=$(python - "$CONFIG_PATH" <<'PY'
import json
import re
import sys

config_path = sys.argv[1]
with open(config_path, "r") as f:
    cfg = json.load(f)

dataset = cfg.get("dataset", "dataset")
protocol = cfg.get("mode", "cand")

def clean(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"

print(f"{clean(dataset)}_{clean(protocol)}")
PY
)

mkdir -p logs/eval
LOG_PATH="logs/eval/${LOG_BASENAME}.log"

nohup python -u eval.py --config "$CONFIG_PATH" --log_file "$LOG_PATH" > /dev/null 2>&1 &
