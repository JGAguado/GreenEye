import json
import os
from datetime import datetime

LOG_FILE = "logs/requests.jsonl"


def log_request(record: dict):
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    record["timestamp"] = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def get_logs(limit=100):
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()[-limit:]
    return [json.loads(l) for l in lines]
