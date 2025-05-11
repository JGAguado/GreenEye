# FILE: tests/test_logger.py
import json
import os

import pytest

from app.logger import get_logs, log_request

LOG_FILE = "logs/requests.jsonl"


@pytest.fixture(autouse=True)
def cleanup_logs():
    # Cleanup before and after each test
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    yield
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)


def test_log_request_creates_file():
    record = {"message": "Test log"}
    log_request(record)
    assert os.path.exists(LOG_FILE)


def test_log_request_writes_record():
    record = {"message": "Test log"}
    log_request(record)
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
    assert len(lines) == 1
    logged_record = json.loads(lines[0])
    assert logged_record["message"] == "Test log"
    assert "timestamp" in logged_record


def test_get_logs_returns_empty_if_no_file():
    logs = get_logs()
    assert logs == []


def test_get_logs_returns_logged_records():
    record1 = {"message": "Log 1"}
    record2 = {"message": "Log 2"}
    log_request(record1)
    log_request(record2)
    logs = get_logs()
    assert len(logs) == 2
    assert logs[0]["message"] == "Log 1"
    assert logs[1]["message"] == "Log 2"


def test_get_logs_respects_limit():
    for i in range(10):
        log_request({"message": f"Log {i}"})
    logs = get_logs(limit=5)
    assert len(logs) == 5
    assert logs[0]["message"] == "Log 5"
    assert logs[-1]["message"] == "Log 9"
