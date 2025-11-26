import json
import os

SESSION_FILE = "pushup_sessions.json"


def load_sessions():
    """Load past pushup sessions from file."""
    if not os.path.exists(SESSION_FILE):
        return []
    try:
        with open(SESSION_FILE, "r") as f:
            return json.load(f)
    except:
        return []


def save_session(pushups_done):
    """Save a new pushup session with total pushups."""
    sessions = load_sessions()
    sessions.append({"pushups": pushups_done})
    with open(SESSION_FILE, "w") as f:
        json.dump(sessions, f, indent=4)


def get_total_pushups():
    """Return total pushups across all sessions."""
    sessions = load_sessions()
    return sum(s["pushups"] for s in sessions)
