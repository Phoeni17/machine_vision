import json
import os

SESSION_FILE = "exercise_sessions.json"  # single file for all exercises


def load_sessions():
    """Load past sessions from JSON file."""
    if not os.path.exists(SESSION_FILE):
        # initialize empty structure
        return {"pushups": [], "squats": []}
    try:
        with open(SESSION_FILE, "r") as f:
            data = json.load(f)
            if "pushups" not in data:
                data["pushups"] = []
            if "squats" not in data:
                data["squats"] = []
            return data
    except (json.JSONDecodeError, FileNotFoundError):
        return {"pushups": [], "squats": []}


def save_session(count, exercise="pushup"):
    """Save a new session for pushups or squats."""
    data = load_sessions()
    if exercise == "pushup":
        data["pushups"].append(count)
    elif exercise == "squat":
        data["squats"].append(count)
    with open(SESSION_FILE, "w") as f:
        json.dump(data, f, indent=4)


def get_total_pushups():
    """Return total pushups across all sessions."""
    data = load_sessions()
    return sum(data.get("pushups", []))


def get_total_squats():
    """Return total squats across all sessions."""
    data = load_sessions()
    return sum(data.get("squats", []))
