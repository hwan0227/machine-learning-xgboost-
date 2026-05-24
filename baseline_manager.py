from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Iterable


BASELINE_FEATURES = [
    "key_count",
    "avg_key_interval",
    "std_key_interval",
    "backspace_ratio",
    "delete_ratio",
    "correction_ratio",
    "pause_ratio",
    "burst_after_pause",
    "repeat_key_ratio",
    "max_idle_time",
    "post_idle_burst_score",
    "backspace_burst_score",
    "post_backspace_fast_typing_score",
    "correction_loop_score",
    "correction_urgency_index",
    "repeat_click_count",
    "mouse_distance",
    "direction_change_count",
    "mouse_jitter",
    "click_interval_std",
    "activity_idle_time",
]

DEFAULT_BASELINE_VALUE = 1.0


def baseline_path(user_id: str, root: str | Path = "data/personal") -> Path:
    return Path(root) / str(user_id) / "baseline.json"


def default_baseline() -> dict:
    defaults = {feature: DEFAULT_BASELINE_VALUE for feature in BASELINE_FEATURES}
    return {
        "version": 1,
        "updated_at": None,
        "quality": "missing",
        "global": defaults.copy(),
        "contexts": {
            "coding": defaults.copy(),
            "document": defaults.copy(),
            "communication": defaults.copy(),
            "unknown": defaults.copy(),
        },
    }


def load_baseline(user_id: str | None, root: str | Path = "data/personal") -> dict:
    if not user_id:
        return default_baseline()
    path = baseline_path(user_id, root)
    if not path.exists():
        return default_baseline()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default_baseline()

    baseline = default_baseline()
    baseline.update({k: v for k, v in data.items() if k not in {"global", "contexts"}})
    baseline["quality"] = data.get("quality", "available")
    if isinstance(data.get("global"), dict):
        baseline["global"].update(data["global"])
    if isinstance(data.get("contexts"), dict):
        for context, values in data["contexts"].items():
            if context not in baseline["contexts"]:
                baseline["contexts"][context] = baseline["global"].copy()
            if isinstance(values, dict):
                baseline["contexts"][context].update(values)
    return baseline


def save_baseline(user_id: str, baseline: dict, root: str | Path = "data/personal") -> Path:
    path = baseline_path(user_id, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = dict(baseline)
    baseline["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _safe_number(value: object) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if number != number:
        return None
    return number


def build_baseline(rows: Iterable[dict]) -> dict:
    baseline = default_baseline()
    rows = list(rows)
    if not rows:
        return baseline
    baseline["quality"] = "low" if len(rows) < 6 else "available"

    for feature in BASELINE_FEATURES:
        values = [_safe_number(row.get(feature)) for row in rows]
        values = [max(v, 0.0) for v in values if v is not None]
        if values:
            baseline["global"][feature] = max(float(median(values)), DEFAULT_BASELINE_VALUE)

    for context in baseline["contexts"]:
        context_rows = [row for row in rows if row.get("task_context") == context]
        for feature in BASELINE_FEATURES:
            values = [_safe_number(row.get(feature)) for row in context_rows]
            values = [max(v, 0.0) for v in values if v is not None]
            if values:
                baseline["contexts"][context][feature] = max(float(median(values)), DEFAULT_BASELINE_VALUE)
            else:
                baseline["contexts"][context][feature] = baseline["global"][feature]

    return baseline


def get_baseline_value(baseline: dict, task_context: str, feature: str, fallback: float = DEFAULT_BASELINE_VALUE) -> float:
    context_values = baseline.get("contexts", {}).get(task_context, {})
    value = context_values.get(feature, baseline.get("global", {}).get(feature, fallback))
    try:
        value = float(value)
    except Exception:
        value = fallback
    return value if value > 0 else fallback


def add_baseline_ratios(features: dict, baseline: dict, task_context: str) -> dict:
    out = dict(features)
    for feature in BASELINE_FEATURES:
        current = _safe_number(features.get(feature))
        if current is None:
            current = 0.0
        base = get_baseline_value(baseline, task_context, feature)
        out[f"{feature}_baseline_value"] = base
        out[f"{feature}_baseline_ratio"] = float(current) / base if base > 0 else 1.0
    return out
