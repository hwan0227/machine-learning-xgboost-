from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from baseline_manager import add_baseline_ratios, load_baseline
from task_context import normalize_task_context


KEY_PRESS_COLS = ["Press_Time", "Time"]
MOUSE_TIME_COLS = ["Time", "Press_Time"]


def categorize_key(key: object) -> str:
    text = str(key).strip().strip('"')
    low = text.lower()
    if "backspace" in low:
        return "backspace"
    if "delete" in low:
        return "delete"
    if "space" in low:
        return "space"
    if "enter" in low or "return" in low:
        return "enter"
    if any(x in low for x in ["shift", "ctrl", "control", "alt", "cmd", "win"]):
        return "modifier"
    if any(x in low for x in ["left", "right", "up", "down", "home", "end", "page"]):
        return "navigation"
    if len(text.strip("'")) == 1:
        char = text.strip("'")
        if char.isalpha():
            return "alpha"
        if char.isdigit():
            return "digit"
        return "punctuation"
    return "other"


def _read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame()


def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _timestamp_series(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    col = _first_existing_col(df, candidates)
    if col is None:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(df[col], errors="coerce").dropna().sort_values()


def _scale_count(value: float, factor: float) -> float:
    return float(max(0.0, min(10.0, value * factor)))


def _max_events_in_window(times: list[pd.Timestamp], window_sec: float) -> int:
    if not times:
        return 0
    best = 1
    start = 0
    for end, current in enumerate(times):
        while (current - times[start]).total_seconds() > window_sec:
            start += 1
        best = max(best, end - start + 1)
    return best


def _session_meta(session_dir: Path) -> dict:
    path = session_dir / "session_meta.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def extract_key_features(key_df: pd.DataFrame) -> dict:
    feats = {
        "key_count": 0,
        "avg_key_interval": 0.0,
        "std_key_interval": 0.0,
        "backspace_ratio": 0.0,
        "delete_ratio": 0.0,
        "correction_ratio": 0.0,
        "pause_ratio": 0.0,
        "burst_after_pause": 0.0,
        "repeat_key_ratio": 0.0,
        "max_idle_time": 0.0,
        "post_idle_burst_score": 0.0,
        "backspace_burst_score": 0.0,
        "post_backspace_fast_typing_score": 0.0,
        "correction_loop_score": 0.0,
        "correction_urgency_index": 0.0,
    }
    if key_df.empty:
        return feats

    key_df = key_df.copy()
    key_df.columns = key_df.columns.astype(str).str.strip().str.replace("Relase_Time", "Release_Time")
    time_col = _first_existing_col(key_df, KEY_PRESS_COLS)
    if time_col is None:
        return feats

    if "Key_Category" in key_df.columns:
        categories = key_df["Key_Category"].astype(str).str.lower()
    elif "Key" in key_df.columns:
        categories = key_df["Key"].map(categorize_key)
    else:
        categories = pd.Series(["other"] * len(key_df))

    key_df["Time"] = pd.to_datetime(key_df[time_col], errors="coerce")
    key_df["category"] = categories
    key_df = key_df.dropna(subset=["Time"]).sort_values("Time")
    if key_df.empty:
        return feats

    feats["key_count"] = int(len(key_df))
    intervals = key_df["Time"].diff().dt.total_seconds().dropna()
    if not intervals.empty:
        feats["avg_key_interval"] = float(intervals.mean())
        feats["std_key_interval"] = float(intervals.std(ddof=0)) if len(intervals) > 1 else 0.0
        feats["pause_ratio"] = float((intervals >= 2.0).mean())
        feats["max_idle_time"] = float(intervals.max())

    key_count = max(feats["key_count"], 1)
    is_backspace = key_df["category"].eq("backspace")
    is_delete = key_df["category"].eq("delete")
    is_correction = is_backspace | is_delete
    feats["backspace_ratio"] = float(is_backspace.sum()) / key_count
    feats["delete_ratio"] = float(is_delete.sum()) / key_count
    feats["correction_ratio"] = float(is_correction.sum()) / key_count
    feats["repeat_key_ratio"] = float(key_df["category"].eq(key_df["category"].shift()).sum()) / key_count

    times = key_df["Time"].tolist()
    backspace_times = key_df.loc[is_backspace, "Time"].tolist()
    feats["backspace_burst_score"] = _scale_count(_max_events_in_window(backspace_times, 2.0), 2.0)

    burst_counts = []
    post_backspace_counts = []
    correction_loops = 0
    categories_list = key_df["category"].tolist()
    for idx, row in enumerate(key_df.itertuples(index=False)):
        current_time = getattr(row, "Time")
        if idx > 0:
            idle = (current_time - times[idx - 1]).total_seconds()
            if idle >= 2.0:
                burst_counts.append(sum(0 <= (t - current_time).total_seconds() <= 5.0 for t in times[idx:]))
        if categories_list[idx] == "backspace":
            post_count = 0
            saw_input = False
            saw_second_backspace = False
            for j in range(idx + 1, len(times)):
                delta = (times[j] - current_time).total_seconds()
                if delta > 5.0:
                    break
                if categories_list[j] != "backspace":
                    saw_input = True
                    if delta <= 3.0:
                        post_count += 1
                elif saw_input:
                    saw_second_backspace = True
            post_backspace_counts.append(post_count)
            if saw_input and saw_second_backspace:
                correction_loops += 1

    feats["burst_after_pause"] = float(max(burst_counts) if burst_counts else 0.0)
    feats["post_idle_burst_score"] = _scale_count(feats["burst_after_pause"], 1.5)
    feats["post_backspace_fast_typing_score"] = _scale_count(max(post_backspace_counts) if post_backspace_counts else 0.0, 1.5)
    feats["correction_loop_score"] = _scale_count(correction_loops, 2.5)
    feats["correction_urgency_index"] = round(
        0.4 * feats["backspace_burst_score"]
        + 0.3 * feats["post_backspace_fast_typing_score"]
        + 0.3 * feats["correction_loop_score"],
        2,
    )
    return feats


def extract_mouse_features(mouse_df: pd.DataFrame) -> dict:
    feats = {
        "repeat_click_count": 0,
        "mouse_distance": 0.0,
        "direction_change_count": 0,
        "mouse_jitter": 0.0,
        "click_interval_std": 0.0,
        "mouse_event_count": 0,
        "mouse_move_count": 0,
        "mouse_click_count": 0,
        "mouse_total_distance": 0.0,
        "mouse_distance_mean": 0.0,
        "mouse_distance_std": 0.0,
        "mouse_speed_mean": 0.0,
        "mouse_speed_std": 0.0,
        "mouse_speed_max": 0.0,
    }
    if mouse_df.empty:
        return feats

    mouse_df = mouse_df.copy()
    mouse_df.columns = mouse_df.columns.astype(str).str.strip()
    required_cols = {"Time", "Event_Type", "X", "Y"}
    if not required_cols.issubset(mouse_df.columns):
        return feats

    mouse_df["Time"] = pd.to_datetime(mouse_df["Time"], errors="coerce")
    mouse_df["X"] = pd.to_numeric(mouse_df["X"], errors="coerce")
    mouse_df["Y"] = pd.to_numeric(mouse_df["Y"], errors="coerce")
    mouse_df = mouse_df.dropna(subset=["Time", "X", "Y"]).sort_values("Time")
    if mouse_df.empty:
        return feats

    feats["mouse_event_count"] = int(len(mouse_df))
    event_series = mouse_df["Event_Type"].astype(str).str.lower()
    move_df = mouse_df[event_series.eq("move")].copy()
    click_df = mouse_df[event_series.str.contains("click")].copy()
    feats["mouse_move_count"] = int(len(move_df))
    feats["mouse_click_count"] = int(len(click_df))

    if len(mouse_df) >= 2:
        dx = mouse_df["X"].diff()
        dy = mouse_df["Y"].diff()
        dt = mouse_df["Time"].diff().dt.total_seconds()
        dist = np.sqrt(dx**2 + dy**2)
        valid = (dt > 0) & dist.notna()
        dist_valid = dist[valid]
        dt_valid = dt[valid]
        speed = dist_valid / dt_valid
        if not dist_valid.empty:
            feats["mouse_distance"] = float(dist_valid.sum())
            feats["mouse_total_distance"] = feats["mouse_distance"]
            feats["mouse_distance_mean"] = float(dist_valid.mean())
            feats["mouse_distance_std"] = float(dist_valid.std(ddof=0)) if len(dist_valid) > 1 else 0.0
        if not speed.empty:
            feats["mouse_speed_mean"] = float(speed.mean())
            feats["mouse_speed_std"] = float(speed.std(ddof=0)) if len(speed) > 1 else 0.0
            feats["mouse_speed_max"] = float(speed.max())

    if len(move_df) >= 3:
        vectors = np.column_stack([move_df["X"].diff().fillna(0), move_df["Y"].diff().fillna(0)])
        angles = np.arctan2(vectors[:, 1], vectors[:, 0])
        angle_diff = np.abs(np.diff(angles))
        angle_diff = np.minimum(angle_diff, 2 * math.pi - angle_diff)
        direction_changes = int((angle_diff > (math.pi / 2)).sum())
        feats["direction_change_count"] = direction_changes
        feats["mouse_jitter"] = _scale_count(direction_changes / max(len(move_df), 1), 20.0)

    if len(click_df) >= 2:
        click_intervals = click_df["Time"].diff().dt.total_seconds().dropna()
        feats["click_interval_std"] = float(click_intervals.std(ddof=0)) if len(click_intervals) > 1 else 0.0
        close_time = click_intervals <= 0.5
        dx = click_df["X"].diff().abs().fillna(9999)
        dy = click_df["Y"].diff().abs().fillna(9999)
        close_pos = (dx <= 8) & (dy <= 8)
        feats["repeat_click_count"] = int((close_time & close_pos.iloc[1:]).sum())

    return feats


def extract_activity_idle_time(key_df: pd.DataFrame, mouse_df: pd.DataFrame, duration_sec: int) -> float:
    key_times = _timestamp_series(key_df, KEY_PRESS_COLS)
    mouse_times = _timestamp_series(mouse_df, MOUSE_TIME_COLS)
    all_times = pd.concat([key_times, mouse_times]).sort_values()
    if all_times.empty:
        return float(duration_sec)
    if len(all_times) == 1:
        return float(duration_sec)
    return float(all_times.diff().dt.total_seconds().dropna().max())


def build_feature_dict(
    session_dir: str | Path,
    duration_sec: int = 300,
    task_context: str | None = None,
    user_id: str | None = None,
    baseline_root: str | Path = "data/personal",
) -> dict:
    session_dir = Path(session_dir)
    meta = _session_meta(session_dir)
    task_context = normalize_task_context(task_context or meta.get("task_context"))
    user_id = user_id or meta.get("user_id")

    key_df = _read_tsv(session_dir / "keystrokes.tsv")
    mouse_df = _read_tsv(session_dir / "mousedata.tsv")

    features = {
        "task_context": task_context,
        "window_seconds": duration_sec,
        "daylight_mode": -1,
    }
    features.update(extract_key_features(key_df))
    features.update(extract_mouse_features(mouse_df))
    features["activity_idle_time"] = extract_activity_idle_time(key_df, mouse_df, duration_sec)

    # Backward-compatible aliases for models trained before the context-aware pipeline.
    features["key_event_count"] = features["key_count"]
    features["key_unique_event_count"] = 0
    features["mouse_total_distance"] = features["mouse_distance"]

    baseline = load_baseline(user_id, baseline_root)
    features = add_baseline_ratios(features, baseline, task_context)
    features["baseline_quality"] = baseline.get("quality", "missing")
    return features


def build_feature_row(
    session_dir: str | Path,
    duration_sec: int = 300,
    task_context: str | None = None,
    user_id: str | None = None,
    baseline_root: str | Path = "data/personal",
) -> pd.DataFrame:
    return pd.DataFrame([build_feature_dict(session_dir, duration_sec, task_context, user_id, baseline_root)])
