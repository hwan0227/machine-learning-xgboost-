from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from pynput import keyboard, mouse

from feature_extraction import categorize_key
from task_context import detect_task_context, normalize_task_context


keyboard_rows = []
mouse_rows = []
key_press_times = {}


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")


def get_daylight_label() -> str:
    hour = datetime.now().hour
    if 5 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 21:
        return "evening"
    return "night"


def on_key_press(key) -> None:
    key_id = str(key)
    key_press_times[key_id] = now_str()


def on_key_release(key) -> None:
    key_id = str(key)
    press_time = key_press_times.pop(key_id, None)
    if press_time is None:
        return

    keyboard_rows.append(
        {
            "Key_Category": categorize_key(key_id),
            "Press_Time": press_time,
            "Release_Time": now_str(),
            "Daylight": get_daylight_label(),
        }
    )


def on_move(x, y) -> None:
    mouse_rows.append(
        {
            "Time": now_str(),
            "Event_Type": "move",
            "X": x,
            "Y": y,
            "Daylight": get_daylight_label(),
        }
    )


def on_click(x, y, button, pressed) -> None:
    if not pressed:
        return
    mouse_rows.append(
        {
            "Time": now_str(),
            "Event_Type": f"click_{button}",
            "X": x,
            "Y": y,
            "Daylight": get_daylight_label(),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=str, required=True)
    parser.add_argument("--session_name", type=str, required=True)
    parser.add_argument("--duration_sec", type=int, default=300)
    parser.add_argument("--label", type=float, choices=[0, 0.5, 1], required=True)
    parser.add_argument("--output_root", type=str, default="data/personal")
    parser.add_argument("--task_context", type=str, default=None)
    args = parser.parse_args()

    task_context = normalize_task_context(args.task_context) if args.task_context else detect_task_context()
    session_dir = Path(args.output_root) / args.user_id / args.session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] task_context = {task_context}")
    print(f"[INFO] recording for {args.duration_sec} seconds...")
    print("[INFO] start using keyboard/mouse now")

    kb_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click)

    kb_listener.start()
    mouse_listener.start()

    time.sleep(args.duration_sec)

    kb_listener.stop()
    mouse_listener.stop()

    key_df = pd.DataFrame(keyboard_rows)
    mouse_df = pd.DataFrame(mouse_rows)

    key_df.to_csv(session_dir / "keystrokes.tsv", sep="\t", index=False, encoding="utf-8-sig")
    mouse_df.to_csv(session_dir / "mousedata.tsv", sep="\t", index=False, encoding="utf-8-sig")

    (session_dir / "label.txt").write_text(str(args.label), encoding="utf-8")
    (session_dir / "session_meta.json").write_text(
        json.dumps(
            {
                "version": 1,
                "user_id": args.user_id,
                "session_name": args.session_name,
                "duration_sec": args.duration_sec,
                "task_context": task_context,
                "label": args.label,
                "collected_at": datetime.now().isoformat(timespec="seconds"),
                "privacy": "stores key categories and timing only; active window title is not stored",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"[DONE] saved session to: {session_dir}")
    print(f"[INFO] keyboard events: {len(key_df)}")
    print(f"[INFO] mouse events: {len(mouse_df)}")


if __name__ == "__main__":
    main()
