from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes


CONTEXT_KEYWORDS = {
    "coding": [
        "visual studio code",
        "vscode",
        "pycharm",
        "intellij",
        "terminal",
        "powershell",
        "cmd",
        "github",
        "stackoverflow",
        "stack overflow",
    ],
    "document": [
        "word",
        "hwp",
        "한글",
        "notion",
        "google docs",
        "powerpoint",
        "excel",
    ],
    "communication": [
        "gmail",
        "outlook",
        "slack",
        "teams",
        "discord",
        "kakaotalk",
        "messenger",
    ],
}

VALID_CONTEXTS = {"coding", "document", "communication", "unknown"}


def classify_window_title(title: str | None) -> str:
    text = (title or "").lower()
    for context, keywords in CONTEXT_KEYWORDS.items():
        if any(keyword.lower() in text for keyword in keywords):
            return context
    return "unknown"


def normalize_task_context(context: str | None) -> str:
    context = (context or "unknown").strip().lower()
    return context if context in VALID_CONTEXTS else "unknown"


def get_active_window_title() -> str:
    """Return the active window title for classification only. Do not persist it."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        return buffer.value
    except Exception:
        return ""


def detect_task_context() -> str:
    return classify_window_title(get_active_window_title())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()
    if args.title is None:
        print(detect_task_context())
    else:
        print(classify_window_title(args.title))


if __name__ == "__main__":
    main()
