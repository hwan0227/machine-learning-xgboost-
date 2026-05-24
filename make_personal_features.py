from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from baseline_manager import build_baseline, save_baseline
from feature_extraction import build_feature_dict


def read_label(session_dir: Path) -> float:
    label_path = session_dir / "label.txt"
    if not label_path.exists():
        raise FileNotFoundError(f"label.txt not found: {session_dir}")
    label = float(label_path.read_text(encoding="utf-8").strip())
    if label not in {0.0, 1.0}:
        raise ValueError("training label must be 0 or 1; 0.5 focus labels are excluded")
    return label


def process_session(session_dir: Path, user_id: str) -> dict:
    row = {
        "user_id": user_id,
        "window_start": np.nan,
        "window_end": np.nan,
        "Fatigue_Val": np.nan,
        "PAM_Val": np.nan,
        "Energy_Val": np.nan,
        "Pleasant_Val": np.nan,
    }
    label = read_label(session_dir)
    row["Stress_Val_raw"] = label
    row["Stress_Val"] = label
    row.update(build_feature_dict(session_dir, user_id=user_id))
    return row


def main() -> None:
    root = Path("data/personal")
    if not root.exists():
        raise FileNotFoundError("data/personal 폴더가 없습니다.")

    rows = []
    baseline_rows_by_user: dict[str, list[dict]] = {}

    for user_dir in root.iterdir():
        if not user_dir.is_dir():
            continue

        for session_dir in user_dir.iterdir():
            if not session_dir.is_dir():
                continue
            try:
                row = process_session(session_dir, user_dir.name)
                rows.append(row)
                if not session_dir.name.lower().startswith("test_"):
                    baseline_rows_by_user.setdefault(user_dir.name, []).append(row)
            except Exception as e:
                print(f"[WARN] skipped: {session_dir} -> {e}")

    if not rows:
        raise ValueError("처리할 개인 세션 데이터가 없습니다.")

    for user_id, user_rows in baseline_rows_by_user.items():
        baseline = build_baseline(user_rows)
        path = save_baseline(user_id, baseline, root)
        print(f"[DONE] baseline saved: {path}")

    # Baseline files were just updated, so rebuild rows with fresh baseline ratios.
    rebuilt_rows = []
    for user_id in sorted({row["user_id"] for row in rows}):
        session_base = root / user_id
        for session_dir in session_base.iterdir():
            if session_dir.is_dir() and (session_dir / "label.txt").exists():
                try:
                    rebuilt = process_session(session_dir, user_id)
                    rebuilt_rows.append(rebuilt)
                except Exception as e:
                    print(f"[WARN] skipped on rebuild: {session_dir} -> {e}")
    if rebuilt_rows:
        rows = rebuilt_rows

    df = pd.DataFrame(rows)
    out_path = Path("data/personal_features.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[DONE] saved: {out_path}")
    print(df.head())
    print(df.shape)
    print(df["Stress_Val"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
