import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def safe_read_tsv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="utf-8-sig")
    df.columns = df.columns.astype(str).str.strip()

    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    print(f"\n[DEBUG] file: {path}")
    print("[DEBUG] columns:", df.columns.tolist())

    return df


def to_datetime_column(df: pd.DataFrame, col: str = "Time") -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip()

    if col not in df.columns:
        candidates = [c for c in df.columns if c.lower() == col.lower()]
        if candidates:
            col = candidates[0]
        else:
            raise KeyError(f"'{col}' column not found. available columns = {df.columns.tolist()}")

    df[col] = pd.to_datetime(df[col], errors="coerce")
    df = df.dropna(subset=[col])

    if col != "Time":
        df = df.rename(columns={col: "Time"})

    return df


def map_stress_label(val) -> int:
    if pd.isna(val):
        return -1

    val = str(val).strip().lower()
    val = val.replace(" ", "_")

    mapping = {
        "f_good": 0,
        "f_great": 0,
        "neutral": 0,

        "s_stressed": 1,
        "v_stressed": 1,
    }

    return mapping.get(val, -1)

def encode_daylight(val: str) -> int:
    if pd.isna(val):
        return -1
    val = str(val).strip().lower()
    mapping = {
        "morning": 0,
        "afternoon": 1,
        "evening": 2,
        "night": 3,
    }
    return mapping.get(val, -1)


def extract_mouse_features(mouse_df: pd.DataFrame, speed_df: pd.DataFrame) -> dict:
    feats = {
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

    if not mouse_df.empty:
        feats["mouse_event_count"] = len(mouse_df)

        if "Event_Type" in mouse_df.columns:
            event_series = mouse_df["Event_Type"].astype(str)
            feats["mouse_move_count"] = (event_series.str.lower() == "move").sum()
            feats["mouse_click_count"] = event_series.str.lower().str.contains("click").sum()

        if {"X", "Y"}.issubset(mouse_df.columns):
            temp = mouse_df.copy()
            temp["X"] = pd.to_numeric(temp["X"], errors="coerce")
            temp["Y"] = pd.to_numeric(temp["Y"], errors="coerce")
            temp = temp.dropna(subset=["X", "Y"]).sort_values("Time")

            if len(temp) >= 2:
                dx = temp["X"].diff()
                dy = temp["Y"].diff()
                dist = np.sqrt(dx ** 2 + dy ** 2).dropna()

                if len(dist) > 0:
                    feats["mouse_total_distance"] = float(dist.sum())
                    feats["mouse_distance_mean"] = float(dist.mean())
                    feats["mouse_distance_std"] = float(dist.std(ddof=0)) if len(dist) > 1 else 0.0

    if not speed_df.empty and "Speed(ms)" in speed_df.columns:
        s = pd.to_numeric(speed_df["Speed(ms)"], errors="coerce").dropna()
        if len(s) > 0:
            feats["mouse_speed_mean"] = float(s.mean())
            feats["mouse_speed_std"] = float(s.std(ddof=0)) if len(s) > 1 else 0.0
            feats["mouse_speed_max"] = float(s.max())

    return feats


def extract_key_features(key_df: pd.DataFrame) -> dict:
    feats = {
        "key_event_count": 0,
        "key_unique_event_count": 0,
    }

    if key_df.empty:
        return feats

    feats["key_event_count"] = len(key_df)

    possible_cols = ["Event_Type", "Key", "Pressed_Key", "Key_Name"]
    found_col = None
    for c in possible_cols:
        if c in key_df.columns:
            found_col = c
            break

    if found_col is not None:
        feats["key_unique_event_count"] = key_df[found_col].astype(str).nunique()

    return feats


def process_user_folder(user_dir: Path, window_minutes: int) -> pd.DataFrame:
    keystrokes_path = user_dir / "keystrokes.tsv"
    mousedata_path = user_dir / "mousedata.tsv"
    mouse_speed_path = user_dir / "mouse_mov_speeds.tsv"
    usercondition_path = user_dir / "usercondition.tsv"

    key_df = safe_read_tsv(keystrokes_path)
    mouse_df = safe_read_tsv(mousedata_path)
    speed_df = safe_read_tsv(mouse_speed_path)
    cond_df = safe_read_tsv(usercondition_path)

    # keystrokes 전용 처리
    key_df = key_df.copy()

    # 컬럼명 오타 보정
    key_df.columns = key_df.columns.str.replace("Relase_Time", "Release_Time")

    # datetime 변환
    key_df["Press_Time"] = pd.to_datetime(key_df["Press_Time"], errors="coerce")
    key_df["Release_Time"] = pd.to_datetime(key_df["Release_Time"], errors="coerce")

    # NaN 제거
    key_df = key_df.dropna(subset=["Press_Time", "Release_Time"])

    # 기준 컬럼을 Time으로 통일 (핵심)
    key_df = key_df.rename(columns={"Press_Time": "Time"})
    mouse_df = to_datetime_column(mouse_df, "Time")
    speed_df = to_datetime_column(speed_df, "Time")
    cond_df = to_datetime_column(cond_df, "Time")

    rows = []
    window_delta = pd.Timedelta(minutes=window_minutes)

    user_id = user_dir.name

    for _, cond_row in cond_df.iterrows():
        end_time = cond_row["Time"]
        start_time = end_time - window_delta

        key_win = key_df[(key_df["Time"] >= start_time) & (key_df["Time"] <= end_time)].copy()
        mouse_win = mouse_df[(mouse_df["Time"] >= start_time) & (mouse_df["Time"] <= end_time)].copy()
        speed_win = speed_df[(speed_df["Time"] >= start_time) & (speed_df["Time"] <= end_time)].copy()
        print(
            "[DEBUG] raw Stress_Val =",
            cond_row.get("Stress_Val", np.nan),
            "-> mapped =",
            map_stress_label(cond_row.get("Stress_Val", np.nan))
        )

        row = {
            "user_id": user_id,
            "window_start": start_time,
            "window_end": end_time,
            "window_seconds": int(window_delta.total_seconds()),
            "daylight_mode": encode_daylight(cond_row.get("Daylight", np.nan)),
            "Fatigue_Val": cond_row.get("Fatigue_Val", np.nan),
            "PAM_Val": cond_row.get("PAM_Val", np.nan),
            "Stress_Val_raw": cond_row.get("Stress_Val", np.nan),
            "Stress_Val": map_stress_label(cond_row.get("Stress_Val", np.nan)),
            "Energy_Val": cond_row.get("Energy_Val", np.nan),
            "Pleasant_Val": cond_row.get("Pleasant_Val", np.nan),
        }

        row.update(extract_key_features(key_win))
        row.update(extract_mouse_features(mouse_win, speed_win))

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="data")
    parser.add_argument("--output_csv", type=str, default="data/kaggle_features.csv")
    parser.add_argument("--window_minutes", type=int, default=5)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    user_dirs = [p for p in input_root.iterdir() if p.is_dir() and p.name.lower().startswith("user")]

    all_dfs = []
    for user_dir in user_dirs:
        print(f"[INFO] processing: {user_dir}")
        user_features = process_user_folder(user_dir, args.window_minutes)
        all_dfs.append(user_features)

    if not all_dfs:
        raise ValueError("No user folders found under input_root.")

    final_df = pd.concat(all_dfs, ignore_index=True)

    final_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] saved: {args.output_csv}")
    print(final_df.head())
    print(final_df.shape)
    print(final_df.columns.tolist())


if __name__ == "__main__":
    main()


