from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import requests

from task_context import detect_task_context, normalize_task_context


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PERSONAL_DIR = DATA_DIR / "personal"
MODELS_DIR = BASE_DIR / "models"

COLLECT_SCRIPT = BASE_DIR / "collect_behavior_features.py"
MAKE_PERSONAL_SCRIPT = BASE_DIR / "make_personal_features.py"
TRAIN_SCRIPT = BASE_DIR / "train_personalized_model.py"
PREDICT_SCRIPT = BASE_DIR / "predict_single_session.py"

KAGGLE_FEATURES = DATA_DIR / "kaggle_features.csv"
PERSONAL_FEATURES = DATA_DIR / "personal_features.csv"
FINAL_MERGED = DATA_DIR / "final_merged_features.csv"
BEST_MODEL = MODELS_DIR / "best_stress_model.pkl"
PREDICTION_RESULT = BASE_DIR / "prediction_result.json"

TARGET_PER_LABEL = 3
SESSION_SEC = 10
PERSONAL_WEIGHT = 5.0
SERVER_URL = "http://127.0.0.1:5000/api/send_data"

DEPARTMENTS = [
    "1. 인사부",
    "2. 재무부",
    "3. 마케팅부",
    "4. 생산부",
    "5. 연구개발부",
    "6. 구매부",
    "7. IT부",
    "8. 법무부",
    "9. 영업부",
]


def select_department() -> str:
    print("\n===== 부서 선택 =====")
    for dept in DEPARTMENTS:
        print(dept)
    choice = input("부서 번호 입력 (1~9): ").strip()
    if choice.isdigit() and 1 <= int(choice) <= 9:
        return DEPARTMENTS[int(choice) - 1]
    print("[WARN] 잘못 입력해서 기본값 7. IT부로 설정합니다.")
    return "7. IT부"


def make_user_id(dept: str, employee_id: str) -> str:
    dept_clean = dept.split(". ", 1)[1] if ". " in dept else dept
    return f"{dept_clean.replace(' ', '_')}_{employee_id.strip().replace(' ', '_')}"


def run_python(script_path: Path, args: list[str]) -> int:
    cmd = [sys.executable, str(script_path)] + args
    print("\n[RUN]", " ".join(cmd))
    return subprocess.run(cmd, cwd=BASE_DIR).returncode


def run_python_capture(script_path: Path, args: list[str]) -> tuple[int, str]:
    cmd = [sys.executable, str(script_path)] + args
    print("\n[RUN]", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    if output:
        print(output)
    return result.returncode, output


def send_to_server(dept: str, result: dict) -> None:
    payload = {
        "dept": dept,
        "score": float(result["final_score"]),
        "label": float(result["class_value"]),
        "state": result["state"],
        "class_value": result["class_value"],
        "task_context": result["task_context"],
        "confidence": result["confidence"],
        "correction_urgency_index": result["correction_urgency_index"],
    }
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=3)
        if response.status_code == 200:
            print("[DONE] 경영진 대시보드로 결과를 전송했습니다.")
        else:
            print(f"[WARN] 서버 전송 실패: status={response.status_code}")
    except Exception as e:
        print(f"[WARN] 서버 연결 실패: {e}")


def merge_features() -> bool:
    if not KAGGLE_FEATURES.exists():
        print(f"[ERROR] 파일 없음: {KAGGLE_FEATURES}")
        return False
    if not PERSONAL_FEATURES.exists():
        print(f"[ERROR] 파일 없음: {PERSONAL_FEATURES}")
        return False
    kaggle_df = pd.read_csv(KAGGLE_FEATURES, encoding="utf-8-sig")
    personal_df = pd.read_csv(PERSONAL_FEATURES, encoding="utf-8-sig")
    pd.concat([kaggle_df, personal_df], ignore_index=True).to_csv(FINAL_MERGED, index=False, encoding="utf-8-sig")
    print("[DONE] 개인 데이터와 기존 데이터를 병합했습니다.")
    return True


def collect_one_session(user_id: str, session_name: str, duration_sec: int, label: float, task_context: str | None = None) -> bool:
    args = [
        "--user_id", user_id,
        "--session_name", session_name,
        "--duration_sec", str(duration_sec),
        "--label", str(label),
    ]
    if task_context:
        args.extend(["--task_context", normalize_task_context(task_context)])
    return run_python(COLLECT_SCRIPT, args) == 0


def count_sessions(user_id: str) -> tuple[int, int, int]:
    user_dir = PERSONAL_DIR / user_id
    if not user_dir.exists():
        return 0, 0, 0
    zero_count = one_count = total = 0
    for session_dir in user_dir.iterdir():
        label_path = session_dir / "label.txt"
        if not session_dir.is_dir() or session_dir.name.lower().startswith("test_") or not label_path.exists():
            continue
        try:
            label = float(label_path.read_text(encoding="utf-8").strip())
        except Exception:
            continue
        total += 1
        if label == 0:
            zero_count += 1
        elif label == 1:
            one_count += 1
    return total, zero_count, one_count


def next_session_name(user_id: str, prefix: str = "session") -> str:
    user_dir = PERSONAL_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    used_numbers = []
    for session_dir in user_dir.iterdir():
        if session_dir.is_dir() and session_dir.name.startswith(f"{prefix}_"):
            try:
                used_numbers.append(int(session_dir.name.split("_")[1]))
            except Exception:
                pass
    n = 1
    while n in used_numbers:
        n += 1
    return f"{prefix}_{n:02d}"


def ask_binary_label(prompt: str) -> int:
    while True:
        value = input(prompt).strip()
        if value in {"0", "1"}:
            return int(value)
        print("[WARN] 0 또는 1 중 하나만 입력하세요.")


def onboarding_collect(user_id: str) -> None:
    print("\n[INFO] 초기/추가 개인화 데이터 수집을 시작합니다.")
    _, zero_count, one_count = count_sessions(user_id)
    print(f"[STATUS] 정상 {zero_count}/{TARGET_PER_LABEL}, 스트레스 가능성 높음 {one_count}/{TARGET_PER_LABEL}")
    if zero_count >= TARGET_PER_LABEL and one_count >= TARGET_PER_LABEL:
        answer = input("초기 개인화 데이터가 이미 충분합니다. 그래도 추가 데이터를 1개 더 수집하시겠습니까? (y/n): ").strip().lower()
        if answer != "y":
            print("[INFO] 추가 데이터 수집을 취소했습니다.")
            return
    session_name = next_session_name(user_id, prefix="session")
    task_context = detect_task_context()
    print(f"[STEP] {session_name}, context={task_context}, {SESSION_SEC}초 측정")
    if not collect_one_session(user_id, session_name, SESSION_SEC, 0, task_context):
        print("[ERROR] 세션 수집 실패")
        return
    label = ask_binary_label("방금 수집한 상태를 선택하세요 (0=정상 / 비스트레스, 1=스트레스 가능성 높음): ")
    (PERSONAL_DIR / user_id / session_name / "label.txt").write_text(str(label), encoding="utf-8")
    print(f"[DONE] {session_name} 저장 완료 -> label={label}")


def train_personal_model(user_id: str) -> None:
    print("\n[STEP] 개인 행동 데이터 feature 생성")
    if run_python(MAKE_PERSONAL_SCRIPT, []) != 0:
        print("[ERROR] 개인 feature 생성 실패")
        return
    print("\n[STEP] 기존 데이터와 개인 데이터 병합")
    if not merge_features():
        return
    print("\n[STEP] 개인 데이터 가중치를 적용해 모델 학습")
    code = run_python(
        TRAIN_SCRIPT,
        [
            "--input_csv", str(FINAL_MERGED),
            "--personal_user_ids", user_id,
            "--personal_weight", str(PERSONAL_WEIGHT),
        ],
    )
    print("[DONE] 개인화 모델 학습 완료" if code == 0 else "[ERROR] 모델 학습 실패")


def ask_self_report() -> dict[str, float]:
    questions = [
        ("tension", "현재 긴장감이 어느 정도인가요? 0~10: "),
        ("pressure_hurry", "현재 부담감/조급함이 어느 정도인가요? 0~10: "),
        ("control", "현재 통제감/여유가 어느 정도인가요? 0~10: "),
    ]
    answers = {}
    for key, prompt in questions:
        while True:
            try:
                value = float(input(prompt).strip())
            except Exception:
                value = -1
            if 0 <= value <= 10:
                answers[key] = value
                break
            print("[WARN] 0~10 사이 숫자를 입력하세요.")
    return answers


def predict_current_state(user_id: str, dept: str) -> None:
    test_dir = PERSONAL_DIR / user_id
    test_dir.mkdir(parents=True, exist_ok=True)
    session_name = next_session_name(user_id, prefix="test")
    task_context = detect_task_context()

    print("\n[STEP] 빠른 3문항 자가 응답")
    report = ask_self_report()
    print(f"[STEP] 현재 상태 측정: {SESSION_SEC}초, context={task_context}")
    if not collect_one_session(user_id, session_name, SESSION_SEC, 0, task_context):
        print("[ERROR] 현재 상태 측정 실패")
        return

    code, _ = run_python_capture(
        PREDICT_SCRIPT,
        [
            "--session_dir", str(PERSONAL_DIR / user_id / session_name),
            "--model_path", str(BEST_MODEL),
            "--duration_sec", str(SESSION_SEC),
            "--dept", dept,
            "--user_id", user_id,
            "--task_context", task_context,
            "--tension", str(report["tension"]),
            "--pressure_hurry", str(report["pressure_hurry"]),
            "--control", str(report["control"]),
        ],
    )
    if code != 0 or not PREDICTION_RESULT.exists():
        print("[ERROR] 예측 실패")
        return

    result = json.loads(PREDICTION_RESULT.read_text(encoding="utf-8"))
    print("\n===== 분석 결과 =====")
    print(f"작업 맥락: {result['task_context']}")
    print(f"최종 상태: {result['state']}")
    print(f"class_value: {result['class_value']}")
    print(f"최종 점수: {result['final_score']}/10")
    print(f"신뢰도: {result['confidence']}%")
    print("판단 근거:")
    for reason in result.get("top_reasons", []):
        print(f"- {reason}")
    print(result.get("interpretation", ""))

    feedback = input("이 분류가 맞나요? (y/n): ").strip().lower()
    if feedback == "y":
        (PERSONAL_DIR / user_id / session_name / "label.txt").write_text(str(result["class_value"]), encoding="utf-8")
        send_to_server(dept, result)
    else:
        actual = ask_binary_label("실제 상태 입력 (0=정상, 1=스트레스 가능성 높음): ")
        (PERSONAL_DIR / user_id / session_name / "label.txt").write_text(str(actual), encoding="utf-8")
        print("[INFO] 실제 피드백 저장 — 다음 학습에 반영됩니다.")


def get_user_context() -> tuple[str, str]:
    dept = select_department()
    employee_id = input("직원 ID 입력 (예: member1): ").strip() or "member1"
    user_id = make_user_id(dept, employee_id)
    print(f"[INFO] 부서: {dept}")
    print(f"[INFO] 사용자 ID: {user_id}")
    return dept, user_id


def main() -> None:
    print("=" * 60)
    print("PC 작업 맥락 기반 개인화 스트레스 가능성 분석 시스템")
    print("=" * 60)
    while True:
        print("\n메뉴")
        print("1. 초기/추가 개인화 데이터 수집")
        print("2. 개인화 모델 학습")
        print("3. 현재 스트레스 가능성 측정")
        print("4. 종료")
        choice = input("번호 선택: ").strip()
        if choice == "1":
            _, user_id = get_user_context()
            onboarding_collect(user_id)
        elif choice == "2":
            _, user_id = get_user_context()
            train_personal_model(user_id)
        elif choice == "3":
            dept, user_id = get_user_context()
            predict_current_state(user_id, dept)
        elif choice == "4":
            print("프로그램을 종료합니다.")
            break
        else:
            print("[WARN] 1~4 중에서 선택하세요.")


if __name__ == "__main__":
    main()
