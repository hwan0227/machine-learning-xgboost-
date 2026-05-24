"""
스트레스 탐지 프로그램 통합 GUI: 수집·학습·예측·피드백·서버 전송의 GUI 버전.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
import webbrowser
import json
import re
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
import pandas as pd
import requests
from PIL import Image, ImageDraw, ImageFont
from tkinter import messagebox, simpledialog
import tkinter as tk
from task_context import detect_task_context, normalize_task_context

# =========================================================
# 경로·상수 (main.py와 동일)
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PERSONAL_DIR = DATA_DIR / "personal"
MODELS_DIR = BASE_DIR / "models"
IMG_DIR = BASE_DIR / "img"

COLLECT_SCRIPT = BASE_DIR / "collect_behavior_features.py"
MAKE_PERSONAL_SCRIPT = BASE_DIR / "make_personal_features.py"
TRAIN_SCRIPT = BASE_DIR / "train_personalized_model.py"
PREDICT_SCRIPT = BASE_DIR / "predict_single_session.py"
EXECUTIVE_SERVER_SCRIPT = BASE_DIR / "executive_server.py"

KAGGLE_FEATURES = DATA_DIR / "kaggle_features.csv"
PERSONAL_FEATURES = DATA_DIR / "personal_features.csv"
FINAL_MERGED = DATA_DIR / "final_merged_features.csv"
BEST_MODEL = MODELS_DIR / "best_stress_model.pkl"
PREDICTION_RESULT = BASE_DIR / "prediction_result.json"

TARGET_PER_LABEL = 3
SESSION_SEC = 10
PERSONAL_WEIGHT = 5.0
SERVER_URL = "http://127.0.0.1:5000/api/send_data"
EXECUTIVE_REPORT_URL = "http://127.0.0.1:5000/api/get_executive_report"

DEPT_DEFINITIONS = [
    {"id": 1, "name": "1. 인사부", "img": "choice_1.jpg", "mode": "agent"},
    {"id": 2, "name": "2. 재무부", "img": "choice_2.jpg", "mode": "agent"},
    {"id": 3, "name": "3. 마케팅부", "img": "choice_3.jpg", "mode": "agent"},
    {"id": 4, "name": "4. 생산부", "img": "choice_4.jpg", "mode": "agent"},
    {"id": 5, "name": "5. 연구개발부", "img": "choice_5.jpg", "mode": "agent"},
    {"id": 6, "name": "6. 구매부", "img": "choice_6.jpg", "mode": "agent"},
    {"id": 7, "name": "7. IT부", "img": "choice_7.jpg", "mode": "agent"},
    {"id": 8, "name": "8. 법무부", "img": "choice_8.jpg", "mode": "agent"},
    {"id": 9, "name": "9. 영업부", "img": "choice_9.jpg", "mode": "agent"},
    {"id": 10, "name": "10. 경영진 모드", "img": "choice_10.jpg", "mode": "executive"},
]


def update_dashboard_data(result: dict | float) -> None:
    file_path = BASE_DIR / "chart_data.js"
    try:
        if not file_path.exists(): return
        content = file_path.read_text(encoding="utf-8")
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match: return
        data = json.loads(json_match.group())

        now = datetime.now()
        if isinstance(result, dict):
            exact_score = float(result.get("final_score", result.get("latest_score", 0.0)))
            data["latest_result"] = result
            data["task_context"] = result.get("task_context", "unknown")
            data["state"] = result.get("state", "정상 / 비스트레스")
            data["class_value"] = result.get("class_value", 0)
            data["confidence"] = result.get("confidence", 0)
            data["self_report_score"] = result.get("self_report_score", 0)
            data["behavior_anomaly_score"] = result.get("behavior_anomaly_score", 0)
            data["persistence_score"] = result.get("persistence_score", 0)
            data["correction_urgency_index"] = result.get("correction_urgency_index", 0)
            data["baseline_quality"] = result.get("baseline_quality", "missing")
            data["top_reasons"] = result.get("top_reasons", [])
            data["interpretation"] = result.get("interpretation", "")
        else:
            exact_score = float(result)
        # 그릇이 부족하면 자동으로 채우는 방어 코드
        if "latest_score" not in data: data["latest_score"] = 0.0
        if "counts" not in data["daily"]: data["daily"]["counts"] = [0] * len(data["daily"]["data"])

        # 현재 점수 배달
        data["latest_score"] = float(exact_score)

        # 시간대별 평균 집계
        idx = (now.hour // 3)
        if idx < len(data["daily"]["data"]):
            old_avg = data["daily"]["data"][idx]
            old_count = data["daily"]["counts"][idx]
            new_count = old_count + 1
            new_avg = ((old_avg * old_count) + exact_score) / new_count
            data["daily"]["data"][idx] = round(new_avg, 1)
            data["daily"]["counts"][idx] = new_count

        file_path.write_text(f"const chartData = {json.dumps(data, indent=4, ensure_ascii=False)};", encoding="utf-8")
        print(f"[SUCCESS] 개인 대시보드 업데이트 완료: {exact_score}점")
    except Exception as e:
        print(f"[ERROR] 대시보드 업데이트 실패: {e}")


def make_user_id(dept: str, employee_id: str) -> str:
    dept_clean = dept.split(". ", 1)[1] if ". " in dept else dept
    dept_clean = dept_clean.replace(" ", "_")
    employee_id = employee_id.strip().replace(" ", "_")
    return f"{dept_clean}_{employee_id}"


def run_python(script_path: Path, args: list[str]) -> int:
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(cmd, cwd=BASE_DIR)
    return result.returncode


def run_python_capture(script_path: Path, args: list[str]) -> tuple[int, str]:
    cmd = [sys.executable, str(script_path)] + args
    result = subprocess.run(
        cmd,
        cwd=BASE_DIR,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    output = ""
    if result.stdout:
        output += result.stdout
    if result.stderr:
        output += result.stderr
    return result.returncode, output


def parse_prediction(output: str) -> tuple[float | None, float | None]:
    pred = None
    score = None
    for line in output.splitlines():
        line = line.strip()
        if line.startswith("[PREDICTION]"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    pred = float(parts[1])
                except Exception:
                    pass
        if "스트레스 점수" in line and "/10" in line:
            try:
                score_text = line.split("=")[-1].replace("/10", "").strip()
                score = float(score_text)
            except Exception:
                pass
    if pred is not None and score is None:
        score = 8.0 if pred == 1 else 3.0
    return pred, score


def send_to_server(dept: str, score: float, label: float, result: dict | None = None) -> None:
    payload = {
        "dept": dept,
        "score": float(score),
        "label": float(label),
    }
    if result:
        payload.update({
            "state": result.get("state"),
            "class_value": result.get("class_value"),
            "task_context": result.get("task_context"),
            "confidence": result.get("confidence"),
            "correction_urgency_index": result.get("correction_urgency_index"),
        })
    try:
        response = requests.post(SERVER_URL, json=payload, timeout=3)
        if response.status_code == 200:
            print("[DONE] 경영진 대시보드로 결과를 전송했습니다.")
            print(f"[INFO] 전송 데이터: {payload}")
        else:
            print(f"[WARN] 서버 전송 실패: status={response.status_code}")
    except Exception as e:
        print(f"[WARN] 서버 연결 실패: {e}")
        print("[INFO] executive_server.py가 실행 중인지 확인하세요.")


def merge_features() -> bool:
    if not KAGGLE_FEATURES.exists():
        print(f"[ERROR] 파일 없음: {KAGGLE_FEATURES}")
        return False
    if not PERSONAL_FEATURES.exists():
        print(f"[ERROR] 파일 없음: {PERSONAL_FEATURES}")
        return False
    kaggle_df = pd.read_csv(KAGGLE_FEATURES, encoding="utf-8-sig")
    personal_df = pd.read_csv(PERSONAL_FEATURES, encoding="utf-8-sig")
    final_df = pd.concat([kaggle_df, personal_df], ignore_index=True)
    final_df.to_csv(FINAL_MERGED, index=False, encoding="utf-8-sig")
    print("[DONE] 개인 데이터와 기존 데이터를 병합했습니다.")
    return True


def collect_one_session(user_id: str, session_name: str, duration_sec: int, label: float, task_context: str | None = None) -> bool:
    args = [
        "--user_id",
        user_id,
        "--session_name",
        session_name,
        "--duration_sec",
        str(duration_sec),
        "--label",
        str(label),
    ]
    if task_context:
        args.extend(["--task_context", normalize_task_context(task_context)])
    code = run_python(
        COLLECT_SCRIPT,
        args,
    )
    return code == 0


def count_sessions(user_id: str) -> tuple[int, int, int]:
    user_dir = PERSONAL_DIR / user_id
    if not user_dir.exists():
        return 0, 0, 0
    zero_count = 0
    one_count = 0
    total = 0
    for session_dir in user_dir.iterdir():
        if not session_dir.is_dir():
            continue
        if session_dir.name.lower().startswith("test_"):
            continue
        label_path = session_dir / "label.txt"
        if not label_path.exists():
            continue
        try:
            lab = float(label_path.read_text(encoding="utf-8").strip())
            total += 1
            if lab == 0:
                zero_count += 1
            elif lab == 1:
                one_count += 1
        except Exception:
            pass
    return total, zero_count, one_count


def count_focus_label_sessions(user_id: str) -> int:
    user_dir = PERSONAL_DIR / user_id
    if not user_dir.exists():
        return 0
    focus_count = 0
    for session_dir in user_dir.iterdir():
        if not session_dir.is_dir():
            continue
        if session_dir.name.lower().startswith("test_"):
            continue
        label_path = session_dir / "label.txt"
        if not label_path.exists():
            continue
        try:
            lab = float(label_path.read_text(encoding="utf-8").strip())
            if lab == 0.5:
                focus_count += 1
        except Exception:
            pass
    return focus_count


def next_session_name(user_id: str, prefix: str = "session") -> str:
    user_dir = PERSONAL_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    used_numbers: list[int] = []
    for session_dir in user_dir.iterdir():
        if not session_dir.is_dir():
            continue
        name = session_dir.name.strip().lower()
        if name.startswith(f"{prefix}_"):
            try:
                num = int(name.split("_")[1])
                used_numbers.append(num)
            except Exception:
                pass
    n = 1
    while n in used_numbers:
        n += 1
    return f"{prefix}_{n:02d}"


def write_session_label_txt(user_id: str, session_name: str, label: int | str) -> Path:
    """collect 실패 등으로 세션 디렉터리가 없을 때도 FileNotFoundError 없이 label.txt를 기록한다."""
    label_path = PERSONAL_DIR / user_id / session_name / "label.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(str(label), encoding="utf-8")
    return label_path


class MentalCareIntegratedApp(ctk.CTk):
    """일반 직원은 부서 선택 후 개인 메뉴로, 경영진은 바로 집계 대시보드로 이동한다."""

    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.title("문서·코딩 작업 기반 스트레스 탐지 프로그램")
        self.geometry("1280x800")
        self.minsize(1000, 700)
        self._center_window(1280, 800)
        self.configure(fg_color="#eef3f8")

        self._dept_id: int = 0
        self._dept_name: str = ""
        self._user_id: str = ""
        self._ctk_dept_images: list[ctk.CTkImage] = []
        self._action_buttons: list[ctk.CTkButton] = []
        self._executive_server_process: subprocess.Popen | None = None
        self._result_labels: dict[str, ctk.CTkLabel] = {}
        self._reason_labels: list[ctk.CTkLabel] = []
        self._interpretation_label: ctk.CTkLabel | None = None
        self._progress_labels: dict[str, ctk.CTkLabel] = {}
        self._feedback_buttons: list[ctk.CTkButton] = []
        self._feedback_actions_frame: ctk.CTkFrame | None = None
        self._feedback_result_label: ctk.CTkLabel | None = None
        self._feedback_status_label: ctk.CTkLabel | None = None
        self._latest_prediction_session: str | None = None
        self._latest_prediction_result: dict | None = None
        self._state_box: ctk.CTkFrame | None = None
        self._log_box: ctk.CTkTextbox | None = None
        self._log_visible = False

        self._container = ctk.CTkFrame(self, fg_color="#eef3f8")
        self._container.pack(fill="both", expand=True)

        self._selection_parent: ctk.CTkFrame | None = None
        self._show_department_selection()

        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _center_window(self, width: int, height: int) -> None:
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def _clear_container(self) -> None:
        for child in self._container.winfo_children():
            child.destroy()
        self._action_buttons.clear()
        self._result_labels.clear()
        self._reason_labels.clear()
        self._progress_labels.clear()
        self._feedback_buttons.clear()
        self._feedback_actions_frame = None
        self._feedback_result_label = None
        self._feedback_status_label = None
        self._latest_prediction_session = None
        self._latest_prediction_result = None
        self._interpretation_label = None
        self._state_box = None
        self._log_box = None

    def _show_department_selection(self) -> None:
        self._clear_container()
        self._dept_id = 0
        self._dept_name = ""
        self._user_id = ""
        self._selection_parent = ctk.CTkFrame(self._container, fg_color="#eef3f8")
        self._selection_parent.pack(fill="both", expand=True)
        self._build_department_selection(self._selection_parent)

    def _show_employee_id_screen(self) -> None:
        self._clear_container()
        self._build_employee_id_screen()

    def _show_main_workspace(self) -> None:
        self._clear_container()
        self._build_main_workspace()

    def _confirm_back_to_department_selection(self) -> None:
        yes = messagebox.askyesno(
            "부서/사용자 다시 선택",
            "현재 사용자를 나가고 부서 선택 화면으로 돌아가시겠습니까?",
            parent=self,
        )
        if yes:
            self._show_department_selection()

    def _main_thread_call(self, fn):
        """워커 스레드에서 메인 스레드로 UI 호출 후 결과 대기."""
        box: dict = {}
        evt = threading.Event()

        def wrap() -> None:
            try:
                box["ret"] = fn()
            except Exception as e:
                box["err"] = e
            finally:
                evt.set()

        self.after(0, wrap)
        evt.wait()
        if "err" in box:
            raise box["err"]
        return box.get("ret")

    def log_to_console(self, msg: str) -> None:
        def append() -> None:
            if getattr(self, "_log_box", None) is None:
                return
            time_str = datetime.now().strftime("%H:%M:%S")
            self._log_box.insert("end", f"[{time_str}] {msg}\n")
            self._log_box.see("end")

        self.after(0, append)

    def _set_actions_enabled(self, enabled: bool) -> None:
        def apply() -> None:
            state = "normal" if enabled else "disabled"
            for b in self._action_buttons:
                try:
                    b.configure(state=state)
                except Exception:
                    pass

        self.after(0, apply)

    def _load_and_text_overlay(self, img_path: Path, text: str) -> Image.Image:
        img = Image.open(img_path)
        img = img.resize((200, 130), Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(img)
        font_path = Path(r"C:\Windows\Fonts\malgunbd.ttf")
        if font_path.exists():
            font = ImageFont.truetype(str(font_path), 16)
        else:
            font = ImageFont.load_default()
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        text_x = (img.width - text_w) / 2
        text_y = (img.height - text_h) / 2
        for dx, dy in [(-1, -1), (1, 1), (-1, 1), (1, -1), (0, -1), (0, 1), (-1, 0), (1, 0)]:
            draw.text((text_x + dx, text_y + dy), text, font=font, fill="black")
        draw.text((text_x, text_y), text, font=font, fill="white")
        return img

    def _build_department_selection(self, parent: ctk.CTkFrame) -> None:
        header = ctk.CTkFrame(parent, fg_color="#ffffff", corner_radius=18, border_width=1, border_color="#dce5f2")
        header.pack(fill="x", padx=34, pady=(26, 18))
        header.grid_columnconfigure(0, weight=1)
        header.grid_columnconfigure(1, weight=0)
        left = ctk.CTkFrame(header, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=26, pady=24)
        ctk.CTkLabel(
            left,
            text="부서 선택",
            font=ctk.CTkFont(family="Malgun Gothic", size=14, weight="bold"),
            text_color="#2563eb",
        ).pack(anchor="w")
        ctk.CTkLabel(
            left,
            text="문서·코딩 작업 기반 스트레스 탐지 프로그램",
            font=ctk.CTkFont(family="Malgun Gothic", size=34, weight="bold"),
            text_color="#182230",
        ).pack(anchor="w", pady=(3, 6))
        ctk.CTkLabel(
            left,
            text="문서 작성과 코딩 작업 중 나타나는 입력 패턴과 자가 체크를 바탕으로 현재 상태를 분석합니다.",
            font=ctk.CTkFont(family="Malgun Gothic", size=15),
            text_color="#667085",
            wraplength=760,
            justify="left",
        ).pack(anchor="w")

        executive = ctk.CTkFrame(header, fg_color="#f3f7ff", corner_radius=14, border_width=1, border_color="#d9e6ff")
        executive.grid(row=0, column=1, sticky="nsew", padx=(0, 22), pady=22)
        ctk.CTkLabel(
            executive,
            text="경영진 조회",
            font=ctk.CTkFont(family="Malgun Gothic", size=17, weight="bold"),
            text_color="#182230",
        ).pack(anchor="e", padx=18, pady=(16, 4))
        ctk.CTkLabel(
            executive,
            text="개인 측정 없이 부서별 평균만 확인",
            font=ctk.CTkFont(family="Malgun Gothic", size=12),
            text_color="#667085",
        ).pack(anchor="e", padx=18)
        ctk.CTkButton(
            executive,
            text="경영진 대시보드 열기",
            width=190,
            height=38,
            fg_color="#2563eb",
            hover_color="#1d4ed8",
            command=self.open_executive_dashboard,
        ).pack(anchor="e", padx=18, pady=(16, 16))

        title_row = ctk.CTkFrame(parent, fg_color="transparent")
        title_row.pack(fill="x", padx=36, pady=(0, 8))
        ctk.CTkLabel(
            title_row,
            text="일반 직원 부서 선택",
            font=ctk.CTkFont(family="Malgun Gothic", size=20, weight="bold"),
            text_color="#182230",
        ).pack(side="left")
        ctk.CTkLabel(
            title_row,
            text="소속 부서를 선택하면 개인 분석 화면으로 이동합니다.",
            font=ctk.CTkFont(family="Malgun Gothic", size=13),
            text_color="#667085",
        ).pack(side="left", padx=12)

        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.pack(fill="both", expand=True, padx=24, pady=(0, 24))

        grid = ctk.CTkFrame(scroll, fg_color="transparent")
        grid.pack(fill="x", expand=True)
        for col in range(3):
            grid.grid_columnconfigure(col, weight=1, uniform="dept")

        dept_meta = {
            1: ("HR", "구성원 지원과 조직 운영"),
            2: ("FIN", "예산·정산·재무 업무"),
            3: ("MKT", "캠페인과 시장 커뮤니케이션"),
            4: ("OPS", "생산·운영 프로세스"),
            5: ("R&D", "연구개발과 기술 검토"),
            6: ("BUY", "구매·계약·협력사 관리"),
            7: ("IT", "개발·시스템·기술 지원"),
            8: ("LAW", "법무 검토와 리스크 관리"),
            9: ("SALES", "영업 활동과 고객 대응"),
        }
        departments = [dept for dept in DEPT_DEFINITIONS if dept.get("mode") != "executive"]
        for i, dept in enumerate(departments):
            row, col = divmod(i, 3)
            tag, desc = dept_meta.get(int(dept["id"]), ("DEPT", "부서 업무"))
            card = ctk.CTkFrame(grid, fg_color="#ffffff", corner_radius=16, border_width=1, border_color="#dce5f2")
            card.grid(row=row, column=col, padx=12, pady=12, sticky="nsew")
            ctk.CTkLabel(
                card,
                text=tag,
                width=66,
                height=32,
                fg_color="#eaf1ff",
                corner_radius=9,
                text_color="#2563eb",
                font=ctk.CTkFont(family="Malgun Gothic", size=13, weight="bold"),
            ).pack(anchor="w", padx=18, pady=(18, 10))
            ctk.CTkLabel(
                card,
                text=str(dept["name"]),
                font=ctk.CTkFont(family="Malgun Gothic", size=20, weight="bold"),
                text_color="#182230",
            ).pack(anchor="w", padx=18)
            ctk.CTkLabel(
                card,
                text=desc,
                font=ctk.CTkFont(family="Malgun Gothic", size=13),
                text_color="#667085",
                wraplength=260,
                justify="left",
            ).pack(anchor="w", padx=18, pady=(6, 16))
            ctk.CTkButton(
                card,
                text="이 부서 선택",
                height=38,
                fg_color="#ffffff",
                hover_color="#eef4ff",
                text_color="#2563eb",
                border_width=1,
                border_color="#c7d7fe",
                font=ctk.CTkFont(family="Malgun Gothic", size=14, weight="bold"),
                command=lambda d=dept: self._on_department_chosen(d),
            ).pack(fill="x", padx=18, pady=(0, 18))

    def _on_department_chosen(self, dept: dict) -> None:
        self._dept_id = int(dept["id"])
        self._dept_name = str(dept["name"])
        if self._dept_id == 10 or dept.get("mode") == "executive":
            self.open_executive_dashboard()
            return
        self._show_employee_id_screen()

    def _build_employee_id_screen(self) -> None:
        frame = ctk.CTkFrame(self._container, fg_color="#eef3f8")
        frame.pack(fill="both", expand=True, padx=40, pady=40)

        card = ctk.CTkFrame(frame, fg_color="#ffffff", corner_radius=18, border_width=1, border_color="#dce5f2")
        card.pack(expand=True)
        ctk.CTkLabel(
            card,
            text="사용자 확인",
            font=ctk.CTkFont(family="Malgun Gothic", size=14, weight="bold"),
            text_color="#2563eb",
        ).pack(anchor="w", padx=34, pady=(30, 4))
        ctk.CTkLabel(
            card,
            text=f"선택 부서: {self._dept_name}",
            font=ctk.CTkFont(family="Malgun Gothic", size=24, weight="bold"),
            text_color="#182230",
        ).pack(anchor="w", padx=34, pady=(0, 8))

        ctk.CTkLabel(
            card,
            text="개인화 데이터와 측정 결과를 구분하기 위한 직원 ID 또는 이름을 입력하세요.",
            font=ctk.CTkFont(family="Malgun Gothic", size=16),
            text_color="#667085",
            wraplength=460,
            justify="left",
        ).pack(anchor="w", padx=34, pady=(0, 18))

        entry = ctk.CTkEntry(card, width=460, height=42, placeholder_text="예: member1")
        entry.pack(padx=34, pady=(0, 16))

        def on_next() -> None:
            raw = entry.get().strip()
            if not raw:
                raw = "member1"
            self._user_id = make_user_id(self._dept_name, raw)
            self._show_main_workspace()

        ctk.CTkButton(
            card,
            text="다음 — 메인 메뉴로",
            font=ctk.CTkFont(family="Malgun Gothic", size=16, weight="bold"),
            width=460,
            height=44,
            command=on_next,
        ).pack(padx=34, pady=(0, 10))
        ctk.CTkButton(
            card,
            text="← 부서 선택으로 돌아가기",
            font=ctk.CTkFont(family="Malgun Gothic", size=14, weight="bold"),
            width=460,
            height=38,
            fg_color="#ffffff",
            hover_color="#eef2f7",
            text_color="#344054",
            border_width=1,
            border_color="#d0d5dd",
            command=self._show_department_selection,
        ).pack(padx=34, pady=(0, 30))

    def _build_main_workspace(self) -> None:
        self.title(f"스트레스 탐지 프로그램 — {self._dept_name} / {self._user_id}")

        main = ctk.CTkScrollableFrame(self._container, fg_color="#eef3f8")
        main.pack(fill="both", expand=True, padx=28, pady=20)

        self._action_buttons.clear()
        hero = ctk.CTkFrame(main, fg_color="#ffffff", corner_radius=18, border_width=1, border_color="#dce5f2")
        hero.pack(fill="x", pady=(0, 18))
        hero.grid_columnconfigure(0, weight=1)
        hero.grid_columnconfigure(1, weight=0)

        hero_left = ctk.CTkFrame(hero, fg_color="transparent")
        hero_left.grid(row=0, column=0, sticky="nsew", padx=24, pady=22)
        ctk.CTkLabel(
            hero_left,
            text="문서·코딩 작업 기반",
            font=ctk.CTkFont(family="Malgun Gothic", size=14, weight="bold"),
            text_color="#2563eb",
        ).pack(anchor="w")
        ctk.CTkLabel(
            hero_left,
            text="문서·코딩 작업 기반 스트레스 탐지 프로그램",
            font=ctk.CTkFont(family="Malgun Gothic", size=34, weight="bold"),
            text_color="#182230",
        ).pack(anchor="w", pady=(2, 6))
        ctk.CTkLabel(
            hero_left,
            text="문서 작성과 코딩 중 나타나는 입력 패턴과 자가 체크를 바탕으로 현재 스트레스 상태를 분석합니다.",
            font=ctk.CTkFont(family="Malgun Gothic", size=15),
            text_color="#667085",
            wraplength=720,
            justify="left",
        ).pack(anchor="w")
        info_row = ctk.CTkFrame(hero_left, fg_color="transparent")
        info_row.pack(anchor="w", pady=(16, 0))
        ctk.CTkLabel(
            info_row,
            text=f"부서  {self._dept_name}",
            font=ctk.CTkFont(family="Malgun Gothic", size=14, weight="bold"),
            text_color="#16803c",
        ).pack(side="left", padx=(0, 18))
        ctk.CTkLabel(
            info_row,
            text=f"사용자  {self._user_id}",
            font=ctk.CTkFont(family="Malgun Gothic", size=14, weight="bold"),
            text_color="#344054",
        ).pack(side="left")

        hero_right = ctk.CTkFrame(hero, fg_color="#f3f7ff", corner_radius=14, border_width=1, border_color="#d9e6ff")
        hero_right.grid(row=0, column=1, sticky="nsew", padx=(0, 22), pady=22)
        ctk.CTkLabel(
            hero_right,
            text="개인 분석 모드",
            font=ctk.CTkFont(family="Malgun Gothic", size=15, weight="bold"),
            text_color="#2563eb",
        ).pack(anchor="e", padx=18, pady=(16, 6))
        ctk.CTkLabel(
            hero_right,
            text="최종 상태는 3단계만 표시",
            font=ctk.CTkFont(family="Malgun Gothic", size=13),
            text_color="#667085",
        ).pack(anchor="e", padx=18)
        ctk.CTkButton(
            hero_right,
            text="부서/사용자 다시 선택",
            width=190,
            height=36,
            fg_color="#ffffff",
            hover_color="#eef2f7",
            text_color="#344054",
            border_width=1,
            border_color="#d0d5dd",
            command=self._confirm_back_to_department_selection,
        ).pack(anchor="e", padx=18, pady=(18, 16))

        action_title = ctk.CTkFrame(main, fg_color="transparent")
        action_title.pack(fill="x", pady=(0, 8))
        ctk.CTkLabel(
            action_title,
            text="진행 순서",
            font=ctk.CTkFont(family="Malgun Gothic", size=20, weight="bold"),
            text_color="#182230",
        ).pack(side="left")
        ctk.CTkLabel(
            action_title,
            text="1단계부터 3단계까지 차례대로 진행하면 됩니다.",
            font=ctk.CTkFont(family="Malgun Gothic", size=13),
            text_color="#667085",
        ).pack(side="left", padx=12)

        btn_frame = ctk.CTkFrame(main, fg_color="transparent")
        btn_frame.pack(fill="x", pady=(0, 16))
        btn_frame.grid_columnconfigure(0, weight=1, uniform="actions")
        btn_frame.grid_columnconfigure(1, weight=1, uniform="actions")
        btn_frame.grid_columnconfigure(2, weight=1, uniform="actions")

        def action_card(row: int, col: int, tag: str, title: str, desc: str, color: str, hover: str, command) -> None:
            card = ctk.CTkFrame(btn_frame, fg_color="#ffffff", corner_radius=16, border_width=1, border_color="#dce5f2")
            card.grid(row=row, column=col, sticky="nsew", padx=(0 if col == 0 else 8, 0 if col == 2 else 8), pady=7)
            ctk.CTkLabel(
                card,
                text=tag,
                width=54,
                height=28,
                fg_color="#eef4ff",
                corner_radius=8,
                text_color=color,
                font=ctk.CTkFont(family="Malgun Gothic", size=12, weight="bold"),
            ).pack(anchor="w", padx=18, pady=(16, 8))
            ctk.CTkLabel(
                card,
                text=title,
                font=ctk.CTkFont(family="Malgun Gothic", size=18, weight="bold"),
                text_color="#182230",
            ).pack(anchor="w", padx=18)
            ctk.CTkLabel(
                card,
                text=desc,
                font=ctk.CTkFont(family="Malgun Gothic", size=13),
                text_color="#667085",
                wraplength=470,
                justify="left",
            ).pack(anchor="w", padx=18, pady=(5, 14))
            button = ctk.CTkButton(
                card,
                text="실행하기",
                height=38,
                fg_color=color,
                hover_color=hover,
                font=ctk.CTkFont(family="Malgun Gothic", size=14, weight="bold"),
                command=command,
            )
            button.pack(fill="x", padx=18, pady=(0, 16))
            self._action_buttons.append(button)

        action_card(0, 0, "STEP 1", "초기/추가 개인화 데이터 수집", "개인 기준선과 학습용 행동 데이터를 한 세션씩 수집합니다.", "#2563eb", "#1d4ed8", self._on_onboarding_clicked)
        action_card(0, 1, "STEP 2", "개인화 모델 학습", "수집된 0/1 라벨 데이터를 바탕으로 개인 모델을 갱신합니다.", "#2563eb", "#1d4ed8", self._on_train_clicked)
        action_card(0, 2, "STEP 3", "현재 스트레스 가능성 측정", "빠른 자가 체크 후 현재 문서·코딩 작업 상태를 3단계로 분석합니다.", "#0f766e", "#115e59", self._on_predict_clicked)

        aux_bar = ctk.CTkFrame(main, fg_color="#f8fbff", corner_radius=16, border_width=1, border_color="#dce5f2")
        aux_bar.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(
            aux_bar,
            text="최근 결과와 추이는 개인 대시보드에서 확인할 수 있습니다.",
            font=ctk.CTkFont(family="Malgun Gothic", size=14),
            text_color="#667085",
        ).pack(side="left", padx=18, pady=14)
        dashboard_button = ctk.CTkButton(
            aux_bar,
            text="개인 대시보드 열기",
            width=170,
            height=34,
            fg_color="#ffffff",
            hover_color="#eef4ff",
            text_color="#2563eb",
            border_width=1,
            border_color="#c7d7fe",
            font=ctk.CTkFont(family="Malgun Gothic", size=14, weight="bold"),
            command=self._open_personal_dashboard,
        )
        dashboard_button.pack(side="right", padx=18, pady=12)
        self._action_buttons.append(dashboard_button)

        result_card = ctk.CTkFrame(main, fg_color="#ffffff", corner_radius=18, border_width=1, border_color="#dce5f2")
        result_card.pack(fill="x", pady=(0, 16))
        result_card.grid_columnconfigure(0, weight=1, uniform="result")
        result_card.grid_columnconfigure(1, weight=1, uniform="result")
        ctk.CTkLabel(
            result_card,
            text="최근 결과 요약",
            font=ctk.CTkFont(family="Malgun Gothic", size=21, weight="bold"),
            text_color="#182230",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=22, pady=(18, 10))

        score_frame = ctk.CTkFrame(result_card, fg_color="#f7faff", corner_radius=14, border_width=1, border_color="#e6edf7")
        score_frame.grid(row=1, column=0, sticky="nsew", padx=(18, 8), pady=(0, 18))
        reason_frame = ctk.CTkFrame(result_card, fg_color="#f7faff", corner_radius=14, border_width=1, border_color="#e6edf7")
        reason_frame.grid(row=1, column=1, sticky="nsew", padx=(8, 18), pady=(0, 18))
        reason_frame.grid_propagate(False)
        reason_frame.configure(height=340)
        score_frame.grid_columnconfigure(0, weight=1)
        score_frame.grid_columnconfigure(1, weight=1)

        self._result_labels.clear()
        state_box = ctk.CTkFrame(score_frame, fg_color="#eefdf3", corner_radius=12, border_width=1, border_color="#c8f3d6")
        state_box.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(12, 8))
        self._state_box = state_box
        ctk.CTkLabel(state_box, text="최종 상태", font=ctk.CTkFont(size=13, weight="bold"), text_color="#667085").pack(anchor="w", padx=16, pady=(12, 0))
        state_value = ctk.CTkLabel(
            state_box,
            text="아직 측정 전",
            font=ctk.CTkFont(family="Malgun Gothic", size=27, weight="bold"),
            text_color="#16803c",
        )
        state_value.pack(anchor="w", padx=16, pady=(4, 14))
        self._result_labels["state"] = state_value

        score_big = ctk.CTkFrame(score_frame, fg_color="#ffffff", corner_radius=12, border_width=1, border_color="#e5e9f0")
        score_big.grid(row=1, column=0, sticky="nsew", padx=(12, 6), pady=6)
        ctk.CTkLabel(score_big, text="최종 점수", font=ctk.CTkFont(size=13, weight="bold"), text_color="#667085").pack(anchor="w", padx=16, pady=(14, 0))
        final_score_label = ctk.CTkLabel(score_big, text="-", font=ctk.CTkFont(family="Malgun Gothic", size=34, weight="bold"), text_color="#182230")
        final_score_label.pack(anchor="w", padx=16, pady=(3, 12))
        self._result_labels["final_score"] = final_score_label

        confidence_big = ctk.CTkFrame(score_frame, fg_color="#ffffff", corner_radius=12, border_width=1, border_color="#e5e9f0")
        confidence_big.grid(row=1, column=1, sticky="nsew", padx=(6, 12), pady=6)
        ctk.CTkLabel(confidence_big, text="신뢰도", font=ctk.CTkFont(size=13, weight="bold"), text_color="#667085").pack(anchor="w", padx=16, pady=(14, 0))
        confidence_label = ctk.CTkLabel(confidence_big, text="-", font=ctk.CTkFont(family="Malgun Gothic", size=34, weight="bold"), text_color="#182230")
        confidence_label.pack(anchor="w", padx=16, pady=(3, 12))
        self._result_labels["confidence"] = confidence_label

        ctk.CTkLabel(
            score_frame,
            text="상세 지표",
            font=ctk.CTkFont(family="Malgun Gothic", size=14, weight="bold"),
            text_color="#667085",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=14, pady=(10, 0))
        metric_defs = [
            ("class_value", "분류 결과", "-"),
            ("self_report_score", "자가 체크", "-"),
            ("behavior_anomaly_score", "입력 안정도", "-"),
            ("persistence_score", "작업 지속성", "-"),
            ("correction_urgency_index", "급한 수정 패턴", "-"),
            ("baseline_quality", "기준 데이터", "-"),
        ]
        for idx, (key, label, value) in enumerate(metric_defs):
            box = ctk.CTkFrame(score_frame, fg_color="#ffffff", corner_radius=10, border_width=1, border_color="#e5e9f0")
            box.grid(row=3 + idx // 2, column=idx % 2, sticky="nsew", padx=(12 if idx % 2 == 0 else 6, 6 if idx % 2 == 0 else 12), pady=6)
            ctk.CTkLabel(box, text=label, font=ctk.CTkFont(size=12, weight="bold"), text_color="#667085").pack(anchor="w", padx=13, pady=(10, 0))
            value_label = ctk.CTkLabel(box, text=value, font=ctk.CTkFont(family="Malgun Gothic", size=18, weight="bold"), text_color="#182230")
            value_label.pack(anchor="w", padx=13, pady=(3, 10))
            self._result_labels[key] = value_label

        ctk.CTkLabel(
            reason_frame,
            text="판단 근거 / 해석",
            font=ctk.CTkFont(family="Malgun Gothic", size=21, weight="bold"),
            text_color="#182230",
        ).pack(anchor="w", padx=18, pady=(16, 8))
        self._reason_labels = []
        for _ in range(5):
            row = ctk.CTkFrame(reason_frame, fg_color="#ffffff", corner_radius=10, border_width=1, border_color="#dce5f2")
            row.pack(anchor="w", fill="x", padx=18, pady=3)
            label = ctk.CTkLabel(row, text="-", anchor="w", justify="left", wraplength=520, text_color="#344054", font=ctk.CTkFont(size=14))
            label.pack(anchor="w", fill="x", padx=12, pady=8)
            self._reason_labels.append(label)
        self._interpretation_label = ctk.CTkLabel(
            reason_frame,
            text="",
            anchor="w",
            justify="left",
            wraplength=520,
            text_color="#475467",
            font=ctk.CTkFont(family="Malgun Gothic", size=14),
        )
        self._interpretation_label.pack(anchor="w", fill="x", padx=20, pady=(12, 6))
        meta_line = ctk.CTkFrame(reason_frame, fg_color="transparent")
        meta_line.pack(anchor="w", fill="x", padx=18, pady=(4, 12))
        self._result_labels["formula"] = ctk.CTkLabel(meta_line, text="분석 방식: 자가 체크와 입력 패턴 종합", anchor="w", justify="left", wraplength=300, text_color="#667085", font=ctk.CTkFont(size=12))
        self._result_labels["formula"].pack(side="left", padx=(0, 12))
        self._result_labels["measured_at"] = ctk.CTkLabel(meta_line, text="측정 시간: -", anchor="w", justify="left", wraplength=220, text_color="#667085", font=ctk.CTkFont(size=12))
        self._result_labels["measured_at"].pack(side="left")
        self._result_labels["adjustment"] = ctk.CTkLabel(reason_frame, text="", anchor="w", justify="left", wraplength=520, text_color="#c2410c", font=ctk.CTkFont(size=12, weight="bold"))
        self._result_labels["adjustment"].pack(anchor="w", fill="x", padx=20, pady=(0, 12))

        feedback_card = ctk.CTkFrame(main, fg_color="#f0f9ff", corner_radius=18, border_width=1, border_color="#bae6fd")
        feedback_card.pack(fill="x", pady=(0, 16))
        feedback_card.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            feedback_card,
            text="측정 결과를 개인 기준 데이터에 반영",
            font=ctk.CTkFont(family="Malgun Gothic", size=18, weight="bold"),
            text_color="#0f172a",
        ).grid(row=0, column=0, sticky="w", padx=18, pady=(16, 4))
        self._feedback_result_label = ctk.CTkLabel(
            feedback_card,
            text="측정 후 이번 결과가 여기에 표시됩니다.",
            font=ctk.CTkFont(family="Malgun Gothic", size=18, weight="bold"),
            text_color="#0f172a",
            anchor="w",
            justify="left",
            wraplength=760,
        )
        self._feedback_result_label.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 4))
        self._feedback_status_label = ctk.CTkLabel(
            feedback_card,
            text="측정 후 결과를 개인 기준 데이터에 반영할 수 있습니다.",
            font=ctk.CTkFont(family="Malgun Gothic", size=14),
            text_color="#475467",
            anchor="w",
            justify="left",
            wraplength=760,
        )
        self._feedback_status_label.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 14))

        progress_card = ctk.CTkFrame(main, fg_color="#ffffff", corner_radius=18, border_width=1, border_color="#dce5f2")
        progress_card.pack(fill="x", pady=(0, 16))
        ctk.CTkLabel(
            progress_card,
            text="개인화 데이터 진행상황",
            font=ctk.CTkFont(family="Malgun Gothic", size=19, weight="bold"),
            text_color="#182230",
        ).grid(row=0, column=0, columnspan=4, sticky="w", padx=18, pady=(15, 8))
        self._progress_labels: dict[str, ctk.CTkLabel] = {}
        progress_defs = [
            ("normal", "정상 / 비스트레스", "0 / 3"),
            ("stress", "스트레스 가능성 높음", "0 / 3"),
            ("remaining", "남은 데이터", "정상 3개, 스트레스 3개"),
            ("percent", "전체 진행률", "0%"),
            ("session", "현재 세션", "-"),
            ("next", "다음 안내", "초기 수집을 시작하세요"),
        ]
        for idx, (key, title, value) in enumerate(progress_defs):
            box = ctk.CTkFrame(progress_card, fg_color="#f8fafc", corner_radius=10, border_width=1, border_color="#eef2f7")
            box.grid(row=1 + idx // 3, column=idx % 3, sticky="nsew", padx=8, pady=6)
            ctk.CTkLabel(box, text=title, text_color="#667085", font=ctk.CTkFont(size=12, weight="bold")).pack(anchor="w", padx=12, pady=(10, 0))
            label = ctk.CTkLabel(box, text=value, font=ctk.CTkFont(family="Malgun Gothic", size=15, weight="bold"), text_color="#182230")
            label.pack(anchor="w", padx=12, pady=(3, 10))
            self._progress_labels[key] = label
        for col in range(3):
            progress_card.grid_columnconfigure(col, weight=1)
        self._update_onboarding_progress()

        log_header = ctk.CTkFrame(main, fg_color="transparent")
        log_header.pack(fill="x", pady=(8, 2))
        ctk.CTkLabel(log_header, text="상태 로그", text_color="#667085", font=ctk.CTkFont(size=14, weight="bold")).pack(side="left")
        ctk.CTkButton(
            log_header,
            text="로그 보기/숨기기",
            width=130,
            height=28,
            fg_color="#ffffff",
            hover_color="#eef2f7",
            text_color="#344054",
            border_width=1,
            border_color="#d0d5dd",
            command=self._toggle_log_box,
        ).pack(side="right")

        self._log_visible = False
        self._log_box = ctk.CTkTextbox(
            main,
            height=120,
            fg_color="#ffffff",
            text_color="#344054",
            font=ctk.CTkFont(family="Consolas", size=12),
            wrap="word",
        )

        self.log_to_console(f"부서: {self._dept_name}")
        self.log_to_console(f"사용자 ID: {self._user_id}")
        self.log_to_console(f"세션 수집 시간: {SESSION_SEC}초 (버튼 실행 시에만 수집)")
        self.log_to_console("메뉴에서 작업을 선택하세요.")

    def _toggle_log_box(self) -> None:
        if self._log_box is None:
            return
        if self._log_visible:
            self._log_box.pack_forget()
            self._log_visible = False
        else:
            self._log_box.pack(fill="x", pady=(0, 8))
            self._log_visible = True

    def _display_class_value(self, value) -> str:
        try:
            numeric = float(value)
        except Exception:
            return "-"
        if numeric == 1:
            return "스트레스 가능성 높음"
        if numeric == 0.5:
            return "집중 입력 상태"
        return "정상 / 비스트레스"

    def _display_baseline_quality(self, value) -> str:
        text = str(value or "").lower()
        if text in {"missing", "low", ""}:
            return "기준 데이터 보완 필요"
        if text in {"available", "ok", "good"}:
            return "기준 데이터 충분"
        return "기준 데이터 확인 중"

    def _friendly_reason(self, text: str) -> str:
        replacements = {
            "자가 응답 점수가 낮음": "자가 체크 결과가 안정 범위입니다",
            "자가 응답 점수가 높음": "자가 체크에서 부담감이 높게 나타났습니다",
            "자가 응답 점수가 매우 높음": "자가 체크에서 긴장과 부담이 매우 높게 나타났습니다",
            "행동 이상 점수가 낮음": "입력 패턴이 전반적으로 안정적입니다",
            "수정 조급성 지수가 낮음": "급한 수정 패턴이 두드러지지 않았습니다",
            "마우스 불안정성이 낮음": "마우스 사용 패턴이 안정적입니다",
            "개인 기준선 데이터가 부족함": "개인 기준 데이터가 아직 충분하지 않습니다",
            "개인 기준선 데이터가 부족하여 자가 응답과 현재 행동 패턴을 더 크게 반영함": "개인 기준 데이터가 부족해 자가 체크와 현재 입력 패턴을 더 크게 반영했습니다",
            "행동 근거는 부족하여 신뢰도는 낮게 표시": "입력 패턴 근거가 충분하지 않아 신뢰도를 낮게 표시했습니다",
            "자가 응답은 높지만 행동 근거가 부족함": "자가 체크는 높지만 입력 패턴 근거는 아직 약합니다",
            "mouse_jitter가 기준선 대비 증가": "마우스 움직임의 흔들림이 평소보다 증가했습니다",
            "반복 클릭이 기준선 대비 증가": "반복 클릭이 평소보다 늘었습니다",
            "Backspace 연타 후 빠른 재입력과 재수정 패턴 감지": "삭제 후 빠른 재입력과 재수정 흐름이 감지되었습니다",
        }
        result = str(text)
        for before, after in replacements.items():
            result = result.replace(before, after)
        return result

    def _friendly_interpretation(self, text: str) -> str:
        if not text:
            return ""
        replacements = {
            "현재 입력 패턴은 개인 기준선과 크게 벗어나지 않아 정상 작업 상태로 판단함.": "현재 입력 패턴은 평소 작업 특성과 큰 차이가 없어 안정적인 상태로 판단됩니다.",
            "고민 후 빠르게 작성한 패턴으로 판단하여 스트레스 가능성 높음으로 분류하지 않음.": "고민 후 빠르게 입력한 집중 흐름으로 보이며, 스트레스 가능성이 높은 상태로 보지는 않았습니다.",
            "자가 응답이 매우 높아 스트레스 가능성 높음으로 분류하되, 행동 근거가 부족한 경우 신뢰도를 낮게 해석함.": "자가 체크에서 부담이 높게 나타나 스트레스 가능성을 높게 보되, 입력 패턴 근거가 약한 경우 신뢰도는 낮게 해석합니다.",
        }
        result = str(text)
        for before, after in replacements.items():
            result = result.replace(before, after)
        return result

    def _update_result_card(self, result: dict) -> None:
        def apply() -> None:
            if not self._result_labels:
                return
            debug = result.get("scoring_debug", {})
            values = {
                "state": result.get("state", "-"),
                "class_value": self._display_class_value(result.get("class_value", "-")),
                "final_score": f"{result.get('final_score', '-')}/10",
                "confidence": f"{result.get('confidence', '-')}%",
                "self_report_score": result.get("self_report_score", "-"),
                "behavior_anomaly_score": result.get("behavior_anomaly_score", "-"),
                "persistence_score": result.get("persistence_score", "-"),
                "correction_urgency_index": result.get("correction_urgency_index", "-"),
                "baseline_quality": self._display_baseline_quality(result.get("baseline_quality", "-")),
            }
            state = result.get("state")
            color = "#22c55e"
            box_color = "#eefdf3"
            box_border = "#c8f3d6"
            if state == "집중 입력 상태":
                color = "#3b82f6"
                box_color = "#eef4ff"
                box_border = "#c7d7fe"
            elif state == "스트레스 가능성 높음":
                color = "#f97316"
                box_color = "#fff7ed"
                box_border = "#fed7aa"
            if self._state_box is not None:
                self._state_box.configure(fg_color=box_color, border_color=box_border)
            for key, value in values.items():
                label = self._result_labels.get(key)
                if label:
                    label.configure(text=str(value), text_color=color if key == "state" else "#182230")
            reasons = [self._friendly_reason(reason) for reason in result.get("top_reasons", [])[:5]]
            for idx, label in enumerate(self._reason_labels):
                label.configure(text=f"- {reasons[idx]}" if idx < len(reasons) else "-")
            if self._interpretation_label:
                self._interpretation_label.configure(text=self._friendly_interpretation(result.get("interpretation", "")))
            formula_label = self._result_labels.get("formula")
            if formula_label:
                formula_label.configure(text="분석 방식: 자가 체크와 입력 패턴 종합")
            measured_label = self._result_labels.get("measured_at")
            if measured_label:
                measured_label.configure(text=f"측정 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            adjustment_label = self._result_labels.get("adjustment")
            if adjustment_label:
                reason = result.get("score_adjustment_reason") or debug.get("score_adjustment_reason", "")
                adjustment_label.configure(text=f"점수 보정: {reason}" if result.get("score_adjusted") and reason else "")

        self.after(0, apply)

    def _update_onboarding_progress(self, session_name: str = "-", next_hint: str | None = None) -> None:
        def apply() -> None:
            if not self._progress_labels:
                return
            _, zero_count, one_count = count_sessions(self._user_id)
            focus_excluded = count_focus_label_sessions(self._user_id)
            need_zero = max(0, TARGET_PER_LABEL - zero_count)
            need_one = max(0, TARGET_PER_LABEL - one_count)
            total_goal = TARGET_PER_LABEL * 2
            current = min(zero_count, TARGET_PER_LABEL) + min(one_count, TARGET_PER_LABEL)
            percent = round(current / total_goal * 100)
            hint = next_hint
            if hint is None:
                if need_zero > 0 and need_one > 0:
                    hint = "정상 또는 스트레스 가능성 데이터를 추가로 수집하세요"
                elif need_zero > 0:
                    hint = "다음에는 정상 / 비스트레스 라벨이 필요합니다"
                elif need_one > 0:
                    hint = "다음에는 스트레스 가능성 높음 라벨이 필요합니다"
                else:
                    hint = "초기 데이터 충분, 추가 데이터 수집 가능"
            if focus_excluded:
                hint += f" (0.5 집중 입력 라벨 {focus_excluded}개는 학습용 카운트에서 제외)"
            self._progress_labels["normal"].configure(text=f"{zero_count} / {TARGET_PER_LABEL}")
            self._progress_labels["stress"].configure(text=f"{one_count} / {TARGET_PER_LABEL}")
            self._progress_labels["remaining"].configure(text=f"정상 {need_zero}개, 스트레스 {need_one}개")
            self._progress_labels["percent"].configure(text=f"{percent}%")
            self._progress_labels["session"].configure(text=session_name)
            self._progress_labels["next"].configure(text=hint)

        self.after(0, apply)

    def _set_feedback_actions(self, enabled: bool, message: str | None = None, result: dict | None = None) -> None:
        def apply() -> None:
            state = "normal" if enabled else "disabled"
            show_buttons = enabled or self._latest_prediction_result is not None
            for idx, button in enumerate(self._feedback_buttons):
                if show_buttons:
                    button.grid(row=0, column=idx, padx=(0 if idx == 0 else 8, 0))
                else:
                    button.grid_remove()
                button.configure(state=state)
            if result is not None:
                self._configure_feedback_recommendation(result)
            if message and self._feedback_status_label is not None:
                self._feedback_status_label.configure(text=message)

        self.after(0, apply)

    def _configure_feedback_recommendation(self, result: dict) -> None:
        state_text = self._display_class_value(result.get("class_value", result.get("state", 0)))
        if self._feedback_result_label is not None:
            self._feedback_result_label.configure(text=f"이번 측정 결과: {state_text}")

    def _show_prediction_feedback_popup(self, result: dict) -> str:
        state_text = self._display_class_value(result.get("class_value", result.get("state", 0)))
        choice = {"value": "skip"}

        dialog = ctk.CTkToplevel(self)
        dialog.title("측정 결과 반영")
        popup_w = 720
        popup_h = 380
        dialog.geometry(f"{popup_w}x{popup_h}")
        dialog.resizable(False, False)
        dialog.configure(fg_color="#eef3f8")
        dialog.transient(self)
        dialog.grab_set()

        self.update_idletasks()
        x = self.winfo_x() + max(0, (self.winfo_width() - popup_w) // 2)
        y = self.winfo_y() + max(0, (self.winfo_height() - popup_h) // 2)
        dialog.geometry(f"{popup_w}x{popup_h}+{x}+{y}")

        card = ctk.CTkFrame(dialog, fg_color="#ffffff", corner_radius=20, border_width=1, border_color="#dce5f2")
        card.pack(fill="both", expand=True, padx=22, pady=22)

        ctk.CTkLabel(
            card,
            text="측정 결과를 개인 기준 데이터에 반영",
            font=ctk.CTkFont(family="Malgun Gothic", size=23, weight="bold"),
            text_color="#182230",
        ).pack(anchor="w", padx=24, pady=(22, 10))

        color = "#16a34a"
        badge_bg = "#eefdf3"
        badge_border = "#bbf7d0"
        if state_text == "집중 입력 상태":
            color = "#2563eb"
            badge_bg = "#eef4ff"
            badge_border = "#c7d7fe"
        elif state_text == "스트레스 가능성 높음":
            color = "#f97316"
            badge_bg = "#fff7ed"
            badge_border = "#fed7aa"

        badge = ctk.CTkFrame(card, fg_color=badge_bg, corner_radius=16, border_width=1, border_color=badge_border)
        badge.pack(fill="x", padx=24, pady=(0, 14))
        ctk.CTkLabel(
            badge,
            text=f"이번 측정 결과: {state_text}",
            font=ctk.CTkFont(family="Malgun Gothic", size=24, weight="bold"),
            text_color=color,
        ).pack(anchor="w", padx=18, pady=16)

        ctk.CTkLabel(
            card,
            text=(
                "현재 결과가 실제 상태와 맞다면 개인 기준 데이터에 반영할 수 있습니다. "
                "저장된 데이터는 다음 개인화 모델 학습 시 기준 데이터로 활용됩니다. "
                "저장하지 않으려면 오른쪽 위 X를 눌러 닫으면 됩니다."
            ),
            font=ctk.CTkFont(family="Malgun Gothic", size=14),
            text_color="#475467",
            justify="left",
            wraplength=640,
        ).pack(anchor="w", padx=26, pady=(0, 18))

        actions = ctk.CTkFrame(card, fg_color="transparent")
        actions.pack(fill="x", padx=24, pady=(2, 12))

        def close_with(value: str) -> None:
            choice["value"] = value
            dialog.grab_release()
            dialog.destroy()

        normal_primary = state_text == "정상 / 비스트레스"
        stress_primary = state_text == "스트레스 가능성 높음"

        normal_text = "추천: 정상 데이터로 반영" if normal_primary else "정상 데이터로 반영"
        stress_text = "추천: 스트레스 가능성 데이터로 반영" if stress_primary else "스트레스 가능성 데이터로 반영"

        def button_style(primary: bool, color_name: str) -> dict:
            if primary and color_name == "green":
                return {"fg_color": "#16a34a", "hover_color": "#15803d", "text_color": "#ffffff", "height": 46}
            if primary and color_name == "orange":
                return {"fg_color": "#f97316", "hover_color": "#ea580c", "text_color": "#ffffff", "height": 46}
            return {
                "fg_color": "#ffffff",
                "hover_color": "#eef2f7",
                "text_color": "#344054",
                "height": 40,
                "border_width": 1,
                "border_color": "#cbd5e1",
            }

        buttons = [
            (normal_text, lambda: close_with("normal"), normal_primary, "green", 300),
            (stress_text, lambda: close_with("stress"), stress_primary, "orange", 300),
        ]
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)
        for idx, (text, command, primary, color_name, width) in enumerate(buttons):
            ctk.CTkButton(
                actions,
                text=text,
                width=width,
                font=ctk.CTkFont(family="Malgun Gothic", size=14, weight="bold"),
                command=command,
                **button_style(primary, color_name),
            ).grid(row=0, column=idx, padx=(0 if idx == 0 else 12, 0), sticky="ew")

        dialog.protocol("WM_DELETE_WINDOW", lambda: close_with("skip"))
        self.wait_window(dialog)
        return choice["value"]

    def _save_current_prediction_label(self, label: int) -> None:
        if not self._user_id or not self._latest_prediction_session:
            self.log_to_console("[INFO] 저장할 측정 세션이 없습니다.")
            self._set_feedback_actions(False, "저장할 측정 결과가 없습니다. 먼저 현재 스트레스 가능성을 측정하세요.")
            return
        try:
            result = self._latest_prediction_result or {}
            write_session_label_txt(self._user_id, self._latest_prediction_session, label)
            update_dashboard_data(result)

            label_text = "정상 / 비스트레스" if label == 0 else "스트레스 가능성 높음"
            saved_text = "정상 데이터" if label == 0 else "스트레스 가능성 데이터"
            self.log_to_console(f"[INFO] 이번 측정 결과를 '{label_text}' 개인화 데이터로 반영했습니다.")
            self.log_to_console("[INFO] 다음 개인화 모델 학습 시 반영됩니다.")

            try:
                predicted = float(result.get("class_value", -1))
                score = float(result.get("final_score", 0))
                if predicted in {0.0, 1.0} and int(predicted) == label:
                    send_to_server(self._dept_name, score, predicted, result)
                    self.log_to_console("[INFO] 경영진 집계에는 부서 단위 요약만 전송했습니다.")
            except Exception:
                self.log_to_console("[INFO] 경영진 서버 전송은 건너뛰었습니다.")

            self._update_onboarding_progress(
                self._latest_prediction_session,
                "반영 완료. 초기 데이터가 충분하면 개인화 모델 학습을 다시 진행할 수 있습니다",
            )
            self._set_feedback_actions(False, f"{saved_text}로 반영되었습니다. 다음 모델 학습 시 개인 기준에 포함됩니다.")
        except Exception as e:
            self.log_to_console(f"[ERROR] 결과 반영 실패: {e}")
            self._set_feedback_actions(True, "결과 반영 중 오류가 발생했습니다. 다시 선택해 주세요.")

    def _skip_current_prediction_save(self) -> None:
        self.log_to_console("[INFO] 이번 측정 결과는 개인화 데이터에 저장하지 않았습니다.")
        self._set_feedback_actions(False, "이번 결과는 저장하지 않았습니다.")

    def _open_personal_dashboard(self) -> None:
        try:
            html_path = BASE_DIR / "dashboard.html"
            if not html_path.is_file():
                self.log_to_console(f"[오류] HTML 없음: {html_path}")
                return
            self.log_to_console(f"브라우저: {html_path.name}")
            webbrowser.open(html_path.resolve().as_uri())
        except Exception as e:
            self.log_to_console(f"[오류] 대시보드: {e}")

    def _is_executive_server_running(self) -> bool:
        try:
            response = requests.get(EXECUTIVE_REPORT_URL, timeout=0.8)
            return response.status_code == 200
        except Exception:
            return False

    def _start_executive_server_if_needed(self) -> bool:
        if self._is_executive_server_running():
            return True
        if not EXECUTIVE_SERVER_SCRIPT.exists():
            return False
        try:
            kwargs = {
                "cwd": BASE_DIR,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if sys.platform.startswith("win") and hasattr(subprocess, "CREATE_NO_WINDOW"):
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            self._executive_server_process = subprocess.Popen(
                [sys.executable, str(EXECUTIVE_SERVER_SCRIPT)],
                **kwargs,
            )
            for _ in range(12):
                if self._is_executive_server_running():
                    return True
                time.sleep(0.25)
        except Exception:
            return False
        return self._is_executive_server_running()

    def open_executive_dashboard(self) -> None:
        html_path = BASE_DIR / "executive_dashboard.html"
        if not html_path.is_file():
            messagebox.showerror("경영진 대시보드", "경영진 대시보드를 열 수 없습니다. executive_dashboard.html을 확인해주세요.")
            return

        server_ready = self._start_executive_server_if_needed()
        try:
            webbrowser.open(html_path.resolve().as_uri())
            if getattr(self, "_log_box", None) is not None:
                self.log_to_console("경영진 대시보드를 열었습니다. 개인 측정 없이 부서별 집계만 표시합니다.")
                if not server_ready:
                    self.log_to_console("경영진 서버 연결이 없으면 대시보드에 예시 데이터가 표시될 수 있습니다.")
        except Exception:
            messagebox.showwarning(
                "경영진 대시보드",
                "경영진 대시보드를 열 수 없습니다. executive_server.py를 실행해주세요.",
            )

    def _on_onboarding_clicked(self) -> None:
        self._set_actions_enabled(False)
        threading.Thread(target=self._onboarding_worker, daemon=True, name="Onboarding").start()

    def _onboarding_worker(self) -> None:
        user_id = self._user_id
        try:
            self.log_to_console("[INFO] 초기/추가 개인화 데이터 수집 시작")
            self.log_to_console(f"[INFO] 목표: 비스트레스 {TARGET_PER_LABEL}개, 스트레스 {TARGET_PER_LABEL}개")
            self._update_onboarding_progress("-", "현재 진행상황 확인 중")
            _, zero_count, one_count = count_sessions(user_id)
            focus_excluded = count_focus_label_sessions(user_id)
            need_zero = max(0, TARGET_PER_LABEL - zero_count)
            need_one = max(0, TARGET_PER_LABEL - one_count)
            self.log_to_console(
                f"[STATUS] 비스트레스 {zero_count}개, 스트레스 {one_count}개 "
                f"(필요: {need_zero} / {need_one})"
            )
            if focus_excluded:
                self.log_to_console(f"[INFO] 0.5 집중 입력 라벨 세션 {focus_excluded}개는 학습용 0/1 카운트에서 제외했습니다.")
            if zero_count >= TARGET_PER_LABEL and one_count >= TARGET_PER_LABEL:
                proceed = self._main_thread_call(
                    lambda: messagebox.askyesno(
                        "추가 데이터 수집",
                        "초기 개인화 데이터가 이미 충분합니다.\n그래도 추가 데이터를 1개 더 수집하시겠습니까?",
                        parent=self,
                    )
                )
                if not proceed:
                    self.log_to_console("[INFO] 추가 데이터 수집을 취소했습니다.")
                    self._update_onboarding_progress("-", "초기 데이터 충분, 추가 데이터 수집 가능")
                    return

            session_name = next_session_name(user_id, prefix="session")
            self._update_onboarding_progress(session_name, "현재 세션 수집 중")
            self.log_to_console(f"[STEP] {session_name} — {SESSION_SEC}초간 키보드·마우스 사용")
            ok = collect_one_session(user_id, session_name, SESSION_SEC, 0)
            if not ok:
                self.log_to_console("[ERROR] 세션 수집 실패")
                self._update_onboarding_progress(session_name, "세션 수집 실패")
                return

            def ask_label() -> int | None:
                return simpledialog.askinteger(
                    "수집 데이터 상태 선택",
                    "방금 수집한 상태를 선택하세요.\n0 = 정상 / 비스트레스\n1 = 스트레스 가능성 높음\n\n※ 집중 입력 상태는 시스템이 자동으로 분류합니다.",
                    minvalue=0,
                    maxvalue=1,
                    parent=self,
                )

            label = self._main_thread_call(ask_label)
            if label is None:
                self.log_to_console("[INFO] 사용자가 취소했습니다. label.txt 미저장")
                self._update_onboarding_progress(session_name, "라벨 입력 취소")
                return

            write_session_label_txt(user_id, session_name, label)
            self.log_to_console(f"[DONE] {session_name} 저장 → label={label}")
            self._update_onboarding_progress(session_name, "라벨 저장 완료")
        except Exception as e:
            self.log_to_console(f"[ERROR] 온보딩: {e}")
        finally:
            self._set_actions_enabled(True)

    def _on_train_clicked(self) -> None:
        self._set_actions_enabled(False)
        threading.Thread(target=self._train_worker, daemon=True, name="Train").start()

    def _train_worker(self) -> None:
        user_id = self._user_id
        try:
            self.log_to_console("[INFO] 개인화 모델 학습 시작")
            self.log_to_console("[STEP] 개인 행동 데이터 특징 생성")
            code = run_python(MAKE_PERSONAL_SCRIPT, [])
            if code != 0:
                self.log_to_console("[ERROR] 개인 특징 생성 실패")
                return
            self.log_to_console("[STEP] 데이터 병합")
            if not merge_features():
                self.log_to_console("[ERROR] 병합 실패")
                return
            self.log_to_console("[STEP] 모델 학습 (subprocess)")
            code = run_python(
                TRAIN_SCRIPT,
                [
                    "--input_csv",
                    str(FINAL_MERGED),
                    "--personal_user_ids",
                    user_id,
                    "--personal_weight",
                    str(PERSONAL_WEIGHT),
                ],
            )
            if code != 0:
                self.log_to_console("[ERROR] 모델 학습 실패")
                return
            self.log_to_console("[DONE] 개인화 모델 학습 완료")
            self._update_onboarding_progress("-", "모델 학습 완료")
        except Exception as e:
            self.log_to_console(f"[ERROR] 학습: {e}")
        finally:
            self._set_actions_enabled(True)

    def _on_predict_clicked(self) -> None:
        self._set_actions_enabled(False)
        threading.Thread(target=self._predict_worker, daemon=True, name="Predict").start()

    def _ask_self_report(self) -> dict[str, float] | None:
        dialog = ctk.CTkToplevel(self)
        dialog.title("빠른 자가 체크")
        dialog.geometry("520x520")
        dialog.minsize(500, 420)
        dialog.configure(fg_color="#f6f8fb")
        dialog.transient(self)
        dialog.grab_set()

        result: dict[str, float] | None = None
        vars_ = {
            "tension": tk.DoubleVar(value=3),
            "pressure_hurry": tk.DoubleVar(value=3),
            "control": tk.DoubleVar(value=7),
            "workload": tk.DoubleVar(value=5),
            "hurry": tk.DoubleVar(value=5),
            "irritability": tk.DoubleVar(value=5),
        }

        ctk.CTkLabel(
            dialog,
            text="빠른 자가 체크",
            font=ctk.CTkFont(family="Malgun Gothic", size=22, weight="bold"),
        ).pack(anchor="w", padx=24, pady=(22, 4))
        ctk.CTkLabel(
            dialog,
            text="측정 버튼을 눌렀을 때만 입력합니다. 기본값 그대로 빠르게 진행할 수 있습니다.",
            text_color="#667085",
            wraplength=460,
        ).pack(anchor="w", padx=24, pady=(0, 14))

        def add_slider(parent, key: str, text: str) -> None:
            row = ctk.CTkFrame(parent, fg_color="transparent")
            row.pack(fill="x", padx=24, pady=8)
            value_label = ctk.CTkLabel(row, text=f"{vars_[key].get():.0f}", width=28)
            value_label.pack(side="right", padx=(8, 0))
            ctk.CTkLabel(row, text=text, width=230, anchor="w").pack(side="left")
            slider = ctk.CTkSlider(row, from_=0, to=10, variable=vars_[key], number_of_steps=10)
            slider.pack(side="left", fill="x", expand=True, padx=10)

            def refresh(_value=None) -> None:
                value_label.configure(text=f"{vars_[key].get():.0f}")

            slider.configure(command=refresh)

        add_slider(dialog, "tension", "현재 긴장감")
        add_slider(dialog, "pressure_hurry", "부담감/조급함")
        add_slider(dialog, "control", "여유감/통제감")

        detail_frame = ctk.CTkFrame(dialog, fg_color="#ffffff", corner_radius=8, border_width=1, border_color="#e5e9f0")
        detail_open = tk.BooleanVar(value=False)

        def toggle_detail() -> None:
            if detail_open.get():
                detail_frame.pack_forget()
                detail_open.set(False)
                detail_btn.configure(text="상세 체크 열기")
            else:
                detail_frame.pack(fill="x", padx=18, pady=(8, 4))
                detail_open.set(True)
                detail_btn.configure(text="상세 체크 닫기")

        detail_btn = ctk.CTkButton(
            dialog,
            text="상세 체크 열기",
            fg_color="#ffffff",
            hover_color="#eef2f7",
            text_color="#344054",
            border_width=1,
            border_color="#d0d5dd",
            command=toggle_detail,
        )
        detail_btn.pack(anchor="w", padx=24, pady=(8, 4))
        add_slider(detail_frame, "workload", "업무/과제 부담감")
        add_slider(detail_frame, "hurry", "조급함")
        add_slider(detail_frame, "irritability", "짜증/예민함")

        button_row = ctk.CTkFrame(dialog, fg_color="transparent")
        button_row.pack(fill="x", padx=24, pady=(18, 20))

        def finish() -> None:
            nonlocal result
            result = {
                "tension": float(vars_["tension"].get()),
                "pressure_hurry": float(vars_["pressure_hurry"].get()),
                "control": float(vars_["control"].get()),
                "detailed": bool(detail_open.get()),
            }
            if detail_open.get():
                result.update({
                    "workload": float(vars_["workload"].get()),
                    "hurry": float(vars_["hurry"].get()),
                    "irritability": float(vars_["irritability"].get()),
                })
            dialog.destroy()

        def cancel() -> None:
            dialog.destroy()

        ctk.CTkButton(button_row, text="빠른 측정 시작", height=42, command=finish).pack(side="right", padx=(8, 0))
        ctk.CTkButton(button_row, text="취소", height=42, fg_color="#4b5563", hover_color="#6b7280", command=cancel).pack(side="right")
        dialog.protocol("WM_DELETE_WINDOW", cancel)
        self.wait_window(dialog)
        return result

    def _predict_worker(self) -> None:
        user_id = self._user_id
        dept = self._dept_name
        try:
            test_dir = PERSONAL_DIR / user_id
            test_dir.mkdir(parents=True, exist_ok=True)
            session_name = next_session_name(user_id, prefix="test")
            task_context = detect_task_context()

            self.log_to_console("[STEP] 빠른 3문항 자가 체크 후 측정을 시작합니다.")
            self_report = self._main_thread_call(self._ask_self_report)
            if self_report is None:
                self.log_to_console("[INFO] 자가 응답 입력 취소 — 측정 중단")
                return

            self.log_to_console(f"[INFO] 측정 시작 — {session_name}, {SESSION_SEC}초, context={task_context}")
            ok = collect_one_session(user_id, session_name, SESSION_SEC, 0, task_context)
            if not ok:
                self.log_to_console("[ERROR] 현재 상태 측정 실패")
                return

            self.log_to_console("[STEP] 문서·코딩 작업 기반 스트레스 상태 분석")
            predict_args = [
                "--session_dir",
                str(PERSONAL_DIR / user_id / session_name),
                "--model_path",
                str(BEST_MODEL),
                "--duration_sec",
                str(SESSION_SEC),
                "--dept",
                dept,
                "--user_id",
                user_id,
                "--task_context",
                task_context,
                "--tension",
                str(self_report["tension"]),
                "--control",
                str(self_report["control"]),
            ]
            if self_report.get("detailed"):
                predict_args.extend([
                    "--workload",
                    str(self_report["workload"]),
                    "--hurry",
                    str(self_report["hurry"]),
                    "--irritability",
                    str(self_report["irritability"]),
                ])
            else:
                predict_args.extend([
                    "--pressure_hurry",
                    str(self_report["pressure_hurry"]),
                ])
            code, output = run_python_capture(
                PREDICT_SCRIPT,
                predict_args,
            )
            if code != 0:
                self.log_to_console("[ERROR] 예측 실패")
                self.log_to_console(output[:2000] if output else "")
                return

            if not PREDICTION_RESULT.exists():
                self.log_to_console("[ERROR] prediction_result.json 생성 실패")
                return
            result = json.loads(PREDICTION_RESULT.read_text(encoding="utf-8"))
            pred = float(result["class_value"])
            score = float(result["final_score"])

            debug = result.get("scoring_debug", {})
            self.log_to_console(f"[RESULT] 최종 상태: {result['state']}")
            self.log_to_console(f"[RESULT] 점수: {score} / 10")
            if result.get("score_adjusted"):
                self.log_to_console(f"[RESULT] 원점수: {result.get('raw_final_score', '-')} / 10")
                self.log_to_console(f"[RESULT] 점수 보정: {result.get('score_adjustment_reason', '-')}")
            self.log_to_console(f"[RESULT] 분류 결과: {self._display_class_value(pred)}")
            self.log_to_console(f"[RESULT] 신뢰도: {result['confidence']}%")
            self.log_to_console(f"[RESULT] 적용 공식: {debug.get('final_score_formula', '-')}")
            for reason in result.get("top_reasons", [])[:5]:
                self.log_to_console(f"[RESULT] 근거: {reason}")
            top_debug = sorted(debug.get("feature_ratios", []), key=lambda item: item.get("final_feature_score", 0), reverse=True)[:3]
            for row in top_debug:
                self.log_to_console(
                    f"[DEBUG] {row['feature']}: 현재 {row['current']} / 기준선 {row['baseline']} / "
                    f"ratio {row['ratio']} / score {row['final_feature_score']}"
                )
            self.log_to_console(result.get("interpretation", ""))
            self._update_result_card(result)
            self._latest_prediction_session = session_name
            self._latest_prediction_result = result
            update_dashboard_data(result)
            self._set_feedback_actions(False, "측정 결과가 준비되었습니다. 팝업에서 개인 기준 데이터 반영 여부를 선택하세요.", result)
            self.log_to_console("[INFO] 측정 결과 반영 선택 팝업을 표시합니다.")

            feedback_choice = self._main_thread_call(lambda: self._show_prediction_feedback_popup(result))
            if feedback_choice == "normal":
                self._save_current_prediction_label(0)
            elif feedback_choice == "stress":
                self._save_current_prediction_label(1)
            else:
                self._skip_current_prediction_save()
        except Exception as e:
            self.log_to_console(f"[ERROR] 측정/예측: {e}")
        finally:
            self._set_actions_enabled(True)


def main() -> None:
    app = MentalCareIntegratedApp()
    app.mainloop()


if __name__ == "__main__":
    main()
