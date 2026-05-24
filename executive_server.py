from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import json
import os

app = Flask(__name__)
CORS(app) # HTML 파일에서 서버로 통신할 수 있게 보안 해제

DATA_FILE = "executive_data.json"

# [수정] 9개 부서의 데이터가 섞이지(통합되지) 않도록 독립적으로 찍어내는 함수
def create_default_stats():
    return {
        "history": [0.0, 0.0, 0.0, 0.0], # 4주치 과거 데이터
        "current_week_sum": 0.0,
        "current_week_count": 0,
        "last_week_no": datetime.now().isocalendar()[1], # 현재 주차
        "trend": "0",
        "class_counts": {"0": 0, "0.5": 0, "1": 0},
        "last_measured_at": None,
    }

# 서버 메모리에 저장될 부서별 실시간 데이터베이스
department_db = {
    "1. 인사부": create_default_stats(),
    "2. 재무부": create_default_stats(),
    "3. 마케팅부": create_default_stats(),
    "4. 생산부": create_default_stats(),
    "5. 연구개발부": create_default_stats(),
    "6. 구매부": create_default_stats(),
    "7. IT부": create_default_stats(),
    "8. 법무부": create_default_stats(),
    "9. 영업부": create_default_stats()
}

def load_db():
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 데이터 로드 실패: {e}")
    
    # 파일이 없으면 초기 데이터 생성
    depts = ["1. 인사부", "2. 재무부", "3. 마케팅부", "4. 생산부", 
             "5. 연구개발부", "6. 구매부", "7. IT부", "8. 법무부", "9. 영업부"]
    return {name: create_default_stats() for name in depts}

# [추가] 현재 메모리의 데이터를 파일로 저장하는 함수
def save_db():
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(department_db, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"❌ 데이터 저장 실패: {e}")

# 서버 시작 시 데이터 로드
department_db = load_db()

# [API 1] 직원 PC에서 데이터를 전송받는 네트워크 주소 (POST)
@app.route('/api/send_data', methods=['POST'])
def receive_data():
    data = request.json # 직원 PC에서 보낸 JSON 데이터 파싱
    dept_name = data.get("dept")
    score = float(data.get("score", 0))
    class_value = str(data.get("class_value", data.get("label", 0)))
    if class_value == "0.0":
        class_value = "0"
    if class_value not in {"0", "0.5", "1"}:
        class_value = "0"

    # 서버 DB에 누적 계산
    if dept_name in department_db:
        dept = department_db[dept_name]
        dept.setdefault("class_counts", {"0": 0, "0.5": 0, "1": 0})
        now_week = datetime.now().isocalendar()[1]

        # [핵심] 주차가 바뀌었을 때 (Rolling)
        if now_week != dept["last_week_no"]:
            # 현재 주차 평균 계산
            this_week_avg = round(dept["current_week_sum"] / dept["current_week_count"], 1) if dept["current_week_count"] > 0 else dept["history"][-1]
            # 히스토리 업데이트: 제일 오래된 주(index 0) 삭제, 방금 끝난 주 추가
            dept["history"].pop(0)
            dept["history"].append(this_week_avg)
            # 초기화
            dept["current_week_sum"] = 0.0
            dept["current_week_count"] = 0
            dept["class_counts"] = {"0": 0, "0.5": 0, "1": 0}
            dept["last_week_no"] = now_week
        
        dept["current_week_sum"] += score
        dept["current_week_count"] += 1
        dept["class_counts"][class_value] = dept["class_counts"].get(class_value, 0) + 1
        dept["last_measured_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # [수정] 점수를 받을 때마다 즉시 파일에 저장!
        save_db()

    print(f"📡 [데이터 수신 성공] {dept_name}: {score}점 (누적 횟수: {department_db.get(dept_name, {}).get('current_week_count', 0)})")
    return jsonify({"status": "success", "message": "서버 수신 완료"})

# [API 2] 경영진 대시보드에 통합 데이터를 쏴주는 주소 (GET)
@app.route('/api/get_executive_report', methods=['GET'])
def send_report():
    report_list = []
    
    for dept_name, stats in department_db.items():
        stats.setdefault("class_counts", {"0": 0, "0.5": 0, "1": 0})
        if stats["current_week_count"] > 0:
            avg_score = round(stats["current_week_sum"] / stats["current_week_count"], 1)
        else:
            # 이번 주 데이터가 아직 없으면 가장 최근(저번 주) 점수를 기본값으로 표시
            avg_score = stats["history"][-1] 
        
        total = max(1, sum(int(v) for v in stats["class_counts"].values()))
        report_list.append({
            "dept_name": dept_name,
            "avg_score": avg_score,
            "history": stats["history"], # [추가됨] 대시보드 그래프를 그리기 위해 과거 4주치 기록도 같이 보냄
            "trend": stats["trend"],
            "class_counts": stats["class_counts"],
            "normal_ratio": round(stats["class_counts"].get("0", 0) / total * 100, 1),
            "focus_ratio": round(stats["class_counts"].get("0.5", 0) / total * 100, 1),
            "stress_ratio": round(stats["class_counts"].get("1", 0) / total * 100, 1),
            "last_measured_at": stats.get("last_measured_at") or "-",
        })
        
    return jsonify(report_list) # 경영진 HTML로 데이터 전송

if __name__ == '__main__':
    print("===================================================")
    print(" 🏢 [Mental Care] 전사 중앙 수집 서버(API) 가동 중...")
    print(" 🌐 네트워크 주소: http://127.0.0.1:5000")
    print("===================================================")
    app.run(port=5000, debug=True)
