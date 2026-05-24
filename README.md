# PC 작업 맥락 기반 개인화 스트레스 가능성 분석 시스템

# 스트레스 탐지 프로그램

PC 작업 맥락 기반 개인화 스트레스 가능성 분석 시스템입니다.

이 프로젝트는 키보드/마우스 입력만으로 스트레스를 확정하는 도구가 아닙니다.
PC 작업 맥락 안에서 정상 입력, 집중 입력, 스트레스성 조급 입력을 구분하고,
자가 응답, 개인 기준선, 행동 이상, 무입력 예외 처리, 지속성을 함께 해석해
스트레스 가능성과 판단 근거를 제공합니다.

## 최종 상태

최종 출력 state는 아래 3개만 사용합니다.

- `0`: 정상 / 비스트레스
- `0.5`: 집중 입력 상태
- `1`: 스트레스 가능성 높음

판단 보류, 모니터링 대기, 주관적 부담 같은 개념은 최종 state로 쓰지 않고,
내부 처리나 `top_reasons` 문구로만 사용합니다.

## 주요 파일

- `main_system.py`: GUI 실행 파일
- `main.py`: 콘솔 실행 파일
- `collect_behavior_features.py`: 키보드/마우스 로그 수집
- `feature_extraction.py`: 학습/예측 공통 feature 추출
- `baseline_manager.py`: 개인별, 작업 맥락별 baseline 관리
- `task_context.py`: 활성 창을 분류해 `coding`, `document`, `communication`, `unknown`만 저장
- `stress_decision.py`: 최종 3단계 판정 로직
- `make_personal_features.py`: 개인 세션 feature CSV 생성 및 baseline 갱신
- `train_personalized_model.py`: 보조 모델 학습
- `predict_single_session.py`: 단일 세션 분석 및 `prediction_result.json` 출력
- `dashboard.html`, `chart_data.js`: 개인 대시보드
- `executive_server.py`, `executive_dashboard.html`: 부서 단위 대시보드

## 실행

```bash
pip install -r requirements.txt
python main_system.py
```

콘솔 버전:

```bash
python main.py
```

## 데이터 흐름

1. 사용자가 부서와 직원 ID를 선택합니다.
2. 측정 전에 5문항 자가 응답을 입력합니다.
3. 수집기는 실제 입력 문자와 활성 창 제목 원문을 저장하지 않고, 키 범주와 입력 시각, 마우스 통계, `task_context`만 저장합니다.
4. `feature_extraction.py`가 키 입력 간격, 수정 조급성 지수, 마우스 불안정성, 무입력 시간 등을 공통 계산합니다.
5. `baseline_manager.py`가 `data/personal/{user_id}/baseline.json` 기준선을 읽고 baseline 대비 ratio를 붙입니다.
6. 기존 모델 확률은 `behavior_anomaly_score`의 보조 신호로만 반영됩니다.
7. `stress_decision.py`가 최종 상태 3개 중 하나를 결정합니다.
8. `prediction_result.json`과 `chart_data.js`가 대시보드에 반영됩니다.

## 개인정보 보호

저장하지 않는 항목:

- 사용자가 입력한 실제 문장
- 비밀번호
- 메일 제목 원문
- 활성 창 제목 원문

저장하는 항목:

- 키 입력 시간
- 키 종류 범주
- 입력 간격
- 백스페이스/삭제 비율
- 클릭 수와 마우스 이동 통계
- `task_context`

실제 `data/` 폴더, 원본 `.tsv`, `.csv`, `.pkl`, 예측 결과 파일은 GitHub에 올리지 않습니다.
테스트용 `sample_data/`만 버전 관리합니다.
