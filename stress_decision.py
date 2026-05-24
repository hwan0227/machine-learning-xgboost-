from __future__ import annotations

import json
from pathlib import Path


STATE_NORMAL = "정상 / 비스트레스"
STATE_FOCUS = "집중 입력 상태"
STATE_STRESS = "스트레스 가능성 높음"
VALID_STATES = {STATE_NORMAL, STATE_FOCUS, STATE_STRESS}

CLASS_VALUES = {
    STATE_NORMAL: 0,
    STATE_FOCUS: 0.5,
    STATE_STRESS: 1,
}

ANOMALY_FEATURES = [
    "backspace_ratio",
    "delete_ratio",
    "correction_ratio",
    "std_key_interval",
    "pause_ratio",
    "burst_after_pause",
    "repeat_key_ratio",
    "repeat_click_count",
    "direction_change_count",
    "mouse_jitter",
    "click_interval_std",
    "correction_urgency_index",
    "correction_loop_score",
    "post_idle_burst_score",
]

ANOMALY_WEIGHTS = {
    "correction_urgency_index": 3.0,
    "correction_loop_score": 2.0,
    "repeat_click_count": 2.0,
    "mouse_jitter": 2.0,
    "click_interval_std": 1.5,
    "backspace_ratio": 1.5,
    "delete_ratio": 1.2,
    "correction_ratio": 1.5,
    "post_idle_burst_score": 1.2,
}

DAMPENED_BY_CONTEXT = {
    "key_count",
    "burst_after_pause",
    "std_key_interval",
    "pause_ratio",
    "post_idle_burst_score",
}

NOT_DAMPENED = {
    "correction_urgency_index",
    "repeat_click_count",
    "mouse_jitter",
    "click_interval_std",
    "correction_loop_score",
}


def clamp(value: float, low: float = 0.0, high: float = 10.0) -> float:
    return max(low, min(high, float(value)))


def self_report_score(
    tension: float,
    workload: float | None = None,
    hurry: float | None = None,
    irritability: float | None = None,
    control: float = 10.0,
    pressure_hurry: float | None = None,
) -> float:
    if pressure_hurry is not None:
        return round(clamp((tension + pressure_hurry + (10 - control)) / 3.0), 2)
    workload = 0.0 if workload is None else workload
    hurry = 0.0 if hurry is None else hurry
    irritability = 0.0 if irritability is None else irritability
    return round(clamp((tension + workload + hurry + irritability + (10 - control)) / 5.0), 2)


def ratio_to_score(ratio: float) -> float:
    try:
        ratio = float(ratio)
    except Exception:
        return 0.0
    if ratio <= 1.3:
        return 0.0
    if ratio >= 3.0:
        return 10.0
    return round((ratio - 1.3) / (3.0 - 1.3) * 10.0, 2)


def context_dampening(task_context: str, feature: str) -> float:
    if feature not in DAMPENED_BY_CONTEXT or feature in NOT_DAMPENED:
        return 1.0
    if task_context == "coding":
        return 0.75
    if task_context == "document":
        return 0.85
    if task_context == "communication":
        return 0.95
    return 1.0


def calculate_rule_based_behavior_score_with_debug(features: dict, task_context: str) -> tuple[float, dict, list[dict]]:
    scores = {}
    debug_rows = []
    for feature in ANOMALY_FEATURES:
        ratio = features.get(f"{feature}_baseline_ratio", 1.0)
        raw_score = ratio_to_score(ratio)
        damping = context_dampening(task_context, feature)
        score = raw_score * damping
        scores[feature] = round(clamp(score), 2)

    if "correction_urgency_index" in features:
        scores["correction_urgency_index"] = max(
            scores.get("correction_urgency_index", 0.0),
            clamp(float(features.get("correction_urgency_index", 0.0))),
        )
    if "mouse_jitter" in features:
        scores["mouse_jitter"] = max(scores.get("mouse_jitter", 0.0), clamp(float(features.get("mouse_jitter", 0.0))))

    if not scores:
        return 0.0, scores, debug_rows
    weighted_sum = 0.0
    weight_sum = 0.0
    for feature, score in scores.items():
        weight = ANOMALY_WEIGHTS.get(feature, 1.0)
        weighted_sum += score * weight
        weight_sum += weight
        ratio = features.get(f"{feature}_baseline_ratio", 1.0)
        damping = context_dampening(task_context, feature)
        debug_rows.append(
            {
                "feature": feature,
                "current": round(float(features.get(feature, 0.0)), 4),
                "baseline": round(float(features.get(f"{feature}_baseline_value", 1.0)), 4),
                "ratio": round(float(ratio), 4),
                "raw_score": round(ratio_to_score(ratio), 2),
                "weight": weight,
                "damping": damping,
                "final_feature_score": round(score, 2),
            }
        )
    return round(weighted_sum / weight_sum, 2), scores, debug_rows


def calculate_rule_based_behavior_score(features: dict, task_context: str) -> tuple[float, dict]:
    score, scores, _debug = calculate_rule_based_behavior_score_with_debug(features, task_context)
    return score, scores


def combine_behavior_score(rule_score: float, model_probability_score: float | None) -> float:
    if model_probability_score is None:
        return round(clamp(rule_score), 2)
    return round(clamp(0.6 * rule_score + 0.4 * clamp(model_probability_score)), 2)


def load_history(path: str | Path = "stress_history.json") -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(history: list[dict], path: str | Path = "stress_history.json") -> None:
    path = Path(path)
    path.write_text(json.dumps(history[-30:], indent=2, ensure_ascii=False), encoding="utf-8")


def calculate_persistence_score(history: list[dict], behavior_anomaly_score: float, correction_urgency_index: float) -> float:
    recent = history[-3:]
    repeated_stress = sum(1 for item in recent if item.get("class_value") == 1)
    repeated_urgency = sum(1 for item in recent if float(item.get("correction_urgency_index", 0.0)) >= 6.0)
    score = 0.0
    score += repeated_stress * 2.0
    score += repeated_urgency * 1.5
    if behavior_anomaly_score >= 6.0:
        score += 2.0
    if correction_urgency_index >= 6.0:
        score += 2.0
    return round(clamp(score), 2)


def previous_result(history: list[dict]) -> dict:
    if not history:
        return {
            "state": STATE_NORMAL,
            "class_value": 0,
            "final_score": 0.0,
            "confidence": 60,
        }
    last = history[-1]
    if last.get("state") not in VALID_STATES:
        last["state"] = STATE_NORMAL
        last["class_value"] = 0
    return last


def _high_count(self_score: float, behavior_score: float, persistence_score: float) -> int:
    return sum(
        [
            self_score >= 6.5,
            behavior_score >= 6.0,
            persistence_score >= 6.0,
        ]
    )


def _confidence(final_score: float, high_count: int, state: str) -> int:
    if state == STATE_STRESS:
        return int(max(70, min(92, 68 + high_count * 7 + (final_score - 7.0) * 4)))
    if state == STATE_FOCUS:
        return int(max(68, min(86, 78 - abs(final_score - 3.5) * 2)))
    return int(max(70, min(90, 88 - final_score * 2)))


def _result_confidence(final_score: float, high_count: int, state: str, self_score: float, behavior_score: float) -> int:
    confidence = _confidence(final_score, high_count, state)
    if state == STATE_STRESS and self_score >= 9.5 and behavior_score < 3.5:
        return int(max(52, min(confidence, 64)))
    return confidence


def decide_state(
    *,
    task_context: str,
    self_report_score_value: float,
    behavior_anomaly_score: float,
    persistence_score: float,
    model_probability_score: float | None,
    features: dict,
    feature_scores: dict,
    history: list[dict],
    scoring_debug: dict | None = None,
) -> dict:
    activity_idle_time = float(features.get("activity_idle_time", 0.0))
    correction_urgency = float(features.get("correction_urgency_index", 0.0))
    history_count = len(history)
    baseline_quality = str(features.get("baseline_quality", "missing"))
    if history_count == 0:
        self_weight = 0.65 if baseline_quality not in {"missing", "low"} else 0.72
        behavior_weight = 1.0 - self_weight
        persistence_weight = 0.0
        formula = f"{self_weight:.2f}*self + {behavior_weight:.2f}*behavior"
    elif history_count < 3:
        self_weight = 0.55 if baseline_quality not in {"missing", "low"} else 0.62
        behavior_weight = 0.35
        persistence_weight = 1.0 - self_weight - behavior_weight
        formula = f"{self_weight:.2f}*self + {behavior_weight:.2f}*behavior + {persistence_weight:.2f}*persistence"
    else:
        self_weight = 0.50
        behavior_weight = 0.30
        persistence_weight = 0.20
        formula = "0.50*self + 0.30*behavior + 0.20*persistence"
    raw_final_score = round(
        self_weight * self_report_score_value
        + behavior_weight * behavior_anomaly_score
        + persistence_weight * persistence_score,
        2,
    )
    final_score = raw_final_score

    reasons: list[str] = []
    triggered_rules: list[str] = []
    previous = previous_result(history)

    if 60 <= activity_idle_time < 300:
        result = dict(previous)
        scoring_debug = scoring_debug or {}
        scoring_debug.update(
            {
                "history_count": history_count,
                "baseline_quality": baseline_quality,
                "final_score_formula": "held_previous_state_due_to_idle",
                "self_report_score": round(self_report_score_value, 2),
                "behavior_anomaly_score": round(behavior_anomaly_score, 2),
                "persistence_score": round(persistence_score, 2),
                "triggered_rules": ["60 <= activity_idle_time < 300"],
            }
        )
        result.update(
            {
                "task_context": task_context,
                "model_probability_score": model_probability_score,
                "activity_idle_time": round(activity_idle_time, 2),
                "correction_urgency_index": round(correction_urgency, 2),
                "top_reasons": ["긴 무입력 이후 입력 재개 패턴 확인"],
                "interpretation": "1분 이상 무입력 이후의 구간이므로 새 점수를 확정하지 않고 직전 상태를 유지함.",
                "baseline_quality": baseline_quality,
                "scoring_debug": scoring_debug,
            }
        )
        return _with_required_fields(result, self_report_score_value, behavior_anomaly_score, persistence_score)

    if activity_idle_time >= 300:
        result = dict(previous)
        scoring_debug = scoring_debug or {}
        scoring_debug.update(
            {
                "history_count": history_count,
                "baseline_quality": baseline_quality,
                "final_score_formula": "held_previous_state_due_to_idle",
                "self_report_score": round(self_report_score_value, 2),
                "behavior_anomaly_score": round(behavior_anomaly_score, 2),
                "persistence_score": round(persistence_score, 2),
                "triggered_rules": ["activity_idle_time >= 300"],
            }
        )
        result.update(
            {
                "task_context": task_context,
                "model_probability_score": model_probability_score,
                "activity_idle_time": round(activity_idle_time, 2),
                "correction_urgency_index": round(correction_urgency, 2),
                "top_reasons": ["5분 이상 활동 없음으로 해당 구간 분석 제외"],
                "interpretation": "5분 이상 활동이 없어 학습/평가 구간에서 제외하고 직전 상태를 유지함.",
                "baseline_quality": baseline_quality,
                "scoring_debug": scoring_debug,
            }
        )
        return _with_required_fields(result, self_report_score_value, behavior_anomaly_score, persistence_score)

    high_count = _high_count(self_report_score_value, behavior_anomaly_score, persistence_score)
    repeat_click_high = feature_scores.get("repeat_click_count", 0.0) >= 6.0 or float(features.get("repeat_click_count", 0.0)) >= 3
    jitter_high = feature_scores.get("mouse_jitter", 0.0) >= 6.0 or float(features.get("mouse_jitter", 0.0)) >= 6.0
    stress_by_core_rule = raw_final_score >= 7.0 and high_count >= 2
    stress_by_urgency_rule = (
        self_report_score_value >= 7.0
        and correction_urgency >= 6.0
    )
    stress_by_strong_self_rule = (
        (self_report_score_value >= 8.5 and behavior_anomaly_score >= 4.0)
        or (self_report_score_value >= 9.0 and behavior_anomaly_score >= 3.0)
    )
    stress_by_extreme_self_rule = self_report_score_value >= 9.0

    focus_like_input = (
        task_context in {"coding", "document"}
        and self_report_score_value <= 4.5
        and behavior_anomaly_score >= 1.0
        and correction_urgency < 4.0
        and not repeat_click_high
        and not jitter_high
        and (
            features.get("key_count_baseline_ratio", 1.0) >= 1.3
            or features.get("burst_after_pause_baseline_ratio", 1.0) >= 1.3
            or features.get("post_idle_burst_score", 0.0) >= 3.0
        )
    )

    if focus_like_input:
        state = STATE_FOCUS
        triggered_rules.append("focus_like_input")
        reasons = [
            "고민 후 입력량과 burst_after_pause는 증가함",
            "correction_urgency_index가 낮음",
            "반복 클릭과 mouse_jitter가 낮음",
            "자가 응답 점수가 낮음",
            f"{task_context} 작업에서는 고민 후 빠른 입력이 자연스러운 집중 패턴으로 해석됨",
        ]
        interpretation = "고민 후 빠르게 작성한 패턴으로 판단하여 스트레스 가능성 높음으로 분류하지 않음."
    elif stress_by_core_rule or stress_by_urgency_rule or stress_by_strong_self_rule or stress_by_extreme_self_rule:
        state = STATE_STRESS
        score_floor = 7.0
        score_adjustment_reason = "자가 응답 점수가 매우 높아 스트레스 가능성 점수를 보정함"
        if stress_by_extreme_self_rule:
            triggered_rules.append("self_report_score >= 9.0")
            if behavior_anomaly_score >= 3.0:
                score_floor = max(score_floor, 7.2)
        if stress_by_strong_self_rule:
            triggered_rules.append("strong self-report with behavioral support")
            score_floor = max(score_floor, 7.2)
        if stress_by_core_rule:
            triggered_rules.append("final_score >= 7.0 and at least two high signals")
        if stress_by_urgency_rule:
            triggered_rules.append("correction_urgency_index >= 6.0 and self_report_score >= 7.0")
            score_floor = max(score_floor, 7.3)
            score_adjustment_reason = "자가 응답과 수정 조급성 지수가 함께 높아 스트레스 가능성 점수를 보정함"
        if raw_final_score < score_floor:
            final_score = round(score_floor, 2)
        if self_report_score_value >= 9.0:
            reasons.append("자가 응답 점수가 매우 높음")
        elif self_report_score_value >= 6.5:
            reasons.append("자가 응답 점수가 높음")
        if behavior_anomaly_score < 3.0:
            reasons.append("자가 응답은 매우 높지만 행동 근거는 부족하여 신뢰도를 낮게 표시")
        elif self_report_score_value >= 9.0:
            reasons.append("행동 이상 근거가 일부 확인됨")
        if baseline_quality in {"missing", "low"}:
            reasons.append("개인 기준선 데이터가 부족하여 자가 응답과 현재 행동 패턴을 더 크게 반영함")
        if correction_urgency >= 6.0:
            reasons.append(f"{task_context} 작업 중 correction_urgency_index가 기준선 대비 증가")
            reasons.append("Backspace 연타 후 빠른 재입력과 재수정 패턴 감지")
        if repeat_click_high:
            reasons.append("반복 클릭이 기준선 대비 증가")
        if jitter_high:
            reasons.append("mouse_jitter가 기준선 대비 증가")
        if persistence_score >= 6.0:
            reasons.append("최근 패턴이 반복됨")
        interpretation = "자가 응답이 매우 높아 스트레스 가능성 높음으로 분류하되, 행동 근거가 부족한 경우 신뢰도를 낮게 해석함."
    else:
        state = STATE_NORMAL
        reasons = [
            "자가 응답 점수가 낮음" if self_report_score_value < 4.5 else "자가 응답은 높지만 행동 근거가 부족함",
            "행동 이상 점수가 낮음" if behavior_anomaly_score < 6.0 else "행동 이상 점수만으로는 스트레스 가능성 높음 조건을 충족하지 않음",
            "수정 조급성 지수가 낮음" if correction_urgency < 4.0 else "수정 조급성 지수가 단독 신호로만 관찰됨",
            "마우스 불안정성이 낮음" if not jitter_high else "마우스 불안정성이 단독 신호로만 관찰됨",
        ]
        if baseline_quality in {"missing", "low"}:
            reasons.append("개인 기준선 데이터가 부족함")
        interpretation = "현재 입력 패턴은 개인 기준선과 크게 벗어나지 않아 정상 작업 상태로 판단함."

    scoring_debug = scoring_debug or {}
    scoring_debug.update(
        {
            "history_count": history_count,
            "baseline_quality": baseline_quality,
            "final_score_formula": formula,
            "raw_final_score": raw_final_score,
            "self_report_score": round(self_report_score_value, 2),
            "behavior_anomaly_score": round(behavior_anomaly_score, 2),
            "persistence_score": round(persistence_score, 2),
            "triggered_rules": triggered_rules,
        }
    )
    score_adjusted = state == STATE_STRESS and final_score > raw_final_score
    if score_adjusted:
        scoring_debug["score_adjustment_reason"] = score_adjustment_reason

    return {
        "task_context": task_context,
        "state": state,
        "class_value": CLASS_VALUES[state],
        "raw_final_score": raw_final_score,
        "final_score": final_score,
        "score_adjusted": score_adjusted,
        "score_adjustment_reason": score_adjustment_reason if score_adjusted else "",
        "confidence": _result_confidence(final_score, high_count, state, self_report_score_value, behavior_anomaly_score),
        "self_report_score": round(self_report_score_value, 2),
        "behavior_anomaly_score": round(behavior_anomaly_score, 2),
        "persistence_score": round(persistence_score, 2),
        "model_probability_score": None if model_probability_score is None else round(model_probability_score, 2),
        "activity_idle_time": round(activity_idle_time, 2),
        "correction_urgency_index": round(correction_urgency, 2),
        "baseline_quality": baseline_quality,
        "top_reasons": reasons[:5],
        "interpretation": interpretation,
        "scoring_debug": scoring_debug,
    }


def _with_required_fields(result: dict, self_score: float, behavior_score: float, persistence_score: float) -> dict:
    state = result.get("state") if result.get("state") in VALID_STATES else STATE_NORMAL
    result["state"] = state
    result["class_value"] = CLASS_VALUES[state]
    result.setdefault("final_score", 0.0)
    result.setdefault("raw_final_score", result.get("final_score", 0.0))
    if state == STATE_STRESS and float(result.get("final_score", 0.0)) < 7.0:
        result["raw_final_score"] = result.get("final_score", 0.0)
        result["final_score"] = 7.0
        result["score_adjusted"] = True
        result["score_adjustment_reason"] = "스트레스 가능성 높음 상태와 점수 기준을 일치시키기 위해 보정함"
    else:
        result.setdefault("score_adjusted", False)
        result.setdefault("score_adjustment_reason", "")
    result.setdefault("confidence", 60)
    result["self_report_score"] = round(self_score, 2)
    result["behavior_anomaly_score"] = round(behavior_score, 2)
    result["persistence_score"] = round(persistence_score, 2)
    return result
