const chartData = {
    "latest_score": 7.2,
    "task_context": "unknown",
    "state": "스트레스 가능성 높음",
    "class_value": 1,
    "confidence": 75,
    "self_report_score": 10.0,
    "behavior_anomaly_score": 3.51,
    "persistence_score": 0.0,
    "correction_urgency_index": 3.8,
    "baseline_quality": "missing",
    "raw_final_score": 0,
    "score_adjusted": false,
    "score_adjustment_reason": "",
    "top_reasons": [
        "자가 응답 점수가 매우 높음",
        "행동 이상 근거가 일부 확인됨",
        "개인 기준선 데이터가 부족하여 자가 응답과 현재 행동 패턴을 더 크게 반영함"
    ],
    "interpretation": "자가 응답이 매우 높아 스트레스 가능성 높음으로 분류하되, 행동 근거가 부족한 경우 신뢰도를 낮게 해석함.",
    "latest_result": {
        "task_context": "unknown",
        "state": "스트레스 가능성 높음",
        "class_value": 1,
        "raw_final_score": 6.05,
        "final_score": 7.2,
        "score_adjusted": true,
        "score_adjustment_reason": "자가 응답 점수가 매우 높아 스트레스 가능성 점수를 보정함",
        "confidence": 75,
        "self_report_score": 10.0,
        "behavior_anomaly_score": 3.51,
        "persistence_score": 0.0,
        "model_probability_score": 5.75,
        "activity_idle_time": 1.37,
        "correction_urgency_index": 3.8,
        "baseline_quality": "missing",
        "top_reasons": [
            "자가 응답 점수가 매우 높음",
            "행동 이상 근거가 일부 확인됨",
            "개인 기준선 데이터가 부족하여 자가 응답과 현재 행동 패턴을 더 크게 반영함"
        ],
        "interpretation": "자가 응답이 매우 높아 스트레스 가능성 높음으로 분류하되, 행동 근거가 부족한 경우 신뢰도를 낮게 해석함.",
        "scoring_debug": {
            "feature_ratios": [
                {
                    "feature": "backspace_ratio",
                    "current": 0.0092,
                    "baseline": 1.0,
                    "ratio": 0.0092,
                    "raw_score": 0.0,
                    "weight": 1.5,
                    "damping": 1.0,
                    "final_feature_score": 0.0
                },
                {
                    "feature": "delete_ratio",
                    "current": 0.0,
                    "baseline": 1.0,
                    "ratio": 0.0,
                    "raw_score": 0.0,
                    "weight": 1.2,
                    "damping": 1.0,
                    "final_feature_score": 0.0
                },
                {
                    "feature": "correction_ratio",
                    "current": 0.0092,
                    "baseline": 1.0,
                    "ratio": 0.0092,
                    "raw_score": 0.0,
                    "weight": 1.5,
                    "damping": 1.0,
                    "final_feature_score": 0.0
                },
                {
                    "feature": "std_key_interval",
                    "current": 0.1333,
                    "baseline": 1.0,
                    "ratio": 0.1333,
                    "raw_score": 0.0,
                    "weight": 1.0,
                    "damping": 1.0,
                    "final_feature_score": 0.0
                },
                {
                    "feature": "pause_ratio",
                    "current": 0.0,
                    "baseline": 1.0,
                    "ratio": 0.0,
                    "raw_score": 0.0,
                    "weight": 1.0,
                    "damping": 1.0,
                    "final_feature_score": 0.0
                },
                {
                    "feature": "burst_after_pause",
                    "current": 0.0,
                    "baseline": 1.0,
                    "ratio": 0.0,
                    "raw_score": 0.0,
                    "weight": 1.0,
                    "damping": 1.0,
                    "final_feature_score": 0.0
                },
                {
                    "feature": "repeat_key_ratio",
                    "current": 0.7156,
                    "baseline": 1.0,
                    "ratio": 0.7156,
                    "raw_score": 0.0,
                    "weight": 1.0,
                    "damping": 1.0,
                    "final_feature_score": 0.0
                },
                {
                    "feature": "repeat_click_count",
                    "current": 0.0,
                    "baseline": 1.0,
                    "ratio": 0.0,
                    "raw_score": 0.0,
                    "weight": 2.0,
                    "damping": 1.0,
                    "final_feature_score": 0.0
                },
                {
                    "feature": "direction_change_count",
                    "current": 29.0,
                    "baseline": 1.0,
                    "ratio": 29.0,
                    "raw_score": 10.0,
                    "weight": 1.0,
                    "damping": 1.0,
                    "final_feature_score": 10.0
                },
                {
                    "feature": "mouse_jitter",
                    "current": 1.0902,
                    "baseline": 1.0,
                    "ratio": 1.0902,
                    "raw_score": 0.0,
                    "weight": 2.0,
                    "damping": 1.0,
                    "final_feature_score": 1.09
                },
                {
                    "feature": "click_interval_std",
                    "current": 0.0,
                    "baseline": 1.0,
                    "ratio": 0.0,
                    "raw_score": 0.0,
                    "weight": 1.5,
                    "damping": 1.0,
                    "final_feature_score": 0.0
                },
                {
                    "feature": "correction_urgency_index",
                    "current": 3.8,
                    "baseline": 1.0,
                    "ratio": 3.8,
                    "raw_score": 10.0,
                    "weight": 3.0,
                    "damping": 1.0,
                    "final_feature_score": 10.0
                },
                {
                    "feature": "correction_loop_score",
                    "current": 0.0,
                    "baseline": 1.0,
                    "ratio": 0.0,
                    "raw_score": 0.0,
                    "weight": 2.0,
                    "damping": 1.0,
                    "final_feature_score": 0.0
                },
                {
                    "feature": "post_idle_burst_score",
                    "current": 0.0,
                    "baseline": 1.0,
                    "ratio": 0.0,
                    "raw_score": 0.0,
                    "weight": 1.2,
                    "damping": 1.0,
                    "final_feature_score": 0.0
                }
            ],
            "history_count": 4,
            "baseline_quality": "missing",
            "final_score_formula": "0.50*self + 0.30*behavior + 0.20*persistence",
            "raw_final_score": 6.05,
            "self_report_score": 10.0,
            "behavior_anomaly_score": 3.51,
            "persistence_score": 0.0,
            "triggered_rules": [
                "self_report_score >= 9.0",
                "strong self-report with behavioral support"
            ],
            "score_adjustment_reason": "자가 응답 점수가 매우 높아 스트레스 가능성 점수를 보정함"
        },
        "dept": "1. 인사부",
        "model_prediction": 1,
        "rule_based_behavior_score": 2.02
    },
    "daily": {
        "labels": [
            "00:00",
            "03:00",
            "06:00",
            "09:00",
            "12:00",
            "15:00",
            "18:00",
            "21:00"
        ],
        "data": [
            0,
            0,
            0,
            0,
            0,
            3.3,
            0,
            0
        ],
        "counts": [
            0,
            0,
            0,
            0,
            0,
            8,
            0,
            0
        ]
    },
    "weekly": {
        "labels": [
            "월",
            "화",
            "수",
            "목",
            "금",
            "토",
            "일"
        ],
        "data": [
            0,
            0,
            0,
            0,
            0,
            0,
            0
        ]
    },
    "monthly": {
        "labels": [
            "1주차",
            "2주차",
            "3주차",
            "4주차"
        ],
        "data": [
            0,
            0,
            0,
            0
        ]
    }
};