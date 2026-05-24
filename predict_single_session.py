from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    import joblib
except ImportError:
    joblib = None

from feature_extraction import build_feature_dict, build_feature_row
from stress_decision import (
    calculate_persistence_score,
    calculate_rule_based_behavior_score_with_debug,
    combine_behavior_score,
    decide_state,
    load_history,
    save_history,
    self_report_score,
)
from task_context import normalize_task_context


def stress_probability_from_model(model, X: pd.DataFrame) -> float | None:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        proba = model.predict_proba(X)[0]
    except Exception as e:
        print(f"[WARN] model probability skipped: {e}")
        return None

    classes = list(getattr(model, "classes_", []))
    if not classes and hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        classes = list(getattr(classifier, "classes_", []))

    stress_index = None
    candidates = (2, 1.0, 1) if len(classes) > 2 else (1.0, 1, 2)
    for candidate in candidates:
        if candidate in classes:
            stress_index = classes.index(candidate)
            break
    if stress_index is None:
        stress_index = len(proba) - 1
    return round(float(proba[stress_index]) * 10.0, 2)


def load_model_score(model_path: Path, X: pd.DataFrame) -> tuple[int | None, float | None]:
    if joblib is None:
        print("[WARN] joblib not installed; model probability skipped")
        return None, None
    if not model_path.exists():
        print(f"[WARN] model not found: {model_path}")
        return None, None
    try:
        model = joblib.load(model_path)
        pred = int(model.predict(X)[0])
        return pred, stress_probability_from_model(model, X)
    except Exception as e:
        print(f"[WARN] model prediction skipped: {e}")
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session_dir", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="models/best_stress_model.pkl")
    parser.add_argument("--duration_sec", type=int, default=300)
    parser.add_argument("--dept", type=str, default="7. IT부")
    parser.add_argument("--user_id", type=str, default=None)
    parser.add_argument("--task_context", type=str, default=None)
    parser.add_argument("--baseline_root", type=str, default="data/personal")
    parser.add_argument("--history_path", type=str, default="stress_history.json")
    parser.add_argument("--output_json", type=str, default="prediction_result.json")
    parser.add_argument("--tension", type=float, default=0.0)
    parser.add_argument("--pressure_hurry", type=float, default=None)
    parser.add_argument("--workload", type=float, default=0.0)
    parser.add_argument("--hurry", type=float, default=0.0)
    parser.add_argument("--irritability", type=float, default=0.0)
    parser.add_argument("--control", type=float, default=10.0)
    parser.add_argument("--no_history_update", action="store_true")
    args = parser.parse_args()

    session_dir = Path(args.session_dir)
    model_path = Path(args.model_path)
    task_context = normalize_task_context(args.task_context)

    features = build_feature_dict(
        session_dir,
        duration_sec=args.duration_sec,
        task_context=task_context if args.task_context else None,
        user_id=args.user_id,
        baseline_root=args.baseline_root,
    )
    task_context = features["task_context"]
    X = build_feature_row(
        session_dir,
        duration_sec=args.duration_sec,
        task_context=task_context,
        user_id=args.user_id,
        baseline_root=args.baseline_root,
    )

    model_pred, model_probability_score = load_model_score(model_path, X)
    rule_behavior_score, feature_scores, feature_ratio_debug = calculate_rule_based_behavior_score_with_debug(features, task_context)
    behavior_anomaly_score = combine_behavior_score(rule_behavior_score, model_probability_score)
    self_score = self_report_score(
        tension=args.tension,
        workload=args.workload,
        hurry=args.hurry,
        irritability=args.irritability,
        control=args.control,
        pressure_hurry=args.pressure_hurry,
    )
    history = load_history(args.history_path)
    persistence = calculate_persistence_score(
        history,
        behavior_anomaly_score,
        float(features.get("correction_urgency_index", 0.0)),
    )

    result = decide_state(
        task_context=task_context,
        self_report_score_value=self_score,
        behavior_anomaly_score=behavior_anomaly_score,
        persistence_score=persistence,
        model_probability_score=model_probability_score,
        features=features,
        feature_scores=feature_scores,
        history=history,
        scoring_debug={"feature_ratios": feature_ratio_debug},
    )
    result["dept"] = args.dept
    result["model_prediction"] = model_pred
    result["rule_based_behavior_score"] = rule_behavior_score

    output_path = Path(args.output_json)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_history_update and result.get("activity_idle_time", 0) < 300:
        history.append(result)
        save_history(history, args.history_path)

    print("[INFO] dept =", args.dept)
    print(f"[INFO] task_context = {result['task_context']}")
    print(f"[INFO] self_report_score = {result['self_report_score']}/10")
    print(f"[INFO] behavior_anomaly_score = {result['behavior_anomaly_score']}/10")
    print(f"[INFO] persistence_score = {result['persistence_score']}/10")
    print(f"[INFO] final_score = {result['final_score']}/10")
    print(f"[INFO] state = {result['state']}")
    print(f"[INFO] class_value = {result['class_value']}")
    print(f"[INFO] prediction_result = {output_path}")
    top_debug = sorted(feature_ratio_debug, key=lambda item: item.get("final_feature_score", 0), reverse=True)[:3]
    for row in top_debug:
        print(
            "[DEBUG] "
            f"{row['feature']}: 현재 {row['current']} / 기준선 {row['baseline']} / "
            f"ratio {row['ratio']} / score {row['final_feature_score']}"
        )
    print("[PREDICTION]", result["class_value"])


if __name__ == "__main__":
    main()
