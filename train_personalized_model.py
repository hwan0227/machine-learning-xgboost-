import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# XGBoost는 선택사항
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def load_feature_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    print(f"[INFO] loaded: {path}")
    print(f"[INFO] shape: {df.shape}")
    print("[INFO] columns:", df.columns.tolist())
    return df


def build_sample_weights(df: pd.DataFrame, personal_user_ids=None, personal_weight=3.0):
    """
    기본은 전부 1.
    나중에 personal_user_ids에 해당하는 user_id에 더 큰 가중치 부여 가능.
    """
    weights = np.ones(len(df), dtype=float)

    if personal_user_ids is None:
        return weights

    personal_user_ids = set(str(x) for x in personal_user_ids)

    if "user_id" in df.columns:
        mask = df["user_id"].astype(str).isin(personal_user_ids)
        weights[mask] = personal_weight

    return weights


def prepare_xy(df: pd.DataFrame):
    df = df.copy()

    if "Stress_Val" not in df.columns:
        raise ValueError("'Stress_Val' column not found.")

    df["Stress_Val"] = pd.to_numeric(df["Stress_Val"], errors="coerce")
    df = df[df["Stress_Val"].isin([0, 1])].copy()

    if df.empty:
        raise ValueError(
            "No usable rows after filtering Stress_Val to {0,1}. "
            "Check data/kaggle_features.csv -> Stress_Val column values."
        )

    class_map = {0.0: 0, 1.0: 1}
    df["Stress_Class"] = df["Stress_Val"].map(class_map)

    drop_cols = [
        "Stress_Val",
        "Stress_Val_raw",
        "Stress_Class",
        "window_start",
        "window_end",
        "user_id",
        "Fatigue_Val",
        "PAM_Val",
        "Energy_Val",
        "Pleasant_Val",
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()
    y = df["Stress_Class"].astype(int).copy()

    return X, y, df


def split_columns(X: pd.DataFrame):
    categorical_cols = []
    numeric_cols = []

    for col in X.columns:
        if X[col].dtype == "object":
            categorical_cols.append(col)
        else:
            numeric_cols.append(col)

    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols):
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

    return preprocessor


def evaluate_model(name, model, X_train, X_test, y_train, y_test, sample_weight=None):
    print("\n" + "=" * 60)
    print(f"[MODEL] {name}")

    if sample_weight is not None:
        model.fit(X_train, y_train, classifier__sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"[RESULT] accuracy = {acc:.4f}")
    print(f"[RESULT] f1-score = {f1:.4f}")
    print("[RESULT] confusion matrix:")
    print(cm)
    print("[RESULT] classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    return {
        "name": name,
        "model": model,
        "accuracy": acc,
        "f1": f1,
        "confusion_matrix": cm,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="data/kaggle_features.csv")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    # 나중에 개인 데이터 user_id 가 따로 생기면 사용
    # 예: --personal_user_ids my1 my2 --personal_weight 3
    parser.add_argument("--personal_user_ids", nargs="*", default=None)
    parser.add_argument("--personal_weight", type=float, default=3.0)

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_csv(input_csv)
    X, y, clean_df = prepare_xy(df)

    print(f"[INFO] usable rows after label filtering: {len(clean_df)}")
    print("[INFO] label distribution:")
    print(y.value_counts(dropna=False))

    numeric_cols, categorical_cols = split_columns(X)

    print("[INFO] numeric columns:", numeric_cols)
    print("[INFO] categorical columns:", categorical_cols)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, clean_df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y if len(y.unique()) > 1 else None
    )

    train_weights = build_sample_weights(
        df_train,
        personal_user_ids=args.personal_user_ids,
        personal_weight=args.personal_weight
    )

    print(f"[INFO] train size: {len(X_train)}, test size: {len(X_test)}")
    print(f"[INFO] sample weight unique values: {np.unique(train_weights)}")

    results = []

    # 1) Logistic Regression
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            random_state=args.random_state,
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    results.append(
        evaluate_model(
            "LogisticRegression",
            lr_pipeline,
            X_train, X_test, y_train, y_test,
            sample_weight=None  # LR에도 줄 수 있지만 1차는 비교 단순화
        )
    )

    # 2) Random Forest
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=args.random_state,
            class_weight="balanced"
        ))
    ])

    results.append(
        evaluate_model(
            "RandomForest",
            rf_pipeline,
            X_train, X_test, y_train, y_test,
            sample_weight=train_weights
        )
    )

    # 3) XGBoost
    if HAS_XGBOOST:
        xgb_pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=args.random_state,
                eval_metric="logloss"
            ))
        ])

        results.append(
            evaluate_model(
                "XGBoost",
                xgb_pipeline,
                X_train, X_test, y_train, y_test,
                sample_weight=train_weights
            )
        )
    else:
        print("\n[INFO] xgboost not installed -> XGBoost skipped.")

    # 최고 모델 선택
    best_result = max(results, key=lambda x: x["f1"])
    best_name = best_result["name"]
    best_model = best_result["model"]

    print("\n" + "=" * 60)
    print(f"[BEST MODEL] {best_name}")
    print(f"[BEST F1] {best_result['f1']:.4f}")
    print(f"[BEST ACC] {best_result['accuracy']:.4f}")

    # 전체 데이터로 best model 재학습
    final_weights = build_sample_weights(
        clean_df,
        personal_user_ids=args.personal_user_ids,
        personal_weight=args.personal_weight
    )

    if best_name in ["RandomForest", "XGBoost"]:
        best_model.fit(X, y, classifier__sample_weight=final_weights)
    else:
        best_model.fit(X, y)

    save_path = model_dir / "best_stress_model.pkl"
    joblib.dump(best_model, save_path)

    print(f"[DONE] saved best model to: {save_path}")


if __name__ == "__main__":
    main()
