# model_cv_tune.py
# Porównanie modeli na Stratified 5-fold CV + szybki tuning (RandomizedSearch) + holdout
# Wymaga: pandas, numpy, scikit-learn, joblib, (opcjonalnie) xgboost
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve
)
from sklearn.pipeline import Pipeline
from scipy.stats import randint as sp_randint, uniform as sp_uniform, loguniform as sp_loguniform

# Import z Twojego pipeline’u (spójne przygotowanie danych i preprocesor!)
from loan_model import (
    load_data, clean_dataframe, engineer_features,
    build_preprocessor, build_model,
    TARGET_COL,
    NUM_CONT_FEATURES_DEFAULT, NUM_BIN_FEATURES_DEFAULT, CAT_FEATURES_DEFAULT
)

# XGBoost opcjonalnie
try:
    from xgboost import XGBClassifier  # noqa: F401
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False


def build_feature_lists(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str]]:
    """Weź tylko te kolumny, które faktycznie są w danych."""
    num_cont = [c for c in NUM_CONT_FEATURES_DEFAULT if c in df.columns]
    num_bin = [c for c in NUM_BIN_FEATURES_DEFAULT if c in df.columns]
    cat = [c for c in CAT_FEATURES_DEFAULT if c in df.columns]
    return num_cont, num_bin, cat


def cv_scores_to_row(name: str, cvres: Dict[str, np.ndarray]) -> Dict[str, str]:
    """Zamień wyniki cross_validate na słownik 'mean±std' (string), żeby ładnie zapisać do CSV."""
    row = {"model": name}
    for k, v in cvres.items():
        if not k.startswith("test_"):
            continue
        met = k.replace("test_", "")
        row[met] = f"{np.mean(v):.4f}±{np.std(v):.4f}"
    return row


def get_param_distributions(model_key: str) -> Dict[str, object]:
    """Przestrzenie do RandomizedSearch (skromne, szybkie)."""
    if model_key == "logreg":
        return {
            "clf__C": sp_loguniform(1e-3, 10.0),
            "clf__l1_ratio": sp_uniform(0.0, 1.0),
        }
    elif model_key == "rf":
        return {
            "clf__n_estimators": sp_randint(300, 900),
            "clf__max_depth": sp_randint(4, 20),
            "clf__min_samples_split": sp_randint(2, 10),
            "clf__min_samples_leaf": sp_randint(1, 5),
            "clf__max_features": ["sqrt", "log2", None],
        }
    elif model_key == "xgb":
        return {
            "clf__n_estimators": sp_randint(300, 900),
            "clf__max_depth": sp_randint(3, 8),
            "clf__learning_rate": sp_loguniform(0.01, 0.3),
            "clf__subsample": sp_uniform(0.6, 0.4),
            "clf__colsample_bytree": sp_uniform(0.6, 0.4),
            "clf__reg_lambda": sp_loguniform(1e-3, 10.0),
            "clf__gamma": sp_loguniform(1e-8, 1e-1),
        }
    else:
        return {}


def evaluate_on_holdout(pipe: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, object]:
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, digits=3),
    }


def main():
    parser = argparse.ArgumentParser(description="Porównanie modeli (5-fold CV) + RandomizedSearch + holdout")
    parser.add_argument("--data", required=True, help="Ścieżka do CSV z danymi")
    parser.add_argument("--sep", default=",", help="Separator CSV")
    parser.add_argument("--outdir", default="outputs/cv", help="Folder wyjściowy")
    parser.add_argument("--models", default="logreg,rf,xgb", help="Jakie modele porównać (lista: logreg,rf,xgb)")
    parser.add_argument("--cv", type=int, default=5, help="Liczba foldów (domyślnie 5)")
    parser.add_argument("--n_iter", type=int, default=30, help="Iteracje RandomizedSearch (domyślnie 30)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Rozmiar holdoutu (domyślnie 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="Ziarno losowe")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    models_dir = os.path.join(args.outdir, "models"); os.makedirs(models_dir, exist_ok=True)

    # 1) Dane i przygotowanie spójne z loan_model.py
    df = load_data(args.data, args.sep)
    df = clean_dataframe(df)
    df = engineer_features(df)

    if TARGET_COL not in df.columns:
        print("[ERROR] Brak kolumny celu (Loan_Status).")
        sys.exit(1)

    df = df.dropna(subset=[TARGET_COL]).copy()
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    # listy cech pod preprocesor
    num_cont, num_bin, cat = build_feature_lists(df)
    pre = build_preprocessor(num_cont, num_bin, cat)

    # 2) Train/Holdout (holdout NIE bierze udziału w CV/tuningu)
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # 3) Definicja modeli
    want_models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    all_models: Dict[str, Pipeline] = {}

    for key in want_models:
        if key == "xgb" and not XGB_AVAILABLE:
            print("[WARN] xgboost nie jest zainstalowany — pomijam XGB.")
            continue
        name, clf = build_model(key)
        pipe = Pipeline([("pre", pre), ("clf", clf)])
        all_models[key] = pipe

    if not all_models:
        print("[ERROR] Nie zdefiniowano żadnego modelu do porównania.")
        sys.exit(1)

    # 4) CV — baseline (bez tuningu)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    scoring = ["roc_auc", "accuracy", "precision", "recall", "f1"]

    rows_baseline = []
    cv_raw_results = {}
    for key, pipe in all_models.items():
        cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)
        rows_baseline.append(cv_scores_to_row(f"{key}_baseline", cvres))
        cv_raw_results[f"{key}_baseline"] = {k: list(map(float, v)) for k, v in cvres.items() if k.startswith("test_")}

    baseline_df = pd.DataFrame(rows_baseline)
    baseline_csv = os.path.join(args.outdir, "cv_baseline_summary.csv")
    baseline_df.to_csv(baseline_csv, index=False)

    # 5) RandomizedSearchCV — tuning pod ROC-AUC
    tuned_pipes: Dict[str, Pipeline] = {}
    tuned_params: Dict[str, dict] = {}
    rows_tuned = []

    for key, base_pipe in all_models.items():
        param_distributions = get_param_distributions(key)
        if not param_distributions:
            tuned_pipes[key] = base_pipe.fit(X_train, y_train)
            tuned_params[key] = {"info": "no_param_search"}
            cvres = cross_validate(base_pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
            rows_tuned.append(cv_scores_to_row(f"{key}_tuned", cvres))
            cv_raw_results[f"{key}_tuned"] = {k: list(map(float, v)) for k, v in cvres.items() if k.startswith("test_")}
            continue

        rs = RandomizedSearchCV(
            estimator=base_pipe,
            param_distributions=param_distributions,
            n_iter=args.n_iter,
            scoring="roc_auc",
            n_jobs=-1,
            cv=cv,
            refit=True,
            random_state=args.random_state,
            verbose=0,
        )
        rs.fit(X_train, y_train)
        tuned_pipes[key] = rs.best_estimator_
        tuned_params[key] = rs.best_params_

        cvres = cross_validate(rs.best_estimator_, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        rows_tuned.append(cv_scores_to_row(f"{key}_tuned", cvres))
        cv_raw_results[f"{key}_tuned"] = {k: list(map(float, v)) for k, v in cvres.items() if k.startswith("test_")}

    tuned_df = pd.DataFrame(rows_tuned)
    tuned_csv = os.path.join(args.outdir, "cv_tuned_summary.csv")
    tuned_df.to_csv(tuned_csv, index=False)

    # 6) Wybór zwycięzcy po średnim ROC-AUC z CV (tuned)
    def parse_mean(value: str) -> float:
        try:
            return float(value.split("±")[0])
        except Exception:
            return -np.inf

    tuned_df["_auc"] = tuned_df.get("roc_auc", pd.Series(dtype=str)).apply(parse_mean)
    if tuned_df["_auc"].isna().all():
        print("[ERROR] Brak kolumny roc_auc w wynikach tuned.")
        sys.exit(1)

    best_row = tuned_df.iloc[int(np.argmax(tuned_df["_auc"]))]
    best_name = str(best_row["model"]).replace("_tuned", "")
    best_pipe = tuned_pipes[best_name]

    # 7) Ewaluacja zwycięzcy na holdoucie (próg 0.5)
    hold_metrics = evaluate_on_holdout(best_pipe, X_hold, y_hold)

    # 7a) DODANE: próg Youdena na holdoucie + metryki
    y_proba_hold = best_pipe.predict_proba(X_hold)[:, 1]
    fpr, tpr, thr = roc_curve(y_hold, y_proba_hold)
    j = tpr - fpr
    thr_youden = float(thr[int(np.argmax(j))])
    y_pred_y = (y_proba_hold >= thr_youden).astype(int)
    youden_metrics = {
        "threshold_youden": thr_youden,
        "precision": float(precision_score(y_hold, y_pred_y, zero_division=0)),
        "recall": float(recall_score(y_hold, y_pred_y, zero_division=0)),
        "f1": float(f1_score(y_hold, y_pred_y, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_hold, y_pred_y).tolist(),
        "report": classification_report(y_hold, y_pred_y, digits=3),
    }

    print(f"\n=== HOLDOUT — próg Youdena = {thr_youden:.3f} ===")
    print(f"{'precision':>10}: {youden_metrics['precision']:.4f}")
    print(f"{'recall':>10}: {youden_metrics['recall']:.4f}")
    print(f"{'f1':>10}: {youden_metrics['f1']:.4f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:", np.array(youden_metrics["confusion_matrix"]))
    print("\nClassification report (Youden):\n", youden_metrics["report"])

    # 8) Zapis wyników i modelu
    hold_json = os.path.join(args.outdir, "best_holdout_metrics.json")
    with open(hold_json, "w", encoding="utf-8") as f:
        json.dump(hold_metrics, f, ensure_ascii=False, indent=2)

    hold_youden_json = os.path.join(args.outdir, "best_holdout_youden_metrics.json")
    with open(hold_youden_json, "w", encoding="utf-8") as f:
        json.dump(youden_metrics, f, ensure_ascii=False, indent=2)

    with open(os.path.join(args.outdir, "cv_raw_results.json"), "w", encoding="utf-8") as f:
        json.dump(cv_raw_results, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.outdir, "best_params.json"), "w", encoding="utf-8") as f:
        json.dump(tuned_params, f, ensure_ascii=False, indent=2)

    model_path = os.path.join(models_dir, f"best_{best_name}_tuned.joblib")
    joblib.dump(best_pipe, model_path)

    # 9) Konsola — szybkie podsumowanie
    print("\n=== PODSUMOWANIE CV (baseline) ===")
    print(baseline_df.to_string(index=False))
    print("\n=== PODSUMOWANIE CV (tuned) ===")
    print(tuned_df.drop(columns=["_auc"]).to_string(index=False))
    print(f"\n>>> ZWYCIĘZCA: {best_name} (średni ROC-AUC z CV najwyższy)")
    print("\n=== HOLDOUT (20%) — zwycięzca (próg 0.5) ===")
    for k in ["accuracy", "roc_auc", "precision", "recall", "f1"]:
        print(f"{k:>10}: {hold_metrics[k]:.4f}")
    print("Confusion matrix [[TN, FP],[FN, TP]]:", np.array(hold_metrics["confusion_matrix"]))
    print("\nClassification report:\n", hold_metrics["report"])
    print(f"\nZapisano model: {model_path}")
    print(f"Wyniki: {args.outdir}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Domyślka pod PyCharm — porówna logreg i RF (XGB jeśli masz)
        sys.argv += [
            "--data", "data/loan_data.csv",
            "--sep", ",",
            "--outdir", "outputs/cv",
            "--models", "logreg,rf,xgb",
            "--cv", "5",
            "--n_iter", "30",
            "--test_size", "0.2",
            "--random_state", "42",
        ]
    main()
