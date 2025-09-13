# model_calibrate.py
# Kalibracja prawdopodobieństw (sigmoid/isotonic) z ewaluacją na holdoucie
# Wymaga: pandas, numpy, scikit-learn, joblib, (opcjonalnie) xgboost
import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, brier_score_loss, log_loss, roc_curve,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Import z Twojego pipeline’u
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
    num_cont = [c for c in NUM_CONT_FEATURES_DEFAULT if c in df.columns]
    num_bin = [c for c in NUM_BIN_FEATURES_DEFAULT if c in df.columns]
    cat = [c for c in CAT_FEATURES_DEFAULT if c in df.columns]
    return num_cont, num_bin, cat


def rebuild_tuned_pipeline(model_key: str, preprocessor, best_params: Dict[str, Dict[str, object]]) -> Pipeline:
    """Odtwórz pipeline z najlepszymi hiperparametrami (z best_params.json)."""
    model_key = model_key.lower()
    if model_key == "xgb" and not XGB_AVAILABLE:
        print("[WARN] xgboost nie jest zainstalowany — przełączam na logreg.", file=sys.stderr)
        model_key = "logreg"

    _, clf = build_model(model_key)
    pipe = Pipeline([("pre", preprocessor), ("clf", clf)])

    params_for_model = best_params.get(model_key)
    if params_for_model:
        pipe.set_params(**params_for_model)
    else:
        print(f"[INFO] Brak wpisu '{model_key}' w best_params.json — używam ustawień domyślnych.", file=sys.stderr)
    return pipe


def evaluate_thresholds(y_true: np.ndarray, proba: np.ndarray, thr: float, rule: str = ">=") -> Dict[str, object]:
    """
    Policz metryki dla zadanego progu.
    rule: '>=' (domyślnie) lub '>' — UWAGA: precision_recall_curve używa reguły 'p > thr'.
    """
    if rule not in (">", ">="):
        rule = ">="
    pred = (proba > thr).astype(int) if rule == ">" else (proba >= thr).astype(int)
    return {
        "threshold": float(thr),
        "decision_rule": rule,
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "report": classification_report(y_true, pred, digits=3),
    }


def pick_youden(y_true: np.ndarray, proba: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, proba)
    j = tpr - fpr
    return float(thr[int(np.argmax(j))])


def find_threshold_for_precision(y_true: np.ndarray, proba: np.ndarray, target_precision: float) -> Dict[str, object]:
    """
    Znajdź najniższy próg, przy którym precision >= target_precision
    (SPÓJNIE z precision_recall_curve: reguła klasyfikacji 'p > thr').
    Jeśli nieosiągalne, zwróć próg o maksymalnej precision (z adnotacją).
    """
    prec, rec, thr = precision_recall_curve(y_true, proba)  # thr rośnie; prec/rec mają len(thr)+1
    if thr.size == 0:
        t = 0.5
        return {
            "meets_target": False,
            "threshold": float(t),
            "metrics": evaluate_thresholds(y_true, proba, t, rule=">")
        }

    mask = (prec[1:] >= target_precision)
    if np.any(mask):
        i = int(np.argmax(mask))  # pierwszy próg spełniający warunek
        t = float(thr[i])
        mets = evaluate_thresholds(y_true, proba, t, rule=">")
        meets = mets["precision"] >= target_precision  # sanity check tej samej reguły
        return {
            "meets_target": bool(meets),
            "threshold": t,
            "metrics": mets
        }
    else:
        # nieosiągalne — wybierz próg maksymalizujący precision (spośród prec[1:])
        i = int(np.argmax(prec[1:]))
        t = float(thr[i])
        mets = evaluate_thresholds(y_true, proba, t, rule=">")
        return {
            "meets_target": False,
            "threshold": t,
            "metrics": mets,
            "best_precision": float(prec[i + 1])  # odpowiada thr[i]
        }


def plot_reliability(y_true: np.ndarray, proba: np.ndarray, out_png: str):
    """Reliability diagram."""
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    # krzywa
    plt.plot(mean_pred, frac_pos, marker="o", linewidth=1)
    # idealna kalibracja
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Przewidywane prawdopodobieństwo")
    plt.ylabel("Zaobserwowany odsetek (Y)")
    plt.title("Reliability diagram (10-quantile bins)")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def plot_hist(proba: np.ndarray, out_png: str):
    plt.figure(figsize=(6, 4))
    plt.hist(proba, bins=20)
    plt.xlabel("Przewidywane prawdopodobieństwo")
    plt.ylabel("Liczność")
    plt.title("Histogram przewidywanych prawdopodobieństw")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Kalibracja prawdopodobieństw (sigmoid/isotonic) + ewaluacja holdout")
    parser.add_argument("--data", required=True, help="Ścieżka do CSV z danymi")
    parser.add_argument("--sep", default=",", help="Separator CSV")
    parser.add_argument("--outdir", default="outputs/calib", help="Folder wyjściowy")
    parser.add_argument("--model", default="xgb", choices=["xgb", "rf", "logreg"], help="Model do kalibracji")
    parser.add_argument("--best_params", default="outputs/cv/best_params.json", help="Plik z najlepszymi parametrami")
    parser.add_argument("--calib", default="sigmoid", choices=["sigmoid", "isotonic"], help="Metoda kalibracji")
    parser.add_argument("--cv", type=int, default=5, help="Foldy do kalibracji krzyżowej (CalibratedClassifierCV)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Rozmiar holdoutu")
    parser.add_argument("--random_state", type=int, default=42, help="Ziarno losowe")
    # NOWE:
    parser.add_argument("--target_precision", type=float, default=None,
                        help="Jeśli podane (np. 0.90), wyznacza próg osiągający co najmniej taką precyzję na holdoucie.")
    parser.add_argument("--save_model", type=str, default=None,
                        help="Ścieżka do zapisu skalibrowanego modelu .joblib. "
                             "Domyślnie outputs/calib/models/<model>_calibrated_<metoda>.joblib")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    imgdir = os.path.join(args.outdir, "img"); os.makedirs(imgdir, exist_ok=True)
    models_dir = os.path.join(args.outdir, "models"); os.makedirs(models_dir, exist_ok=True)

    # 1) Dane
    df = load_data(args.data, args.sep)
    df = clean_dataframe(df)
    df = engineer_features(df)

    if TARGET_COL not in df.columns:
        print("[ERROR] Brak kolumny celu (Loan_Status).")
        sys.exit(1)

    df = df.dropna(subset=[TARGET_COL]).copy()
    y = df[TARGET_COL].astype(int)
    X = df.drop(columns=[TARGET_COL])

    # 2) Split: train/holdout
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    # 3) Preprocessor i tuned pipeline
    num_cont, num_bin, cat = build_feature_lists(df)
    pre = build_preprocessor(num_cont, num_bin, cat)

    try:
        with open(args.best_params, "r", encoding="utf-8") as f:
            best_params_all = json.load(f)
    except Exception:
        best_params_all = {}

    base_pipe = rebuild_tuned_pipeline(args.model, pre, best_params_all)

    # 4) Kalibracja krzyżowa na treningu
    #    Uwaga: kalibrujemy CAŁY pipeline (pre + model), to bezpieczne i spójne.
    calib_clf = CalibratedClassifierCV(
        estimator=base_pipe,
        method=args.calib,
        cv=args.cv  # k-fold na treningu, bez wycieku na holdout
    )
    calib_clf.fit(X_train, y_train)

    # 5) Ewaluacja na holdoucie
    proba_hold = calib_clf.predict_proba(X_hold)[:, 1]

    # metryki probabilistyczne
    auc = float(roc_auc_score(y_hold, proba_hold))
    brier = float(brier_score_loss(y_hold, proba_hold))
    try:
        ll = float(log_loss(y_hold, proba_hold, labels=[0, 1]))
    except Exception:
        ll = float("nan")

    # metryki klasyfikacyjne dla thr=0.5 i thr=Youden (reguła >=, jak dotąd)
    metrics_thr05 = evaluate_thresholds(y_hold.values, proba_hold, thr=0.5, rule=">=")
    thr_youden = pick_youden(y_hold.values, proba_hold)
    metrics_thrY = evaluate_thresholds(y_hold.values, proba_hold, thr=thr_youden, rule=">=")

    # (NOWE) próg pod docelową precyzję — spójnie z PR: reguła '>'
    precision_target_block: Optional[Dict[str, object]] = None
    if args.target_precision is not None:
        precision_target_block = find_threshold_for_precision(
            y_hold.values, proba_hold, target_precision=float(args.target_precision)
        )

    # 6) Wykresy kalibracji
    plot_reliability(y_hold.values, proba_hold, os.path.join(imgdir, "reliability_holdout.png"))
    plot_hist(proba_hold, os.path.join(imgdir, "proba_hist_holdout.png"))

    # 7) Zapisy — podsumowanie JSON
    summary = {
        "model": args.model,
        "calibration_method": args.calib,
        "cv_folds": args.cv,
        "holdout_metrics_prob": {
            "roc_auc": auc,
            "brier": brier,
            "log_loss": ll
        },
        "holdout_metrics_thr_0.50": metrics_thr05,
        "holdout_metrics_thr_youden": metrics_thrY
    }
    if precision_target_block is not None:
        summary["target_precision"] = float(args.target_precision)
        summary["threshold_for_target_precision"] = {
            "meets_target": precision_target_block["meets_target"],
            "threshold": float(precision_target_block["threshold"]),
            "metrics": precision_target_block["metrics"],
            **({"best_precision": precision_target_block["best_precision"]}
               if not precision_target_block["meets_target"] else {})
        }

    with open(os.path.join(args.outdir, "calibration_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 8) Zapis skalibrowanego modelu do joblib
    save_path = args.save_model
    if save_path is None or not save_path.strip():
        save_path = os.path.join(models_dir, f"{args.model}_calibrated_{args.calib}.joblib")
    joblib.dump(calib_clf, save_path)

    # 9) Konsola — skrót
    print("\n=== KALIBRACJA — PODSUMOWANIE (holdout) ===")
    print(f"metoda: {args.calib}, folds={args.cv}")
    print(f"ROC-AUC: {auc:.4f} | Brier: {brier:.4f} | LogLoss: {ll:.4f}")

    print("\n[thr=0.50] (reguła >=)")
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"{k:>10}: {metrics_thr05[k]:.4f}")
    print("CM [[TN,FP],[FN,TP]]:", np.array(metrics_thr05["confusion_matrix"]))

    print(f"\n[thr=Youden={metrics_thrY['threshold']:.3f}] (reguła >=)")
    for k in ["accuracy", "precision", "recall", "f1"]:
        print(f"{k:>10}: {metrics_thrY[k]:.4f}")
    print("CM [[TN,FP],[FN,TP]]:", np.array(metrics_thrY["confusion_matrix"]))

    if precision_target_block is not None:
        thr_tp = precision_target_block["threshold"]
        mtp = precision_target_block["metrics"]
        status = "OSIĄGNIĘTY" if precision_target_block["meets_target"] else "NIEOSIĄGALNY"
        print(f"\n[thr dla precision>={args.target_precision:.2f}] — {status} @ {thr_tp:.3f} (reguła >)")
        for k in ["accuracy", "precision", "recall", "f1"]:
            print(f"{k:>10}: {mtp[k]:.4f}")
        print("CM [[TN,FP],[FN,TP]]:", np.array(mtp["confusion_matrix"]))
        if not precision_target_block["meets_target"]:
            print(f"(max precision osiągalne ≈ {precision_target_block['best_precision']:.4f})")

    print(f"\nWykresy: {os.path.join(imgdir, 'reliability_holdout.png')} | {os.path.join(imgdir, 'proba_hist_holdout.png')}")
    print(f"Zapisano: {os.path.join(args.outdir, 'calibration_summary.json')}")
    print(f"Model: {save_path}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Domyślka pod PyCharm
        sys.argv += [
            "--data", "data/loan_data.csv",
            "--sep", ",",
            "--outdir", "outputs/calib",
            "--model", "xgb",
            "--best_params", "outputs/cv/best_params.json",
            "--calib", "sigmoid",          # lub "isotonic" (na małych próbkach możliwy overfit — ale CV=5 pomaga)
            "--cv", "5",
            "--test_size", "0.2",
            "--random_state", "42",
            "--target_precision", "0.87"
        ]
    main()
