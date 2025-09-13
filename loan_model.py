# loan_model.py
# Baseline do przewidywania akceptacji wniosku kredytowego (Y/N).
# Wymagania: pandas, numpy, scikit-learn, matplotlib, joblib (xgboost opcjonalnie)

import argparse
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, RocCurveDisplay, roc_curve
)
import matplotlib.pyplot as plt

# xgboost opcjonalnie
try:
    from xgboost import XGBClassifier  # type: ignore
    XGB_AVAILABLE = True
except Exception:
    XGB_AVAILABLE = False

TARGET_COL = "Loan_Status"
ID_COL = "Loan_ID"

# Używamy logów + ratio zamiast surowych dochodów/kwot, żeby ograniczyć współliniowość
NUM_CONT_FEATURES_DEFAULT = [
    "Loan_Amount_Term", "Dependents",
    "LoanToIncome",
    "log_ApplicantIncome", "log_CoapplicantIncome",
    "log_LoanAmount"
]
NUM_BIN_FEATURES_DEFAULT = ["Credit_History"]

CAT_FEATURES_DEFAULT = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]


def load_data(path: str, sep: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=sep)
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Dependents: "3+" -> 3
    if "Dependents" in df.columns:
        df["Dependents"] = (
            df["Dependents"].astype(str).str.strip()
            .replace({"3+": "3", "nan": np.nan, "None": np.nan, "": np.nan})
        )
        df["Dependents"] = pd.to_numeric(df["Dependents"], errors="coerce")

    # Cel: Y/N -> 1/0 (bez FutureWarning)
    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip().str.upper().map({"Y": 1, "N": 0})
        invalid = ~df[TARGET_COL].isin([0, 1])
        if invalid.any():
            df.loc[invalid, TARGET_COL] = np.nan

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cechy domenowe + log-transformy, z ochroną przed dzieleniem przez zero."""
    df = df.copy()

    # Suma dochodów
    if {"ApplicantIncome", "CoapplicantIncome"}.issubset(df.columns):
        df["TotalIncome"] = df[["ApplicantIncome", "CoapplicantIncome"]].sum(axis=1, min_count=1)
    elif "ApplicantIncome" in df.columns:
        df["TotalIncome"] = df["ApplicantIncome"]
    else:
        df["TotalIncome"] = np.nan

    # Gdy TotalIncome <= 0 → NaN (żeby uniknąć nielogicznych ratio)
    df.loc[(df["TotalIncome"].notna()) & (df["TotalIncome"] <= 0), "TotalIncome"] = np.nan

    # Ratio: LoanAmount / TotalIncome tylko gdy TotalIncome > 0
    if "LoanAmount" in df.columns:
        df["LoanToIncome"] = np.where(
            (df["TotalIncome"] > 0),
            df["LoanAmount"] / df["TotalIncome"],
            np.nan
        )
    else:
        df["LoanToIncome"] = np.nan

    # Logi
    for col in ["ApplicantIncome", "CoapplicantIncome", "TotalIncome", "LoanAmount"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    return df


def build_preprocessor(
    num_cont_features: List[str],
    num_bin_features: List[str],
    cat_features: List[str]
) -> ColumnTransformer:
    # Ciągłe: mediana + indykator braków + skaler
    num_cont_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler())
    ])
    # Binarne numeryczne (Credit_History): najczęstsza
    num_bin_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_cont", num_cont_pipe, num_cont_features),
            ("num_bin", num_bin_pipe, num_bin_features),
            ("cat", cat_pipe, cat_features),
        ]
    )
    return preprocessor


def build_model(model_name: str) -> Tuple[str, object]:
    model_name = model_name.lower()
    if model_name == "logreg":
        # Elastic-net stabilizuje współczynniki (mniej „dziwnych” znaków przy kolinearności)
        model = LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.2,
            C=1.0
        )
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )
    elif model_name == "xgb":
        if not XGB_AVAILABLE:
            print("Ostrzeżenie: xgboost nie jest zainstalowany. Używam Logistic Regression.", file=sys.stderr)
            return build_model("logreg")
        model = XGBClassifier(
            n_estimators=600,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            objective="binary:logistic",
            eval_metric="auc"
        )
    else:
        raise ValueError(f"Nieznany model: {model_name}. Użyj: logreg | rf | xgb")
    return model_name, model


def pick_threshold_by_youden(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y_true, y_proba)
    j = tpr - fpr
    best_ix = np.argmax(j)
    return float(thr[best_ix])


def train_and_evaluate(
    df: pd.DataFrame,
    num_cont_features: List[str],
    num_bin_features: List[str],
    cat_features: List[str],
    model_name: str,
    test_size: float,
    random_state: int,
    plots: bool,
    threshold_eval: float
) -> Tuple[Pipeline, Dict[str, float]]:
    df = df.dropna(subset=[TARGET_COL]).copy()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    preprocessor = build_preprocessor(num_cont_features, num_bin_features, cat_features)
    name, base_model = build_model(model_name)
    pipe = Pipeline(steps=[("pre", preprocessor), ("clf", base_model)])
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]

    # Raport dla zadanego progu (domyślnie 0.5 lub jak podasz w argumencie)
    y_pred_eval = (y_proba >= threshold_eval).astype(int)
    metrics_eval = {
        "threshold_eval": threshold_eval,
        "accuracy": accuracy_score(y_test, y_pred_eval),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "precision": precision_score(y_test, y_pred_eval, zero_division=0),
        "recall": recall_score(y_test, y_pred_eval, zero_division=0),
        "f1": f1_score(y_test, y_pred_eval, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_eval).tolist(),
        "report": classification_report(y_test, y_pred_eval, digits=3)
    }

    print(f"\n=== WYNIKI (threshold={threshold_eval:.3f}) ===")
    for k, v in metrics_eval.items():
        if k in ["confusion_matrix", "report", "threshold_eval"]:
            continue
        print(f"{k:>14}: {v:.4f}")
    print("\nMacierz pomyłek [[TN, FP],[FN, TP]]:")
    print(np.array(metrics_eval["confusion_matrix"]))
    print("\nRaport klasyfikacji:")
    print(metrics_eval["report"])

    # Sugestia progu (Youden J)
    thr = pick_threshold_by_youden(y_test.values, y_proba)
    y_pred_j = (y_proba >= thr).astype(int)
    metrics_j = {
        "threshold_opt": thr,
        "accuracy_opt": accuracy_score(y_test, y_pred_j),
        "precision_opt": precision_score(y_test, y_pred_j, zero_division=0),
        "recall_opt": recall_score(y_test, y_pred_j, zero_division=0),
        "f1_opt": f1_score(y_test, y_pred_j, zero_division=0),
        "confusion_matrix_opt": confusion_matrix(y_test, y_pred_j).tolist(),
    }

    print(f"\n=== SUGEROWANY PRÓG (Youden J) = {thr:.3f} ===")
    print(f"{'accuracy':>14}: {metrics_j['accuracy_opt']:.4f}")
    print(f"{'precision':>14}: {metrics_j['precision_opt']:.4f}")
    print(f"{'recall':>14}: {metrics_j['recall_opt']:.4f}")
    print(f"{'f1':>14}: {metrics_j['f1_opt']:.4f}")
    print("Macierz pomyłek [[TN, FP],[FN, TP]]:", np.array(metrics_j["confusion_matrix_opt"]))

    if plots:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"ROC — {name.upper()}")
        plt.show()

    try:
        show_feature_importance(pipe)
    except Exception as e:
        print(f"(Info) Nie pokazano ważności cech: {e}")

    metrics_all = {**metrics_eval, **metrics_j}
    return pipe, metrics_all


def get_feature_names_from_column_transformer(pre: ColumnTransformer) -> List[str]:
    """Nazwy cech po transformacji, w tym indykatory braków (__missing)."""
    output_features: List[str] = []
    for name, transformer, cols in pre.transformers_:
        if name == "remainder":
            continue
        if name == "num_cont":
            imputer: SimpleImputer = transformer.named_steps["imputer"]
            cols = list(cols)
            output_features.extend(cols)
            if hasattr(imputer, "indicator_") and imputer.indicator_ is not None:
                miss_idx = getattr(imputer.indicator_, "features_", None)
                if miss_idx is not None:
                    for idx in miss_idx:
                        if 0 <= idx < len(cols):
                            output_features.append(f"{cols[idx]}__missing")
        elif name == "num_bin":
            output_features.extend(list(cols))
        elif name == "cat":
            ohe = transformer.named_steps["ohe"]
            output_features.extend(list(ohe.get_feature_names_out(cols)))
    return output_features


def show_feature_importance(pipe: Pipeline) -> None:
    pre = pipe.named_steps["pre"]
    feat_names = get_feature_names_from_column_transformer(pre)
    clf = pipe.named_steps["clf"]

    if hasattr(clf, "coef_"):
        coefs = pd.Series(clf.coef_.ravel(), index=feat_names)
        pos = coefs[coefs > 0].sort_values(ascending=False).head(10)
        neg = coefs[coefs < 0].sort_values().head(10)

        def fmt(sr: pd.Series) -> pd.DataFrame:
            return pd.DataFrame({
                "coef": sr.round(4),
                "odds_ratio=exp(coef)": np.exp(sr).round(3)
            })

        print("\nTop + cechy (większa → większa szansa akceptacji):")
        print(fmt(pos))
        print("\nTop − cechy (większa → mniejsza szansa akceptacji):")
        print(fmt(neg))

    elif hasattr(clf, "feature_importances_"):
        fi = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False)
        print("\nTop 15 ważności cech:")
        print(fi.head(15).round(4))
    else:
        print("(Info) Klasyfikator nie udostępnia współczynników/ważności.")


def score_new(file_path: str, sep: str, pipeline_path: str, out_path: str, threshold_score: float) -> None:
    pipe: Pipeline = joblib.load(pipeline_path)
    df_new = load_data(file_path, sep)
    df_new = clean_dataframe(df_new)
    df_new = engineer_features(df_new)

    cols_drop = [c for c in [TARGET_COL] if c in df_new.columns]
    X_new = df_new.drop(columns=cols_drop) if cols_drop else df_new.copy()

    proba = pipe.predict_proba(X_new)[:, 1]
    pred = (proba >= threshold_score).astype(int)  # polityka progu dla outputu

    out = X_new.copy()
    out["Pred_Prob_Approved"] = proba
    out[f"Pred_Approved_{threshold_score:.2f}"] = pred

    if ID_COL in df_new.columns:
        out.insert(0, ID_COL, df_new[ID_COL])

    out.to_csv(out_path, index=False)
    print(f"Zapisano predykcje do: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Model akceptacji wniosku kredytowego")
    parser.add_argument("--data", type=str, required=True, help="Ścieżka do CSV z danymi uczącymi")
    parser.add_argument("--sep", type=str, default=",", help="Separator CSV (domyślnie ',')")
    parser.add_argument("--model", type=str, default="logreg", choices=["logreg", "rf", "xgb"], help="Wybór modelu")
    parser.add_argument("--test_size", type=float, default=0.2, help="Udział testu (domyślnie 0.2)")
    parser.add_argument("--random_state", type=int, default=42, help="Ziarno losowe")
    parser.add_argument("--save_model", type=str, default="loan_approval_model.joblib", help="Gdzie zapisać pipeline")
    parser.add_argument("--plots", action="store_true", help="Pokaż wykres ROC")
    parser.add_argument("--score", type=str, default=None, help="CSV z nowymi wnioskami do wyliczenia predykcji")
    parser.add_argument("--score_out", type=str, default="predictions.csv", help="Gdzie zapisać predykcje dla --score")
    parser.add_argument("--threshold_eval", type=float, default=0.5, help="Próg do raportu metryk")
    parser.add_argument("--threshold_score", type=float, default=0.5, help="Próg do predykcji w pliku wyjściowym")
    args = parser.parse_args()

    df = load_data(args.data, args.sep)
    df = clean_dataframe(df)
    df = engineer_features(df)

    # Złóż finalne listy dostępnych cech
    num_cont = [c for c in NUM_CONT_FEATURES_DEFAULT if c in df.columns]
    num_bin = [c for c in NUM_BIN_FEATURES_DEFAULT if c in df.columns]
    cat = [c for c in CAT_FEATURES_DEFAULT if c in df.columns]

    # Szybka walidacja
    required = set([TARGET_COL, ID_COL] + num_bin + cat)
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"UWAGA: Brakuje kolumn: {missing}. Kontynuuję, ale model może nie działać optymalnie.", file=sys.stderr)

    pipe, metrics = train_and_evaluate(
        df=df,
        num_cont_features=num_cont,
        num_bin_features=num_bin,
        cat_features=cat,
        model_name=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
        plots=args.plots,
        threshold_eval=args.threshold_eval
    )

    joblib.dump(pipe, args.save_model)
    print(f"\nZapisano wytrenowany pipeline do: {args.save_model}")

    if args.score:
        score_new(args.score, args.sep, args.save_model, args.score_out, args.threshold_score)


# Domyślne parametry dla PyCharm (bez wpisywania argumentów)
if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv += [
            "--data", "data/loan_data.csv",
            "--model", "logreg",
            "--plots",
            "--threshold_eval", "0.5",
            "--threshold_score", "0.53"  # np. blisko Youdena z Twojego wyniku (0.526)
        ]
    main()
