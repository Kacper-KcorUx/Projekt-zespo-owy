# eda_credit_risk.py
# EDA dla zbioru wniosków kredytowych — spójne z pipeline'm w loan_model.py
# Wymagania: pandas, numpy, matplotlib, (opcjonalnie) seaborn, scipy, statsmodels (dla VIF)

import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set(context="notebook", style="whitegrid")
    HAS_SNS = True
except Exception:
    HAS_SNS = False

from scipy import stats
from pandas.api import types as pdt

# --- Importy z modelu ----------------------------------------------------------

try:
    from loan_model import (
        load_data, clean_dataframe, engineer_features,
        TARGET_COL
    )
except Exception as e:
    print(f"[WARN] Nie udało się zaimportować z loan_model.py: {e}\n"
          f"Upewnij się, że pliki są w tym samym folderze.", file=sys.stderr)
    raise

# --- Narzędzia -----------------------------------------------------------------

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def to_md(df: pd.DataFrame, index: bool = False) -> str:
    """Bezpieczny markdown: gdy brak 'tabulate' lub stary pandas — fallback na to_string()."""
    try:
        return df.to_markdown(index=index)
    except Exception:
        return df.to_string(index=index)

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér’s V dla dwóch zmiennych kategorycznych (x vs y)."""
    tbl = pd.crosstab(x, y)
    if tbl.size == 0 or tbl.shape[0] < 2 or tbl.shape[1] < 2:
        return np.nan
    chi2 = stats.chi2_contingency(tbl, correction=False)[0]
    n = tbl.to_numpy().sum()
    r, k = tbl.shape
    denom = min(r - 1, k - 1)
    if n == 0 or denom <= 0:
        return np.nan
    return float(np.sqrt((chi2 / n) / denom))

def mann_whitney_effect(a: pd.Series, b: pd.Series):
    """Efekt dla Mann–Whitney (A=Y, B=N): różnica median + r z Z-score."""
    a = pd.Series(a).dropna().astype(float)
    b = pd.Series(b).dropna().astype(float)
    med_diff = float(np.nanmedian(a) - np.nanmedian(b)) if len(a) and len(b) else np.nan
    try:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        mu = len(a) * len(b) / 2.0
        sigma = np.sqrt(len(a) * len(b) * (len(a) + len(b) + 1) / 12.0)
        z = (u - mu) / sigma if sigma > 0 else np.nan
        r = abs(z) / np.sqrt(len(a) + len(b)) if (len(a) + len(b)) > 0 else np.nan
    except Exception:
        p, r = np.nan, np.nan
    return med_diff, p, r

def vif_table(df_num: pd.DataFrame):
    """VIF dla liczb (opcjonalnie, jeśli statsmodels dostępne)."""
    try:
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except Exception:
        return None
    X = df_num.dropna().copy()
    if X.shape[0] < 10 or X.shape[1] < 2:
        return None
    X = sm.add_constant(X, has_constant="add")
    cols = [c for c in X.columns if c != "const"]
    vifs = []
    for i, c in enumerate(cols):
        try:
            vifs.append((c, float(variance_inflation_factor(X[cols].values, i))))
        except Exception:
            vifs.append((c, np.nan))
    return pd.DataFrame(vifs, columns=["feature", "VIF"]).sort_values("VIF", ascending=False)

def is_high_cardinality(series: pd.Series, n: int) -> bool:
    """Czy kolumna ma wysoką krotność? Wyklucz z asocjacji (np. Loan_ID)."""
    return series.nunique(dropna=True) > max(30, int(0.2 * n))

# --- Główna logika -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="EDA dla wniosków kredytowych")
    parser.add_argument("--data", required=True, help="Ścieżka do CSV z danymi")
    parser.add_argument("--sep", default=",", help="Separator CSV (domyślnie ',')")
    parser.add_argument("--outdir", default="outputs/eda", help="Folder wyjściowy")
    args = parser.parse_args()

    outdir = ensure_dir(args.outdir)
    imgdir = ensure_dir(os.path.join(outdir, "img"))
    tbldir = ensure_dir(os.path.join(outdir, "tables"))

    # 1) Wczytanie i spójne przygotowanie
    df = load_data(args.data, args.sep)
    df = clean_dataframe(df)
    df = engineer_features(df)

    # 2) Podstawy: kształt, dtypes, braki
    shape_info = f"shape: {df.shape[0]} wierszy × {df.shape[1]} kolumn"
    miss = df.isna().mean().sort_values(ascending=False)
    miss_df = pd.DataFrame({"missing_pct": (miss * 100).round(2)})
    miss_path = os.path.join(tbldir, "missing.csv"); miss_df.to_csv(miss_path)

    # 3) Balans klas
    if TARGET_COL in df.columns:
        cls = df[TARGET_COL].dropna().astype(int)
        cls_counts = cls.value_counts().rename({0: "N", 1: "Y"})
        cls_rate = float(cls.mean())
    else:
        print("[INFO] Brak kolumny celu — przerywam analizy targetowe.")
        cls_counts, cls_rate = pd.Series(dtype=int), np.nan

    # 4) Podział kolumn na typy
    cat_cols = [c for c in df.columns if pdt.is_object_dtype(df[c]) or pdt.is_categorical_dtype(df[c])]
    # num_cols: wszystko poza kategoriami, celem i ID
    num_cols = [c for c in df.columns if c not in cat_cols + [TARGET_COL, "Loan_ID"]]

    # 5) Opis statystyczny liczb
    desc = df[num_cols].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
    desc_path = os.path.join(tbldir, "describe_numeric.csv"); desc.to_csv(desc_path)

    # 6) Korelacje liczbowe (Pearson) + heatmapa
    corr_cols = [c for c in num_cols] + ([TARGET_COL] if TARGET_COL in df.columns else [])
    corr = df[corr_cols].corr(method="pearson")
    corr_path = os.path.join(tbldir, "corr_numeric.csv"); corr.to_csv(corr_path)
    plt.figure(figsize=(10, 8))
    if HAS_SNS:
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    else:
        plt.imshow(corr, cmap="coolwarm"); plt.colorbar()
        plt.title("Correlation matrix (Pearson)")
    plt.tight_layout()
    plt.savefig(os.path.join(imgdir, "corr_heatmap.png")); plt.close()

    # 7) Cramér’s V dla kategorycznych vs cel — bez ID / wysokiej krotności
    cramers_rows = []
    cramers_df = pd.DataFrame(columns=["feature", "cramers_v"])
    if TARGET_COL in df.columns:
        n = len(df)
        cat_for_assoc = [
            c for c in cat_cols
            if c.lower() != "loan_id" and not is_high_cardinality(df[c], n)
        ]
        for c in cat_for_assoc:
            try:
                v = cramers_v(df[c].astype(str), df[TARGET_COL].astype(int))
            except Exception:
                v = np.nan
            cramers_rows.append((c, v))
        cramers_df = pd.DataFrame(cramers_rows, columns=["feature", "cramers_v"]).sort_values(
            "cramers_v", ascending=False
        )
    cramers_path = os.path.join(tbldir, "cramers_v_target.csv"); cramers_df.to_csv(cramers_path, index=False)

    # 8) Testy istotności
    tests_num = []
    tests_cat = []
    if TARGET_COL in df.columns:
        y = df[TARGET_COL].astype(float)
        # liczby — Mann–Whitney
        for c in num_cols:
            a = df.loc[y == 1, c].replace([np.inf, -np.inf], np.nan).dropna()
            b = df.loc[y == 0, c].replace([np.inf, -np.inf], np.nan).dropna()
            if len(a) > 3 and len(b) > 3:
                med_diff, p, r = mann_whitney_effect(a, b)
                tests_num.append((c, float(np.nanmedian(a)), float(np.nanmedian(b)), med_diff, p, r))
        # kategorie — chi-kwadrat (tylko sensowne)
        n = len(df)
        cat_for_assoc = [
            c for c in cat_cols
            if c.lower() != "loan_id" and not is_high_cardinality(df[c], n)
        ]
        for c in cat_for_assoc:
            tbl = pd.crosstab(df[c], y)
            if tbl.shape[0] > 1 and tbl.shape[1] > 1:
                try:
                    chi2, p, dof, _ = stats.chi2_contingency(tbl)
                except Exception:
                    chi2, p, dof = np.nan, np.nan, np.nan
                tests_cat.append((c, chi2, dof, p, int(tbl.values.sum())))
    tests_num_df = pd.DataFrame(
        tests_num, columns=["feature", "median_Y", "median_N", "median_diff", "p_value", "effect_r"]
    ).sort_values("p_value")
    tests_cat_df = pd.DataFrame(
        tests_cat, columns=["feature", "chi2", "dof", "p_value", "n"]
    ).sort_values("p_value")
    tests_num_df.to_csv(os.path.join(tbldir, "tests_numeric_mannwhitney.csv"), index=False)
    tests_cat_df.to_csv(os.path.join(tbldir, "tests_categorical_chi2.csv"), index=False)

    # 9) VIF (opcjonalnie)
    vif_df = vif_table(df[num_cols])
    if vif_df is not None:
        vif_df.to_csv(os.path.join(tbldir, "vif.csv"), index=False)

    # 10) Profil LoanToIncome — kwantyle (automatycznie dopasowane do rozkładu)
    lti_profile = None
    if "LoanToIncome" in df.columns and TARGET_COL in df.columns:
        valid = df[["LoanToIncome", TARGET_COL]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) > 0:
            # Spróbuj 7 kwantyli, w razie duplikatów zmniejsz liczbę koszyków
            binned = None
            for q in (7, 6, 5, 4):
                try:
                    binned = pd.qcut(valid["LoanToIncome"], q=q, duplicates="drop")
                    if binned.nunique() >= 3:
                        break
                except ValueError:
                    continue
            if binned is None:
                # awaryjnie: 3 koszyki równej szerokości
                edges = np.linspace(valid["LoanToIncome"].min(), valid["LoanToIncome"].max(), 4)
                binned = pd.cut(valid["LoanToIncome"], bins=np.unique(edges), include_lowest=True)
            g = valid.groupby(binned)[TARGET_COL]
            lti_profile = pd.DataFrame({"count": g.size(), "approve_rate": g.mean()}).reset_index()
            lti_profile.to_csv(os.path.join(tbldir, "loan_to_income_profile.csv"), index=False)

            # Wykres słupkowy profilu LTI
            ax = lti_profile.set_index("LoanToIncome").approve_rate.plot(kind="bar", figsize=(8, 4))
            ax.set_ylabel("Approve rate (mean of Y)")
            ax.set_xlabel("LoanToIncome — kwantyle")
            plt.tight_layout()
            plt.savefig(os.path.join(imgdir, "loan_to_income_profile.png")); plt.close()

    # 11) Wykresy: rozkłady + boxploty vs target
    def save_hist(col: str):
        plt.figure(figsize=(6, 4))
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if HAS_SNS:
            sns.histplot(series, kde=True, bins=30)
        else:
            plt.hist(series, bins=30)
        plt.title(f"Distribution: {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(imgdir, f"dist_{col}.png")); plt.close()

    def save_box_vs_target(col: str):
        if TARGET_COL not in df.columns:
            return
        plt.figure(figsize=(6, 4))
        d0 = df.loc[df[TARGET_COL] == 0, col].replace([np.inf, -np.inf], np.nan)
        d1 = df.loc[df[TARGET_COL] == 1, col].replace([np.inf, -np.inf], np.nan)
        if HAS_SNS:
            sns.boxplot(x=df[TARGET_COL], y=df[col])
            try:
                sns.stripplot(x=df[TARGET_COL], y=df[col], size=2, alpha=0.4, color="k")
            except Exception:
                pass
        else:
            plt.boxplot([d0.dropna(), d1.dropna()], labels=["0", "1"])
        plt.title(f"{col} vs Loan_Status")
        plt.tight_layout()
        plt.savefig(os.path.join(imgdir, f"box_{col}_vs_target.png")); plt.close()

    top_num_for_plots = [
        c for c in ["LoanToIncome", "log_LoanAmount", "log_ApplicantIncome", "log_CoapplicantIncome", "Loan_Amount_Term"]
        if c in df.columns
    ]
    for c in top_num_for_plots:
        save_hist(c)
        save_box_vs_target(c)

    # Kategoryczne: stacked bar (udział Y/N) — tylko bez high-card
    if TARGET_COL in df.columns:
        n = len(df)
        cats_for_plot = [
            c for c in ["Property_Area", "Education", "Married", "Self_Employed", "Gender"]
            if c in df.columns and not is_high_cardinality(df[c], n)
        ]
        for c in cats_for_plot:
            ct = pd.crosstab(df[c], df[TARGET_COL], normalize="index")
            ct.to_csv(os.path.join(tbldir, f"stack_{c}.csv"))
            ax = ct.plot(kind="bar", stacked=True, figsize=(6, 4))
            ax.set_ylabel("Udział")
            ax.set_xlabel(c)
            plt.title(f"{c}: udział Y/N")
            plt.tight_layout()
            plt.savefig(os.path.join(imgdir, f"stack_{c}.png")); plt.close()

    # 12) Raport markdown
    report_path = os.path.join(outdir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# EDA — raport skrótowy\n\n")
        f.write(f"- Plik wejściowy: `{args.data}`\n")
        f.write(f"- {shape_info}\n\n")
        if TARGET_COL in df.columns:
            f.write("## Balans klas\n\n")
            f.write(f"- Liczność: Y={int(cls_counts.get('Y', 0))}, N={int(cls_counts.get('N', 0))}\n")
            f.write(f"- Udział Y: {cls_rate:.3f}\n\n")
        f.write("## Braki danych (TOP 10)\n\n")
        f.write(to_md(miss_df.reset_index().rename(columns={'index':'feature'}).head(10), index=False) + "\n\n")
        f.write("## Najsilniejsze powiązania liczbowe (|r|, TOP 10) z celem\n\n")
        if TARGET_COL in df.columns and TARGET_COL in corr.columns:
            r_abs = corr[TARGET_COL].drop(TARGET_COL, errors="ignore").abs().sort_values(ascending=False)
            f.write(
                to_md(r_abs.reset_index().rename(columns={'index': 'feature', TARGET_COL: 'abs_pearson_r'}).head(10),
                      index=False) + "\n\n")
        f.write("## Cramér’s V dla kategorycznych (TOP 10)\n\n")
        if len(cramers_df) > 0:
            f.write(to_md(cramers_df.head(10), index=False) + "\n\n")
        f.write("## Testy istotności — liczby (Mann–Whitney, sort p)\n\n")
        if not tests_num_df.empty:
            f.write(to_md(tests_num_df.head(10), index=False) + "\n\n")
        f.write("## Testy istotności — kategorie (chi2, sort p)\n\n")
        if not tests_cat_df.empty:
            f.write(to_md(tests_cat_df.head(10), index=False) + "\n\n")
        if lti_profile is not None:
            f.write("## Profil LoanToIncome (akceptacja w koszykach — kwantyle)\n\n")
            f.write(to_md(lti_profile, index=False) + "\n\n")
        if vif_df is not None:
            f.write("## VIF (multikolinearność)\n\n")
            f.write(to_md(vif_df.head(15), index=False) + "\n\n")
        f.write("## Pliki pomocnicze\n\n")
        f.write(f"- Wykresy: `{imgdir}`\n")
        f.write(f"- Tabele: `{tbldir}`\n")

    print("\n[EDA] Zakończono. Najważniejsze pliki:")
    print(f"- Raport: {report_path}")
    print(f"- Heatmapa korelacji: {os.path.join(imgdir, 'corr_heatmap.png')}")
    print(f"- Braki: {os.path.join(tbldir, 'missing.csv')}")
    print(f"- Korelacje: {os.path.join(tbldir, 'corr_numeric.csv')}")
    print(f"- Cramér’s V: {os.path.join(tbldir, 'cramers_v_target.csv')}")
    print(f"- Testy num: {os.path.join(tbldir, 'tests_numeric_mannwhitney.csv')}")
    print(f"- Testy kat: {os.path.join(tbldir, 'tests_categorical_chi2.csv')}")
    if lti_profile is not None:
        print(f"- Profil LTI: {os.path.join(tbldir, 'loan_to_income_profile.csv')}")
        print(f"- Wykres LTI: {os.path.join(imgdir, 'loan_to_income_profile.png')}")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # domyślki pod PyCharm
        sys.argv += ["--data", "data/loan_data.csv", "--sep", ",", "--outdir", "outputs/eda"]
    main()
