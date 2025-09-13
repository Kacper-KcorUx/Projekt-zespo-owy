# EDA — raport skrótowy

- Plik wejściowy: `data/loan_data.csv`
- shape: 381 wierszy × 19 kolumn

## Balans klas

- Liczność: Y=271, N=110
- Udział Y: 0.711

## Braki danych (TOP 10)

| feature          |   missing_pct |
|:-----------------|--------------:|
| Credit_History   |          7.87 |
| Self_Employed    |          5.51 |
| Loan_Amount_Term |          2.89 |
| Dependents       |          2.1  |
| Gender           |          1.31 |
| Married          |          0    |
| Loan_ID          |          0    |
| ApplicantIncome  |          0    |
| Education        |          0    |
| LoanAmount       |          0    |

## Najsilniejsze powiązania liczbowe (|r|, TOP 10) z celem

| feature               |   abs_pearson_r |
|:----------------------|----------------:|
| Credit_History        |      0.618937   |
| log_CoapplicantIncome |      0.135874   |
| LoanToIncome          |      0.0690556  |
| log_TotalIncome       |      0.0658835  |
| Loan_Amount_Term      |      0.0477435  |
| LoanAmount            |      0.0412196  |
| log_LoanAmount        |      0.029894   |
| ApplicantIncome       |      0.0101669  |
| CoapplicantIncome     |      0.00901652 |
| Dependents            |      0.00708471 |

## Cramér’s V dla kategorycznych (TOP 10)

| feature       |   cramers_v |
|:--------------|------------:|
| Property_Area |   0.168128  |
| Gender        |   0.133169  |
| Married       |   0.0924727 |
| Education     |   0.0555858 |
| Self_Employed |   0.0541886 |

## Testy istotności — liczby (Mann–Whitney, sort p)

| feature               |   median_Y |   median_N |   median_diff |     p_value |   effect_r |
|:----------------------|-----------:|-----------:|--------------:|------------:|-----------:|
| Credit_History        |    1       |    0       |    1          | 5.30548e-31 |  0.394815  |
| CoapplicantIncome     | 1210       |    0       | 1210          | 0.0149887   |  0.118462  |
| log_CoapplicantIncome |    7.0992  |    0       |    7.0992     | 0.0149887   |  0.118462  |
| TotalIncome           | 4600       | 4594.5     |    5.5        | 0.226952    |  0.0619265 |
| log_TotalIncome       |    8.43403 |    8.43283 |    0.00119924 | 0.226952    |  0.0619265 |
| log_LoanAmount        |    4.7185  |    4.67283 |    0.04567    | 0.281898    |  0.0551422 |
| LoanAmount            |  111       |  106       |    5          | 0.281898    |  0.0551422 |
| Loan_Amount_Term      |  360       |  360       |    0          | 0.352783    |  0.030576  |
| ApplicantIncome       | 3276       | 3418       | -142          | 0.527488    |  0.0323964 |
| log_ApplicantIncome   |    8.09468 |    8.1371  |   -0.0424197  | 0.527488    |  0.0323964 |

## Testy istotności — kategorie (chi2, sort p)

| feature       |       chi2 |   dof |   p_value |   n |
|:--------------|-----------:|------:|----------:|----:|
| Property_Area | 10.7698    |     2 | 0.0045854 | 381 |
| Married       |  2.85504   |     1 | 0.0910881 | 381 |
| Education     |  0.917226  |     1 | 0.338204  | 381 |
| Gender        |  0.177444  |     1 | 0.673579  | 376 |
| Self_Employed |  0.0057598 |     1 | 0.939504  | 360 |

## Profil LoanToIncome (akceptacja w koszykach — kwantyle)

| LoanToIncome      |   count |   approve_rate |
|:------------------|--------:|---------------:|
| (0.00152, 0.0163] |      55 |       0.709091 |
| (0.0163, 0.0203]  |      54 |       0.685185 |
| (0.0203, 0.0227]  |      54 |       0.722222 |
| (0.0227, 0.0247]  |      55 |       0.690909 |
| (0.0247, 0.0271]  |      54 |       0.814815 |
| (0.0271, 0.0298]  |      54 |       0.703704 |
| (0.0298, 0.0692]  |      55 |       0.654545 |

## Pliki pomocnicze

- Wykresy: `outputs/eda\img`
- Tabele: `outputs/eda\tables`
