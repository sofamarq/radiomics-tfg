"""Recalculo minimo del hold-out 80/20 para RF y GB (el intento previo solo persistio ROC en el log).
Necesario para completar la tabla de la Etapa 4 (columnas C PR / C+A+R PR).
Mismo split seed=42 que Etapas 2-4. inner cv=3, n_jobs=2 (version liviana). Logistic NO se toca."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from run_stage2 import OUT_DIR, RANDOM_STATE, build_cohort
from run_stage4 import MODELS_S4, make_estimator4, make_grid4

merged, y, rad_cols, _ = build_cohort()
idx = np.arange(len(merged))
tr, te = train_test_split(idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

rows = []
for algo in ("random_forest", "gradient_boosting"):
    r = {"algorithm": algo}
    for m, blocks in MODELS_S4.items():
        gs = GridSearchCV(make_estimator4(algo, blocks, rad_cols), make_grid4(algo, blocks),
                          scoring="roc_auc", cv=inner, n_jobs=2, refit=True)
        gs.fit(merged.iloc[tr], y.iloc[tr])
        p = gs.predict_proba(merged.iloc[te])[:, 1]
        tag = "C" if m == "C" else "C_A_R"
        r[f"roc_{tag}"] = roc_auc_score(y.iloc[te], p)
        r[f"pr_{tag}"] = average_precision_score(y.iloc[te], p)
        print(f"{algo} {m}: ROC={r[f'roc_{tag}']:.3f} PR={r[f'pr_{tag}']:.3f}")
    r["delta_roc"] = r["roc_C_A_R"] - r["roc_C"]
    r["delta_pr"] = r["pr_C_A_R"] - r["pr_C"]
    rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "stage4light_holdout_rfgb.csv", index=False)
print(df.to_string(index=False))
print("saved stage4light_holdout_rfgb.csv")
