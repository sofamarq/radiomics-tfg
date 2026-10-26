"""Comparación exploratoria del número de componentes principales en C+A+R (Tabla 3.3).

Validación cruzada anidada única 5 × 3 (semilla 42), con las mismas particiones para todas las
configuraciones. Regresión logística; C se selecciona en la validación interna (ROC-AUC). Métricas sobre las
predicciones fuera de pliegue. No es comparable con la validación anidada repetida 10 × 5 × 5.

Configuraciones: grilla oficial de componentes (selección interna) y 2, 3, 19 y 30 componentes fijos.
Salida: resultados/pca_exploratorio_5x3.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "pipeline_v2"))
from run_stage2 import build_cohort  # noqa: E402
from run_stage4 import make_estimator4, make_grid4  # noqa: E402

merged, y, rad_cols, _ = build_cohort()
yv = y.values
CAR = ["C", "A", "R"]
outer = StratifiedKFold(5, shuffle=True, random_state=42)
inner = StratifiedKFold(3, shuffle=True, random_state=42)


def run(nombre, ncomp=None):
    grid = make_grid4("logistic", CAR)
    if ncomp is not None:
        grid["preprocessor__R__pca__n_components"] = [ncomp]
    oof = np.full(len(yv), np.nan)
    ncs = []
    for tr, te in outer.split(merged, yv):
        gs = GridSearchCV(make_estimator4("logistic", CAR, rad_cols), grid, scoring="roc_auc", cv=inner,
                          n_jobs=-1).fit(merged.iloc[tr], y.iloc[tr])
        oof[te] = gs.predict_proba(merged.iloc[te])[:, 1]
        ncs.append(int(gs.best_estimator_.named_steps["preprocessor"].named_transformers_["R"]
                       .named_steps["pca"].n_components_))
    r = {"configuracion": nombre, "roc_oof": roc_auc_score(yv, oof), "pr_oof": average_precision_score(yv, oof),
         "n_comp": "/".join(map(str, ncs))}
    print(r, flush=True)
    return r


filas = [run("Grilla oficial"), run("PCA 2 fijos", 2), run("PCA 3 fijos", 3), run("PCA 19 fijos", 19),
         run("PCA 30 fijos", 30)]
(AQUI / "resultados").mkdir(exist_ok=True)
pd.DataFrame(filas).to_csv(AQUI / "resultados" / "pca_exploratorio_5x3.csv", index=False)
