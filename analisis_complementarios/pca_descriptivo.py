"""PCA descriptivo global del bloque R (Figura 3.2).

Ajuste único sobre la cohorte completa (N = 281): imputación por mediana, eliminación de varianza cero y
filtro |r| > 0,95, estandarización y PCA. Es un análisis descriptivo, independiente de los PCA ajustados
dentro de cada partición durante la validación de los modelos.

Salida: resultados/scree.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "pipeline_v2"))
from run_stage2 import CORR_THRESHOLD, CorrelationFilter, build_cohort  # noqa: E402

merged, y, rad_cols, _ = build_cohort()
X = SimpleImputer(strategy="median").fit_transform(merged[rad_cols])
cf = CorrelationFilter(CORR_THRESHOLD).fit(X)
X = StandardScaler().fit_transform(cf.transform(X))
ev = PCA().fit(X).explained_variance_ratio_
cum = np.cumsum(ev)
out = {"n_rad_in": len(rad_cols), "n_after_corr": int(cf.n_out_), "ev": ev[:60].tolist(), "cum": cum[:60].tolist(),
       "n70": int(np.searchsorted(cum, .70) + 1), "n80": int(np.searchsorted(cum, .80) + 1),
       "n90": int(np.searchsorted(cum, .90) + 1), "cum10": float(cum[9])}
(AQUI / "resultados").mkdir(exist_ok=True)
json.dump(out, open(AQUI / "resultados" / "scree.json", "w"), indent=1)
print({k: v for k, v in out.items() if k not in ("ev", "cum")})
