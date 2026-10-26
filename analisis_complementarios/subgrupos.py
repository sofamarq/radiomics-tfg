"""Análisis exploratorio post hoc por subgrupo (Tabla 3.9): luminal frente a no luminal.

Usa las predicciones fuera de pliegue promediadas de los modelos globales (Etapa 3, 10 × 5 × 5); no reentrena.
Subtipo subrogado por inmunohistoquímica: luminal = (RE+ o RP+) y HER2−; no luminal = HER2+ o triple negativo.
Bootstrap pareado a nivel paciente dentro de cada subgrupo (5000 remuestreos, semilla 20260909, mismo
procedimiento que la Etapa 3).

Entradas: ../pipeline_v2/stage3_oof_aggregated.csv, stage3_oof_long.csv, stage3_bootstrap.csv
Salidas: resultados/grupos_metricas.csv, resultados/grupos_por_repeticion.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as ap, roc_auc_score as roc

AQUI = Path(__file__).resolve().parent
PIPE = AQUI.parent / "pipeline_v2"
RES = AQUI / "resultados"
sys.path.insert(0, str(PIPE))
from run_stage2 import ID_COL, build_cohort  # noqa: E402
from run_stage3 import BOOT_SEED, N_BOOT  # noqa: E402

M = ["C", "C+R", "C+A+R"]
ADD = ["C+R", "C+A+R"]

merged, _, _, _ = build_cohort()
d = merged[[ID_COL, "ER_bin", "PR_bin", "HER2_bin"]].rename(columns={ID_COL: "patient_id"}).copy()
hr = (d.ER_bin == 1) | (d.PR_bin == 1)
d["subtipo"] = np.select([d.HER2_bin == 1, hr & (d.HER2_bin == 0), (~hr) & (d.HER2_bin == 0)],
                         ["HER2+", "Luminal HR+/HER2−", "Triple negativo"], default="Sin clasificar")
sub = d[["patient_id", "subtipo"]].copy()
sub["grupo"] = np.where(sub.subtipo == "Luminal HR+/HER2−", "Luminal", "No luminal")

agg = pd.read_csv(PIPE / "stage3_oof_aggregated.csv").merge(sub, on="patient_id")
long = pd.read_csv(PIPE / "stage3_oof_long.csv").merge(sub, on="patient_id")
nl = agg[agg.grupo == "No luminal"]
assert len(nl) == 162 and nl.y_true.sum() == 75, (len(nl), nl.y_true.sum())

GROUPS = {"No luminal (HER2+ ∪ TN)": ("grupo", "No luminal"), "Luminal": ("grupo", "Luminal"),
          "HER2+": ("subtipo", "HER2+"), "Triple negativo": ("subtipo", "Triple negativo"), "Global": (None, None)}


def sel(df, g):
    return df if GROUPS[g][0] is None else df[df[GROUPS[g][0]] == GROUPS[g][1]]


def metrics(y, P):
    return {f"{k}_{m}": f(y, P[m]) for m in M for k, f in (("roc", roc), ("pr", ap))}


def boot_draws(a, rng):
    y = a.y_true.values
    P = {m: a[m].values for m in M}
    out = []
    while len(out) < N_BOOT:
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        r = {m: (roc(y[i], P[m][i]), ap(y[i], P[m][i])) for m in M}
        out.append([r[m][j] - r["C"][j] for m in ADD for j in (0, 1)])
    return np.array(out)   # columnas: C+R roc, C+R pr, C+A+R roc, C+A+R pr


rows, reps = [], []
for g in GROUPS:
    a = sel(agg, g)
    met = metrics(a.y_true.values, a)
    draws = boot_draws(a, np.random.default_rng(BOOT_SEED))
    rec = {"grupo": g, "N": len(a), "positivos": int(a.y_true.sum()), "prevalencia": a.y_true.mean(), **met}
    for j, (m, k) in enumerate([(m, k) for m in ADD for k in ("roc", "pr")]):
        rec[f"d_{k}_{m}"] = met[f"{k}_{m}"] - met[f"{k}_C"]
        rec[f"d_{k}_{m}_lo"], rec[f"d_{k}_{m}_hi"] = np.percentile(draws[:, j], [2.5, 97.5])
    for r, gg in sel(long, g).groupby("repetition"):
        mm = metrics(gg.y_true.values, gg)
        reps.append({"grupo": g, "repetition": r, **mm,
                     **{f"d_{k}_{m}": mm[f"{k}_{m}"] - mm[f"{k}_C"] for m in ADD for k in ("roc", "pr")}})
    rows.append(rec)
res = pd.DataFrame(rows)
reps = pd.DataFrame(reps)
st = reps.groupby("grupo")[[c for c in reps.columns if c.startswith("d_")]].agg(
    ["mean", "std", lambda s: int((s > 0).sum())])
st.columns = [f"{a}_{ {'mean': 'mean', 'std': 'sd'}.get(b, 'npos')}" for a, b in st.columns]
res = res.merge(st.reset_index(), on="grupo")

# control: el grupo Global reproduce el IC oficial de la Etapa 3
off = pd.read_csv(PIPE / "stage3_bootstrap.csv").set_index("comparison").loc["C vs C+A+R"]
assert abs(res.set_index("grupo").loc["Global", "d_roc_C+A+R_lo"] - off.delta_roc_ci_lo) < 1e-12

RES.mkdir(exist_ok=True)
res.to_csv(RES / "grupos_metricas.csv", index=False)
reps.to_csv(RES / "grupos_por_repeticion.csv", index=False)
print(res[["grupo", "N", "positivos"]].to_string(index=False))
