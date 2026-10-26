"""Tabla descriptiva de la cohorte (N = 281), total y por respuesta, sin pruebas de hipótesis (Tablas 3.1 y A.2).

Usa la cohorte y el tratamiento de faltantes del pipeline (run_stage2.build_cohort).
Salida: resultados/07_caracterizacion_cohorte.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent / "pipeline_v2"))
from run_stage2 import NC_CODE, build_cohort  # noqa: E402

merged, y, _, _ = build_cohort()
C_DESC = ["age", "menopause_post", "race", "ER_bin", "PR_bin", "HER2_bin", "bilateral_bc", "histologic_type",
          "tumor_side_R", "stage_T", "stage_N", "stage_M", "grade_tubule", "grade_nuclear", "grade_mitotic"]
d = merged[["Patient ID"] + C_DESC].copy()
assert len(d) == 281 and int(y.sum()) == 93
assert d["stage_M"].value_counts().to_dict() == {0: 280, 1: 1}
assert int((d["histologic_type"] == NC_CODE).sum()) == 92 and int((d["bilateral_bc"] == NC_CODE).sum()) == 90

HIST = {0: "DCIS", 1: "ductal", 2: "lobular", 3: "metaplásico", 4: "LCIS", 5: "tubular", 6: "mixto",
        7: "micropapilar", 8: "coloide", 9: "mucinoso", 10: "medular", NC_CODE: "NC (no recolectado)"}
RACE = {1: "Blanca", 2: "Negra", 3: "Asiática", 4: "Nativa", 5: "Hispana", 6: "Multirracial",
        7: "Hawaiana/isleña del Pacífico", 8: "Indígena americana"}
d["y"] = y.values
d["subtipo"] = np.select([(d.ER_bin.eq(1) | d.PR_bin.eq(1)) & d.HER2_bin.eq(0), d.HER2_bin.eq(1)],
                         ["Luminal (RH+/HER2−)", "HER2+"], "Triple negativo")
groups = {"Total": d, "No respondedoras": d[d.y == 0], "Respondedoras": d[d.y == 1]}
rows = []


def add(var, cat, fn):
    rows.append({"Variable": var, "Categoría": cat, **{g: fn(gd) for g, gd in groups.items()}})


def npct(mask_fn):
    return lambda gd: f"{int(mask_fn(gd).sum())} ({mask_fn(gd).mean()*100:.1f}%)"


def cat_block(var, col, mapping, missing_label="Sin dato"):
    for k, lab in mapping.items():
        add(var, lab, npct(lambda gd, k=k: gd[col].eq(k)))
    if d[col].isna().any():
        add(var, missing_label, npct(lambda gd: gd[col].isna()))


rows.append({"Variable": "N", "Categoría": "", **{g: str(len(gd)) for g, gd in groups.items()}})
add("Edad (años)", "media ± DE", lambda gd: f"{gd.age.mean():.1f} ± {gd.age.std():.1f}")
add("Edad (años)", "mediana [RIC]",
    lambda gd: f"{gd.age.median():.1f} [{gd.age.quantile(.25):.1f}–{gd.age.quantile(.75):.1f}]")
cat_block("Menopausia", "menopause_post", {0: "Premenopausia", 1: "Posmenopausia"})
cat_block("Raza/etnia", "race", {k: v for k, v in RACE.items() if (d.race == k).any()})
cat_block("RE", "ER_bin", {1: "Positivo", 0: "Negativo"})
cat_block("RP", "PR_bin", {1: "Positivo", 0: "Negativo"})
cat_block("HER2", "HER2_bin", {1: "Positivo", 0: "Negativo"})
cat_block("Subtipo (subrogado IHQ)", "subtipo", {s: s for s in ["Luminal (RH+/HER2−)", "HER2+", "Triple negativo"]})
cat_block("Lateralidad", "tumor_side_R", {0: "Izquierda", 1: "Derecha"})
cat_block("Cáncer bilateral", "bilateral_bc", {0: "No", 1: "Sí", NC_CODE: "NC (no recolectado)"})
cat_block("Estadio T", "stage_T", {k: f"T{k}" for k in sorted(d.stage_T.dropna().unique().astype(int))})
cat_block("Estadio N", "stage_N", {k: f"N{k}" for k in sorted(d.stage_N.dropna().unique().astype(int))})
cat_block("Estadio M (reconstruido; no predictor)", "stage_M", {0: "M0", 1: "M1"})
for col, lab in [("grade_tubule", "Grado tubular"), ("grade_nuclear", "Grado nuclear"),
                 ("grade_mitotic", "Grado mitótico")]:
    cat_block(lab, col, {k: str(k) for k in sorted(d[col].dropna().unique().astype(int))})
cat_block("Tipo histológico", "histologic_type",
          {k: HIST[k] for k in [1, 2, 6, NC_CODE] if (d.histologic_type == k).any()})
if (~d.histologic_type.isin([1, 2, 6, NC_CODE])).any():
    add("Tipo histológico", "Otros", npct(lambda gd: ~gd.histologic_type.isin([1, 2, 6, NC_CODE])))
tabla = pd.DataFrame(rows).set_index(["Variable", "Categoría"])
tabla.columns = [f"{g} (n={len(groups[g])})" for g in groups]
(AQUI / "resultados").mkdir(exist_ok=True)
tabla.to_csv(AQUI / "resultados" / "07_caracterizacion_cohorte.csv")
print(f"Tabla descriptiva: {len(tabla)} filas")
