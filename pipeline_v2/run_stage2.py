"""
ETAPA 2 - Pipeline corregido y reproducible.
TFG Radiomics NACT - Duke-Breast-Cancer-MRI.

Corre dos analisis con EXACTAMENTE el mismo preprocessing/logica:
  A) split historico 80/20 estratificado, random_state=42  -> comparabilidad
  B) Nested Stratified CV 5x5                               -> estimacion principal

Modelo: Logistic Regression L2 (class_weight='balanced'), unico clasificador.
Bloques: C (clinica/patologia basal), M (MRI findings cualitativos basales),
         A (adquisicion/contexto tecnico MRI), R (radiomics).
Exclusiones: O (mama/eco), T (tratamiento), X dudosas, leakage/post-outcome.

Toda imputacion / encoding / scaling / filtro de correlacion / PCA / tuning de C
se ajusta EXCLUSIVAMENTE sobre el train de cada fold.

Salidas: pipeline_v2/results_stage2.csv, pipeline_v2/oof_predictions.csv
No modifica notebook ni bases originales.
"""

from __future__ import annotations
import json
import warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------------------
RANDOM_STATE = 42
CORR_THRESHOLD = 0.95
C_GRID = [0.01, 0.1, 1.0, 10.0]
# Grilla ampliada hacia abajo (2026-09-29): con {5,...,30} la seleccion quedaba en el minimo
NCOMP_GRID = [1, 2, 3, 5, 10, 15, 20, 30]
OUTER_SPLITS = 5
INNER_SPLITS = 5

ROOT = Path(__file__).resolve().parent.parent
DB = Path(os.environ.get("RADIOMICS_DATOS", ROOT / "datos"))   # carpeta con los dos Excel de Duke (configurable)
CLIN_XLSX = DB / "Clinical_and_Other_Features.xlsx"
IMG_XLSX = DB / "Imaging_Features.xlsx"
OUT_DIR = ROOT / "pipeline_v2"

TARGET_COL = "Overall Near-complete Response:  Stricter Definition"
MET_PRES_COL = "Metastatic at Presentation (Outside of Lymph Nodes)"
# Codigo de categoria explicita "NC" (Data Not Collected, leyenda de Duke) para histologic_type y
# bilateral_bc. Significa "dato no recolectado": NO es una categoria clinica ni ausencia de la condicion.
NC_CODE = -99
ID_COL = "Patient ID"

# Modelos a evaluar (orden logico, NO por performance)
MODELS: dict[str, list[str]] = {
    "C":       ["C"],
    "C+M":     ["C", "M"],
    "C+A":     ["C", "A"],
    "R":       ["R"],
    "C+R":     ["C", "R"],
    "C+M+R":   ["C", "M", "R"],
    "C+A+R":   ["C", "A", "R"],
    "C+M+A+R": ["C", "M", "A", "R"],
}

# --------------------------------------------------------------------------------------
# Definicion semantica de variables (NO se usa select_dtypes)
# Claves = nombre de columna final ya limpia. Valor = columna cruda de origen (o None si derivada).
# --------------------------------------------------------------------------------------
C_CONTINUOUS = {"age": None}                                # derivada de 'Date of Birth (Days)'
C_BINARY = {
    "ER_bin": "ER status",
    "PR_bin": "PR status",
    "HER2_bin": "HER2 status",
    "menopause_post": "Menopause (at diagnosis)",           # 0=pre 1=post ; 2(N/A)->missing
    "tumor_side_R": "Tumor Location",                       # L/R ; reconstruida desde 'Position'
}
C_ORDINAL = {
    "stage_T": "Staging(Tumor Size)# [T]",                  # 1..4
    "stage_N": "Staging(Nodes)#(Nx replaced by -1)[N]",     # 0..3 ; -1 (Nx) -> missing
    # stage_M: se reconstruye en build_cohort (Mx -> 'Metastatic at Presentation') pero NO es predictor:
    # quedo practicamente constante (280/281 M0, 1 M1) -> excluida por varianza casi nula (2026-10-01).
    "grade_tubule": "Tumor Grade (Tubule)",                 # 1..3
    "grade_nuclear": "Tumor Grade (Nuclear)",               # 1..3
    "grade_mitotic": "Tumor Grade (Mitotic)",               # 1..3
}
C_NOMINAL = {
    "race": "Race and Ethnicity",                           # 1..8 ; 0 (N/A) -> missing
    "histologic_type": "Histologic type",                   # 0..10 ; vacio -> NC_CODE
    "bilateral_bc": "Bilateral breast cancer",              # 0/1 ; NC -> NC_CODE (nominal de 3 niveles)
}

M_BINARY = {
    "mf_multicentric": "Multicentric/Multifocal",
    "mf_contralateral": "Contralateral Breast Involvement",
    "mf_lymphadenopathy": "Lymphadenopathy or Suspicious Nodes",
    "mf_skin_nipple": "Skin/Nipple Invovlement",
    "mf_pec_chest": "Pec/Chest Involvement",
}

A_CONTINUOUS = {
    "days_to_mri": "Days to MRI (From the Date of Diagnosis)",
    "tr": "TR (Repetition Time)",
    "te": "TE (Echo Time)",
}
A_ORDINAL = {
    "field_strength": "Field Strength (Tesla)",             # code monotono
    "slice_thickness": "Slice Thickness ",
    "matrix_rows": "Rows",
    "flip_angle": "Flip Angle \n",
    "fov": "FOV Computed (Field of View) in cm ",
}
A_BINARY = {"patient_position": "Patient Position During MRI"}
A_NOMINAL = {
    "manufacturer": "Manufacturer",
    "model_name": "Manufacturer Model Name",
    "scan_options": "Scan Options",
    "acq_matrix": "Acquisition Matrix",
    "contrast_agent": "Contrast Agent",
}

BLOCK_TYPES = {
    "C": {"continuous": list(C_CONTINUOUS), "binary": list(C_BINARY),
          "ordinal": list(C_ORDINAL), "nominal": list(C_NOMINAL)},
    "M": {"binary": list(M_BINARY)},
    "A": {"continuous": list(A_CONTINUOUS), "binary": list(A_BINARY),
          "ordinal": list(A_ORDINAL), "nominal": list(A_NOMINAL)},
}


# --------------------------------------------------------------------------------------
# Carga y construccion de la cohorte
# --------------------------------------------------------------------------------------
def _to_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


# El Excel clínico de TCIA tiene una fila superior de grupos ("Patient Information", ...) y algunas columnas
# cuyo nombre está en la fila de grupos o en la leyenda. Se quita esa fila y se asignan nombres únicos a estas
# 16 columnas (índice: nombre de la fila de encabezados en el archivo oficial -> nombre usado en el análisis).
ENCABEZADOS_ARMONIZADOS = {
    23: ("ER", "ER status"), 24: ("PR", "PR status"), 25: ("HER2", "HER2 status"),
    31: ("Tumor Grade", "Tumor Grade (Tubule)"), 32: (None, "Tumor Grade (Nuclear)"),
    33: (None, "Tumor Grade (Mitotic)"),
    38: ("Bilateral Information", "Bilateral breast cancer"), 39: (None, "Bilateral different. Rec status"),
    40: (None, "Bilateral. Side Annotated"),
    41: ("For Other Side If Bilateral", "For Other Side If Bilateral. Side of cancer"),
    42: (None, "Other side oncotype score"), 43: (None, "Other side Norringham grade"),
    44: (None, "Other side ER"), 45: (None, "Other side PR"), 46: (None, "Other side HER2"),
    47: (None, "Other side Mol Subtype"),
}


def leer_clinico(sheet_name: str = "Data") -> pd.DataFrame:
    """Lee una hoja del Excel clínico (sin encabezado) con la fila de nombres de columna en la posición 0."""
    raw = pd.read_excel(CLIN_XLSX, sheet_name=sheet_name, header=None)
    if raw.iat[0, 0] == "Patient Information":          # archivo oficial de TCIA
        raw = raw.iloc[1:].reset_index(drop=True)
        for j, (original, nombre) in ENCABEZADOS_ARMONIZADOS.items():
            actual = raw.iat[0, j]
            assert (pd.isna(actual) if original is None else actual == original), (j, actual)
            raw.iat[0, j] = nombre
    return raw


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = leer_clinico("Data")
    names = raw.iloc[0].tolist()
    clin = raw.iloc[2:].copy()                # fila 1 = leyenda -> se descarta
    clin.columns = names
    clin = clin.reset_index(drop=True)
    rad = pd.read_excel(IMG_XLSX)             # sin fila de leyenda
    return clin, rad


def dedup_radiomics(rad: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Elimina columnas radiomicas exactamente duplicadas (target-blind)."""
    feats = [c for c in rad.columns if c != ID_COL]
    seen: dict[tuple, str] = {}
    keep: list[str] = []
    dropped: list[str] = []
    for c in feats:
        key = tuple(pd.util.hash_pandas_object(rad[c].fillna(-9.99e12), index=False))
        if key in seen:
            dropped.append(c)
        else:
            seen[key] = c
            keep.append(c)
    return rad[[ID_COL] + keep].copy(), dropped


def build_cohort(target_col: str = TARGET_COL,
                  valid_values=(0, 1, 2),
                  positive_values=(1, 2),
                  expected_n: int = 281) -> tuple[pd.DataFrame, pd.Series, list[str], dict]:
    """target_col/valid_values/positive_values permiten reutilizar EXACTAMENTE el mismo
    armado de cohorte (C/M/A/R, dedup radiomics, merge) con una definicion de target
    alternativa (p. ej., analisis de sensibilidad). Por defecto reproduce el target
    principal (Stricter) sin cambios."""
    clin, rad = load_raw()

    neo = _to_num(clin["Received Neoadjuvant Therapy or Not"])
    chemo = _to_num(clin["Neoadjuvant Chemotherapy"])
    tgt = _to_num(clin[target_col])
    mask = (neo == 1) & (chemo == 1) & (tgt.isin(list(valid_values)))
    coh = clin[mask].copy().reset_index(drop=True)
    y = tgt[mask].reset_index(drop=True).isin(list(positive_values)).astype(int)
    y.name = "y"

    out = pd.DataFrame({ID_COL: coh[ID_COL].values})

    # ---- C: derivadas / limpias ----
    out["age"] = -_to_num(coh["Date of Birth (Days)"]) / 365.25
    men = _to_num(coh["Menopause (at diagnosis)"]);  men = men.where(men != 2, np.nan)
    out["menopause_post"] = men
    race = _to_num(coh["Race and Ethnicity"]);  race = race.where(race != 0, np.nan)
    out["race"] = race
    for dst, src in {"ER_bin": "ER status", "PR_bin": "PR status", "HER2_bin": "HER2 status"}.items():
        out[dst] = _to_num(coh[src])
    # histologic_type / bilateral_bc: los faltantes son un bloque "no recolectado" de Duke (NC), sin
    # variable equivalente para reconstruirlos -> categoria explicita NC en vez de imputar con la moda.
    bil_raw = coh["Bilateral breast cancer"]
    out["bilateral_bc"] = _to_num(bil_raw).where(bil_raw.astype(str).str.strip() != "NC", NC_CODE)
    out["histologic_type"] = _to_num(coh["Histologic type"]).fillna(NC_CODE)

    # Tumor side: usar 'Tumor Location' y reconstruir desde 'Position' donde falte (R1 de la auditoria)
    tl = coh["Tumor Location"].astype(str).str.strip().str.upper()
    tl = tl.where(tl.isin(["L", "R"]), np.nan)
    pos_side = (coh["Position"].astype(str).str.strip().str.upper()
                .str.extract(r"^\s*([LR])")[0])
    side = tl.fillna(pos_side)
    out["tumor_side_R"] = side.map({"L": 0.0, "R": 1.0})

    # Staging ordinales; Nx / Mx (-1) -> missing
    out["stage_T"] = _to_num(coh["Staging(Tumor Size)# [T]"])
    n = _to_num(coh["Staging(Nodes)#(Nx replaced by -1)[N]"]);  out["stage_N"] = n.where(n >= 0, np.nan)
    # stage_M: Mx (-1) se recupera con 'Metastatic at Presentation' (misma fuente clinica que el TNM,
    # pretratamiento; concordancia 718/719 con M observado en Duke completo): 0 -> M0, 1 -> M1.
    m = _to_num(coh["Staging(Metastasis)#(Mx -replaced by -1)[M]"])
    out["stage_M"] = m.where(m >= 0, _to_num(coh[MET_PRES_COL]))
    for dst, src in {"grade_tubule": "Tumor Grade (Tubule)",
                     "grade_nuclear": "Tumor Grade (Nuclear)",
                     "grade_mitotic": "Tumor Grade (Mitotic)"}.items():
        out[dst] = _to_num(coh[src])

    # ---- M ----
    for dst, src in M_BINARY.items():
        out[dst] = _to_num(coh[src])

    # ---- A ----
    for dst, src in {**A_CONTINUOUS, **A_ORDINAL, **A_BINARY, **A_NOMINAL}.items():
        out[dst] = _to_num(coh[src])

    # ---- R ----
    rad_dedup, dropped = dedup_radiomics(rad)
    rad_cols = [c for c in rad_dedup.columns if c != ID_COL]
    merged = out.merge(rad_dedup, on=ID_COL, how="inner")

    assert len(merged) == expected_n, f"cohorte reconstruida = {len(merged)}, se esperaba {expected_n}"
    assert not merged[["stage_M", "histologic_type", "bilateral_bc"]].isna().any().any(), \
        "faltantes inesperados en stage_M / histologic_type / bilateral_bc"
    if target_col == TARGET_COL and valid_values == (0, 1, 2) and positive_values == (1, 2):
        assert int(y.sum()) == 93 and int((y == 0).sum()) == 188

    # Mol Subtype solo para referencia/subgrupos (NO predictor)
    meta = {
        "n": len(merged), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
        "radiomics_raw": len([c for c in rad.columns if c != ID_COL]),
        "radiomics_dupes_removed": len(dropped),
        "radiomics_after_dedup": len(rad_cols),
        "dupes_removed_names": dropped,
        "mol_subtype": _to_num(coh["Mol Subtype"]).tolist(),
    }
    return merged, y, rad_cols, meta


# --------------------------------------------------------------------------------------
# Transformers y armado de pipeline
# --------------------------------------------------------------------------------------
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Elimina varianza cero y una de cada par con |Pearson| > threshold.
    Se ajusta SOLO con el train que recibe en fit (dentro del pipeline/fold)."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        p = X.shape[1]
        std = X.std(axis=0)
        drop = set(np.where(std == 0)[0].tolist())          # varianza cero
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.abs(np.corrcoef(X, rowvar=False))
        corr = np.nan_to_num(corr)
        for j in range(p):
            if j in drop:
                continue
            for i in range(j):
                if i in drop:
                    continue
                if corr[i, j] > self.threshold:
                    drop.add(j)
                    break
        self.keep_ = np.array([k for k in range(p) if k not in drop], dtype=int)
        self.n_in_, self.n_out_ = p, len(self.keep_)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


def make_preprocessor(blocks: list[str], rad_cols: list[str]) -> ColumnTransformer:
    cont, binc, ordc, nomc = [], [], [], []
    for b in ("C", "M", "A"):
        if b in blocks:
            t = BLOCK_TYPES[b]
            cont += t.get("continuous", [])
            binc += t.get("binary", [])
            ordc += t.get("ordinal", [])
            nomc += t.get("nominal", [])

    transformers = []
    if cont:
        transformers.append(("cont", Pipeline([("imp", SimpleImputer(strategy="median")),
                                               ("sc", StandardScaler())]), cont))
    if binc:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), binc))
    if ordc:
        transformers.append(("ord", Pipeline([("imp", SimpleImputer(strategy="median")),
                                              ("sc", StandardScaler())]), ordc))
    if nomc:
        transformers.append(("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                              ("ohe", OneHotEncoder(handle_unknown="ignore",
                                                                    sparse_output=False))]), nomc))
    if "R" in blocks:
        transformers.append(("R", Pipeline([("imp", SimpleImputer(strategy="median")),
                                            ("corr", CorrelationFilter(CORR_THRESHOLD)),
                                            ("sc", StandardScaler()),
                                            ("pca", PCA(random_state=RANDOM_STATE))]), rad_cols))
    return ColumnTransformer(transformers, remainder="drop")


def make_estimator(blocks: list[str], rad_cols: list[str]) -> Pipeline:
    return Pipeline([
        ("preprocessor", make_preprocessor(blocks, rad_cols)),
        ("clf", LogisticRegression(penalty="l2", class_weight="balanced",
                                   max_iter=5000, random_state=RANDOM_STATE)),
    ])


def make_grid(blocks: list[str]) -> dict:
    grid = {"clf__C": C_GRID}
    if "R" in blocks:
        grid["preprocessor__R__pca__n_components"] = NCOMP_GRID
    return grid


def n_features_out(fitted_pipe: Pipeline, X_train: pd.DataFrame) -> int:
    return int(fitted_pipe.named_steps["preprocessor"].transform(X_train).shape[1])


# --------------------------------------------------------------------------------------
# Analisis A - split historico 80/20
# --------------------------------------------------------------------------------------
def run_split_analysis(merged: pd.DataFrame, y: pd.Series, rad_cols: list[str]) -> pd.DataFrame:
    idx = np.arange(len(merged))
    tr, te = train_test_split(idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    Xtr, Xte = merged.iloc[tr], merged.iloc[te]
    ytr, yte = y.iloc[tr], y.iloc[te]
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for name, blocks in MODELS.items():
        gs = GridSearchCV(make_estimator(blocks, rad_cols), make_grid(blocks),
                          scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)
        gs.fit(Xtr, ytr)
        proba = gs.predict_proba(Xte)[:, 1]
        best = gs.best_estimator_
        rows.append({
            "analysis": "split_8020",
            "model": name,
            "roc_auc": roc_auc_score(yte, proba),
            "pr_auc": average_precision_score(yte, proba),
            "n_train": len(tr),
            "n_test": len(te),
            "n_features_final": n_features_out(best, Xtr),
            "best_C": gs.best_params_["clf__C"],
            "best_n_components": gs.best_params_.get("preprocessor__R__pca__n_components", None),
            "cv_roc_auc_inner": gs.best_score_,
        })
        print(f"  [A] {name:9s} ROC={rows[-1]['roc_auc']:.3f}  PR={rows[-1]['pr_auc']:.3f}  "
              f"C={rows[-1]['best_C']}  nPC={rows[-1]['best_n_components']}  p={rows[-1]['n_features_final']}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Analisis B - Nested Stratified CV 5x5 (analisis principal)
# --------------------------------------------------------------------------------------
def run_nested_cv(merged: pd.DataFrame, y: pd.Series, rad_cols: list[str]):
    outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    yv = y.values

    oof = pd.DataFrame({"patient_id": merged[ID_COL].values, "y_true": yv})
    summary = {}
    for name, blocks in MODELS.items():
        oof_p = np.full(len(merged), np.nan)
        fold_roc, fold_pr, chosen = [], [], []
        for k, (tri, tei) in enumerate(outer.split(merged, yv)):
            gs = GridSearchCV(make_estimator(blocks, rad_cols), make_grid(blocks),
                              scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)
            gs.fit(merged.iloc[tri], y.iloc[tri])
            p = gs.predict_proba(merged.iloc[tei])[:, 1]
            oof_p[tei] = p
            fold_roc.append(roc_auc_score(yv[tei], p))
            fold_pr.append(average_precision_score(yv[tei], p))
            chosen.append({"fold": k,
                           "C": gs.best_params_["clf__C"],
                           "n_components": gs.best_params_.get("preprocessor__R__pca__n_components", None)})
        oof[name] = oof_p
        summary[name] = {
            "roc_auc_oof": roc_auc_score(yv, oof_p),
            "pr_auc_oof": average_precision_score(yv, oof_p),
            "roc_outer_mean": float(np.mean(fold_roc)),
            "roc_outer_sd": float(np.std(fold_roc)),
            "pr_outer_mean": float(np.mean(fold_pr)),
            "pr_outer_sd": float(np.std(fold_pr)),
            "fold_roc": [round(x, 4) for x in fold_roc],
            "fold_pr": [round(x, 4) for x in fold_pr],
            "chosen_params": chosen,
        }
        s = summary[name]
        print(f"  [B] {name:9s} OOF ROC={s['roc_auc_oof']:.3f}  OOF PR={s['pr_auc_oof']:.3f}  "
              f"outerROC={s['roc_outer_mean']:.3f}+-{s['roc_outer_sd']:.3f}")
    return oof, summary


# --------------------------------------------------------------------------------------
# Ensamblado de resultados
# --------------------------------------------------------------------------------------
def assemble_results(df_a: pd.DataFrame, summary_b: dict, prevalence: float) -> pd.DataFrame:
    roc_c = summary_b["C"]["roc_auc_oof"]
    pr_c = summary_b["C"]["pr_auc_oof"]
    rows = []
    for name in MODELS:
        s = summary_b[name]
        rows.append({
            "analysis": "nested_cv",
            "model": name,
            "roc_auc": s["roc_auc_oof"],
            "delta_roc_vs_C": (np.nan if name == "R" else s["roc_auc_oof"] - roc_c),
            "pr_auc": s["pr_auc_oof"],
            "delta_pr_vs_C": (np.nan if name == "R" else s["pr_auc_oof"] - pr_c),
            "roc_outer_mean": s["roc_outer_mean"],
            "roc_outer_sd": s["roc_outer_sd"],
            "pr_outer_mean": s["pr_outer_mean"],
            "pr_outer_sd": s["pr_outer_sd"],
            "n_train": None, "n_test": None, "n_features_final": None,
            "best_C": None, "best_n_components": None, "cv_roc_auc_inner": None,
        })
    df_b = pd.DataFrame(rows)
    allr = pd.concat([df_a, df_b], ignore_index=True)
    allr["prevalence"] = round(prevalence, 4)
    return allr


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Construyendo cohorte...")
    merged, y, rad_cols, meta = build_cohort()
    prevalence = meta["n_pos"] / meta["n"]
    print(f"  N={meta['n']}  pos={meta['n_pos']}  neg={meta['n_neg']}  prevalencia={prevalence:.3f}")
    print(f"  radiomics: {meta['radiomics_raw']} -> dedup -{meta['radiomics_dupes_removed']} "
          f"-> {meta['radiomics_after_dedup']}")

    print("\nAnalisis A - split historico 80/20 (comparabilidad)...")
    df_a = run_split_analysis(merged, y, rad_cols)

    print("\nAnalisis B - Nested CV 5x5 (principal)...")
    oof, summary_b = run_nested_cv(merged, y, rad_cols)

    results = assemble_results(df_a, summary_b, prevalence)
    results.to_csv(OUT_DIR / "results_stage2.csv", index=False)
    oof.to_csv(OUT_DIR / "oof_predictions.csv", index=False)
    with open(OUT_DIR / "nested_cv_details.json", "w") as fh:
        json.dump({"meta": {k: v for k, v in meta.items() if k not in ("dupes_removed_names", "mol_subtype")},
                   "summary": summary_b}, fh, indent=2)

    print(f"\nGuardado en {OUT_DIR}/:")
    print("  results_stage2.csv, oof_predictions.csv, nested_cv_details.json")


if __name__ == "__main__":
    main()
