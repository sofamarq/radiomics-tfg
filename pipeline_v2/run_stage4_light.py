"""
ETAPA 4 - version LIVIANA.
Comparacion exploratoria RF y Gradient Boosting (Logistic se reutiliza de Etapa 3).

Motivo: la ejecucion 10x5x5 de run_stage4.py fue interrumpida por costo computacional.
Decision (por COSTO, no por performance):
  - Logistic Regression: queda respaldada por la Etapa 3 (Repeated Nested CV 10x5x5 + bootstrap).
    NO se recalcula. Se reutilizan sus resultados.
  - Random Forest y Gradient Boosting: se evaluan con un Nested CV UNICO 5x3 exploratorio,
    de a un algoritmo y secuencialmente (sin repeticiones, sin bootstrap).

Esquema para RF y GB:
  - outer StratifiedKFold(5, shuffle, random_state=42)
  - inner StratifiedKFold(3, shuffle, random_state=42)   [3 folds internos = version liviana]
  - una sola ejecucion; comparar C vs C+A+R
  - GridSearchCV n_jobs=2 (limite), outer folds recorridos con un for secuencial (sin paralelizar)
  - grillas IDENTICAS a la preespecificacion de Etapa 4 (run_stage4.ALGOS); no se modifican por performance
  - todo preprocessing / correlation filter / PCA / tuning dentro del train de cada fold
  - se guarda el CSV inmediatamente despues de terminar CADA algoritmo

Datos, variables, target, pipeline V2: CONGELADOS (run_stage2.build_cohort).
No modifica notebook, bases, ni run_stage2/3/4.py.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from run_stage2 import ID_COL, OUT_DIR, RANDOM_STATE, build_cohort
from run_stage4 import ALGOS, MODELS_S4, NCOMP_GRID, make_estimator4, make_grid4

OUTER_SPLITS = 5
INNER_SPLITS = 3
GS_N_JOBS = 2

# --- referencia Logistic (Etapas 2/3) y hold-out RF/GB: se LEEN de los CSV ya generados ---
# (antes estaban escritos a mano; se leen para que la tabla combinada quede sincronizada con
#  los resultados vigentes. No se recalcula nada de Logistic.)
def _load_refs():
    from sklearn.metrics import average_precision_score as _ap, roc_auc_score as _roc
    st2 = pd.read_csv(OUT_DIR / "results_stage2.csv")
    ho = st2[st2.analysis == "split_8020"].set_index("model")
    agg = pd.read_csv(OUT_DIR / "stage3_oof_aggregated.csv")
    summ = pd.read_csv(OUT_DIR / "stage3_summary.csv").set_index("model")
    bt = pd.read_csv(OUT_DIR / "stage3_bootstrap.csv").set_index("comparison").loc["C vs C+A+R"]
    logistic_ref = {
        "holdout": {"roc_C": ho.loc["C", "roc_auc"], "roc_C_A_R": ho.loc["C+A+R", "roc_auc"],
                    "pr_C": ho.loc["C", "pr_auc"], "pr_C_A_R": ho.loc["C+A+R", "pr_auc"]},
        "cv_oof": {"roc_C": _roc(agg.y_true, agg["C"]), "roc_C_A_R": _roc(agg.y_true, agg["C+A+R"]),
                   "pr_C": _ap(agg.y_true, agg["C"]), "pr_C_A_R": _ap(agg.y_true, agg["C+A+R"])},
        "cv_perrep": {"roc_C_mean": summ.loc["C", "roc_auc_mean"], "roc_C_sd": summ.loc["C", "roc_auc_sd"],
                      "roc_C_A_R_mean": summ.loc["C+A+R", "roc_auc_mean"], "roc_C_A_R_sd": summ.loc["C+A+R", "roc_auc_sd"],
                      "pr_C_mean": summ.loc["C", "pr_auc_mean"], "pr_C_sd": summ.loc["C", "pr_auc_sd"],
                      "pr_C_A_R_mean": summ.loc["C+A+R", "pr_auc_mean"], "pr_C_A_R_sd": summ.loc["C+A+R", "pr_auc_sd"]},
        "cv_boot": {"delta_roc": bt.delta_roc_point, "delta_roc_ci": f"[{bt.delta_roc_ci_lo:+.3f}, {bt.delta_roc_ci_hi:+.3f}]",
                    "delta_roc_incl0": bool(bt.delta_roc_includes_0),
                    "delta_pr": bt.delta_pr_point, "delta_pr_ci": f"[{bt.delta_pr_ci_lo:+.3f}, {bt.delta_pr_ci_hi:+.3f}]",
                    "delta_pr_incl0": bool(bt.delta_pr_includes_0)},
        "scheme": "Etapa 3: Repeated Nested CV 10x5x5 + bootstrap pareado (5000)",
    }
    hr = pd.read_csv(OUT_DIR / "stage4light_holdout_rfgb.csv").set_index("algorithm")   # stage4light_holdout.py
    holdout_rfgb = {a: {k: float(hr.loc[a, k]) for k in ("roc_C", "roc_C_A_R", "pr_C", "pr_C_A_R")}
                    for a in ("random_forest", "gradient_boosting")}
    return logistic_ref, holdout_rfgb


def _chosen(gs, blocks):
    p = gs.best_params_
    rec = {k.replace("clf__", ""): v for k, v in p.items() if k.startswith("clf__")}
    rec["n_components"] = p.get("preprocessor__R__pca__n_components", None)
    if "R" in blocks:
        rec["n_radiomics_after_corr"] = int(
            gs.best_estimator_.named_steps["preprocessor"]
            .named_transformers_["R"].named_steps["corr"].n_out_)
    return rec


def single_nested_cv(algo, merged, y, rad_cols):
    yv = y.values
    n = len(merged)
    outer = StratifiedKFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    res = {"algorithm": algo, "models": {}}
    oof = {m: np.full(n, np.nan) for m in MODELS_S4}

    for m, blocks in MODELS_S4.items():
        fold_roc, fold_pr, chosen = [], [], []
        for fi, (tri, tei) in enumerate(outer.split(merged, yv)):   # for secuencial, sin paralelizar
            gs = GridSearchCV(make_estimator4(algo, blocks, rad_cols),
                              make_grid4(algo, blocks),
                              scoring="roc_auc", cv=inner, n_jobs=GS_N_JOBS, refit=True)
            gs.fit(merged.iloc[tri], y.iloc[tri])
            p = gs.predict_proba(merged.iloc[tei])[:, 1]
            oof[m][tei] = p
            fold_roc.append(roc_auc_score(yv[tei], p))
            fold_pr.append(average_precision_score(yv[tei], p))
            chosen.append({"fold": fi, **_chosen(gs, blocks)})
            print(f"    {algo} {m} fold {fi + 1}/{OUTER_SPLITS}  ROC={fold_roc[-1]:.3f} PR={fold_pr[-1]:.3f}")
        res["models"][m] = {
            "fold_roc": [round(x, 4) for x in fold_roc],
            "fold_pr": [round(x, 4) for x in fold_pr],
            "roc_mean": float(np.mean(fold_roc)), "roc_sd": float(np.std(fold_roc, ddof=1)),
            "pr_mean": float(np.mean(fold_pr)), "pr_sd": float(np.std(fold_pr, ddof=1)),
            "roc_oof": float(roc_auc_score(yv, oof[m])),
            "pr_oof": float(average_precision_score(yv, oof[m])),
            "chosen_params": chosen,
        }
    a, c = res["models"]["C+A+R"], res["models"]["C"]
    dr = np.array(a["fold_roc"]) - np.array(c["fold_roc"])
    dp = np.array(a["fold_pr"]) - np.array(c["fold_pr"])
    res["delta"] = {
        "roc_fold": [round(x, 4) for x in dr], "pr_fold": [round(x, 4) for x in dp],
        "roc_mean": float(dr.mean()), "roc_sd": float(dr.std(ddof=1)),
        "pr_mean": float(dp.mean()), "pr_sd": float(dp.std(ddof=1)),
        "roc_oof": a["roc_oof"] - c["roc_oof"], "pr_oof": a["pr_oof"] - c["pr_oof"],
    }
    return res, oof


def save_algo(res, oof, merged, y):
    algo = res["algorithm"]
    rows = []
    for m in MODELS_S4:
        mm = res["models"][m]
        for fi in range(OUTER_SPLITS):
            rows.append({"algorithm": algo, "model": m, "outer_fold": fi,
                         "roc_auc": mm["fold_roc"][fi], "pr_auc": mm["fold_pr"][fi]})
    pd.DataFrame(rows).to_csv(OUT_DIR / f"stage4light_{algo}_folds.csv", index=False)

    summ = []
    for m in MODELS_S4:
        mm = res["models"][m]
        summ.append({"algorithm": algo, "model": m,
                     "roc_mean": mm["roc_mean"], "roc_sd": mm["roc_sd"],
                     "pr_mean": mm["pr_mean"], "pr_sd": mm["pr_sd"],
                     "roc_oof": mm["roc_oof"], "pr_oof": mm["pr_oof"]})
    summ.append({"algorithm": algo, "model": "delta(C+A+R - C)",
                 "roc_mean": res["delta"]["roc_mean"], "roc_sd": res["delta"]["roc_sd"],
                 "pr_mean": res["delta"]["pr_mean"], "pr_sd": res["delta"]["pr_sd"],
                 "roc_oof": res["delta"]["roc_oof"], "pr_oof": res["delta"]["pr_oof"]})
    pd.DataFrame(summ).to_csv(OUT_DIR / f"stage4light_{algo}_summary.csv", index=False)

    od = pd.DataFrame({"patient_id": merged[ID_COL].values, "y_true": y.values})
    for m in MODELS_S4:
        od[m] = oof[m]
    od.to_csv(OUT_DIR / f"stage4light_{algo}_oof.csv", index=False)

    with open(OUT_DIR / f"stage4light_{algo}_details.json", "w") as fh:
        json.dump(res, fh, indent=2, default=str)
    print(f"  >> guardado stage4light_{algo}_*.csv / .json")


def build_combined(rf_res, gb_res):
    def cv_row(algo, holdout, cv):
        return {
            "algorithm": algo,
            "holdout_roc_C": holdout["roc_C"], "holdout_roc_C_A_R": holdout["roc_C_A_R"],
            "holdout_pr_C": holdout["pr_C"], "holdout_pr_C_A_R": holdout["pr_C_A_R"],
            "holdout_delta_roc": (None if holdout["roc_C"] is None else holdout["roc_C_A_R"] - holdout["roc_C"]),
            "holdout_delta_pr": (None if holdout["pr_C"] is None else holdout["pr_C_A_R"] - holdout["pr_C"]),
            "cv_roc_C": cv["roc_C"], "cv_roc_C_A_R": cv["roc_C_A_R"],
            "cv_pr_C": cv["pr_C"], "cv_pr_C_A_R": cv["pr_C_A_R"],
            "cv_delta_roc": cv["roc_C_A_R"] - cv["roc_C"],
            "cv_delta_pr": cv["pr_C_A_R"] - cv["pr_C"],
        }
    rows = [cv_row("logistic", LOGISTIC_REF["holdout"], LOGISTIC_REF["cv_oof"])]
    for algo, r in (("random_forest", rf_res), ("gradient_boosting", gb_res)):
        cv = {"roc_C": r["models"]["C"]["roc_oof"], "roc_C_A_R": r["models"]["C+A+R"]["roc_oof"],
              "pr_C": r["models"]["C"]["pr_oof"], "pr_C_A_R": r["models"]["C+A+R"]["pr_oof"]}
        rows.append(cv_row(algo, HOLDOUT_RFGB[algo], cv))
    comb = pd.DataFrame(rows)
    comb.to_csv(OUT_DIR / "stage4light_combined.csv", index=False)
    return comb


def main():
    global LOGISTIC_REF, HOLDOUT_RFGB
    LOGISTIC_REF, HOLDOUT_RFGB = _load_refs()
    OUT_DIR.mkdir(exist_ok=True)
    print("Cohorte congelada (identica a Etapas 2-3)...")
    merged, y, rad_cols, meta = build_cohort()
    print(f"  N={meta['n']}  prevalencia={meta['n_pos']/meta['n']:.3f}  R pool={meta['radiomics_after_dedup']}")
    print(f"  Esquema RF/GB: Nested CV UNICO outer={OUTER_SPLITS} x inner={INNER_SPLITS}, seed={RANDOM_STATE}, "
          f"GS n_jobs={GS_N_JOBS}, outer secuencial. Logistic: reutilizada de Etapa 3.")

    print("\n[1/2] Random Forest ...")
    rf_res, rf_oof = single_nested_cv("random_forest", merged, y, rad_cols)
    save_algo(rf_res, rf_oof, merged, y)

    print("\n[2/2] Gradient Boosting ...")
    gb_res, gb_oof = single_nested_cv("gradient_boosting", merged, y, rad_cols)
    save_algo(gb_res, gb_oof, merged, y)

    comb = build_combined(rf_res, gb_res)

    oof_all = pd.DataFrame({"patient_id": merged[ID_COL].values, "y_true": y.values})
    for m in MODELS_S4:
        oof_all[f"random_forest__{m}"] = rf_oof[m]
        oof_all[f"gradient_boosting__{m}"] = gb_oof[m]
    oof_all.to_csv(OUT_DIR / "stage4light_oof.csv", index=False)

    with open(OUT_DIR / "stage4light_details.json", "w") as fh:
        json.dump({
            "config": {"outer": OUTER_SPLITS, "inner": INNER_SPLITS, "seed": RANDOM_STATE,
                       "gs_n_jobs": GS_N_JOBS, "repetitions": 1, "bootstrap": False,
                       "grids": {k: {kk: [str(x) for x in vv] for kk, vv in v["grid"].items()}
                                 for k, v in ALGOS.items() if k in ("random_forest", "gradient_boosting")},
                       "ncomp_grid": NCOMP_GRID},
            "logistic_ref": LOGISTIC_REF,
            "holdout_rfgb": HOLDOUT_RFGB,
            "random_forest": rf_res,
            "gradient_boosting": gb_res,
            "combined": comb.to_dict(orient="records"),
        }, fh, indent=2, default=str)

    print("\n===== RESUMEN (CV: Logistic=Etapa3 OOF 10x5x5 ; RF/GB=OOF 5x3 unico) =====")
    for _, r in comb.iterrows():
        hd = "" if r.holdout_delta_roc is None else f"{r.holdout_delta_roc:+.3f}"
        print(f"{r.algorithm:17s} | HO ROC C {r.holdout_roc_C} -> C+A+R {r.holdout_roc_C_A_R} (d {hd}) | "
              f"CV ROC C {r.cv_roc_C:.3f} -> C+A+R {r.cv_roc_C_A_R:.3f} (dROC {r.cv_delta_roc:+.3f}) "
              f"| CV dPR {r.cv_delta_pr:+.3f}")
    print(f"\nGuardado en {OUT_DIR}/  (stage4light_*.csv / .json)")


if __name__ == "__main__":
    main()
