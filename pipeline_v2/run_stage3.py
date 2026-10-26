"""
ETAPA 3 - Validacion de robustez.
TFG Radiomics NACT - Duke-Breast-Cancer-MRI.

Reutiliza EXACTAMENTE el pipeline V2 de Etapa 2 (run_stage2.py). No cambia
variables, target, PCA, preprocessing ni estrategia de radiomics.

Modelos evaluados: C, C+R, C+A+R  (Logistic Regression L2).

  A) Hold-out 80/20 (random_state=42): se REUTILIZAN los valores de Etapa 2
     (pipeline_v2/results_stage2.csv), no se recalcula ni se buscan seeds.

  B) Repeated Nested CV: outer StratifiedKFold(5) x inner StratifiedKFold(5),
     10 repeticiones con seeds predefinidos (42..51), nunca elegidos por performance.
     Toda transformacion (imputacion/encoding/scaling/filtro correlacion/PCA/tuning C)
     se ajusta dentro del training del fold. El outer test no participa de ninguna decision.

  Incertidumbre: bootstrap PAREADO a nivel paciente sobre las predicciones OOF
  agregadas (promedio entre repeticiones), 5000 resamples, seed fijo.

Salidas (en pipeline_v2/):
  - stage3_oof_long.csv          probas OOF por paciente y repeticion
  - stage3_oof_aggregated.csv    proba OOF promediada entre repeticiones, por paciente
  - stage3_per_repetition.csv    ROC/PR por repeticion y modelo
  - stage3_summary.csv           media/SD/rango/Delta por modelo
  - stage3_bootstrap.csv         Delta ROC/PR + IC95% (C vs C+R, C vs C+A+R)
  - stage3_stability.csv         C, n_components, n_radiomics tras filtro por fold
  - stage3_details.json          volcado completo
No modifica notebook ni bases originales.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from run_stage2 import (build_cohort, make_estimator, make_grid, ID_COL,
                        OUT_DIR)

MODELS_S3 = {
    "C":     ["C"],
    "C+R":   ["C", "R"],
    "C+A+R": ["C", "A", "R"],
}
REP_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]   # predefinidos, NO elegidos por performance
N_OUTER = 5
N_INNER = 5
N_BOOT = 5000
BOOT_SEED = 20260909


def _fit_predict_fold(blocks, rad_cols, Xtr, ytr, Xte, inner_seed):
    gs = GridSearchCV(make_estimator(blocks, rad_cols), make_grid(blocks),
                      scoring="roc_auc",
                      cv=StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=inner_seed),
                      n_jobs=-1, refit=True)
    gs.fit(Xtr, ytr)
    proba = gs.predict_proba(Xte)[:, 1]
    best = gs.best_estimator_
    n_rad = None
    if "R" in blocks:
        corr = best.named_steps["preprocessor"].named_transformers_["R"].named_steps["corr"]
        n_rad = int(corr.n_out_)
    return proba, {
        "C": gs.best_params_["clf__C"],
        "n_components": gs.best_params_.get("preprocessor__R__pca__n_components", None),
        "n_radiomics_after_corr": n_rad,
    }


def run_repeated_nested_cv(merged, y, rad_cols):
    yv = y.values
    n = len(merged)
    long_rows = []
    per_rep = []
    stability = []
    # oof_by_model_rep[model] -> array (n_rep, n) de probas OOF
    oof_stack = {m: np.full((len(REP_SEEDS), n), np.nan) for m in MODELS_S3}

    for ri, seed in enumerate(REP_SEEDS):
        outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
        oof_rep = {m: np.full(n, np.nan) for m in MODELS_S3}
        for fi, (tri, tei) in enumerate(outer.split(merged, yv)):
            Xtr, Xte = merged.iloc[tri], merged.iloc[tei]
            ytr = y.iloc[tri]
            for m, blocks in MODELS_S3.items():
                proba, chosen = _fit_predict_fold(blocks, rad_cols, Xtr, ytr, Xte, inner_seed=seed)
                oof_rep[m][tei] = proba
                stability.append({"model": m, "repetition": ri, "seed": seed, "outer_fold": fi,
                                  **chosen})
        for m in MODELS_S3:
            oof_stack[m][ri] = oof_rep[m]
            roc = roc_auc_score(yv, oof_rep[m])
            pr = average_precision_score(yv, oof_rep[m])
            per_rep.append({"model": m, "repetition": ri, "seed": seed,
                            "roc_auc": roc, "pr_auc": pr})
        for idx in range(n):
            row = {"repetition": ri, "seed": seed,
                   "patient_id": merged[ID_COL].iloc[idx], "y_true": int(yv[idx])}
            for m in MODELS_S3:
                row[m] = oof_rep[m][idx]
            long_rows.append(row)
        print(f"  rep {ri + 1}/{len(REP_SEEDS)} (seed {seed}) hecha")

    # ---- agregado: promedio de proba OOF entre repeticiones ----
    agg = pd.DataFrame({"patient_id": merged[ID_COL].values, "y_true": yv})
    for m in MODELS_S3:
        agg[m] = oof_stack[m].mean(axis=0)

    return (pd.DataFrame(long_rows), pd.DataFrame(per_rep),
            pd.DataFrame(stability), agg, oof_stack)


def summarize_per_rep(per_rep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    piv_roc = per_rep.pivot(index="repetition", columns="model", values="roc_auc")
    piv_pr = per_rep.pivot(index="repetition", columns="model", values="pr_auc")
    for m in MODELS_S3:
        r, p = piv_roc[m], piv_pr[m]
        dr = (r - piv_roc["C"]) if m != "C" else pd.Series(np.nan, index=r.index)
        dp = (p - piv_pr["C"]) if m != "C" else pd.Series(np.nan, index=p.index)
        rows.append({
            "model": m,
            "roc_auc_mean": r.mean(), "roc_auc_sd": r.std(ddof=1),
            "roc_auc_min": r.min(), "roc_auc_max": r.max(),
            "pr_auc_mean": p.mean(), "pr_auc_sd": p.std(ddof=1),
            "pr_auc_min": p.min(), "pr_auc_max": p.max(),
            "delta_roc_vs_C_mean": dr.mean(), "delta_roc_vs_C_sd": dr.std(ddof=1),
            "delta_pr_vs_C_mean": dp.mean(), "delta_pr_vs_C_sd": dp.std(ddof=1),
        })
    return pd.DataFrame(rows)


def aggregated_metrics(agg: pd.DataFrame) -> dict:
    y = agg["y_true"].values
    out = {}
    for m in MODELS_S3:
        out[m] = {"roc_auc_agg": roc_auc_score(y, agg[m].values),
                  "pr_auc_agg": average_precision_score(y, agg[m].values)}
    return out


def paired_bootstrap(agg: pd.DataFrame) -> pd.DataFrame:
    y = agg["y_true"].values
    n = len(y)
    rng = np.random.default_rng(BOOT_SEED)
    probs = {m: agg[m].values for m in MODELS_S3}
    # point estimates (muestra completa)
    roc0 = {m: roc_auc_score(y, probs[m]) for m in MODELS_S3}
    pr0 = {m: average_precision_score(y, probs[m]) for m in MODELS_S3}

    draws = {("roc", m): [] for m in ("C+R", "C+A+R")}
    draws.update({("pr", m): [] for m in ("C+R", "C+A+R")})
    b = 0
    while b < N_BOOT:
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        rc = roc_auc_score(yb, probs["C"][idx]); pc = average_precision_score(yb, probs["C"][idx])
        for m in ("C+R", "C+A+R"):
            draws[("roc", m)].append(roc_auc_score(yb, probs[m][idx]) - rc)
            draws[("pr", m)].append(average_precision_score(yb, probs[m][idx]) - pc)
        b += 1

    rows = []
    for m in ("C+R", "C+A+R"):
        for metric, r0, p0 in (("roc", roc0, pr0),):
            pass
        d_roc = np.array(draws[("roc", m)]); d_pr = np.array(draws[("pr", m)])
        rows.append({
            "comparison": f"C vs {m}",
            "delta_roc_point": roc0[m] - roc0["C"],
            "delta_roc_ci_lo": np.percentile(d_roc, 2.5),
            "delta_roc_ci_hi": np.percentile(d_roc, 97.5),
            "delta_roc_includes_0": bool(np.percentile(d_roc, 2.5) <= 0 <= np.percentile(d_roc, 97.5)),
            "delta_pr_point": pr0[m] - pr0["C"],
            "delta_pr_ci_lo": np.percentile(d_pr, 2.5),
            "delta_pr_ci_hi": np.percentile(d_pr, 97.5),
            "delta_pr_includes_0": bool(np.percentile(d_pr, 2.5) <= 0 <= np.percentile(d_pr, 97.5)),
            "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
        })
    return pd.DataFrame(rows)


def summarize_stability(stab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MODELS_S3:
        s = stab[stab.model == m]
        rec = {"model": m, "n_folds": len(s),
               "C_values": json.dumps(s["C"].value_counts().sort_index().to_dict()),
               "C_median": float(np.median(s["C"]))}
        if "R" in MODELS_S3[m]:
            rec["n_components_values"] = json.dumps(
                s["n_components"].dropna().astype(int).value_counts().sort_index().to_dict())
            rec["n_components_median"] = float(np.median(s["n_components"].dropna()))
            rr = s["n_radiomics_after_corr"].dropna().astype(int)
            rec["n_radiomics_after_corr_median"] = float(np.median(rr))
            rec["n_radiomics_after_corr_min"] = int(rr.min())
            rec["n_radiomics_after_corr_max"] = int(rr.max())
        rows.append(rec)
    return pd.DataFrame(rows)


def load_holdout_8020() -> pd.DataFrame:
    r = pd.read_csv(OUT_DIR / "results_stage2.csv")
    r = r[(r.analysis == "split_8020") & (r.model.isin(MODELS_S3))][
        ["model", "roc_auc", "pr_auc", "n_features_final", "best_C", "best_n_components"]]
    return r.reset_index(drop=True)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Cohorte...")
    merged, y, rad_cols, meta = build_cohort()
    prevalence = meta["n_pos"] / meta["n"]
    print(f"  N={meta['n']}  prevalencia={prevalence:.3f}  R pool={meta['radiomics_after_dedup']}")

    print("Repeated Nested CV (10 x 5x5)...")
    long_df, per_rep, stab, agg, _ = run_repeated_nested_cv(merged, y, rad_cols)

    summ = summarize_per_rep(per_rep)
    agg_m = aggregated_metrics(agg)
    boot = paired_bootstrap(agg)
    stab_s = summarize_stability(stab)

    long_df.to_csv(OUT_DIR / "stage3_oof_long.csv", index=False)
    agg.to_csv(OUT_DIR / "stage3_oof_aggregated.csv", index=False)
    per_rep.to_csv(OUT_DIR / "stage3_per_repetition.csv", index=False)
    summ.to_csv(OUT_DIR / "stage3_summary.csv", index=False)
    boot.to_csv(OUT_DIR / "stage3_bootstrap.csv", index=False)
    stab.to_csv(OUT_DIR / "stage3_stability_folds.csv", index=False)
    stab_s.to_csv(OUT_DIR / "stage3_stability.csv", index=False)

    holdout = load_holdout_8020()
    holdout.to_csv(OUT_DIR / "stage3_holdout_8020.csv", index=False)

    with open(OUT_DIR / "stage3_details.json", "w") as fh:
        json.dump({
            "config": {"rep_seeds": REP_SEEDS, "n_outer": N_OUTER, "n_inner": N_INNER,
                       "n_boot": N_BOOT, "boot_seed": BOOT_SEED},
            "prevalence": prevalence,
            "holdout_8020": holdout.to_dict(orient="records"),
            "aggregated_oof_metrics": agg_m,
            "per_repetition_summary": summ.to_dict(orient="records"),
            "bootstrap": boot.to_dict(orient="records"),
            "stability": stab_s.to_dict(orient="records"),
        }, fh, indent=2)

    print("\n--- Repeated Nested CV (media +- SD, [min,max]) ---")
    for _, r in summ.iterrows():
        print(f"  {r['model']:6s} ROC {r.roc_auc_mean:.3f}+-{r.roc_auc_sd:.3f} "
              f"[{r.roc_auc_min:.3f},{r.roc_auc_max:.3f}]  "
              f"PR {r.pr_auc_mean:.3f}+-{r.pr_auc_sd:.3f}")
    print("\n--- Agregado OOF ---")
    for m in MODELS_S3:
        print(f"  {m:6s} ROC {agg_m[m]['roc_auc_agg']:.3f}  PR {agg_m[m]['pr_auc_agg']:.3f}")
    print("\n--- Bootstrap pareado (Delta + IC95%) ---")
    for _, r in boot.iterrows():
        print(f"  {r['comparison']:12s} dROC {r.delta_roc_point:+.3f} "
              f"[{r.delta_roc_ci_lo:+.3f},{r.delta_roc_ci_hi:+.3f}] inc0={r.delta_roc_includes_0} | "
              f"dPR {r.delta_pr_point:+.3f} [{r.delta_pr_ci_lo:+.3f},{r.delta_pr_ci_hi:+.3f}] "
              f"inc0={r.delta_pr_includes_0}")
    print("\n--- Estabilidad ---")
    for _, r in stab_s.iterrows():
        print(f"  {r['model']:6s} C={r['C_values']}  " +
              (f"nPC={r.get('n_components_values','-')}  nRad~{r.get('n_radiomics_after_corr_median','-')}"
               f" [{r.get('n_radiomics_after_corr_min','-')},{r.get('n_radiomics_after_corr_max','-')}]"
               if 'R' in MODELS_S3[r['model']] else ""))
    print(f"\nGuardado en {OUT_DIR}/")


if __name__ == "__main__":
    main()
