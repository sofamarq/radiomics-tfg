"""
ETAPA 4 - Comparacion preespecificada de algoritmos.
TFG Radiomics NACT - Duke-Breast-Cancer-MRI.

Objetivo: ver si el aporte incremental de A+R al modelo clinico depende del algoritmo.
Algoritmos FIJADOS antes de ver resultados: Logistic Regression L2, Random Forest,
Gradient Boosting (sklearn). Conjuntos de predictores: C  vs  C+A+R.

Datos, variables, target, reglas NC/NP, reconstrucciones, exclusiones, de-dup de R:
CONGELADOS e identicos a Etapas 2-3 (se reutiliza run_stage2.build_cohort).

Esquema A: hold-out 80/20 (random_state=42, MISMOS pacientes que Etapas 2-3).
Esquema B: Repeated Nested CV outer5 x inner5, 10 repeticiones, seeds 42-51
           (LAS MISMAS de Etapa 3 -> comparaciones pareadas sobre las mismas particiones).
Incertidumbre: bootstrap pareado a nivel paciente sobre OOF agregado, 5000 resamples,
               seed 20260909 (el mismo de Etapa 3).

Grillas preespecificadas.
No se amplian modelos ni grillas segun performance.

Salidas en pipeline_v2/: stage4_*.csv, stage4_details.json
No modifica notebook, bases originales, ni run_stage2.py / run_stage3.py.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from run_stage2 import (BLOCK_TYPES, CorrelationFilter, ID_COL, NCOMP_GRID, OUT_DIR,
                        RANDOM_STATE, build_cohort)

MODELS_S4 = {"C": ["C"], "C+A+R": ["C", "A", "R"]}
REP_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]      # IDENTICAS a Etapa 3
N_OUTER = 5
N_INNER = 5
N_BOOT = 5000
BOOT_SEED = 20260909
# NCOMP_GRID se importa de run_stage2 (misma grilla de PCA para los tres algoritmos)

ALGOS = {
    "logistic": dict(
        make=lambda: LogisticRegression(penalty="l2", class_weight="balanced",
                                        max_iter=5000, random_state=RANDOM_STATE),
        grid={"clf__C": [0.01, 0.1, 1.0, 10.0]},
        scale_dense=True),
    "random_forest": dict(
        make=lambda: RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=1),
        grid={"clf__max_depth": [None, 3, 5],
              "clf__min_samples_leaf": [2, 5, 10],
              "clf__max_features": ["sqrt", 0.5],
              "clf__class_weight": [None, "balanced"]},
        scale_dense=False),
    "gradient_boosting": dict(
        make=lambda: GradientBoostingClassifier(random_state=RANDOM_STATE),
        grid={"clf__n_estimators": [100, 200],
              "clf__learning_rate": [0.05, 0.1],
              "clf__max_depth": [2, 3],
              "clf__min_samples_leaf": [10]},
        scale_dense=False),
}


def make_preprocessor4(blocks, rad_cols, scale_dense):
    cont, binc, ordc, nomc = [], [], [], []
    for b in ("C", "A"):
        if b in blocks:
            t = BLOCK_TYPES[b]
            cont += t.get("continuous", [])
            binc += t.get("binary", [])
            ordc += t.get("ordinal", [])
            nomc += t.get("nominal", [])
    tr = []
    if cont:
        steps = [("imp", SimpleImputer(strategy="median"))]
        if scale_dense:
            steps.append(("sc", StandardScaler()))
        tr.append(("cont", Pipeline(steps), cont))
    if binc:
        tr.append(("bin", SimpleImputer(strategy="most_frequent"), binc))
    if ordc:
        steps = [("imp", SimpleImputer(strategy="median"))]
        if scale_dense:
            steps.append(("sc", StandardScaler()))
        tr.append(("ord", Pipeline(steps), ordc))
    if nomc:
        tr.append(("nom", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), nomc))
    if "R" in blocks:
        tr.append(("R", Pipeline([("imp", SimpleImputer(strategy="median")),
                                  ("corr", CorrelationFilter(0.95)),
                                  ("sc", StandardScaler()),
                                  ("pca", PCA(random_state=RANDOM_STATE))]), rad_cols))
    return ColumnTransformer(tr, remainder="drop")


def make_estimator4(algo, blocks, rad_cols):
    spec = ALGOS[algo]
    return Pipeline([("preprocessor", make_preprocessor4(blocks, rad_cols, spec["scale_dense"])),
                     ("clf", spec["make"]())])


def make_grid4(algo, blocks):
    grid = dict(ALGOS[algo]["grid"])
    if "R" in blocks:
        grid["preprocessor__R__pca__n_components"] = NCOMP_GRID
    return grid


def _gs(algo, blocks, rad_cols, Xtr, ytr, inner_seed):
    gs = GridSearchCV(make_estimator4(algo, blocks, rad_cols), make_grid4(algo, blocks),
                      scoring="roc_auc",
                      cv=StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=inner_seed),
                      n_jobs=-1, refit=True)
    gs.fit(Xtr, ytr)
    return gs


def _chosen(gs, blocks):
    p = gs.best_params_
    rec = {k.replace("clf__", ""): v for k, v in p.items() if k.startswith("clf__")}
    rec["n_components"] = p.get("preprocessor__R__pca__n_components", None)
    n_rad = None
    if "R" in blocks:
        n_rad = int(gs.best_estimator_.named_steps["preprocessor"]
                    .named_transformers_["R"].named_steps["corr"].n_out_)
    rec["n_radiomics_after_corr"] = n_rad
    return rec


# ---------------------------------------------------------------- Esquema A
def run_holdout(merged, y, rad_cols):
    idx = np.arange(len(merged))
    tr, te = train_test_split(idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    Xtr, Xte, ytr, yte = merged.iloc[tr], merged.iloc[te], y.iloc[tr], y.iloc[te]
    rows = []
    for algo in ALGOS:
        res = {}
        for m, blocks in MODELS_S4.items():
            gs = _gs(algo, blocks, rad_cols, Xtr, ytr, inner_seed=RANDOM_STATE)
            proba = gs.predict_proba(Xte)[:, 1]
            res[m] = (roc_auc_score(yte, proba), average_precision_score(yte, proba), _chosen(gs, blocks))
        rocC, prC, _ = res["C"]
        rocAR, prAR, chAR = res["C+A+R"]
        rows.append({"algorithm": algo,
                     "roc_C": rocC, "pr_C": prC,
                     "roc_C_A_R": rocAR, "pr_C_A_R": prAR,
                     "delta_roc": rocAR - rocC, "delta_pr": prAR - prC,
                     "chosen_C": json.dumps(res["C"][2], default=str),
                     "chosen_C_A_R": json.dumps(chAR, default=str),
                     "n_train": len(tr), "n_test": len(te)})
        print(f"  [A] {algo:17s} C ROC={rocC:.3f} | C+A+R ROC={rocAR:.3f}  dROC={rocAR-rocC:+.3f} dPR={prAR-prC:+.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Esquema B
def run_repeated_nested(merged, y, rad_cols):
    yv = y.values
    n = len(merged)
    long_rows, per_rep, stability = [], [], []
    # oof_stack[algo][model] -> (n_rep, n)
    oof_stack = {a: {m: np.full((len(REP_SEEDS), n), np.nan) for m in MODELS_S4} for a in ALGOS}

    for ri, seed in enumerate(REP_SEEDS):
        outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
        oof_rep = {a: {m: np.full(n, np.nan) for m in MODELS_S4} for a in ALGOS}
        for fi, (tri, tei) in enumerate(outer.split(merged, yv)):
            Xtr, Xte, ytr = merged.iloc[tri], merged.iloc[tei], y.iloc[tri]
            for algo in ALGOS:
                for m, blocks in MODELS_S4.items():
                    gs = _gs(algo, blocks, rad_cols, Xtr, ytr, inner_seed=seed)
                    oof_rep[algo][m][tei] = gs.predict_proba(Xte)[:, 1]
                    stability.append({"algorithm": algo, "model": m, "repetition": ri,
                                      "seed": seed, "outer_fold": fi, **_chosen(gs, blocks)})
        for algo in ALGOS:
            for m in MODELS_S4:
                oof_stack[algo][m][ri] = oof_rep[algo][m]
                per_rep.append({"algorithm": algo, "model": m, "repetition": ri, "seed": seed,
                                "roc_auc": roc_auc_score(yv, oof_rep[algo][m]),
                                "pr_auc": average_precision_score(yv, oof_rep[algo][m])})
        for idx in range(n):
            row = {"repetition": ri, "seed": seed,
                   "patient_id": merged[ID_COL].iloc[idx], "y_true": int(yv[idx])}
            for algo in ALGOS:
                for m in MODELS_S4:
                    row[f"{algo}__{m}"] = oof_rep[algo][m][idx]
            long_rows.append(row)
        print(f"  [B] repeticion {ri + 1}/{len(REP_SEEDS)} (seed {seed}) lista")

    agg = pd.DataFrame({"patient_id": merged[ID_COL].values, "y_true": yv})
    for algo in ALGOS:
        for m in MODELS_S4:
            agg[f"{algo}__{m}"] = oof_stack[algo][m].mean(axis=0)
    return pd.DataFrame(long_rows), pd.DataFrame(per_rep), pd.DataFrame(stability), agg


def summarize_per_rep(per_rep):
    rows = []
    for algo in ALGOS:
        sub = per_rep[per_rep.algorithm == algo]
        pr_roc = sub.pivot(index="repetition", columns="model", values="roc_auc")
        pr_pr = sub.pivot(index="repetition", columns="model", values="pr_auc")
        d_roc = pr_roc["C+A+R"] - pr_roc["C"]
        d_pr = pr_pr["C+A+R"] - pr_pr["C"]
        for m in MODELS_S4:
            rows.append({"algorithm": algo, "model": m,
                         "roc_mean": pr_roc[m].mean(), "roc_sd": pr_roc[m].std(ddof=1),
                         "roc_min": pr_roc[m].min(), "roc_max": pr_roc[m].max(),
                         "pr_mean": pr_pr[m].mean(), "pr_sd": pr_pr[m].std(ddof=1),
                         "pr_min": pr_pr[m].min(), "pr_max": pr_pr[m].max()})
        rows.append({"algorithm": algo, "model": "delta(C+A+R - C)",
                     "roc_mean": d_roc.mean(), "roc_sd": d_roc.std(ddof=1),
                     "roc_min": d_roc.min(), "roc_max": d_roc.max(),
                     "pr_mean": d_pr.mean(), "pr_sd": d_pr.std(ddof=1),
                     "pr_min": d_pr.min(), "pr_max": d_pr.max()})
    return pd.DataFrame(rows)


def aggregated_metrics(agg):
    y = agg["y_true"].values
    out = {}
    for algo in ALGOS:
        rc = roc_auc_score(y, agg[f"{algo}__C"].values)
        pc = average_precision_score(y, agg[f"{algo}__C"].values)
        ra = roc_auc_score(y, agg[f"{algo}__C+A+R"].values)
        pa = average_precision_score(y, agg[f"{algo}__C+A+R"].values)
        out[algo] = {"roc_C": rc, "pr_C": pc, "roc_C_A_R": ra, "pr_C_A_R": pa,
                     "delta_roc": ra - rc, "delta_pr": pa - pc}
    return out


def paired_bootstrap(agg):
    y = agg["y_true"].values
    n = len(y)
    rng = np.random.default_rng(BOOT_SEED)
    P = {a: {"C": agg[f"{a}__C"].values, "AR": agg[f"{a}__C+A+R"].values} for a in ALGOS}
    d_roc = {a: [] for a in ALGOS}
    d_pr = {a: [] for a in ALGOS}
    b = 0
    while b < N_BOOT:
        i = rng.integers(0, n, n)
        yb = y[i]
        if yb.min() == yb.max():
            continue
        for a in ALGOS:
            d_roc[a].append(roc_auc_score(yb, P[a]["AR"][i]) - roc_auc_score(yb, P[a]["C"][i]))
            d_pr[a].append(average_precision_score(yb, P[a]["AR"][i]) - average_precision_score(yb, P[a]["C"][i]))
        b += 1
    rows = []
    aggm = aggregated_metrics(agg)
    for a in ALGOS:
        dr, dp = np.array(d_roc[a]), np.array(d_pr[a])
        rows.append({"algorithm": a,
                     "delta_roc_point": aggm[a]["delta_roc"],
                     "delta_roc_ci_lo": np.percentile(dr, 2.5), "delta_roc_ci_hi": np.percentile(dr, 97.5),
                     "delta_roc_includes_0": bool(np.percentile(dr, 2.5) <= 0 <= np.percentile(dr, 97.5)),
                     "delta_pr_point": aggm[a]["delta_pr"],
                     "delta_pr_ci_lo": np.percentile(dp, 2.5), "delta_pr_ci_hi": np.percentile(dp, 97.5),
                     "delta_pr_includes_0": bool(np.percentile(dp, 2.5) <= 0 <= np.percentile(dp, 97.5)),
                     "n_boot": N_BOOT, "boot_seed": BOOT_SEED})
    return pd.DataFrame(rows)


def summarize_stability(stab):
    rows = []
    for algo in ALGOS:
        for m in MODELS_S4:
            s = stab[(stab.algorithm == algo) & (stab.model == m)]
            rec = {"algorithm": algo, "model": m, "n_folds": len(s)}
            for col in ("C", "max_depth", "min_samples_leaf", "max_features", "class_weight",
                        "n_estimators", "learning_rate", "n_components"):
                if col in s.columns and s[col].notna().any():
                    vc = s[col].astype(str).value_counts().sort_index().to_dict()
                    rec[col] = json.dumps(vc)
            if "R" in MODELS_S4[m] and s["n_radiomics_after_corr"].notna().any():
                rr = s["n_radiomics_after_corr"].dropna().astype(int)
                rec["n_radiomics_after_corr"] = f"{int(rr.median())} [{int(rr.min())}-{int(rr.max())}]"
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("Cohorte (congelada, identica a Etapas 2-3)...")
    merged, y, rad_cols, meta = build_cohort()
    prev = meta["n_pos"] / meta["n"]
    print(f"  N={meta['n']}  prevalencia={prev:.3f}  R pool={meta['radiomics_after_dedup']}")

    print("Esquema A - hold-out 80/20 ...")
    hold = run_holdout(merged, y, rad_cols)

    print("Esquema B - Repeated Nested CV (3 algoritmos x 2 modelos x 10 x 5x5) ...")
    long_df, per_rep, stab, agg = run_repeated_nested(merged, y, rad_cols)

    summ = summarize_per_rep(per_rep)
    aggm = aggregated_metrics(agg)
    boot = paired_bootstrap(agg)
    stab_s = summarize_stability(stab)

    hold.to_csv(OUT_DIR / "stage4_holdout.csv", index=False)
    per_rep.to_csv(OUT_DIR / "stage4_per_repetition.csv", index=False)
    summ.to_csv(OUT_DIR / "stage4_summary.csv", index=False)
    boot.to_csv(OUT_DIR / "stage4_bootstrap.csv", index=False)
    stab.to_csv(OUT_DIR / "stage4_stability_folds.csv", index=False)
    stab_s.to_csv(OUT_DIR / "stage4_stability.csv", index=False)
    long_df.to_csv(OUT_DIR / "stage4_oof_long.csv", index=False)
    agg.to_csv(OUT_DIR / "stage4_oof_aggregated.csv", index=False)

    combined = []
    for algo in ALGOS:
        h = hold[hold.algorithm == algo].iloc[0]
        sc = summ[(summ.algorithm == algo) & (summ.model == "C")].iloc[0]
        sa = summ[(summ.algorithm == algo) & (summ.model == "C+A+R")].iloc[0]
        sd = summ[(summ.algorithm == algo) & (summ.model == "delta(C+A+R - C)")].iloc[0]
        bt = boot[boot.algorithm == algo].iloc[0]
        combined.append({
            "algorithm": algo,
            "holdout_roc_C": h.roc_C, "holdout_roc_C_A_R": h.roc_C_A_R,
            "holdout_pr_C": h.pr_C, "holdout_pr_C_A_R": h.pr_C_A_R,
            "holdout_delta_roc": h.delta_roc, "holdout_delta_pr": h.delta_pr,
            "ncv_roc_C_mean": sc.roc_mean, "ncv_roc_C_sd": sc.roc_sd,
            "ncv_roc_C_A_R_mean": sa.roc_mean, "ncv_roc_C_A_R_sd": sa.roc_sd,
            "ncv_pr_C_mean": sc.pr_mean, "ncv_pr_C_A_R_mean": sa.pr_mean,
            "ncv_delta_roc_mean": sd.roc_mean, "ncv_delta_roc_sd": sd.roc_sd,
            "ncv_delta_pr_mean": sd.pr_mean, "ncv_delta_pr_sd": sd.pr_sd,
            "oof_roc_C": aggm[algo]["roc_C"], "oof_roc_C_A_R": aggm[algo]["roc_C_A_R"],
            "oof_pr_C": aggm[algo]["pr_C"], "oof_pr_C_A_R": aggm[algo]["pr_C_A_R"],
            "oof_delta_roc": aggm[algo]["delta_roc"], "oof_delta_pr": aggm[algo]["delta_pr"],
            "boot_delta_roc": bt.delta_roc_point,
            "boot_delta_roc_ci": f"[{bt.delta_roc_ci_lo:+.3f}, {bt.delta_roc_ci_hi:+.3f}]",
            "boot_delta_roc_incl0": bt.delta_roc_includes_0,
            "boot_delta_pr": bt.delta_pr_point,
            "boot_delta_pr_ci": f"[{bt.delta_pr_ci_lo:+.3f}, {bt.delta_pr_ci_hi:+.3f}]",
            "boot_delta_pr_incl0": bt.delta_pr_includes_0,
        })
    comb = pd.DataFrame(combined)
    comb.to_csv(OUT_DIR / "stage4_combined.csv", index=False)

    with open(OUT_DIR / "stage4_details.json", "w") as fh:
        json.dump({
            "config": {"rep_seeds": REP_SEEDS, "n_outer": N_OUTER, "n_inner": N_INNER,
                       "n_boot": N_BOOT, "boot_seed": BOOT_SEED, "ncomp_grid": NCOMP_GRID,
                       "grids": {k: {kk: [str(x) for x in vv] for kk, vv in v["grid"].items()}
                                 for k, v in ALGOS.items()}},
            "prevalence": prev,
            "holdout": hold.drop(columns=["chosen_C", "chosen_C_A_R"]).to_dict(orient="records"),
            "per_repetition_summary": summ.to_dict(orient="records"),
            "aggregated_oof": aggm,
            "bootstrap": boot.to_dict(orient="records"),
            "stability": stab_s.to_dict(orient="records"),
            "combined": combined,
        }, fh, indent=2, default=str)

    print("\n===== RESUMEN =====")
    for r in combined:
        print(f"{r['algorithm']:17s} | holdout dROC {r['holdout_delta_roc']:+.3f} dPR {r['holdout_delta_pr']:+.3f}"
              f" | NCV dROC {r['ncv_delta_roc_mean']:+.3f}+-{r['ncv_delta_roc_sd']:.3f}"
              f" | OOF dROC {r['oof_delta_roc']:+.3f} CI{r['boot_delta_roc_ci']} incl0={r['boot_delta_roc_incl0']}"
              f" | OOF dPR {r['oof_delta_pr']:+.3f} CI{r['boot_delta_pr_ci']} incl0={r['boot_delta_pr_incl0']}")
    print(f"\nGuardado en {OUT_DIR}/")


if __name__ == "__main__":
    main()
