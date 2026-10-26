"""
ETAPA 5 - Caracterizacion final del modelo principal.
Modelo principal: Logistic Regression L2  C+A+R.   Referencia: Logistic Regression L2  C.
(RF y GB quedan como comparacion exploratoria de Etapa 4; no se repiten.)

Reutiliza las predicciones OOF agregadas de Etapa 3 (stage3_oof_aggregated.csv) para
discriminacion, calibracion, clasificacion (threshold fijo 0.50) y decision curve analysis.
Para interpretabilidad reajusta los pipelines LR (C y C+A+R) sobre el TRAIN del hold-out 80/20
(seed 42) con los hiperparametros elegidos en el train del hold-out de Etapa 3 (leidos de
stage3_holdout_8020.csv) -> coeficientes/OR + loadings del PCA final.

NO se recalibra, NO se optimiza el threshold como resultado principal, NO se cambia nada del
pipeline V2. No modifica notebook, bases, ni run_stage2/3/4*.py.

Salidas: pipeline_v2/stage5_*.csv, pipeline_v2/figs_etapa5/*.png, stage5_details.json
"""
from __future__ import annotations
import base64
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

from run_stage2 import (BLOCK_TYPES, RANDOM_STATE, build_cohort, make_estimator)
from run_stage4 import make_estimator4, make_preprocessor4   # LR usa scale_dense=True -> identico a V2

P = Path(__file__).resolve().parent
FIGS = P / "figs_etapa5"
FIGS.mkdir(exist_ok=True)
PREV = 93 / 281
EPS = 1e-6

# Hiperparametros elegidos por GridSearch en el TRAIN del hold-out 80/20 (Etapa 3); se leen del CSV
_HO = pd.read_csv(P / "stage3_holdout_8020.csv").set_index("model")
BEST = {"C": {"clf__C": float(_HO.loc["C", "best_C"])},
        "C+A+R": {"clf__C": float(_HO.loc["C+A+R", "best_C"]),
                  "preprocessor__R__pca__n_components": int(_HO.loc["C+A+R", "best_n_components"])}}

FAMILY_RULES = [
    (r"^F\d+_", "F* (difusion/DT)"),
    (r"washin|washout|peak_enhancement|WashinRate|SER|PE_map|enhanc", "cinetica/realce"),
    (r"DFT|Moment_Invariant|Fourier", "Fourier/momentos"),
    (r"vol_cu_mm|Total_.*_vol|volume|Vol", "morfologia/volumen"),
    (r"Homogeneity|Entropy|Contrast|Correlation|Dissimilarity|inverse_difference|"
     r"information_measure|Sum_|Difference_|cluster|Energy|IDM|GLCM", "textura GLCM/1er orden"),
]


def family_of(name: str) -> str:
    for pat, lab in FAMILY_RULES:
        if re.search(pat, name, flags=re.I):
            return lab
    return "otras"


def b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


# ------------------------------------------------------------------ 1. reuse OOF
def load_oof():
    df = pd.read_csv(P / "stage3_oof_aggregated.csv")
    y = df["y_true"].values.astype(int)
    return df["patient_id"].values, y, {"C": df["C"].values, "C+A+R": df["C+A+R"].values}


def load_bootstrap():
    b = pd.read_csv(P / "stage3_bootstrap.csv").set_index("comparison").loc["C vs C+A+R"]
    return {"delta_roc": b.delta_roc_point, "delta_roc_ci": [b.delta_roc_ci_lo, b.delta_roc_ci_hi],
            "delta_pr": b.delta_pr_point, "delta_pr_ci": [b.delta_pr_ci_lo, b.delta_pr_ci_hi]}


# ------------------------------------------------------------------ 2. calibration
def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_stats(y, p):
    brier = brier_score_loss(y, p)
    lp = logit(p)
    # slope: y ~ a + b*logit(p)   (b ideal = 1)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(lp.reshape(-1, 1), y)
    slope = float(lr.coef_[0][0])
    # calibration-in-the-large intercept: fixed slope=1, solve sum(sigmoid(a+logit(p)) - y)=0
    f = lambda a: np.mean(1 / (1 + np.exp(-(a + lp))) - y)
    intercept = float(brentq(f, -10, 10))
    return {"brier": brier, "cal_slope": slope, "cal_intercept": intercept}


# ------------------------------------------------------------------ 3. classification
def clf_metrics(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    acc = (tp + tn) / len(y)
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else np.nan
    return dict(threshold=thr, TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp),
                sensitivity=sens, specificity=spec, PPV=ppv, NPV=npv, accuracy=acc, F1=f1)


def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(thr[np.argmax(j)])


# ------------------------------------------------------------------ 4. DCA
def net_benefit(y, p, pt_grid):
    n = len(y)
    prev = y.mean()
    nb_model, nb_all = [], []
    for pt in pt_grid:
        w = pt / (1 - pt)
        pred = (p >= pt).astype(int)
        tp = np.sum((pred == 1) & (y == 1))
        fp = np.sum((pred == 1) & (y == 0))
        nb_model.append(tp / n - (fp / n) * w)
        nb_all.append(prev - (1 - prev) * w)
    return np.array(nb_model), np.array(nb_all)


# ------------------------------------------------------------------ 5. interpretability
def fit_holdout(model_name, merged, y, rad_cols):
    idx = np.arange(len(merged))
    tr, te = train_test_split(idx, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    blocks = {"C": ["C"], "C+A+R": ["C", "A", "R"]}[model_name]
    est = make_estimator4("logistic", blocks, rad_cols)
    est.set_params(**BEST[model_name])
    est.fit(merged.iloc[tr], y.iloc[tr])
    proba_te = est.predict_proba(merged.iloc[te])[:, 1]
    return est, tr, te, proba_te


def feature_names_and_coef(est, blocks):
    pre = est.named_steps["preprocessor"]
    clf = est.named_steps["clf"]
    names, blk = [], []
    A_cols = set(sum((BLOCK_TYPES["A"].get(k, []) for k in ("continuous", "binary", "ordinal", "nominal")), []))
    for name, trans, cols in pre.transformers_:
        if name in ("R", "remainder") or isinstance(trans, str):
            continue
        if name == "nom":
            ohe = trans.named_steps["ohe"]
            outc = ohe.get_feature_names_out(cols).tolist()
        else:
            outc = list(cols)
        for c in outc:
            base = c.split("_", 1)[0] if name == "nom" else c
            src = c.rsplit("_", 1)[0] if name == "nom" else c
            names.append(c)
            blk.append("A" if src in A_cols else "C")
    n_dense = len(names)
    coefs = clf.coef_[0]
    rows = [{"feature": names[i], "block": blk[i], "coef": float(coefs[i]),
             "odds_ratio": float(np.exp(coefs[i]))} for i in range(n_dense)]
    # PCA components (block R)
    if "R" in blocks:
        for j in range(n_dense, len(coefs)):
            comp = j - n_dense
            rows.append({"feature": f"PCA_R_{comp + 1}", "block": "R (PCA)",
                         "coef": float(coefs[j]), "odds_ratio": float(np.exp(coefs[j]))})
    return pd.DataFrame(rows)


def pca_loadings(est, rad_cols, top=10):
    pre = est.named_steps["preprocessor"]
    Rp = pre.named_transformers_["R"]
    keep = Rp.named_steps["corr"].keep_
    kept_names = [rad_cols[i] for i in keep]
    comps = Rp.named_steps["pca"].components_          # (n_comp, n_kept)
    evr = Rp.named_steps["pca"].explained_variance_ratio_
    out = []
    for ci in range(comps.shape[0]):
        load = comps[ci]
        order = np.argsort(-np.abs(load))[:top]
        for rank, k in enumerate(order):
            out.append({"component": ci + 1, "rank": rank + 1, "radiomic": kept_names[k],
                        "loading": float(load[k]), "abs_loading": float(abs(load[k])),
                        "family": family_of(kept_names[k])})
    return pd.DataFrame(out), evr, len(kept_names)


# ================================================================== plotting
def fig_roc(y, pc, pca):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for lab, p, col in [("C", pc, "#6b7280"), ("C+A+R", pca, "#3b7dd8")]:
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{lab} (AUC={roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("1 - Especificidad"); ax.set_ylabel("Sensibilidad")
    ax.set_title("ROC - OOF agregado (Repeated Nested CV)"); ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIGS / "fig1_roc_oof.png", dpi=200); plt.close(fig)


def fig_pr(y, pc, pca):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for lab, p, col in [("C", pc, "#6b7280"), ("C+A+R", pca, "#3b7dd8")]:
        pr, rc, _ = precision_recall_curve(y, p)
        ax.plot(rc, pr, color=col, lw=2, label=f"{lab} (AP={average_precision_score(y, p):.3f})")
    ax.axhline(PREV, color="#b3261e", ls="--", lw=1, label=f"Azar (prevalencia={PREV:.2f})")
    ax.set_xlabel("Recall (sensibilidad)"); ax.set_ylabel("Precision (PPV)")
    ax.set_title("Precision-Recall - OOF agregado"); ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIGS / "fig2_pr_oof.png", dpi=200); plt.close(fig)


def fig_calibration(y, pc, pca, cal):
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="Calibracion perfecta")
    for lab, p, col in [("C", pc, "#6b7280"), ("C+A+R", pca, "#3b7dd8")]:
        pt, pp = calibration_curve(y, p, n_bins=10, strategy="quantile")
        c = cal[lab]
        ax.plot(pp, pt, "o-", color=col, lw=1.8, ms=5,
                label=f"{lab}: Brier={c['brier']:.3f}, slope={c['cal_slope']:.2f}, int={c['cal_intercept']:+.2f}")
    ax.set_xlabel("Probabilidad predicha (media por decil)")
    ax.set_ylabel("Frecuencia observada de respuesta")
    ax.set_title("Curva de calibracion - OOF agregado"); ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(FIGS / "fig3_calibration_oof.png", dpi=200); plt.close(fig)


def fig_dca(pt, nb_c, nb_car, nb_all):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.plot(pt, nb_all, color="#9aa3ad", lw=1.5, ls="--", label="Tratar a todas")
    ax.axhline(0, color="#444", lw=1.2, label="Tratar a ninguna")
    ax.plot(pt, nb_c, color="#6b7280", lw=2, label="C")
    ax.plot(pt, nb_car, color="#3b7dd8", lw=2, label="C+A+R")
    ax.set_xlabel("Threshold probability (p_t)"); ax.set_ylabel("Net benefit")
    ax.set_title("Decision Curve Analysis - OOF agregado")
    ax.set_ylim(-0.05, PREV + 0.03); ax.set_xlim(0, 0.8); ax.legend(fontsize=9); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(FIGS / "fig4_dca_oof.png", dpi=200); plt.close(fig)


def fig_confusion(y, pc, pca):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    for ax, lab, p in [(axes[0], "C", pc), (axes[1], "C+A+R", pca)]:
        cm = confusion_matrix(y, (p >= .5).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Real 0\n(no resp.)", "Real 1\n(resp.)"])
        ax.set_title(f"{lab}  (threshold 0.50)")
    fig.suptitle("Matrices de confusion - OOF agregado", y=1.03, fontweight="bold")
    fig.tight_layout(); fig.savefig(FIGS / "fig5_confusion_oof.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_coef_C(df_c):
    d = df_c.reindex(df_c.coef.abs().sort_values(ascending=False).index).head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.6, max(4.2, .38 * len(d))))
    cols = ["#3b7dd8" if c > 0 else "#b3261e" for c in d.coef]
    ax.barh(d.feature, d.coef, color=cols)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("Coeficiente logistico (log-odds; variables densas estandarizadas)")
    ax.set_title("Modelo C - coeficientes (mayor |peso|)")
    ax.grid(axis="x", alpha=.3); fig.tight_layout()
    fig.savefig(FIGS / "fig6_coef_C.png", dpi=200); plt.close(fig)


def fig_blocks_CAR(df_car):
    g = df_car.assign(absc=df_car.coef.abs()).groupby("block").agg(
        sum_abs=("absc", "sum"), n=("absc", "size"), max_abs=("absc", "max")).reset_index()
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(g.block, g.sum_abs, color=["#6b7280", "#c98a3b", "#3b7dd8"][:len(g)])
    ax.set_ylabel("Suma de |coef|"); ax.set_title("C+A+R - contribucion por bloque (suma |coef|)")
    for i, r in g.iterrows():
        ax.text(i, r.sum_abs, f"n={int(r.n)}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=.3); fig.tight_layout()
    fig.savefig(FIGS / "fig7_blocks_CAR.png", dpi=200); plt.close(fig)


def fig_pca_interp(load_df, comp_rank):
    top_comps = comp_rank.head(2)["component"].tolist()
    fig, axes = plt.subplots(1, len(top_comps), figsize=(5.6 * len(top_comps), 4.6), squeeze=False)
    fam_col = {"F* (difusion/DT)": "#8e44ad", "cinetica/realce": "#e67e22",
              "Fourier/momentos": "#16a085", "morfologia/volumen": "#2c3e50",
              "textura GLCM/1er orden": "#3b7dd8", "otras": "#9aa3ad"}
    for ax, comp in zip(axes[0], top_comps):
        d = load_df[load_df.component == comp].sort_values("abs_loading").tail(10)
        short = [r[:34] for r in d.radiomic]
        ax.barh(short, d.loading, color=[fam_col.get(f, "#9aa3ad") for f in d.family])
        ax.axvline(0, color="k", lw=.8)
        cr = comp_rank[comp_rank.component == comp].iloc[0]
        ax.set_title(f"PCA_R_{comp}  (coef LR={cr.coef:+.2f}, OR={cr.odds_ratio:.2f})", fontsize=10)
        ax.set_xlabel("loading"); ax.tick_params(axis="y", labelsize=7.5); ax.grid(axis="x", alpha=.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=v) for k, v in fam_col.items()]
    fig.legend(handles, list(fam_col), loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Componentes PCA de R mas influyentes - top loadings por familia", fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(FIGS / "fig8_pca_interp.png", dpi=200); plt.close(fig)


def fig_holdout_roc(y_te, pc_te, pca_te):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    for lab, p, col in [("C", pc_te, "#6b7280"), ("C+A+R", pca_te, "#3b7dd8")]:
        fpr, tpr, _ = roc_curve(y_te, p)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{lab} (AUC={roc_auc_score(y_te, p):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1)
    ax.set_xlabel("1 - Especificidad"); ax.set_ylabel("Sensibilidad")
    ax.set_title("ROC - hold-out 80/20 (analisis inicial, n=57)"); ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(FIGS / "fig9_roc_holdout.png", dpi=200); plt.close(fig)


# ================================================================== main
def main():
    print("Cargando OOF agregado de Etapa 3...")
    _, y, P_ = load_oof()
    pc, pca = P_["C"], P_["C+A+R"]
    boot = load_bootstrap()

    disc = {m: {"roc_oof": float(roc_auc_score(y, P_[m])), "pr_oof": float(average_precision_score(y, P_[m]))}
            for m in ("C", "C+A+R")}
    hold = pd.read_csv(P / "stage3_holdout_8020.csv").set_index("model")
    for m in ("C", "C+A+R"):
        disc[m]["roc_holdout"] = float(hold.loc[m, "roc_auc"])
        disc[m]["pr_holdout"] = float(hold.loc[m, "pr_auc"])
    disc["delta_oof_roc"] = disc["C+A+R"]["roc_oof"] - disc["C"]["roc_oof"]
    disc["delta_oof_pr"] = disc["C+A+R"]["pr_oof"] - disc["C"]["pr_oof"]
    disc["bootstrap"] = boot
    pd.DataFrame([
        {"model": "C", **{k: disc["C"][k] for k in ("roc_oof", "pr_oof", "roc_holdout", "pr_holdout")}},
        {"model": "C+A+R", **{k: disc["C+A+R"][k] for k in ("roc_oof", "pr_oof", "roc_holdout", "pr_holdout")}},
    ]).to_csv(P / "stage5_discrimination.csv", index=False)

    print("Calibracion...")
    cal = {m: calibration_stats(y, P_[m]) for m in ("C", "C+A+R")}
    pd.DataFrame([{"model": m, **cal[m]} for m in ("C", "C+A+R")]).to_csv(P / "stage5_calibration.csv", index=False)
    cc_rows = []
    for m in ("C", "C+A+R"):
        pt_, pp_ = calibration_curve(y, P_[m], n_bins=10, strategy="quantile")
        for a, b in zip(pp_, pt_):
            cc_rows.append({"model": m, "prob_pred": a, "prob_true": b})
    pd.DataFrame(cc_rows).to_csv(P / "stage5_calibration_curve.csv", index=False)

    print("Clasificacion (threshold 0.50)...")
    clf = {m: clf_metrics(y, P_[m], 0.50) for m in ("C", "C+A+R")}
    pd.DataFrame([{"model": m, **clf[m]} for m in ("C", "C+A+R")]).to_csv(P / "stage5_classification_050.csv", index=False)
    # alternativo SOLO descriptivo
    alt = {}
    for m in ("C", "C+A+R"):
        thr = youden_threshold(y, P_[m])
        alt[m] = clf_metrics(y, P_[m], thr)
    pd.DataFrame([{"model": m, "note": "descriptivo - NO resultado principal", **alt[m]}
                 for m in ("C", "C+A+R")]).to_csv(P / "stage5_classification_alt_descriptive.csv", index=False)
    # Umbrales exploratorios (2026-10-24): 0.50 (referencia) vs 0.40 / 0.35 sobre el MISMO OOF agregado.
    # Media +- SD por repeticion (stage3_oof_long.csv) sólo como estabilidad. NO se elige un umbral.
    long_ = pd.read_csv(P / "stage3_oof_long.csv")
    thr_rows = []
    for m in ("C", "C+A+R"):
        for thr in (0.50, 0.40, 0.35):
            r = clf_metrics(y, P_[m], thr)
            r["n_positive_pred"] = r["TP"] + r["FP"]
            per = [clf_metrics(g["y_true"].values, g[m].values, thr) for _, g in long_.groupby("repetition")]
            for k in ("sensitivity", "specificity"):
                r[f"{k}_rep_mean"] = float(np.mean([d[k] for d in per]))
                r[f"{k}_rep_sd"] = float(np.std([d[k] for d in per], ddof=1))
            thr_rows.append({"model": m, **r})
    pd.DataFrame(thr_rows).to_csv(P / "stage5_classification_thresholds.csv", index=False)

    print("Decision Curve Analysis...")
    pt = np.round(np.arange(0.05, 0.76, 0.01), 3)
    nb_c, nb_all = net_benefit(y, pc, pt)
    nb_car, _ = net_benefit(y, pca, pt)
    dca = pd.DataFrame({"pt": pt, "nb_C": nb_c, "nb_C_A_R": nb_car,
                        "nb_treat_all": nb_all, "nb_treat_none": 0.0})
    dca.to_csv(P / "stage5_dca.csv", index=False)
    gap = dca.nb_C_A_R - dca.nb_C
    car_better = dca.pt[gap > 1e-4].tolist()
    c_better = dca.pt[gap < -1e-4].tolist()

    def contiguous(vals):
        if not vals:
            return []
        vals = sorted(vals); out = [[vals[0], vals[0]]]
        for v in vals[1:]:
            if round(v - out[-1][1], 3) <= 0.011:
                out[-1][1] = v
            else:
                out.append([v, v])
        return out

    car_vs_default = dca[(dca.nb_C_A_R > dca.nb_treat_all + 1e-6) & (dca.nb_C_A_R > 1e-6)]
    cu = dca[(dca.nb_C > dca.nb_treat_all + 1e-6) & (dca.nb_C > 1e-6)]
    dca_summary = {
        "pt_grid": [float(dca.pt.min()), float(dca.pt.max())],
        "frac_pt_car_gt_c": float((gap > 1e-4).mean()),
        "frac_pt_c_gt_car": float((gap < -1e-4).mean()),
        "car_gt_c_ranges": contiguous(car_better),
        "c_gt_car_ranges": contiguous(c_better),
        "car_gt_c_maxgap": float(gap.max()),
        "car_gt_c_maxgap_at_pt": float(dca.pt[gap.idxmax()]),
        "c_gt_car_maxgap": float((-gap).max()),
        "c_gt_car_maxgap_at_pt": float(dca.pt[(-gap).idxmax()]),
        "car_useful_range": [float(car_vs_default.pt.min()), float(car_vs_default.pt.max())] if len(car_vs_default) else None,
        "c_useful_range": [float(cu.pt.min()), float(cu.pt.max())] if len(cu) else None,
    }
    with open(P / "stage5_dca_summary.json", "w") as fh:
        json.dump(dca_summary, fh, indent=2)
    pd.DataFrame([{k: json.dumps(v) if isinstance(v, list) else v for k, v in dca_summary.items()}]
                 ).to_csv(P / "stage5_dca_summary.csv", index=False)

    print("Interpretabilidad - reajuste sobre train del hold-out...")
    merged, yfull, rad_cols, _ = build_cohort()
    est_C, tr, te, pc_te = fit_holdout("C", merged, yfull, rad_cols)
    est_CAR, _, _, pca_te = fit_holdout("C+A+R", merged, yfull, rad_cols)
    y_te = yfull.iloc[te].values

    df_c = feature_names_and_coef(est_C, ["C"])
    df_c.to_csv(P / "stage5_coef_C.csv", index=False)
    df_car = feature_names_and_coef(est_CAR, ["C", "A", "R"])
    df_car.to_csv(P / "stage5_coef_CAR.csv", index=False)

    load_df, evr, n_kept = pca_loadings(est_CAR, rad_cols, top=12)
    load_df.to_csv(P / "stage5_pca_loadings.csv", index=False)
    cr = df_car[df_car.block == "R (PCA)"].reset_index(drop=True).copy()
    cr["component"] = cr.index + 1
    cr["explained_var_ratio"] = [float(evr[i]) for i in range(len(cr))]
    cr["abs_coef"] = cr.coef.abs()
    comp_rank = cr.sort_values("abs_coef", ascending=False)[
        ["component", "coef", "odds_ratio", "explained_var_ratio", "abs_coef"]]
    dom_fam = []
    for c in comp_rank.component:
        fam = (load_df[load_df.component == c].groupby("family").abs_loading.sum()
               .sort_values(ascending=False))
        dom_fam.append("; ".join(f"{k} ({v:.2f})" for k, v in fam.head(3).items()))
    comp_rank["familias_dominantes"] = dom_fam
    comp_rank.to_csv(P / "stage5_pca_component_summary.csv", index=False)

    print("Figuras...")
    fig_roc(y, pc, pca); fig_pr(y, pc, pca); fig_calibration(y, pc, pca, cal)
    fig_dca(pt, nb_c, nb_car, nb_all); fig_confusion(y, pc, pca)
    fig_coef_C(df_c); fig_blocks_CAR(df_car); fig_pca_interp(load_df, comp_rank)
    fig_holdout_roc(y_te, pc_te, pca_te)

    details = {
        "prevalence": PREV, "n": int(len(y)), "n_pos": int(y.sum()),
        "discrimination": disc, "calibration": cal,
        "classification_050": clf, "classification_alt_descriptive": alt,
        "dca_summary": dca_summary,
        "holdout_refit_auc": {"C": float(roc_auc_score(y_te, pc_te)),
                              "C+A+R": float(roc_auc_score(y_te, pca_te))},
        "pca": {"n_components": int(est_CAR.named_steps["preprocessor"].named_transformers_["R"]
                                   .named_steps["pca"].n_components_),
                "n_radiomics_after_corr": int(n_kept),
                "explained_variance_ratio": [float(x) for x in evr],
                "component_summary": comp_rank.to_dict(orient="records")},
        "coef_C_top": df_c.reindex(df_c.coef.abs().sort_values(ascending=False).index).head(12).to_dict(orient="records"),
        "coef_CAR": df_car.to_dict(orient="records"),
    }
    with open(P / "stage5_details.json", "w") as fh:
        json.dump(details, fh, indent=2, default=str)

    print("\n=== DISCRIMINACION (OOF agregado) ===")
    for m in ("C", "C+A+R"):
        print(f"  {m:6s} ROC {disc[m]['roc_oof']:.3f}  PR {disc[m]['pr_oof']:.3f}  "
              f"(hold-out ROC {disc[m]['roc_holdout']:.3f} PR {disc[m]['pr_holdout']:.3f})")
    print(f"  dROC OOF {disc['delta_oof_roc']:+.3f}  IC95%[{boot['delta_roc_ci'][0]:+.3f},{boot['delta_roc_ci'][1]:+.3f}]  "
          f"dPR OOF {disc['delta_oof_pr']:+.3f}  IC95%[{boot['delta_pr_ci'][0]:+.3f},{boot['delta_pr_ci'][1]:+.3f}]")
    print("=== CALIBRACION ===")
    for m in ("C", "C+A+R"):
        print(f"  {m:6s} Brier {cal[m]['brier']:.3f}  slope {cal[m]['cal_slope']:.2f}  intercept {cal[m]['cal_intercept']:+.2f}")
    print("=== CLASIFICACION thr=0.50 ===")
    for m in ("C", "C+A+R"):
        c = clf[m]
        print(f"  {m:6s} sens {c['sensitivity']:.2f} spec {c['specificity']:.2f} PPV {c['PPV']:.2f} "
              f"NPV {c['NPV']:.2f} acc {c['accuracy']:.2f} F1 {c['F1']:.2f}  [TN{c['TN']} FP{c['FP']} FN{c['FN']} TP{c['TP']}]")
    print("=== DCA ===", json.dumps(dca_summary, indent=1, default=str))
    print("=== PCA componentes (por |coef| LR) ===")
    print(comp_rank.to_string(index=False))
    print(f"\nGuardado en {P}/  y figuras en {FIGS}/")


if __name__ == "__main__":
    main()
