"""Figuras de datos del capítulo 3 (Resultados).

Fuentes (solo lectura):
  FIG_PCA_VARIANZA      analisis_complementarios/resultados/scree.json  (PCA descriptivo global, 299 caract.)
  FIG_ROC_PR_MODELOS    pipeline_v2/stage3_oof_aggregated.csv         (OOF promediado 10×5×5)
  FIG_DIFERENCIAS_IC    pipeline_v2/stage3_bootstrap.csv              (bootstrap pareado, 5000)
  FIG_CALIBRACION       pipeline_v2/stage5_calibration_curve.csv, stage5_calibration.csv
  FIG_SELECCION_COMPONENTES  pipeline_v2/stage3_stability_folds.csv  (Figura A.3.1)
"""
import numpy as np
import pandas as pd

import comun as k

mpl = k.preparar_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

T, CM = k.TINTA, k.CM


def coma(d):
    return FuncFormatter(lambda v, _: k.num(v, d))


def fig_pca_varianza():
    d = k.leer_json(k.COMPL / "scree.json")
    ev, cum = np.array(d["ev"]), np.array(d["cum"])
    n = 30
    datos = pd.DataFrame({"componente": np.arange(1, len(ev) + 1), "varianza_explicada": ev, "varianza_acumulada": cum})
    datos.to_csv(k.DIR_DATOS / "FIG_PCA_VARIANZA.csv", index=False)
    assert d["n_after_corr"] == 299 and round(cum[9], 3) == 0.564 and round(cum[29], 3) == 0.806

    fig, (a, b) = plt.subplots(1, 2, figsize=(16 * CM, 6.6 * CM), gridspec_kw={"wspace": 0.32})
    x = np.arange(1, n + 1)
    a.bar(x, ev[:n] * 100, width=0.72, color=k.COMP["secuencial_petroleo"][3], edgecolor="none", zorder=3)
    for i in range(3):
        a.text(x[i] + 0.55, ev[i] * 100 + 0.25, k.num(ev[i] * 100, 1).rstrip() + " %", fontsize=6.7,
               color=T["secundaria"], ha="left", va="bottom")
    a.set_xlim(0.2, n + 0.8)
    a.set_ylim(0, 16)
    a.set_xlabel("Componente principal")
    a.set_ylabel("Varianza explicada (%)")
    a.xaxis.set_major_locator(MultipleLocator(5))
    a.yaxis.set_major_formatter(coma(0))
    a.grid(axis="x", visible=False)

    b.plot(x, cum[:n] * 100, color=k.COMP["secuencial_petroleo"][4], linewidth=1.5, marker="o", markersize=2.6,
           markeredgecolor="white", markeredgewidth=0.5, zorder=3)
    for comp in (2, 3, 10, 19, 30):
        yv = cum[comp - 1] * 100
        b.plot(comp, yv, marker="o", markersize=5.2, markerfacecolor="white",
               markeredgecolor=k.COMP["secuencial_petroleo"][5], markeredgewidth=1.1, zorder=4)
        if comp <= 3:
            dx, dy, ha, va = 0.9, -1.0, "left", "top"
        else:
            dx, dy, ha, va = (0.9, -2.6, "left", "top") if comp == 10 else (-0.4, 2.6, "right", "bottom")
        b.text(comp + dx, yv + dy, f"{comp} componentes: {k.num(yv, 1)} %", fontsize=6.7,
               color=T["secundaria"], ha=ha, va=va)
    b.set_xlim(0, n + 1)
    b.set_ylim(0, 100)
    b.set_xlabel("Número de componentes principales")
    b.set_ylabel("Varianza acumulada (%)")
    b.xaxis.set_major_locator(MultipleLocator(5))
    b.yaxis.set_major_formatter(coma(0))
    k.letra_panel(a, "A", x=-0.2)
    k.letra_panel(b, "B", x=-0.2)
    k.guardar(fig, "FIG_PCA_VARIANZA")
    plt.close(fig)


def fig_roc_pr():
    oof = pd.read_csv(k.PIPE / "stage3_oof_aggregated.csv")
    y = oof["y_true"].values
    prev = y.mean()
    fig, (a, b) = plt.subplots(1, 2, figsize=(16 * CM, 7.9 * CM), gridspec_kw={"wspace": 0.3})
    filas = []
    for m in k.ORDEN:
        p = oof[m].values
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        pr, rc, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        filas.append({"modelo": m, "roc_auc": auc, "ap": ap})
        lw = k.et.linea_modelo(m)
        a.plot(fpr, tpr, drawstyle="steps-post", label=k.et.etiqueta(m, "ROC-AUC", auc), **lw)
        b.plot(rc, pr, drawstyle="steps-post", label=k.et.etiqueta(m, "AP", ap), **lw)
    ref = pd.DataFrame(filas)
    # Control: deben coincidir con las cifras oficiales (C 0,750/0,546; C+R 0,749/0,549; C+A+R 0,761/0,577)
    assert [round(v, 3) for v in ref.roc_auc] == [0.750, 0.749, 0.761]
    assert [round(v, 3) for v in ref.ap] == [0.546, 0.549, 0.577]
    ref.to_csv(k.DIR_DATOS / "FIG_ROC_PR_MODELOS_metricas.csv", index=False)
    a.plot([0, 1], [0, 1], **k.et.linea_referencia())
    b.axhline(prev, **k.et.linea_referencia())
    b.text(0.02, prev - 0.02, f"Prevalencia ({k.num(prev, 3)})", fontsize=6.7, color=T["atenuada"], ha="left",
           va="top")
    a.set_xlabel("1 − especificidad")
    a.set_ylabel("Sensibilidad")
    b.set_xlabel("Sensibilidad (exhaustividad)")
    b.set_ylabel("Valor predictivo positivo (precisión)")
    for ax in (a, b):
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        k.et.formato_coma(ax, 1)
    a.legend(loc="lower right")
    b.legend(loc="upper right")
    k.letra_panel(a, "A", x=-0.2)
    k.letra_panel(b, "B", x=-0.2)
    k.guardar(fig, "FIG_ROC_PR_MODELOS")
    plt.close(fig)


def fig_diferencias():
    bs = pd.read_csv(k.PIPE / "stage3_bootstrap.csv")
    bs["modelo"] = bs["comparison"].str.replace("C vs ", "", regex=False)
    bs.to_csv(k.DIR_DATOS / "FIG_DIFERENCIAS_IC.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(16 * CM, 4.6 * CM), sharey=True, gridspec_kw={"wspace": 0.12})
    modelos = ["C+R", "C+A+R"]
    for ax, met, tit in zip(axs, ("roc", "pr"), ("Diferencia de ROC-AUC respecto de C", "Diferencia de AP respecto de C")):
        lim = 0.11
        ax.axvline(0, color=T["eje"], linewidth=0.8, zorder=1)
        for i, m in enumerate(modelos):
            r = bs[bs.modelo == m].iloc[0]
            yv = len(modelos) - 1 - i
            pt, lo, hi = r[f"delta_{met}_point"], r[f"delta_{met}_ci_lo"], r[f"delta_{met}_ci_hi"]
            c = k.MODELOS[m]["color"]
            ax.plot([lo, hi], [yv, yv], color=c, linewidth=k.MODELOS[m]["ancho_linea"] * 0.9, solid_capstyle="round",
                    zorder=3)
            ax.plot(pt, yv, marker=k.MODELOS[m]["marcador"], markersize=5, color=c, markeredgecolor="white",
                    markeredgewidth=0.8, zorder=4)
            ax.text(lim * 0.98, yv + 0.2, f"{k.num(pt, 3, signo=True)} {k.ic(lo, hi)}", fontsize=6.9,
                    color=T["secundaria"], ha="right", va="bottom")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-0.6, len(modelos) - 0.25)
        ax.set_yticks(range(len(modelos)))
        ax.set_yticklabels(modelos[::-1])
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: k.num(v, 2, signo=abs(v) > 1e-9)))
        ax.set_xlabel(tit)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    k.letra_panel(axs[0], "A", x=-0.17, y=1.03)
    k.letra_panel(axs[1], "B", x=-0.06, y=1.03)
    k.guardar(fig, "FIG_DIFERENCIAS_IC")
    plt.close(fig)


def fig_calibracion():
    cur = pd.read_csv(k.PIPE / "stage5_calibration_curve.csv")
    cal = pd.read_csv(k.PIPE / "stage5_calibration.csv").set_index("model")
    cur.to_csv(k.DIR_DATOS / "FIG_CALIBRACION_curva.csv", index=False)
    cal.to_csv(k.DIR_DATOS / "FIG_CALIBRACION_resumen.csv")
    assert round(cal.loc["C+A+R", "brier"], 3) == 0.198 and round(cal.loc["C", "cal_slope"], 3) == 0.838
    fig, ax = plt.subplots(figsize=(9.2 * CM, 8.4 * CM))
    ax.plot([0, 1], [0, 1], **k.et.linea_referencia())
    ax.text(0.97, 0.915, "Calibración perfecta", rotation=45, fontsize=6.7, color=T["atenuada"], ha="right",
            va="bottom", rotation_mode="anchor", transform_rotates_text=True)
    for m in ("C", "C+A+R"):
        d = cur[cur.model == m].sort_values("prob_pred")
        e = k.MODELOS[m]
        lab = (f"{m} (Brier {k.num(cal.loc[m, 'brier'], 3)}; pendiente {k.num(cal.loc[m, 'cal_slope'], 3)})")
        ax.plot(d.prob_pred, d.prob_true, marker=e["marcador"], markersize=3.6, markeredgecolor="white",
                markeredgewidth=0.6, label=lab, **k.et.linea_modelo(m))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    k.et.formato_coma(ax, 1)
    ax.set_xlabel("Probabilidad estimada (media del decil)")
    ax.set_ylabel("Frecuencia observada de respuesta")
    ax.legend(loc="upper left")
    k.guardar(fig, "FIG_CALIBRACION")
    plt.close(fig)


def fig_seleccion_componentes():
    st = pd.read_csv(k.PIPE / "stage3_stability_folds.csv")
    grilla = (1, 2, 3, 5, 10, 15, 20, 30)
    cuentas = {m: [int((st[st.model == m].n_components == q).sum()) for q in grilla] for m in ("C+R", "C+A+R")}
    assert all(sum(v) == 50 for v in cuentas.values()) and cuentas["C+R"][1] == 28 and cuentas["C+A+R"][1] == 17
    pd.DataFrame({"n_components": grilla, **cuentas}).to_csv(k.DIR_DATOS / "FIG_SELECCION_COMPONENTES.csv",
                                                             index=False)
    fig, ax = plt.subplots(figsize=(13 * CM, 6.4 * CM))
    x, w = np.arange(len(grilla)), 0.36
    for j, m in enumerate(("C+R", "C+A+R")):
        xs = x + (j - 0.5) * (w + 0.03)
        ax.bar(xs, cuentas[m], width=w, color=k.MODELOS[m]["color"], edgecolor="none", label=m, zorder=3)
        for xi, n in zip(xs, cuentas[m]):
            if n:
                ax.text(xi, n + 0.5, str(n), ha="center", va="bottom", fontsize=6.7, color=T["secundaria"])
    ax.set_xticks(x, [str(q) for q in grilla])
    ax.set_xlim(-0.6, len(grilla) - 0.4)
    ax.set_ylim(0, 32)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Número de componentes principales seleccionado (k)")
    ax.set_ylabel("Particiones externas (de 50)")
    leg = ax.legend(loc="upper right")
    for t in leg.get_texts():
        if t.get_text() == "C+A+R":
            t.set_fontweight("semibold")
    k.guardar(fig, "FIG_SELECCION_COMPONENTES")
    plt.close(fig)


if __name__ == "__main__":
    fig_seleccion_componentes()
    fig_pca_varianza()
    fig_roc_pr()
    fig_diferencias()
    fig_calibracion()
    print("Figuras de resultados generadas en", k.DIR_FIG)
