"""Esquemas y diagramas (capítulos 1, 2 y 3). Coordenadas en centímetros de impresión.

  FIG_CURVAS_REALCE_DCE   1.5.2  esquema ilustrativo (sin datos de pacientes); base: Mann et al. (2019)
  FIG_FLUJO_RADIOMICO     1.6.2  esquema conceptual; base: Lambin et al. (2012), Gillies et al. (2016), Lee et al. (2020)
  FIG_DISENO_ESTUDIO      2.1    elaboración propia
  FIG_PREPARACION_R       2.6    elaboración propia (parámetros del pipeline: pipeline_v2/run_stage2.py)
  FIG_CV_ANIDADA          2.8    elaboración propia (10 × 5 × 5; semillas 42–51; bootstrap 5000)
  FIG_FLUJO_COHORTE       3.1.1  datos: Clinical_and_Other_Features.xlsx vía pipeline_v2/run_stage2.load_raw
"""
import numpy as np
import pandas as pd

import comun as k

mpl = k.preparar_matplotlib()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

T, D, CM = k.TINTA, k.EST["diagramas"], k.CM
PT_CM = 2.54 / 72          # 1 pt en cm
TAM_T, TAM = 7.6, 6.8      # títulos de caja y texto secundario (pt)


def lienzo(w, h):
    fig = plt.figure(figsize=(w * CM, h * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def caja(ax, x, y, w, h, titulo=None, texto=None, acento=False, fondo=None, borde=None, ha="center",
         tam_t=TAM_T, tam=TAM, color_t=None, radio=0.16, lw=None, pad=0.22, valign="center"):
    est = D["caja_acento"] if acento else D["caja"]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radio}",
                                facecolor=fondo or est["fondo"], edgecolor=borde or est["borde"],
                                linewidth=(lw or est["ancho_borde"]) * 0.75, zorder=2))
    lt = 0 if not titulo else titulo.count("\n") + 1
    ld = 0 if not texto else texto.count("\n") + 1
    alto = lt * tam_t * PT_CM * 1.22 + ld * tam * PT_CM * 1.25 + (0.12 if lt and ld else 0)
    tx = x + w / 2 if ha == "center" else x + pad
    y0 = y + h / 2 + alto / 2 if valign == "center" else y + h - pad
    if titulo:
        ax.text(tx, y0, titulo, ha=ha, va="top", fontsize=tam_t, fontweight="semibold",
                color=color_t or T["primaria"], linespacing=1.15, zorder=3)
        y0 -= lt * tam_t * PT_CM * 1.22 + 0.12
    if texto:
        ax.text(tx, y0, texto, ha=ha, va="top", fontsize=tam, color=T["secundaria"], linespacing=1.2, zorder=3)


def flecha(ax, x1, y1, x2, y2, color=None, lw=None, estilo="-|>", ms=7):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=estilo, mutation_scale=ms, shrinkA=0, shrinkB=0,
                                 color=color or D["flechas"]["color"], linewidth=(lw or D["flechas"]["ancho"]) * 0.75,
                                 zorder=4))


def contenedor(ax, x, y, w, h, titulo, subtitulo=None, acento=False):
    borde = D["caja_acento"]["borde"] if acento else D["caja"]["borde"]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.22", facecolor="white",
                                edgecolor=borde, linewidth=0.9 if acento else 0.75,
                                linestyle="solid" if acento else (0, (3, 2)), zorder=1))
    ax.text(x + 0.25, y + h - 0.22, titulo, ha="left", va="top", fontsize=7.9, fontweight="bold",
            color=k.MODELOS["C+A+R"]["color"] if acento else T["primaria"], zorder=3)
    if subtitulo:
        ax.text(x + 0.25, y + h - 0.62, subtitulo, ha="left", va="top", fontsize=6.6, color=T["atenuada"],
                style="italic", zorder=3)


# --------------------------------------------------------------------- capítulo 1
def fig_curvas_realce():
    fig, ax = plt.subplots(figsize=(13 * CM, 7.2 * CM))
    t = np.linspace(0, 6, 400)
    pre = np.linspace(-0.8, 0, 20)
    # Verde, naranja y violeta: distintos de los colores de modelo; validados para daltonismo (todos los pares)
    curvas = [
        ("Tipo I · realce persistente", 0.6 * (1 - np.exp(-t / 0.9)) + 0.07 * t, "#2F8F4E", "solid"),
        ("Tipo II · meseta", 0.9 * (1 - np.exp(-t / 0.45)), "#F28C38", (0, (5, 2))),
        ("Tipo III · lavado", 1.15 * (1 - np.exp(-t / 0.4)) * np.exp(-0.09 * t), "#8E6BD9", (0, (1.2, 1.6))),
    ]
    for nombre, y, c, ls in curvas:
        ax.plot(np.r_[pre, t], np.r_[np.zeros_like(pre), y], color=c, linewidth=1.5, linestyle=ls,
                solid_capstyle="round", zorder=3)
        ax.text(6.12, y[-1], nombre, color=T["primaria"], fontsize=7, va="center", ha="left")
        ax.plot([5.98, 6.08], [y[-1], y[-1]], color=c, linewidth=1.5, linestyle="solid")
    ax.axvline(2, **k.et.linea_referencia())
    ax.text(-0.07, 0.05, "Administración del contraste", rotation=90, ha="right", va="bottom", fontsize=6.4,
            color=T["secundaria"])
    ax.text(-0.4, 1.17, "Basal", ha="center", va="bottom", fontsize=6.8, color=T["secundaria"])
    ax.text(1.0, 1.17, "Fase inicial", ha="center", va="bottom", fontsize=6.8, color=T["secundaria"])
    ax.text(4.0, 1.17, "Fase tardía", ha="center", va="bottom", fontsize=6.8, color=T["secundaria"])
    ax.axvline(0, color=T["referencia"], linewidth=0.7, zorder=0)
    ax.set_xlim(-0.8, 6)
    ax.set_ylim(-0.02, 1.16)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("Tiempo")
    ax.set_ylabel("Intensidad de señal relativa")
    fig.subplots_adjust(right=0.7, bottom=0.12, top=0.9)
    k.guardar(fig, "FIG_CURVAS_REALCE_DCE")
    plt.close(fig)


def fig_flujo_radiomico():
    W, H = 16, 4.2
    fig, ax = lienzo(W, H)
    etapas = [
        ("Adquisición\nde imágenes", "Equipo, protocolo\ny parámetros de la\nsecuencia"),
        ("Segmentación", "Delimitación de la\nregión de interés\n(tumor o tejido\ncircundante)"),
        ("Procesamiento\nde la imagen", "Remuestreo,\ndiscretización o\nnormalización de\nintensidades"),
        ("Extracción de\ncaracterísticas", "Forma, intensidad,\ntextura y\ncomportamiento\ndel realce"),
        ("Análisis y\nmodelado", "Selección de\ninformación e\nintegración con\ndatos clínicos"),
    ]
    n, gap = len(etapas), 0.55
    w = (W - 0.3 - gap * (n - 1)) / n
    y, h = 1.05, 2.75
    for i, (tit, txt) in enumerate(etapas):
        x = 0.15 + i * (w + gap)
        caja(ax, x, y, w, h, tit, txt, acento=(i == 3))
        ax.text(x + w / 2, y + h + 0.1, str(i + 1), ha="center", va="bottom", fontsize=7, color=T["atenuada"],
                fontweight="semibold")
        if i < n - 1:
            flecha(ax, x + w + 0.07, y + h / 2, x + w + gap - 0.07, y + h / 2)
    ax.plot([0.15, W - 0.15], [0.68, 0.68], color=T["grilla"], linewidth=0.8)
    ax.text(W / 2, 0.45, "Las decisiones de cada etapa pueden modificar los valores de las características extraídas.",
            ha="center", va="top", fontsize=6.7, color=T["secundaria"], style="italic")
    k.guardar(fig, "FIG_FLUJO_RADIOMICO")
    plt.close(fig)


# --------------------------------------------------------------------- capítulo 2
def fig_diseno_estudio():
    W, H = 16, 8.6
    fig, ax = lienzo(W, H)
    contenedor(ax, 0.1, 0.1, 5.6, H - 0.2, "Colección Duke-Breast-Cancer-MRI",
               "Provisto por los autores de la colección (TCIA)")
    izq = [("RM dinámica con contraste\npretratamiento", "922 pacientes"),
           ("Segmentación tumoral y\nextracción de características", "529 características radiómicas\n(Imaging_Features.xlsx)"),
           ("Información clínica, anatomo-\npatológica y de la RM", "Variables basales, de\ntratamiento y de respuesta\n(Clinical_and_Other_Features.xlsx)")]
    hb, yb = 1.95, H - 1.15
    for i, (tit, txt) in enumerate(izq):
        yy = yb - (i + 1) * hb - i * 0.35
        caja(ax, 0.4, yy, 5.0, hb, tit, txt)
    contenedor(ax, 6.3, 0.1, W - 6.4, H - 0.2, "Este trabajo", acento=True)
    der = [("Construcción de la cohorte", "Criterios de inclusión y variable objetivo:\nrespuesta completa o casi completa"),
           ("Organización de predictores pretratamiento", "C clínico-anatomopatológico · M hallazgos cualitativos de RM\nA contexto técnico de adquisición · R radiómica"),
           ("Modelado", "Principal: C, C+R y C+A+R con regresión logística\nExploratorio: otras combinaciones de bloques,\nRandom Forest y Gradient Boosting"),
           ("Validación y evaluación", "Validación cruzada anidada repetida\nDiscriminación, calibración y clasificación")]
    xb, wb, hd, g = 6.65, W - 7.0, 1.62, 0.3
    ytop = H - 0.75
    for i, (tit, txt) in enumerate(der):
        yy = ytop - (i + 1) * hd - i * g
        caja(ax, xb, yy, wb, hd, tit, txt, acento=(i == 2), ha="left", pad=0.3)
        if i < len(der) - 1:
            flecha(ax, xb + wb / 2, yy - 0.03, xb + wb / 2, yy - g + 0.03)
    flecha(ax, 5.45, H / 2, 6.6, H / 2, color=k.MODELOS["C+A+R"]["color"])
    k.guardar(fig, "FIG_DISENO_ESTUDIO")
    plt.close(fig)


def fig_preparacion_r():
    W, H = 16, 7.0
    fig, ax = lienzo(W, H)
    # fila global
    ax.text(0.15, H - 0.12, "Antes de la validación (cohorte completa, sin utilizar la variable objetivo)", ha="left",
            va="top", fontsize=7.2, fontweight="semibold", color=T["primaria"])
    yg, hg = H - 1.75, 1.2
    glob = [("529 características", "preextraídas por Duke"), ("Eliminación de duplicados", "exactos (20 columnas)"),
            ("509 características", "bloque R de partida")]
    wg = 4.4
    xs = [0.15, (W - wg) / 2, W - 0.15 - wg]
    for i, (tit, txt) in enumerate(glob):
        caja(ax, xs[i], yg, wg, hg, tit, txt)
        if i < 2:
            flecha(ax, xs[i] + wg + 0.08, yg + hg / 2, xs[i + 1] - 0.08, yg + hg / 2)
    flecha(ax, xs[2] + wg / 2, yg - 0.05, xs[2] + wg / 2, yg - 0.62)
    # bloque dentro del fold
    yc, hc = 0.1, yg - 0.75
    contenedor(ax, 0.1, yc, W - 0.2, hc, "Dentro de cada partición de entrenamiento",
               "Parámetros estimados solo con los datos de entrenamiento; se aplican luego a la partición de evaluación")
    pasos = [("Imputación", "mediana"), ("Filtrado", "varianza nula y\ncorrelación |r| > 0,95\n(Pearson)"),
             ("Estandarización", "z-score"), ("PCA", "componentes\nevaluados: 1, 2, 3,\n5, 10, 15, 20 o 30"),
             ("Regresión logística L2", "componentes de R\njunto con C (o C+A)")]
    n, gap = len(pasos), 0.42
    wp = (W - 0.7 - gap * (n - 1)) / n
    yp, hp = 0.95, 2.35
    for i, (tit, txt) in enumerate(pasos):
        x = 0.35 + i * (wp + gap)
        caja(ax, x, yp, wp, hp, tit, txt, acento=(i == 3))
        if i < n - 1:
            flecha(ax, x + wp + 0.06, yp + hp / 2, x + wp + gap - 0.06, yp + hp / 2)
    ax.text(W / 2, 0.55, "El número de componentes y el parámetro de regularización de la regresión logística se seleccionan "
            "conjuntamente en la validación interna (ROC-AUC).", ha="center", va="center", fontsize=6.6, color=T["secundaria"],
            style="italic")
    k.guardar(fig, "FIG_PREPARACION_R")
    plt.close(fig)


def _grilla(ax, x0, y0, wc, hc, n, colores_diag, color_resto, etiqueta_filas=None):
    for r in range(n):
        for c in range(n):
            col = colores_diag if c == r else color_resto
            ax.add_patch(Rectangle((x0 + c * wc, y0 + (n - 1 - r) * hc), wc - 0.05, hc - 0.07, facecolor=col,
                                   edgecolor="white", linewidth=0.5, zorder=2))
        if etiqueta_filas:
            ax.text(x0 - 0.12, y0 + (n - 1 - r) * hc + (hc - 0.07) / 2, f"{etiqueta_filas}{r + 1}", ha="right",
                    va="center", fontsize=6.3, color=T["secundaria"])


def fig_cv_anidada():
    W, H = 16, 8.4
    fig, ax = lienzo(W, H)
    col_ent, col_val, col_prueba = (k.COMP["diagrama"]["entrenamiento"], k.COMP["diagrama"]["validacion_interna"],
                                   k.COMP["diagrama"]["prueba_externa"])
    contenedor(ax, 0.1, 2.05, W - 0.2, H - 2.15, "Una repetición (r = 1, …, 10; semillas 42 a 51)",
               "Particiones estratificadas por clase de respuesta")
    # ciclo externo
    ax.text(0.55, H - 1.25, "Ciclo externo · 5 particiones", ha="left", va="top", fontsize=7.2, fontweight="semibold")
    wc, hc = 0.78, 0.62
    xe, ye = 0.95, 3.21
    _grilla(ax, xe, ye, wc, hc, 5, col_prueba, col_ent, etiqueta_filas="Iter. ")
    ax.text(xe + 2.5 * wc, ye - 0.15, "Cada paciente se evalúa una vez\npor repetición", ha="center", va="top",
            fontsize=6.3, color=T["secundaria"])
    # ciclo interno (sobre el entrenamiento externo de una iteración)
    xi, yi, wci, hci = 6.55, 3.75, 0.6, 0.5
    ax.text(xi - 0.3, H - 1.25, "Ciclo interno · 5 particiones", ha="left", va="top", fontsize=7.2,
            fontweight="semibold")
    ax.text(xi - 0.3, H - 1.62, "sobre el entrenamiento externo", ha="left", va="top", fontsize=6.3,
            color=T["secundaria"])
    _grilla(ax, xi, yi, wci, hci, 5, col_val, col_ent)
    flecha(ax, xe + 4 * wc + 0.05, ye + 4 * hc + hc / 2 - 0.03, xi - 0.12, yi + 4 * hci + hci / 2 - 0.03,
           estilo="-|>")
    ax.text(xi + 2.5 * wci, yi - 0.15, "Selección de hiperparámetros:\nmayor ROC-AUC medio interno",
            ha="center", va="top", fontsize=6.3, color=T["secundaria"])
    # pasos a la derecha
    xp, wp = 10.35, W - 10.75
    pasos = [("Ajuste del pipeline completo", "Preprocesamiento + modelo con los\nhiperparámetros elegidos, sobre todo\nel entrenamiento externo"),
             ("Predicción de la partición de prueba", "Probabilidades fuera de pliegue (OOF)\npara los pacientes no utilizados")]
    hp = 1.6
    y1 = H - 1.3 - hp
    caja(ax, xp, y1, wp, hp, *pasos[0], ha="left", pad=0.28)
    y2 = y1 - 0.45 - hp
    caja(ax, xp, y2, wp, hp, *pasos[1], acento=True, ha="left", pad=0.28)
    flecha(ax, xp + wp / 2, y1 - 0.04, xp + wp / 2, y2 + hp + 0.04)
    flecha(ax, xi + 5 * wci + 0.12, yi + 4.5 * hci - 0.03, xp - 0.1, yi + 4.5 * hci - 0.03)
    # leyenda
    for j, (col, lab) in enumerate(((col_ent, "Entrenamiento"), (col_val, "Validación interna"),
                                    (col_prueba, "Prueba (ciclo externo)"))):
        lx, ly = xp + 0.05, 2.95 - j * 0.32
        ax.add_patch(Rectangle((lx, ly), 0.32, 0.22, facecolor=col, edgecolor=D["caja"]["borde"], linewidth=0.4))
        ax.text(lx + 0.45, ly + 0.11, lab, ha="left", va="center", fontsize=6.4, color=T["secundaria"])
    # agregación
    pasos_ag = [("× 10 repeticiones", "10 probabilidades OOF\npor paciente"), ("Promedio por paciente", "una probabilidad\npor paciente y modelo"),
                ("Métricas (N = 281)", "ROC-AUC y AP"), ("Comparación pareada", "bootstrap de las diferencias\nfrente a C (5000 remuestreos)")]
    n, gap = len(pasos_ag), 0.45
    wa = (W - 0.2 - gap * (n - 1)) / n
    for i, (tit, txt) in enumerate(pasos_ag):
        x = 0.1 + i * (wa + gap)
        caja(ax, x, 0.1, wa, 1.45, tit, txt, acento=(i == 1))
        if i < n - 1:
            flecha(ax, x + wa + 0.06, 0.82, x + wa + gap - 0.06, 0.82)
    flecha(ax, 0.1 + wa / 2, 2.0, 0.1 + wa / 2, 1.6)
    k.guardar(fig, "FIG_CV_ANIDADA")
    plt.close(fig)


# --------------------------------------------------------------------- capítulo 3
def datos_flujo_cohorte():
    import sys
    sys.path.insert(0, str(k.PIPE))
    import run_stage2 as s
    clin, _ = s.load_raw()
    neo = s._to_num(clin["Received Neoadjuvant Therapy or Not"])
    qt = s._to_num(clin["Neoadjuvant Chemotherapy"])
    tgt = s._to_num(clin[s.TARGET_COL])
    endo = s._to_num(clin["Neoadjuvant Endocrine Therapy Medications "])
    her2 = s._to_num(clin["Neoadjuvant Anti-Her2 Neu Therapy"])
    sin_qt = (neo == 1) & (qt != 1)
    con_qt = (neo == 1) & (qt == 1)
    d = {"total": len(clin), "neo": int((neo == 1).sum()), "no_neo": int((neo == 2).sum()),
         "neo_sin_dato": int(neo.isna().sum()), "qt": int(con_qt.sum()), "sin_qt": int(sin_qt.sum()),
         "solo_endocrina": int((sin_qt & (endo == 1) & (her2 != 1)).sum()),
         "solo_antiher2": int((sin_qt & (her2 == 1) & (endo != 1)).sum()),
         "codigo3": int((con_qt & (tgt == 3)).sum()), "final": int((con_qt & tgt.isin([0, 1, 2])).sum()),
         "clase1": int((con_qt & tgt.isin([1, 2])).sum()), "clase0": int((con_qt & (tgt == 0)).sum())}
    esperado = {"total": 922, "neo": 312, "no_neo": 581, "neo_sin_dato": 29, "qt": 292, "sin_qt": 20,
                "solo_endocrina": 19, "solo_antiher2": 1, "codigo3": 11, "final": 281, "clase1": 93, "clase0": 188}
    assert d == esperado, d
    pd.Series(d).to_csv(k.DIR_DATOS / "FIG_FLUJO_COHORTE.csv", header=["n"])
    return d


def fig_flujo_cohorte():
    """Figura 3.1: barras proporcionales a las pacientes incluidas en cada paso; las exclusiones salen
    con una flecha hacia una columna separada a la derecha (como en un diagrama de flujo)."""
    d = datos_flujo_cohorte()
    W = 16
    sec = k.COMP["secuencial_petroleo"]
    x0, ancho, hb, paso = 4.6, 6.0, 0.5, 1.45
    xe = x0 + ancho + 0.9                    # columna de exclusiones
    esc = ancho / d["total"]
    H = 0.9 + 3 * paso + hb + 1.0
    fig, ax = lienzo(W, H)
    ys = [H - 0.9 - hb / 2 - i * paso for i in range(4)]   # centro de cada barra

    def rect(x, y, w, h, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", facecolor=color,
                                    edgecolor="none", zorder=2))

    ax.text(x0, H - 0.25, "Pacientes incluidas", ha="left", va="top", fontsize=TAM_T, fontweight="bold",
            color=T["primaria"])
    ax.text(xe, H - 0.25, "Pacientes excluidas", ha="left", va="top", fontsize=TAM_T, fontweight="bold",
            color=T["primaria"])
    ax.plot([x0, W - 0.1], [H - 0.72] * 2, color=k.TINTA["grilla"], linewidth=0.6)

    rotulos = [(d["total"], "Colección Duke-Breast-Cancer-MRI"), (d["neo"], "con tratamiento neoadyuvante"),
               (d["qt"], "con quimioterapia neoadyuvante"), (d["final"], "cohorte analítica")]
    for i, ((n, txt), yc) in enumerate(zip(rotulos, ys)):
        final = i == 3
        if final:
            rect(x0, yc - hb / 2, d["clase1"] * esc - 0.02, hb, sec[5])
            rect(x0 + d["clase1"] * esc + 0.02, yc - hb / 2, d["clase0"] * esc - 0.02, hb, sec[3])
            for xc, m in ((x0 + d["clase1"] * esc / 2, d["clase1"]), (x0 + (d["clase1"] + d["clase0"] / 2) * esc, d["clase0"])):
                ax.text(xc, yc, str(m), ha="center", va="center", fontsize=TAM, fontweight="semibold", color="white",
                        zorder=3)
        else:
            rect(x0, yc - hb / 2, n * esc, hb, sec[2])
        ax.text(x0 - 0.3, yc + 0.13, f"{n} pacientes", ha="right", va="center", fontsize=TAM_T,
                fontweight="bold" if final else "semibold",
                color=k.MODELOS["C+A+R"]["color"] if final else T["primaria"])
        ax.text(x0 - 0.3, yc - 0.2, txt, ha="right", va="center", fontsize=TAM - 0.4, color=T["secundaria"])

    # eje del flujo: línea vertical a la izquierda de las barras, con las salidas hacia las exclusiones
    xs = x0 - 0.12
    ax.plot([xs, xs], [ys[0], ys[3]], color=D["flechas"]["color"], linewidth=D["flechas"]["ancho"] * 0.75, zorder=1)
    exclus = [(d["total"] - d["neo"], f"{d['no_neo']} sin tratamiento neoadyuvante\n{d['neo_sin_dato']} sin dato sobre el tratamiento"),
              (d["sin_qt"], f"Sin quimioterapia: {d['solo_endocrina']} solo terapia\nendocrina y {d['solo_antiher2']} solo anti-HER2"),
              (d["codigo3"], "Sin evaluación anatomopatológica\nde la respuesta (código 3)")]
    for (n, txt), ya, yb in zip(exclus, ys[:-1], ys[1:]):
        ym = (ya + yb) / 2
        flecha(ax, xs, ym, xe - 0.15, ym)
        ax.text(xe, ym + 0.05, f"−{n}", ha="left", va="bottom", fontsize=TAM_T, fontweight="semibold",
                color=T["primaria"])
        ax.text(xe, ym - 0.02, txt, ha="left", va="top", fontsize=TAM - 0.4, color=T["secundaria"], linespacing=1.2)

    yl = ys[3] - hb / 2 - 0.5
    for i, (c, txt) in enumerate([(sec[5], f"Clase 1 · respuesta completa o casi completa ({k.pct(d['clase1'] / d['final'])})"),
                                  (sec[3], f"Clase 0 · sin respuesta completa ni casi completa ({k.pct(d['clase0'] / d['final'])})")]):
        xl = x0 + i * 5.6
        rect(xl, yl - 0.13, 0.26, 0.26, c)
        ax.text(xl + 0.4, yl, txt, ha="left", va="center", fontsize=TAM - 0.4, color=T["secundaria"])
    k.guardar(fig, "FIG_FLUJO_COHORTE")
    plt.close(fig)


if __name__ == "__main__":
    fig_curvas_realce()
    fig_flujo_radiomico()
    fig_diseno_estudio()
    fig_preparacion_r()
    fig_cv_anidada()
    fig_flujo_cohorte()
    print("Diagramas generados en", k.DIR_FIG)
