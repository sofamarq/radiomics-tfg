"""Identidad visual de la tesis para Python: carga estilo_tesis.json y lo aplica a matplotlib.

Uso:
    import estilo_tesis as et
    et.aplicar_matplotlib()
    ax.plot(x, y, **et.linea_modelo("C+A+R"), label=et.etiqueta("C+A+R", "ROC-AUC", 0.761))
    fig.savefig(..., **et.GUARDADO)
"""
import json
from pathlib import Path

EST = json.loads((Path(__file__).resolve().parent / "estilo_tesis.json").read_text(encoding="utf-8"))
MODELOS, ORDEN = EST["modelos"], EST["orden_modelos"]
TINTA, SUP, COMP = EST["tinta"], EST["superficies"], EST["complementarios"]
CM = 1 / 2.54
GUARDADO = {"dpi": EST["dimensiones_A4_cm"]["dpi_exportacion"], "bbox_inches": "tight", "facecolor": SUP["fondo"]}


def num(x, d=None, signo=False):
    """Número con coma decimal y signo menos tipográfico."""
    d = EST["formato_numerico"]["decimales_metricas"] if d is None else d
    s = f"{abs(x):.{d}f}".replace(".", ",")
    return ("−" if x < 0 else "+" if signo else "") + s


def etiqueta(m, metrica, valor):
    return f"{m} ({metrica} {num(valor)})"


def _ls(v):
    return v if isinstance(v, str) else (v[0], tuple(v[1]))


def linea_modelo(m):
    e = MODELOS[m]
    return {"color": e["color"], "linewidth": e["ancho_linea"] * 0.7, "linestyle": _ls(e["mpl_linestyle"]),
            "zorder": 3 + ORDEN.index(m)}


def linea_referencia():
    r = EST["lineas_referencia"]
    return {"color": r["color"], "linewidth": r["ancho"] * 0.7, "linestyle": _ls(r["mpl_linestyle"]), "zorder": 1}


def aplicar_matplotlib():
    import matplotlib as mpl
    from matplotlib import font_manager

    t = EST["tipografia"]
    instaladas = {f.name for f in font_manager.fontManager.ttflist}
    familia = [f for f in [t["familia"], *t["familia_respaldo"]] if f in instaladas] or ["DejaVu Sans"]
    pt = t["tamanos_pt_matplotlib"]
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": familia,
        "font.size": pt["valores_ejes"], "axes.labelsize": pt["titulos_ejes"], "xtick.labelsize": pt["valores_ejes"],
        "ytick.labelsize": pt["valores_ejes"], "legend.fontsize": pt["leyendas"], "axes.labelcolor": TINTA["primaria"],
        "text.color": TINTA["primaria"], "xtick.color": TINTA["secundaria"], "ytick.color": TINTA["secundaria"],
        "axes.edgecolor": EST["ejes"]["color"], "axes.linewidth": EST["ejes"]["ancho"] * 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": EST["grilla"]["color"], "grid.linewidth": EST["grilla"]["ancho"] * 0.7,
        "axes.facecolor": SUP["fondo"], "figure.facecolor": SUP["fondo"], "savefig.facecolor": SUP["fondo"],
        "legend.frameon": True, "legend.edgecolor": EST["leyenda"]["borde"], "legend.framealpha": EST["leyenda"]["opacidad_fondo"],
        "legend.fancybox": False, "axes.formatter.use_locale": False, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.prop_cycle": mpl.cycler(color=[MODELOS[m]["color"] for m in ORDEN]),
    })
    return familia[0]


def formato_coma(ax, d=1):
    """Marcas de ambos ejes con coma decimal."""
    from matplotlib.ticker import FuncFormatter
    f = FuncFormatter(lambda v, _: num(v, d))
    ax.xaxis.set_major_formatter(f)
    ax.yaxis.set_major_formatter(f)
