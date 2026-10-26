"""Punto de entrada único: ejecuta, en orden, las etapas del análisis final y regenera figuras y tablas.

Uso (desde la raíz del repositorio):
    python ejecutar_pipeline.py                      # análisis completo + figuras y tablas
    python ejecutar_pipeline.py --solo-figuras       # solo figuras y tablas, a partir de los resultados guardados
    python ejecutar_pipeline.py --desde etapa3       # retoma desde una etapa
    python ejecutar_pipeline.py --notebook           # además, ejecuta notebook/TFG_Radiomics_Final.ipynb
    python ejecutar_pipeline.py --solo-figuras --eda # figuras y tablas + notebook/TFG_Radiomics_EDA.ipynb

Las etapas no modifican métodos ni parámetros: cada una es un script del proyecto que se ejecuta tal cual.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
PIPE = RAIZ / "pipeline_v2"
COMPL = RAIZ / "analisis_complementarios"
VIS = RAIZ / "elementos_visuales_finales" / "fuentes" / "codigo"

# (nombre, script, carpeta de trabajo, descripción)
ETAPAS = [
    ("etapa2", PIPE / "run_stage2.py", PIPE,
     "Cohorte, bloques y preprocesamiento; CV anidada única 5×5 y hold-out 80/20 (8 combinaciones de bloques)"),
    ("etapa3", PIPE / "run_stage3.py", PIPE,
     "CV anidada repetida 10×5×5 (semillas 42–51) y bootstrap pareado (5000, semilla 20260909)"),
    ("rfgb_holdout", PIPE / "stage4light_holdout.py", PIPE, "Random Forest y Gradient Boosting: hold-out 80/20"),
    ("rfgb_cv", PIPE / "run_stage4_light.py", PIPE, "Random Forest y Gradient Boosting: CV anidada única 5×3"),
    ("etapa5", PIPE / "run_stage5.py", PIPE, "Calibración y clasificación con umbrales 0,50 / 0,40 / 0,35"),
    ("cohorte", COMPL / "caracterizacion_cohorte.py", COMPL, "Tabla descriptiva de la cohorte"),
    ("pca_descriptivo", COMPL / "pca_descriptivo.py", COMPL, "PCA descriptivo global del bloque R"),
    ("pca_exploratorio", COMPL / "pca_exploratorio_5x3.py", COMPL,
     "Comparación exploratoria del número de componentes (CV anidada única 5×3)"),
    ("subgrupos", COMPL / "subgrupos.py", COMPL, "Análisis exploratorio post hoc luminal / no luminal"),
    ("figuras_tablas", VIS / "generar_todo.py", VIS, "Figuras y tablas finales"),
]


def ejecutar(nombre, script, cwd, desc):
    print(f"\n=== [{nombre}] {desc}", flush=True)
    t0 = time.time()
    subprocess.run([sys.executable, str(script)], cwd=cwd, check=True)
    print(f"=== [{nombre}] listo en {(time.time() - t0) / 60:.1f} min", flush=True)


def ejecutar_notebook(nombre="TFG_Radiomics_Final.ipynb"):
    import nbformat
    from nbclient import NotebookClient
    ruta = RAIZ / "notebook" / nombre
    nb = nbformat.read(ruta, as_version=4)
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(RAIZ)}}).execute()
    nbformat.write(nb, ruta)
    print(f"=== [notebook] {nombre} ejecutado y guardado con sus salidas", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    nombres = [e[0] for e in ETAPAS]
    ap.add_argument("--desde", choices=nombres, help="etapa desde la cual retomar")
    ap.add_argument("--solo-figuras", action="store_true", help="solo figuras y tablas, desde los resultados guardados")
    ap.add_argument("--notebook", action="store_true", help="ejecutar también el notebook consolidado")
    ap.add_argument("--eda", action="store_true", help="ejecutar también el notebook de análisis exploratorio")
    a = ap.parse_args()
    inicio = "figuras_tablas" if a.solo_figuras else (a.desde or nombres[0])
    t0 = time.time()
    for e in ETAPAS[nombres.index(inicio):]:
        ejecutar(*e)
    if a.notebook:
        ejecutar_notebook()
    if a.eda:
        ejecutar_notebook("TFG_Radiomics_EDA.ipynb")
    print(f"\nPipeline completo en {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
