"""Regenera todas las figuras y tablas definitivas. Uso: python generar_todo.py (desde esta carpeta) o python ejecutar_pipeline.py --solo-figuras (desde la raíz).

Solo lee resultados ya calculados del repositorio; no entrena modelos (tiempo total < 1 min).
"""
import runpy
from pathlib import Path

aqui = Path(__file__).resolve().parent
for script in ("figuras_diagramas.py", "figuras_resultados.py", "tablas.py"):
    runpy.run_path(str(aqui / script), run_name="__main__")
