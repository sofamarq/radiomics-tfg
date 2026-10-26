# Evaluación del valor incremental de características radiómicas derivadas de resonancia magnética en la predicción de respuesta a quimioterapia neoadyuvante en cáncer de mama

- **Autora:** Sofia Marquevich
- **Carrera:** Ingeniería Biomédica — Trabajo Final de Grado
- **Universidad:** Universidad Austral
- **Año:** 2026

Código y resultados que evalúan si las características radiómicas de la resonancia magnética dinámica con contraste (DCE-MRI) pretratamiento, combinadas con información clínica y de adquisición, mejoran la predicción de la respuesta completa o casi completa a la quimioterapia neoadyuvante respecto de un modelo solo clínico.

## Desenlace, variable objetivo y cohorte

- **Desenlace (Target):** respuesta completa o casi completa a la quimioterapia neoadyuvante.
- **Variable objetivo:** binarización de *Overall Near-complete Response: Stricter Definition* de la colección Duke. Clase 1 = código 1 (respuesta completa) o código 2 (respuesta casi completa); clase 0 = código 0; el código 3 (sin evaluación anatomopatológica) se excluye. La clase 1 no equivale a respuesta patológica completa.
- **Cohorte:** 922 pacientes → 312 con tratamiento neoadyuvante → 292 con quimioterapia neoadyuvante → **281** con desenlace evaluable (93 de clase 1 y 188 de clase 0).
- **Bloques de información:** C (clínico-patológico basal, 14 variables), A (adquisición de la RM, 14 variables) y R (509 características radiómicas tras eliminar 20 columnas duplicadas). Modelos principales: C, C+R y C+A+R.

## Datos

Colección *Duke-Breast-Cancer-MRI* de The Cancer Imaging Archive (TCIA). Los dos archivos de entrada se incluyen sin modificaciones en `datos/`, bajo licencia CC BY-NC 4.0 y sujetos a la política de uso de TCIA; las citas obligatorias están en [`datos/LEEME.md`](datos/LEEME.md). La segmentación y la extracción de las características radiómicas fueron realizadas por los autores de la colección.

## Estructura

```
├── ejecutar_pipeline.py          punto de entrada único
├── requirements.txt
├── datos/                        Excel oficiales de Duke y aviso de licencia (LEEME.md)
├── pipeline_v2/                  análisis principal (scripts) y resultados finales (CSV/JSON)
├── analisis_complementarios/     cohorte descriptiva, PCA descriptivo, PCA exploratorio 5×3, subgrupos
│   └── resultados/
├── notebook/                     TFG_Radiomics_Final.ipynb (alternativa de ejecución del análisis principal)
│                                 TFG_Radiomics_EDA.ipynb (análisis exploratorio de los datos)
└── elementos_visuales_finales/
    ├── figuras/                  figuras finales (PNG, 300 ppp)
    ├── tablas/  anexos/          tablas finales (CSV)
    ├── estilo/                   identidad visual y tipografía Source Sans 3 (con su licencia OFL)
    └── fuentes/                  código de figuras y tablas (codigo/) y datos de cada figura (datos/)
```

| Script | Contenido | Salidas |
|---|---|---|
| `pipeline_v2/run_stage2.py` | Cohorte, variable objetivo, bloques y preprocesamiento; CV anidada única 5×5 y hold-out 80/20 de las 8 combinaciones de bloques | `results_stage2.csv` |
| `pipeline_v2/run_stage3.py` | CV anidada repetida 10×5×5 de C, C+R y C+A+R; probabilidades fuera de pliegue promediadas por paciente; bootstrap pareado | `stage3_*.csv` |
| `pipeline_v2/stage4light_holdout.py`, `run_stage4_light.py` | Random Forest y Gradient Boosting (hold-out y CV anidada única 5×3); grillas en `run_stage4.py` | `stage4light_*.csv` |
| `pipeline_v2/run_stage5.py` | Calibración, matrices de confusión y umbrales 0,50 / 0,40 / 0,35 | `stage5_*.csv` |
| `analisis_complementarios/*.py` | Tabla descriptiva de la cohorte, PCA descriptivo global, comparación exploratoria del número de componentes y análisis post hoc luminal / no luminal | `analisis_complementarios/resultados/` |
| `elementos_visuales_finales/fuentes/codigo/generar_todo.py` | Figuras y tablas finales | `figuras/`, `tablas/`, `anexos/` |

`run_stage5.py` también genera salidas que no se informan en la tesis (curva de decisión, coeficientes, cargas del PCA y figuras auxiliares en `pipeline_v2/figs_etapa5/`, excluida del control de versiones).

## Instalación

Requiere Python 3.11 (resultados obtenidos con Python 3.11.2).

```bash
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Entradas

`datos/Clinical_and_Other_Features.xlsx` y `datos/Imaging_Features.xlsx`. Para usar otra carpeta: `export RADIOMICS_DATOS=/ruta/a/la/carpeta`.

## Ejecución

```bash
python ejecutar_pipeline.py                  # análisis completo + figuras y tablas (≈ 30 min)
python ejecutar_pipeline.py --solo-figuras   # solo figuras y tablas, desde los resultados incluidos (< 1 min)
python ejecutar_pipeline.py --desde etapa3   # retomar desde una etapa
```

La ejecución completa sobrescribe los resultados incluidos en `pipeline_v2/`, `analisis_complementarios/resultados/` y `elementos_visuales_finales/`.

Por defecto las figuras se guardan en PNG y las tablas en CSV. Los formatos adicionales ya implementados se activan con variables de entorno, por ejemplo `FIGURAS_FORMATOS=png,pdf,svg` o `TABLAS_FORMATOS=csv,docx,png` (DOCX requiere `python-docx`). La tipografía Source Sans 3 se registra automáticamente desde `elementos_visuales_finales/estilo/tipografia/`.

### Notebook

`notebook/TFG_Radiomics_Final.ipynb` es una **alternativa de ejecución** del análisis principal, mencionada en la tesis: reproduce de forma autocontenida, desde los dos Excel, la cohorte, el preprocesamiento, el hold-out, la CV anidada repetida 10×5×5, el bootstrap, la calibración y Random Forest / Gradient Boosting, y compara su resultado con `pipeline_v2/stage3_*.csv`. No incluye los análisis complementarios ni genera las figuras y tablas finales. Se incluye con sus salidas; no es necesario ejecutarlo además de `ejecutar_pipeline.py`. Para reejecutarlo: `python ejecutar_pipeline.py --solo-figuras --notebook` (≈ 7 min).

### Análisis exploratorio

`notebook/TFG_Radiomics_EDA.ipynb` es complementario y descriptivo; no es necesario ejecutarlo para correr el pipeline principal. Usa las funciones del pipeline (`pipeline_v2/run_stage2.py`) para la lectura de los Excel, la selección de pacientes y los bloques, por lo que sus cifras coinciden con las del análisis principal. Contiene: estructura de los archivos y armonización de encabezados; integración y controles de calidad; selección de la cohorte y distribución de la variable objetivo; caracterización clínica; datos faltantes, códigos especiales y su tratamiento; reconstrucciones y uso de las columnas clínicas; y caracterización descriptiva del bloque radiómico (duplicados, faltantes y correlaciones). No entrena modelos. Se incluye ejecutado, con sus tablas y gráficos; para reejecutarlo (junto con las figuras y tablas, < 2 min): `python ejecutar_pipeline.py --solo-figuras --eda`. Si se abre en Jupyter, el directorio de trabajo debe ser la raíz del repositorio.

## Salidas: figuras y tablas de la tesis

| Elemento | Archivo |
|---|---|
| Figuras 1.1, 1.2 | `FIG_CURVAS_REALCE_DCE`, `FIG_FLUJO_RADIOMICO` |
| Figuras 2.1–2.3 | `FIG_DISENO_ESTUDIO`, `FIG_PREPARACION_R`, `FIG_CV_ANIDADA` |
| Figuras 3.1–3.5 | `FIG_FLUJO_COHORTE`, `FIG_PCA_VARIANZA`, `FIG_ROC_PR_MODELOS`, `FIG_DIFERENCIAS_IC`, `FIG_CALIBRACION` |
| Figura A.3.1 | `FIG_SELECCION_COMPONENTES` |
| Tablas 1.1, 1.2 | `TAB_SUBTIPOS_MOLECULARES`, `TAB_ESTADO_ARTE` |
| Tabla 2.1 | `TAB_VARIABLE_OBJETIVO` |
| Tablas 3.1–3.9 | `TAB_CARACTERISTICAS_COHORTE`, `TAB_BLOQUES_INFORMACION`, `TAB_PCA_COMPARACION`, `TAB_COMBINACIONES_BLOQUES`, `TAB_MODELOS_PRINCIPALES`, `TAB_RF_GB`, `TAB_MATRICES_CONFUSION`, `TAB_UMBRALES`, `TAB_SUBGRUPOS` |
| Tablas de anexos | `anexos/TAB_A_*` (variables por bloque, cohorte completa, configuración de modelos, hiperparámetros por partición, software, clasificación con umbral 0,50 y hold-out) |

## Reproducibilidad

- **Semillas:** 42 para la partición hold-out, los pliegues de la CV anidada única y los estimadores; 42–51 para las diez repeticiones de la CV anidada 10×5×5; 20260909 para el bootstrap pareado (5000 remuestreos).
- **Sin fuga de información:** imputación, codificación, filtro de correlación (|r| > 0,95), estandarización y PCA se ajustan dentro de cada partición de entrenamiento, como pasos de un `Pipeline` de scikit-learn.
- **Hiperparámetros:** C ∈ {0,01; 0,1; 1; 10} y número de componentes ∈ {1, 2, 3, 5, 10, 15, 20, 30}, seleccionados por ROC-AUC en la validación interna.
- **Controles automáticos:** los scripts verifican N = 281 y la distribución 93/188; el código de las tablas compara las cifras generadas con las de la tesis y se detiene si no coinciden.
- Con las versiones de `requirements.txt`, una ejecución completa desde los Excel reprodujo exactamente todos los resultados incluidos. Otras versiones de scikit-learn o NumPy pueden producir diferencias menores.

## Limitaciones de uso

Trabajo de investigación académica. Los modelos se desarrollaron y validaron internamente en una única cohorte retrospectiva de un solo centro, sin validación externa. No constituyen una herramienta de apoyo a la decisión clínica ni tienen aplicación clínica validada.
