"""Tablas del cuerpo (capítulos 1–3) y de los anexos (CSV por defecto; .docx e imagen opcionales).

Las cifras se leen de los resultados del repositorio (solo lectura) y se controlan contra los valores
citados en la tesis.
"""
import sys

import pandas as pd

import comun as k

sys.path.insert(0, str(k.PIPE))
num, pct, ic = k.num, k.pct, k.ic


def n_pct(n, total, d=1):
    return f"{n} ({pct(n / total, d)})"


# ------------------------------------------------------------------ capítulo 1
def tab_subtipos():
    df = pd.DataFrame([
        ["Luminal (RH+/HER2−)", "RE y/o RP positivos; HER2 negativo",
         "Grupo más frecuente. Incluye luminal A (menor proliferación, comportamiento más favorable) y "
         "luminal B (mayor proliferación); su delimitación varía según los criterios utilizados",
         "Menor frecuencia de respuesta patológica completa; mayor en tumores de alto grado"],
        ["HER2 positivo", "Sobreexpresión de HER2 o amplificación de su gen; RE y RP positivos o negativos",
         "Blanco de tratamientos dirigidos contra HER2",
         "Elevada sensibilidad al tratamiento, en especial cuando el esquema incluye tratamiento anti-HER2"],
        ["Triple negativo", "RE, RP y HER2 negativos", "Sin blancos para terapia endocrina ni anti-HER2",
         "Respuesta patológica completa más frecuente que en los tumores RH+/HER2−"],
    ], columns=["Subtipo", "Definición por inmunohistoquímica", "Características", "Respuesta al tratamiento neoadyuvante"])
    k.guardar_tabla(df, "TAB_SUBTIPOS_MOLECULARES", anchos_cm=[2.8, 3.7, 5.0, 4.5], tam=8.5,
                    nota="RE: receptor de estrógeno; RP: receptor de progesterona; RH: receptores hormonales. "
                         "Clasificación subrogada por inmunohistoquímica, que no reproduce exactamente los subtipos "
                         "definidos por expresión génica.")


def tab_estado_arte():
    df = pd.DataFrame([
        ["Cain et al. (2019)", "Duke; 288 con tratamiento neoadyuvante (64 con pCR)", "pCR",
         "529 características de DCE-MRI de Duke; sin variables clínicas",
         "Conjunto de prueba independiente (división 50/50)",
         "Neoadyuvancia: 0,658; quimioterapia: 0,589; TN/HER2+: 0,707"],
        ["You et al. (2024)", "Duke; 217 con quimioterapia neoadyuvante (cohorte de un estudio multicéntrico)", "pCR",
         "Puntaje radiómico de DCE-MRI y variables clínicas", "No detallada para esta cohorte",
         "Radiómico: 0,59; clínico: 0,68; combinado: 0,72"],
        ["Li et al. (2024)", "Duke; 281 con tratamiento neoadyuvante", "pCR",
         "Características extraídas de T1 precontraste y primera fase poscontraste",
         "Sobremuestreo SMOTE previo a la división 7:3", "LightGBM: 0,823"],
        ["Granzier et al. (2019)", "Revisión sistemática; 16 estudios (35 a 414 pacientes)",
         "Respuesta a la terapia sistémica neoadyuvante", "Radiómica de RM", "Variable entre estudios",
         "Sin estimación agrupada; RQS de 0 % a 41,2 %"],
        ["Kuntjoro et al. (2026)", "Metaanálisis; 11 estudios", "pCR", "Radiómica, clínica y combinación",
         "Agregación por efectos aleatorios", "Agrupado: clínico 0,71; radiómico 0,76; combinado 0,81 (I² = 79 %)"],
    ], columns=["Estudio", "Población", "Desenlace", "Predictores", "Validación", "ROC-AUC informado"])
    k.guardar_tabla(df, "TAB_ESTADO_ARTE", anchos_cm=[2.3, 3.0, 2.3, 2.9, 2.6, 2.9], tam=8,
                    nota="pCR: respuesta patológica completa; RQS: Radiomics Quality Score; TN: triple negativo. "
                         "Los valores no son directamente comparables entre sí ni con este trabajo, por diferencias "
                         "en la definición del desenlace, la cohorte, los predictores y la estrategia de validación.")


# ------------------------------------------------------------------ capítulo 2
def tab_variable_objetivo():
    import run_stage2 as s
    clin, _ = s.load_raw()
    neo = s._to_num(clin["Received Neoadjuvant Therapy or Not"])
    qt = s._to_num(clin["Neoadjuvant Chemotherapy"])
    t = s._to_num(clin[s.TARGET_COL])[(neo == 1) & (qt == 1)]
    n = t.value_counts().to_dict()
    assert {c: n.get(c, 0) for c in (0, 1, 2, 3)} == {0: 188, 1: 64, 2: 29, 3: 11} and t.isna().sum() == 0
    df = pd.DataFrame([
        ["0", "Not complete or near-complete response", "Respuesta no completa ni casi completa", 188, "Clase 0"],
        ["1", "Complete response", "Respuesta completa", 64, "Clase 1"],
        ["2", "Near-complete response", "Respuesta casi completa", 29, "Clase 1"],
        ["3", "Not available", "Evaluación anatomopatológica no disponible", 11, "Excluida"],
        ["NA", "Not applicable", "No aplica", 0, "—"],
    ], columns=["Código", "Categoría en Duke", "Significado", "Pacientes (n = 292)", "Variable objetivo"])
    k.guardar_tabla(df, "TAB_VARIABLE_OBJETIVO", alinear_der=[3], anchos_cm=[1.4, 4.4, 4.6, 2.6, 2.6], tam=8.5,
                    nota="Variable Overall Near-complete Response: Stricter Definition. La categoría casi completa "
                         "reúne la descripción explícita de respuesta casi completa, la presencia exclusiva de "
                         "carcinoma in situ (ductal o lobulillar) y reducciones tumorales del 90 %, 95 % o 99 %. "
                         "Clase 1 = respuesta completa o casi completa (93; 33,1 %); clase 0 = 188 (66,9 %). "
                         "La clase 1 no equivale a respuesta patológica completa.")


# ------------------------------------------------------------------ capítulo 3
def _cohorte():
    df = pd.read_csv(k.COMPL / "07_caracterizacion_cohorte.csv")
    df.columns = ["Variable", "Categoría", "Total (n = 281)", "Sin respuesta (n = 188)", "Con respuesta (n = 93)"]
    df = df[df.Variable != "N"].copy()
    for c in df.columns[2:]:
        df[c] = (df[c].str.replace(".", ",", regex=False).str.replace("%", " %", regex=False)
                 .str.replace(" ± ", " ± ", regex=False))
    return df


def _agrupar(df):
    """Una fila de encabezado por variable (en negrita) y sus categorías debajo."""
    filas, grupos = [], []
    for var, g in df.groupby("Variable", sort=False):
        grupos.append(len(filas))
        if len(g) == 1 or var.startswith("Edad"):
            if var.startswith("Edad"):
                filas.append([var] + [""] * 3)
                for _, r in g.iterrows():
                    filas.append(["   " + r["Categoría"]] + list(r.iloc[2:]))
            else:
                filas.append([var] + list(g.iloc[0, 2:]))
            continue
        filas.append([var] + [""] * 3)
        for _, r in g.iterrows():
            filas.append(["   " + r["Categoría"]] + list(r.iloc[2:]))
    return pd.DataFrame(filas, columns=["Característica"] + list(df.columns[2:])), grupos


def tab_cohorte():
    df = _cohorte()
    # Control contra la tesis (3.1.2)
    def val(v, c, col="Total (n = 281)"):
        return df[(df.Variable == v) & (df["Categoría"] == c)][col].iloc[0]
    assert val("Edad (años)", "media ± DE") == "48,8 ± 10,9" and val("RE", "Positivo").endswith("(58,4 %)")
    assert val("Grado nuclear", "3", "Con respuesta (n = 93)").endswith("(78,5 %)")
    # Versión del cuerpo: raza agrupada (detalle completo en el anexo)
    raza = df[df.Variable == "Raza/etnia"]
    otras = raza[~raza["Categoría"].isin(["Blanca", "Negra", "Sin dato"])]
    suma = [sum(int(x.split(" ")[0]) for x in otras[c]) for c in df.columns[2:]]
    tot = [281, 188, 93]
    fila_otras = pd.DataFrame([["Raza/etnia", "Otras"] + [n_pct(sn, t) for sn, t in zip(suma, tot)]], columns=df.columns)
    cuerpo = pd.concat([df[df.Variable != "Raza/etnia"].iloc[:0]] + [
        pd.concat([raza[raza["Categoría"].isin(["Blanca", "Negra"])], fila_otras, raza[raza["Categoría"] == "Sin dato"]])
        if v == "Raza/etnia" else df[df.Variable == v] for v in df.Variable.unique()])
    t, grupos = _agrupar(cuerpo)
    nota = ("Variables continuas: media ± DE y mediana [RIC]; categóricas: n (%). Con respuesta: respuesta completa o "
            "casi completa. RE: receptor de estrógeno; RP: receptor de progesterona; RH: receptores hormonales; "
            "IHQ: inmunohistoquímica; NC: dato no recolectado en la colección. «Otras» agrupa las categorías asiática, "
            "nativa, hispana, multirracial, hawaiana o isleña del Pacífico e indígena americana (detalle en el anexo). "
            "El estadio M se reconstruyó con la variable Metastatic at Presentation y no se utilizó como predictor. "
            "No se realizaron pruebas de hipótesis.")
    k.guardar_tabla(t, "TAB_CARACTERISTICAS_COHORTE", alinear_der=[1, 2, 3], grupos=grupos,
                    anchos_cm=[5.4, 3.5, 3.5, 3.5], tam=8, nota=nota)
    full, g2 = _agrupar(df)
    k.guardar_tabla(full, "TAB_A_CARACTERISTICAS_COHORTE_COMPLETA", destino="anexos", alinear_der=[1, 2, 3],
                    grupos=g2, anchos_cm=[5.4, 3.5, 3.5, 3.5], tam=8,
                    nota=nota.replace(" «Otras» agrupa las categorías asiática, nativa, hispana, multirracial, "
                                      "hawaiana o isleña del Pacífico e indígena americana (detalle en el anexo).", ""))


def tab_bloques():
    import run_stage2 as s
    n = {b: sum(len(v) for v in s.BLOCK_TYPES[b].values()) for b in ("C", "M", "A")}
    assert n == {"C": 14, "M": 5, "A": 14}
    df = pd.DataFrame([
        ["C · Clínico y anatomopatológico",
         "Edad, raza, estado menopáusico, RE, RP, HER2, estadio T, estadio N, tipo histológico, bilateralidad, "
         "lateralidad y grado histológico (formación tubular, pleomorfismo nuclear y recuento mitótico)",
         "14", "Modelo de referencia; incluido en todos los modelos"],
        ["M · Hallazgos cualitativos de RM",
         "Enfermedad multifocal o multicéntrica, hallazgos contralaterales, linfadenopatías, compromiso de piel o "
         "pezón y compromiso del músculo pectoral o de la pared torácica", "5",
         "Solo en la evaluación inicial de combinaciones"],
        ["A · Contexto técnico de adquisición",
         "Fabricante, modelo del equipo, intensidad de campo, TR, TE, ángulo de inclinación, espesor de corte, filas "
         "de la matriz, matriz de adquisición, campo de visión, opciones de adquisición, agente de contraste, "
         "posición de la paciente y días entre el diagnóstico y la RM", "14",
         "Contexto técnico de R en C+A+R"],
        ["R · Radiómico", "Características cuantitativas de DCE-MRI preextraídas por Duke (tumor y tejido mamario)",
         "529 → 509", "C+R y C+A+R, tras filtrado de correlación y PCA"],
    ], columns=["Bloque", "Contenido", "Variables", "Uso en el análisis"])
    k.guardar_tabla(df, "TAB_BLOQUES_INFORMACION", alinear_der=[2], anchos_cm=[3.4, 7.6, 1.7, 3.3], tam=8.5,
                    nota="TR: tiempo de repetición; TE: tiempo de eco. En R, 529 características originales y 509 tras "
                         "eliminar 20 columnas duplicadas. El listado de variables con su columna de origen y su "
                         "codificación se presenta en el anexo.")


def tab_pca_comparacion():
    ex = pd.read_csv(k.COMPL / "pca_exploratorio_5x3.csv").set_index("configuracion")
    filas = [("Selección en la validación interna", ex.loc["Grilla oficial"], "1 a 3"),
             ("2 componentes fijos", ex.loc["PCA 2 fijos"], "2"),
             ("3 componentes fijos", ex.loc["PCA 3 fijos"], "3"),
             ("19 componentes fijos (≥ 70 % de varianza)", ex.loc["PCA 19 fijos"], "19"),
             ("30 componentes fijos (≥ 80 % de varianza)", ex.loc["PCA 30 fijos"], "30")]
    datos = pd.DataFrame([{"configuracion": f, "roc_auc": r.roc_oof, "ap": r.pr_oof, "componentes": c} for f, r, c in filas])
    datos.to_csv(k.DIR_DATOS / "TAB_PCA_COMPARACION.csv", index=False)
    assert [round(v, 3) for v in datos.roc_auc] == [0.775, 0.775, 0.775, 0.717, 0.708]
    assert [round(v, 3) for v in datos.ap] == [0.587, 0.583, 0.576, 0.524, 0.524]
    df = pd.DataFrame({"Configuración del PCA": datos.configuracion, "Componentes": datos.componentes,
                       "ROC-AUC": [num(v) for v in datos.roc_auc], "AP": [num(v) for v in datos.ap]})
    k.guardar_tabla(df, "TAB_PCA_COMPARACION", alinear_der=[1, 2, 3], anchos_cm=[7.0, 2.6, 2.2, 2.2], tam=9,
                    nota="Modelo C+A+R con regresión logística. Análisis exploratorio con validación cruzada anidada "
                         "única 5 × 3 (semilla 42), con las mismas particiones para todas las configuraciones; el "
                         "parámetro de regularización se seleccionó en la validación interna (ROC-AUC). Métricas sobre "
                         "las predicciones fuera de pliegue. No es comparable con la validación anidada repetida 10 × 5 × 5. "
                         "AP: precisión promedio.")


def tab_combinaciones():
    r = pd.read_csv(k.PIPE / "results_stage2.csv")
    r = r[r.analysis == "nested_cv"].set_index("model")
    orden = ["C", "C+M", "C+A", "R", "C+R", "C+M+R", "C+A+R", "C+M+A+R"]
    r = r.loc[orden]
    r[["roc_auc", "pr_auc", "delta_roc_vs_C", "delta_pr_vs_C"]].to_csv(k.DIR_DATOS / "TAB_COMBINACIONES_BLOQUES.csv")
    assert [round(v, 3) for v in r.roc_auc] == [0.757, 0.744, 0.747, 0.522, 0.767, 0.750, 0.762, 0.740]
    d = lambda v, m: "—" if m in ("C", "R") else num(v, 3, signo=True)  # noqa: E731
    df = pd.DataFrame({"Combinación": orden, "ROC-AUC": [num(v) for v in r.roc_auc], "AP": [num(v) for v in r.pr_auc],
                       "ROC-AUC ": [d(v, m) for v, m in zip(r.delta_roc_vs_C, orden)],
                       "AP ": [d(v, m) for v, m in zip(r.delta_pr_vs_C, orden)]})
    k.guardar_tabla(df, "TAB_COMBINACIONES_BLOQUES", alinear_der=[1, 2, 3, 4], anchos_cm=[3.6, 2.6, 2.6, 3.6, 3.6],
                    encabezado_superior=[("", 1), ("Desempeño", 2), ("Diferencia respecto de C", 2)],
                    tam=9, nota="Regresión logística con validación cruzada anidada única 5 × 5 (semilla 42); métricas "
                                "sobre las predicciones fuera de pliegue (N = 281). C: clínico; M: hallazgos cualitativos "
                                "de RM; A: adquisición; R: radiómico. AP: precisión promedio. Sin intervalos de confianza; "
                                "los valores no son comparables con los de la validación anidada repetida.")


def tab_modelos_principales():
    oof = pd.read_csv(k.PIPE / "stage3_oof_aggregated.csv")
    rep = pd.read_csv(k.PIPE / "stage3_summary.csv").set_index("model")
    from sklearn.metrics import average_precision_score, roc_auc_score
    filas = []
    for m in k.ORDEN:
        roc, ap = roc_auc_score(oof.y_true, oof[m]), average_precision_score(oof.y_true, oof[m])
        filas.append({"modelo": m, "roc_auc": roc, "ap": ap, "roc_rep_media": rep.loc[m, "roc_auc_mean"],
                      "roc_rep_de": rep.loc[m, "roc_auc_sd"], "ap_rep_media": rep.loc[m, "pr_auc_mean"],
                      "ap_rep_de": rep.loc[m, "pr_auc_sd"]})
    datos = pd.DataFrame(filas)
    datos.to_csv(k.DIR_DATOS / "TAB_MODELOS_PRINCIPALES.csv", index=False)
    assert [round(v, 3) for v in datos.roc_auc] == [0.750, 0.749, 0.761]
    df = pd.DataFrame({"Modelo": datos.modelo,
                       "ROC-AUC": [num(v) for v in datos.roc_auc], "AP": [num(v) for v in datos.ap],
                       "ROC-AUC ": [f"{num(a)} ± {num(b)}" for a, b in zip(datos.roc_rep_media, datos.roc_rep_de)],
                       "AP ": [f"{num(a)} ± {num(b)}" for a, b in zip(datos.ap_rep_media, datos.ap_rep_de)]})
    rep = pd.read_csv(k.PIPE / "stage3_per_repetition.csv")
    assert rep.groupby("model").seed.apply(sorted).apply(list).eq([list(range(42, 52))] * 3).all()
    k.guardar_tabla(df, "TAB_MODELOS_PRINCIPALES", alinear_der=[1, 2, 3, 4], anchos_cm=[2.4, 2.2, 2.2, 4.4, 4.4], tam=9,
                    encabezado_superior=[("", 1), ("OOF promediadas (N = 281)", 2),
                                         ("Por repetición: media ± DE (10 repeticiones)", 2)],
                    nota="Regresión logística con validación cruzada anidada repetida 10 × 5 × 5. ROC-AUC y AP: "
                         "calculados sobre las probabilidades fuera de pliegue promediadas por paciente entre las diez "
                         "repeticiones (N = 281; estimación principal). Por repetición: media ± DE muestral de las diez "
                         "estimaciones, una por repetición (semillas 42–51). OOF: fuera de pliegue. AP: precisión promedio; referencia sin capacidad predictiva = prevalencia (0,331).")


def tab_rf_gb():
    d = pd.read_csv(k.PIPE / "stage4light_combined.csv").set_index("algorithm")
    d.to_csv(k.DIR_DATOS / "TAB_RF_GB.csv")
    filas = []
    for alg, nom in (("random_forest", "Random Forest"), ("gradient_boosting", "Gradient Boosting")):
        r = d.loc[alg]
        filas += [[nom, "C", num(r.cv_roc_C), num(r.cv_pr_C), "—", "—"],
                  ["", "C+A+R", num(r.cv_roc_C_A_R), num(r.cv_pr_C_A_R), num(r.cv_delta_roc, signo=True),
                   num(r.cv_delta_pr, signo=True)]]
    assert filas[1][2] == "0,701" and filas[3][2] == "0,748" and filas[2][3] == "0,512"
    df = pd.DataFrame(filas, columns=["Algoritmo", "Modelo", "ROC-AUC", "AP", "Diferencia de ROC-AUC", "Diferencia de AP"])
    k.guardar_tabla(df, "TAB_RF_GB", alinear_der=[2, 3, 4, 5], grupos=[0, 2], anchos_cm=[3.2, 1.6, 1.8, 1.8, 3.6, 3.6],
                    tam=9, nota="Validación cruzada anidada única con 5 particiones externas y 3 internas (semilla 42), "
                                "sin repeticiones; métricas sobre las predicciones fuera de pliegue (N = 281). Diferencias: "
                                "C+A+R menos C con el mismo algoritmo. Por el esquema de validación distinto, estos valores "
                                "no se comparan directamente con los de la regresión logística. AP: precisión promedio.")


def tab_matrices():
    c = pd.read_csv(k.PIPE / "stage5_classification_050.csv").set_index("model")
    c.to_csv(k.DIR_DATOS / "TAB_MATRICES_CONFUSION.csv")
    assert tuple(c.loc["C", ["TN", "FP", "FN", "TP"]]) == (124, 64, 30, 63)
    assert tuple(c.loc["C+A+R", ["TN", "FP", "FN", "TP"]]) == (133, 55, 30, 63)
    df = pd.DataFrame([
        ["Con respuesta completa o casi completa (n = 93)", c.loc["C", "TP"], c.loc["C", "FN"], c.loc["C+A+R", "TP"],
         c.loc["C+A+R", "FN"]],
        ["Sin respuesta completa o casi completa (n = 188)", c.loc["C", "FP"], c.loc["C", "TN"], c.loc["C+A+R", "FP"],
         c.loc["C+A+R", "TN"]],
    ], columns=["Clase observada", "Positiva", "Negativa", "Positiva", "Negativa"])
    m = lambda mod, col: pct(c.loc[mod, col])  # noqa: E731
    k.guardar_tabla(df, "TAB_MATRICES_CONFUSION", alinear_der=[1, 2, 3, 4], anchos_cm=[6.4, 2.4, 2.4, 2.4, 2.4], tam=9,
                    encabezado_superior=[("", 1), ("Predicción de C", 2), ("Predicción de C+A+R", 2)],
                    nota=f"Umbral de 0,50 aplicado a las probabilidades fuera de pliegue promediadas (N = 281). "
                         f"Sensibilidad: {m('C', 'sensitivity')} (C) y {m('C+A+R', 'sensitivity')} (C+A+R); especificidad: "
                         f"{m('C', 'specificity')} y {m('C+A+R', 'specificity')}. Valores predictivos, exactitud y F1: "
                         f"Tabla A.6.")


def tab_a_clasificacion():
    c = pd.read_csv(k.PIPE / "stage5_classification_050.csv").set_index("model")
    c.to_csv(k.DIR_DATOS / "TAB_A_CLASIFICACION_050.csv")
    for mod in ("C", "C+A+R"):
        r = c.loc[mod]
        n = r.TP + r.FP + r.TN + r.FN
        assert n == 281 and abs(r.PPV - r.TP / (r.TP + r.FP)) < 1e-12 and abs(r.NPV - r.TN / (r.TN + r.FN)) < 1e-12
        assert abs(r.accuracy - (r.TP + r.TN) / n) < 1e-12 and abs(r.F1 - 2 * r.TP / (2 * r.TP + r.FP + r.FN)) < 1e-12
    fila = lambda mod: [mod, f"{pct(c.loc[mod, 'PPV'])} ({c.loc[mod, 'TP']}/{c.loc[mod, 'TP'] + c.loc[mod, 'FP']})",  # noqa: E731
                        f"{pct(c.loc[mod, 'NPV'])} ({c.loc[mod, 'TN']}/{c.loc[mod, 'TN'] + c.loc[mod, 'FN']})",
                        f"{pct(c.loc[mod, 'accuracy'])} ({c.loc[mod, 'TP'] + c.loc[mod, 'TN']}/281)", num(c.loc[mod, "F1"])]
    df = pd.DataFrame([fila("C"), fila("C+A+R")], columns=["Modelo", "VPP", "VPN", "Exactitud", "F1"])
    assert list(df.iloc[1, 1:]) == ["53,4 % (63/118)", "81,6 % (133/163)", "69,8 % (196/281)", "0,597"]
    k.guardar_tabla(df, "TAB_A_CLASIFICACION_050", destino="anexos", alinear_der=[1, 2, 3, 4],
                    anchos_cm=[2.0, 3.6, 3.6, 3.6, 1.6], tam=9,
                    nota="Umbral de 0,50 aplicado a las probabilidades fuera de pliegue promediadas (N = 281; 93 con "
                         "respuesta completa o casi completa); valores derivados de las matrices de confusión de la "
                         "Tabla 3.7, donde se informan la sensibilidad y la especificidad. VPP y VPN: valores predictivos "
                         "positivo y negativo; F1: media armónica de la sensibilidad y el VPP.")


def tab_umbrales():
    t = pd.read_csv(k.PIPE / "stage5_classification_thresholds.csv")
    t = t[t.model == "C+A+R"].sort_values("threshold", ascending=False)
    t.to_csv(k.DIR_DATOS / "TAB_UMBRALES.csv", index=False)
    assert [tuple(r) for r in t[["TP", "FP", "TN", "FN"]].values] == [(63, 55, 133, 30), (74, 72, 116, 19), (76, 84, 104, 17)]
    df = pd.DataFrame({"Umbral": [num(v, 2) + (" (referencia)" if v == 0.5 else "") for v in t.threshold],
                       "VP": t.TP, "FN": t.FN, "VN": t.TN, "FP": t.FP,
                       "Sensibilidad": [pct(v) for v in t.sensitivity], "Especificidad": [pct(v) for v in t.specificity],
                       "VPP": [pct(v) for v in t.PPV], "VPN": [pct(v) for v in t.NPV]})
    k.guardar_tabla(df, "TAB_UMBRALES", alinear_der=list(range(1, 9)), anchos_cm=[2.7, 1.1, 1.1, 1.1, 1.1, 2.3, 2.6, 1.9, 1.9],
                    tam=9, nota="Modelo C+A+R; probabilidades fuera de pliegue promediadas (N = 281; 93 con respuesta "
                                "completa o casi completa). El umbral de 0,50 se definió antes del análisis; 0,40 y 0,35 "
                                "son exploratorios e ilustran el compromiso entre sensibilidad y especificidad, sin "
                                "constituir reglas de decisión validadas. VP: verdaderos positivos; FN: falsos negativos; "
                                "VN: verdaderos negativos; FP: falsos positivos; VPP y VPN: valores predictivos positivo "
                                "y negativo.")


def tab_subgrupos():
    g = pd.read_csv(k.COMPL / "grupos_metricas.csv").set_index("grupo")
    g.to_csv(k.DIR_DATOS / "TAB_SUBGRUPOS.csv")
    filas = []
    for grupo, nom in (("Luminal", "Luminal (RH+/HER2−)"), ("No luminal (HER2+ ∪ TN)", "No luminal (HER2+ o triple negativo)")):
        r = g.loc[grupo]
        filas.append([nom, int(r.N), n_pct(int(r.positivos), int(r.N)), num(r.roc_C), num(r["roc_C+A+R"]),
                      f"{num(r['d_roc_C+A+R'], signo=True)} {ic(r['d_roc_C+A+R_lo'], r['d_roc_C+A+R_hi'])}"])
    assert filas[0][5] == "−0,052 [−0,128; 0,017]" and filas[1][5] == "+0,062 [0,006; 0,123]"
    df = pd.DataFrame(filas, columns=["Subgrupo", "N", "Con respuesta", "C", "C+A+R", "Diferencia [IC 95 %]"])
    k.guardar_tabla(df, "TAB_SUBGRUPOS", alinear_der=[1, 2, 3, 4, 5], anchos_cm=[4.8, 1.1, 2.5, 1.8, 1.8, 4.0], tam=9,
                    encabezado_superior=[("", 3), ("ROC-AUC", 2), ("", 1)],
                    nota="Análisis exploratorio post hoc. Se evaluaron las probabilidades fuera de pliegue promediadas "
                         "de los modelos ajustados en la cohorte completa (10 × 5 × 5) dentro de cada subgrupo; no se "
                         "ajustaron modelos por subgrupo. Diferencia: C+A+R menos C; IC 95 % por bootstrap pareado "
                         "(5000 remuestreos). RH: receptores hormonales.")


# ------------------------------------------------------------------ anexos
def tab_a_variables():
    import run_stage2 as s
    desc = {"age": "Edad (años; derivada de Date of Birth)", "ER_bin": "Receptor de estrógeno", "PR_bin": "Receptor de progesterona",
            "HER2_bin": "HER2", "menopause_post": "Posmenopausia", "tumor_side_R": "Lateralidad derecha",
            "stage_T": "Estadio T", "stage_N": "Estadio N", "grade_tubule": "Grado: formación tubular",
            "grade_nuclear": "Grado: pleomorfismo nuclear", "grade_mitotic": "Grado: recuento mitótico",
            "race": "Raza/etnia", "histologic_type": "Tipo histológico", "bilateral_bc": "Cáncer bilateral",
            "mf_multicentric": "Multifocal o multicéntrico", "mf_contralateral": "Compromiso contralateral",
            "mf_lymphadenopathy": "Linfadenopatías o ganglios sospechosos", "mf_skin_nipple": "Compromiso de piel o pezón",
            "mf_pec_chest": "Compromiso pectoral o de la pared torácica", "days_to_mri": "Días entre el diagnóstico y la RM",
            "tr": "Tiempo de repetición", "te": "Tiempo de eco", "patient_position": "Posición de la paciente",
            "field_strength": "Intensidad de campo (T)", "slice_thickness": "Espesor de corte", "matrix_rows": "Filas de la matriz",
            "flip_angle": "Ángulo de inclinación", "fov": "Campo de visión (cm)", "manufacturer": "Fabricante",
            "model_name": "Modelo del equipo", "scan_options": "Opciones de adquisición", "acq_matrix": "Matriz de adquisición",
            "contrast_agent": "Agente de contraste"}
    origen_c = {"age": "Date of Birth (Days)", "ER_bin": "ER status", "PR_bin": "PR status", "HER2_bin": "HER2 status",
                "menopause_post": "Menopause (at diagnosis)", "tumor_side_R": "Tumor Location (y Position)",
                "stage_T": "Staging(Tumor Size)# [T]", "stage_N": "Staging(Nodes)#(Nx replaced by -1)[N]",
                "grade_tubule": "Tumor Grade (Tubule)", "grade_nuclear": "Tumor Grade (Nuclear)",
                "grade_mitotic": "Tumor Grade (Mitotic)", "race": "Race and Ethnicity", "histologic_type": "Histologic type",
                "bilateral_bc": "Bilateral breast cancer"}
    origen = {**origen_c, **s.M_BINARY, **s.A_CONTINUOUS, **s.A_ORDINAL, **s.A_BINARY, **s.A_NOMINAL}
    tipo_es = {"continuous": "Continua", "binary": "Binaria", "ordinal": "Ordinal", "nominal": "Nominal (one-hot)"}
    trat = {"Continua": "Imputación por mediana; estandarización", "Ordinal": "Imputación por mediana; estandarización",
            "Binaria": "Imputación por moda", "Nominal (one-hot)": "Imputación por moda; codificación one-hot"}
    filas = []
    for b in ("C", "M", "A"):
        for t, vs in s.BLOCK_TYPES[b].items():
            for v in vs:
                tp = tipo_es[t]
                tr = trat[tp]
                if v in ("histologic_type", "bilateral_bc"):
                    tr = "Categoría explícita NC (dato no recolectado); codificación one-hot"
                if v == "tumor_side_R":
                    tr = "Reconstruida con Position cuando faltaba"
                filas.append([b if not any(f[0] == b for f in filas) else "", desc[v], v,
                              " ".join(str(origen[v]).split()), tp.replace(" (one-hot)", ""), tr])
    df = pd.DataFrame(filas, columns=["Bloque", "Variable", "Nombre en el análisis", "Columna de Duke", "Tipo", "Tratamiento"])
    assert len(df) == 33
    k.guardar_tabla(df, "TAB_A_VARIABLES_BLOQUES", destino="anexos", anchos_cm=[1.4, 3.1, 2.9, 3.4, 1.7, 3.5], tam=7.5,
                    grupos=[0, 14, 19],
                    nota="Imputación y escalado ajustados dentro de cada partición de entrenamiento. El estadio M "
                         "(reconstruido) se conservó solo con fines descriptivos. Fuente de la codificación: "
                         "pipeline_v2/run_stage2.py.")


def tab_a_configuracion():
    import run_stage2 as s
    import run_stage4 as s4
    assert s.C_GRID == [0.01, 0.1, 1.0, 10.0] and s.NCOMP_GRID == [1, 2, 3, 5, 10, 15, 20, 30]
    rf, gb = s4.ALGOS["random_forest"]["grid"], s4.ALGOS["gradient_boosting"]["grid"]
    assert rf["clf__max_depth"] == [None, 3, 5] and gb["clf__learning_rate"] == [0.05, 0.1]
    kgrid = "1, 2, 3, 5, 10, 15, 20 o 30"
    df = pd.DataFrame([
        ["Regresión logística (L2, ponderación balanceada, máx. 5000 iteraciones)",
         "Parámetro de regularización: 0,01; 0,1; 1; 10. Componentes de R: " + kgrid,
         "Combinaciones de bloques: CV anidada única 5 × 5 (semilla 42). Modelos principales: CV anidada repetida "
         "10 × 5 × 5 (semillas 42 a 51). Hold-out 80/20 estratificado (semilla 42; CV interna de 5 particiones)"],
        ["Random Forest (400 árboles)",
         "Profundidad máxima: sin límite, 3, 5. Mínimo de observaciones por hoja: 2, 5, 10. Variables por división: "
         "raíz cuadrada, 50 %. Ponderación de clases: ninguna, balanceada. Componentes de R: " + kgrid,
         "CV anidada única 5 × 3 (semilla 42)"],
        ["Gradient Boosting",
         "Número de árboles: 100, 200. Tasa de aprendizaje: 0,05; 0,1. Profundidad máxima: 2, 3. Mínimo de "
         "observaciones por hoja: 10 (fijo). Componentes de R: " + kgrid, "CV anidada única 5 × 3 (semilla 42)"],
        ["Comparación pareada", "Bootstrap de las diferencias frente a C: 5000 remuestreos; IC 95 % por percentiles",
         "Semilla 20260909"],
    ], columns=["Algoritmo o procedimiento", "Hiperparámetros evaluados", "Esquema de validación y semillas"])
    k.guardar_tabla(df, "TAB_A_CONFIGURACION_MODELOS", destino="anexos", anchos_cm=[3.8, 6.8, 5.4], tam=8,
                    nota="La selección interna de hiperparámetros utilizó el ROC-AUC medio de las particiones internas. "
                         "CV: validación cruzada. Fuentes: pipeline_v2/run_stage2.py, run_stage3.py, run_stage4.py y "
                         "run_stage4_light.py.")


def tab_a_hiperparametros():
    st = pd.read_csv(k.PIPE / "stage3_stability_folds.csv")
    st.to_csv(k.DIR_DATOS / "TAB_A_HIPERPARAMETROS_PLIEGUES.csv", index=False)
    filas = []
    for m in k.ORDEN:
        g = st[st.model == m]
        assert len(g) == 50
        fila = [m] + [str(int((g.C == c).sum())) for c in (0.01, 0.1, 1.0, 10.0)]
        if g.n_components.notna().any():
            fila += [str(int((g.n_components == q).sum())) for q in (1, 2, 3, 5, 10, 15, 20, 30)]
            r = g.n_radiomics_after_corr
            fila.append(f"{int(r.median())} [{int(r.min())}–{int(r.max())}]")
        else:
            fila += ["—"] * 8 + ["—"]
        filas.append(fila)
    assert filas[1][6] == "28" and filas[2][6] == "17" and filas[2][3] == "41"
    cols = (["Modelo", "0,01", "0,1", "1", "10"] + [str(q) for q in (1, 2, 3, 5, 10, 15, 20, 30)]
            + ["Radiómicas retenidas"])
    k.guardar_tabla(pd.DataFrame(filas, columns=cols), "TAB_A_HIPERPARAMETROS_PLIEGUES", destino="anexos",
                    alinear_der=list(range(1, 14)), anchos_cm=[1.4] + [1.2] * 4 + [0.95] * 8 + [2.2], tam=7.5,
                    encabezado_superior=[("", 1), ("Parámetro de regularización (C)", 4),
                                         ("Componentes principales (k)", 8), ("", 1)],
                    nota="Número de particiones externas (de 50: 10 repeticiones × 5 particiones) en que la validación "
                         "interna seleccionó cada valor. C: parámetro de regularización de la regresión logística; "
                         "k: número de componentes principales del bloque R. Radiómicas retenidas tras el filtrado de "
                         "correlación: mediana [mínimo–máximo]. Fuente: pipeline_v2/stage3_stability_folds.csv.")


def tab_a_software():
    import matplotlib
    import numpy
    import scipy
    import sklearn
    # Versiones con las que se obtuvieron los resultados de la tesis; se avisa si el entorno actual difiere.
    v = {"Python": "3.11.2", "scikit-learn": "1.5.1", "pandas": "2.2.3", "NumPy": "1.26.4", "SciPy": "1.14.1",
         "Matplotlib": "3.9.2"}
    actual = {"Python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "pandas": pd.__version__,
              "NumPy": numpy.__version__, "SciPy": scipy.__version__, "Matplotlib": matplotlib.__version__}
    distintas = {kk: actual[kk] for kk in v if actual[kk] != v[kk]}
    if distintas:
        print("  Aviso: versiones del entorno distintas de las de la tesis:", distintas)
    uso = {"Python": "Lenguaje de programación", "scikit-learn": "Preprocesamiento, PCA, modelos, validación y métricas",
           "pandas": "Organización y manipulación de los datos", "NumPy": "Operaciones numéricas",
           "SciPy": "Procedimientos estadísticos complementarios", "Matplotlib": "Figuras"}
    df = pd.DataFrame([[kk, vv, uso[kk]] for kk, vv in v.items()], columns=["Software", "Versión", "Uso"])
    k.guardar_tabla(df, "TAB_A_SOFTWARE", destino="anexos", anchos_cm=[3.2, 2.2, 8.0], tam=9,
                    nota="Las características radiómicas no se extrajeron en este trabajo: se utilizaron las provistas "
                         "en Imaging_Features.xlsx (Duke).")


def tab_a_holdout():
    r = pd.read_csv(k.PIPE / "results_stage2.csv")
    r = r[r.analysis == "split_8020"].set_index("model")
    orden = ["C", "C+M", "C+A", "R", "C+R", "C+M+R", "C+A+R", "C+M+A+R"]
    r = r.loc[orden]
    assert (r.n_train == 224).all() and (r.n_test == 57).all()
    ho = pd.read_csv(k.PIPE / "stage3_holdout_8020.csv").set_index("model")
    assert all(abs(r.loc[m, c] - ho.loc[m, c]) < 1e-12 for m in ho.index for c in ("roc_auc", "pr_auc"))
    assert (num(r.loc["C+A+R", "roc_auc"]), num(r.loc["C+A+R", "pr_auc"])) == ("0,784", "0,708")
    r[["roc_auc", "pr_auc", "n_train", "n_test"]].to_csv(k.DIR_DATOS / "TAB_A_HOLDOUT.csv")
    df = pd.DataFrame({"Combinación": orden, "ROC-AUC": [num(v) for v in r.roc_auc], "AP": [num(v) for v in r.pr_auc]})
    k.guardar_tabla(df, "TAB_A_HOLDOUT", destino="anexos", alinear_der=[1, 2], anchos_cm=[3.6, 2.6, 2.6], tam=9,
                    nota="Regresión logística. Partición estratificada única: 224 pacientes de entrenamiento y 57 de "
                         "prueba (19 con respuesta completa o casi completa; semilla 42). Hiperparámetros seleccionados "
                         "por validación cruzada estratificada de 5 particiones dentro del entrenamiento; evaluación única "
                         "en el conjunto de prueba. Sin intervalos de confianza; análisis complementario, no comparable "
                         "con la validación anidada repetida. C: clínico; M: hallazgos cualitativos de RM; A: adquisición; "
                         "R: radiómico. AP: precisión promedio.")


if __name__ == "__main__":
    for f in (tab_subtipos, tab_estado_arte, tab_variable_objetivo, tab_cohorte, tab_bloques, tab_pca_comparacion,
              tab_combinaciones, tab_modelos_principales, tab_rf_gb, tab_matrices, tab_umbrales, tab_subgrupos,
              tab_a_variables, tab_a_configuracion, tab_a_hiperparametros, tab_a_software, tab_a_clasificacion, tab_a_holdout):
        f()
        print("ok", f.__name__)
