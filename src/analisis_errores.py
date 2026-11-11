"""Diagnóstico post hoc de las predicciones de 6; no ajusta ni corrige el modelo."""

from dataclasses import dataclass
import sys

import numpy as np
import pandas as pd

if __package__:
    from . import entrenar_modelo as detector
    from . import evaluacion_temporal as temporal
else:
    import entrenar_modelo as detector
    import evaluacion_temporal as temporal


VARIABLES = [*detector.CARACTERISTICAS, "puntaje_anomalia"]


@dataclass
class DiagnosticoErrores:
    movimientos: pd.DataFrame
    por_cuenta: pd.DataFrame
    por_ventana: pd.DataFrame
    por_cuenta_ventana: pd.DataFrame
    distribuciones: pd.DataFrame
    falsos_negativos: pd.DataFrame
    detalle_rafagas: pd.DataFrame
    resumen_rafagas: pd.DataFrame
    entrenamientos: pd.DataFrame


def clasificar_errores(comparacion: pd.DataFrame) -> pd.DataFrame:
    """Conserva las predicciones y agrega TP/FP/TN/FN, sin descartar duplicados."""
    obligatorias = {"transaccion_id", "cuenta_id", "fecha_hora", "ventana",
                    "es_anomalia", "tipo_anomalia", "prediccion_anomalia", *VARIABLES}
    if not comparacion.columns.is_unique or not obligatorias.issubset(comparacion.columns):
        raise ValueError("Faltan columnas de evaluación para analizar errores.")
    datos = comparacion.copy(deep=True)
    if (datos.empty or datos[list(obligatorias)].isna().any().any()
            or not datos.transaccion_id.map(lambda v: isinstance(v, str) and bool(v.strip())).all()
            or datos.transaccion_id.duplicated().any()):
        raise ValueError("Se requieren operaciones evaluadas con IDs únicos y campos completos.")
    if (not datos.es_anomalia.isin([0, 1]).all()
            or not datos.prediccion_anomalia.isin([0, 1]).all()
            or not np.isfinite(datos[VARIABLES].to_numpy(dtype="float64")).all()):
        raise ValueError("Se requieren etiquetas/predicciones binarias y características/puntajes finitos.")
    datos["fecha_hora"] = pd.to_datetime(datos.fecha_hora, format="mixed", errors="raise")
    claves = list(zip(datos.es_anomalia, datos.prediccion_anomalia))
    respuestas = {(0, 0): "TN", (0, 1): "FP", (1, 0): "FN", (1, 1): "TP"}
    datos["resultado_evaluacion"] = [respuestas[clave] for clave in claves]
    return datos.sort_values(["fecha_hora", "transaccion_id"], kind="stable").reset_index(drop=True)


def resumir_falsas_alertas(datos: pd.DataFrame, por: list[str]) -> pd.DataFrame:
    """Distingue FP entre negativos reales, carga FP/total y alertas/total.

    Incluye grupos sin falsas alertas. Si no hay negativos reales, FPR = N/D.
    Las tasas describen las operaciones evaluadas, no toda la actividad bancaria.
    """
    filas = []
    for clave, grupo in datos.groupby(por, sort=True):
        valores = clave if isinstance(clave, tuple) else (clave,)
        fila = dict(zip(por, valores))
        conteos = grupo.resultado_evaluacion.value_counts()
        fila.update({tipo: int(conteos.get(tipo, 0)) for tipo in ("TP", "FP", "TN", "FN")})
        total, normales = len(grupo), fila["FP"] + fila["TN"]
        fila.update(total=total, normales_reales=normales,
                    tasa_fp_normales=fila["FP"] / normales if normales else np.nan,
                    proporcion_fp_total=fila["FP"] / total,
                    tasa_alertas_total=(fila["FP"] + fila["TP"]) / total)
        filas.append(fila)
    return pd.DataFrame(filas).sort_values(["FP", *por], ascending=[False, *([True] * len(por))]).reset_index(drop=True)


def comparar_distribuciones(datos: pd.DataFrame) -> pd.DataFrame:
    """Contrastes descriptivos FP/TN globales y dentro de cada ventana, sin tests causales."""
    filas = []
    grupos = [("agregado", datos), *list(datos.groupby("ventana", sort=True))]
    for ventana, grupo in grupos:
        for respuesta in ("FP", "TN"):
            seleccion = grupo.loc[grupo.resultado_evaluacion.eq(respuesta)]
            for variable in VARIABLES:
                valores = seleccion[variable]
                filas.append({"ventana": ventana, "grupo": respuesta, "variable": variable,
                              "cantidad": len(valores), "media": valores.mean(),
                              "minimo": valores.min(), "q25": valores.quantile(.25),
                              "mediana": valores.median(), "q75": valores.quantile(.75), "maximo": valores.max()})
    return pd.DataFrame(filas)


def reconstruir_rafagas(
    concentraciones: pd.DataFrame, datos: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agrupa etiquetas de concentración por cuenta y gaps consecutivos <=5 min.

    Es una reconstrucción post hoc, no una regla de detección. El encadenamiento
    puede durar más de 5 minutos y no equivale al conteo móvil del modelo. Incluye
    etiquetas fuera de evaluación con sus predicciones ausentes, sin imputarlas.
    Las posiciones cuentan todas las etiquetas del episodio, no solo sus alertas.
    """
    contexto = concentraciones[["transaccion_id", "cuenta_id", "fecha_hora"]].copy()
    if contexto.transaccion_id.isna().any() or contexto.transaccion_id.duplicated().any():
        raise ValueError("Las concentraciones deben tener IDs únicos y completos.")
    contexto["fecha_hora"] = pd.to_datetime(contexto.fecha_hora, format="mixed", errors="raise")
    contexto = contexto.sort_values(["cuenta_id", "fecha_hora", "transaccion_id"], kind="stable")
    nuevo = (contexto.cuenta_id.ne(contexto.cuenta_id.shift())
             | contexto.fecha_hora.diff().gt(pd.Timedelta(minutes=5)))
    contexto["rafaga"] = nuevo.cumsum()
    inicios = contexto.groupby("rafaga", sort=False).first().sort_values(["fecha_hora", "cuenta_id", "transaccion_id"])
    contexto["rafaga"] = contexto.rafaga.map({clave: i for i, clave in enumerate(inicios.index, start=1)})
    contexto = contexto.sort_values(["rafaga", "fecha_hora", "transaccion_id"], kind="stable")
    contexto["posicion"] = contexto.groupby("rafaga", sort=False).cumcount() + 1
    columnas = ["transaccion_id", *VARIABLES, "prediccion_anomalia", "ventana"]
    detalle = contexto.merge(datos[columnas], on="transaccion_id", how="left", validate="one_to_one", sort=False)
    for columna in ("ventana", "prediccion_anomalia"):
        detalle[columna] = detalle[columna].astype("Int64")
    detalle["resultado_correcto"] = detalle.prediccion_anomalia.map({1: "Correcto", 0: "Incorrecto"}).fillna("Sin evaluación")
    filas = []
    for rafaga, grupo in detalle.groupby("rafaga", sort=True):
        alertas = grupo.loc[grupo.prediccion_anomalia.eq(1)]
        primera = alertas.iloc[0] if not alertas.empty else None
        ventanas = sorted(grupo.ventana.dropna().unique())
        filas.append({"rafaga": rafaga, "cuenta_id": grupo.cuenta_id.iloc[0],
                      "inicio": grupo.fecha_hora.min(), "fin": grupo.fecha_hora.max(),
                      "operaciones": len(grupo), "evaluadas": int(grupo.prediccion_anomalia.notna().sum()),
                      "detectadas": len(alertas), "omitidas": int(grupo.prediccion_anomalia.eq(0).sum()),
                      "sin_evaluacion": int(grupo.prediccion_anomalia.isna().sum()),
                      "ventanas": ",".join(str(int(v)) for v in ventanas), "cambio_modelo": len(ventanas) > 1,
                      "primera_alerta_posicion": int(primera.posicion) if primera is not None else np.nan,
                      "primera_alerta_fecha": primera.fecha_hora if primera is not None else pd.NaT,
                      "demora_segundos": (primera.fecha_hora - grupo.fecha_hora.min()).total_seconds()
                      if primera is not None else np.nan})
    columnas_resumen = ["rafaga", "cuenta_id", "inicio", "fin", "operaciones", "evaluadas", "detectadas",
                        "omitidas", "sin_evaluacion", "ventanas", "cambio_modelo", "primera_alerta_posicion",
                        "primera_alerta_fecha", "demora_segundos"]
    return detalle.reset_index(drop=True), pd.DataFrame(filas, columns=columnas_resumen)


def resumir_entrenamientos(resultados: list[detector.ResultadoEntrenamiento]) -> pd.DataFrame:
    """Lee variación en entrenamiento y cortes ya ajustados; no mide importancia causal."""
    filas = []
    for ventana, resultado in enumerate(resultados, start=1):
        cortes = np.zeros(len(detector.CARACTERISTICAS), dtype="int64")
        arboles = cortes.copy()
        for arbol, indices in zip(resultado.modelo.estimators_, resultado.modelo.estimators_features_):
            usados = arbol.tree_.feature
            globales = np.asarray(indices)[usados[usados >= 0]]
            conteos = np.bincount(globales, minlength=len(cortes))
            cortes += conteos
            arboles += (conteos > 0)
        for i, variable in enumerate(detector.CARACTERISTICAS):
            valores = resultado.entrenamiento[variable]
            filas.append({"ventana": ventana, "variable": variable, "n_train": len(valores),
                          "minimo": valores.min(), "mediana": valores.median(), "maximo": valores.max(),
                          "distintos": valores.nunique(), "cortes": int(cortes[i]),
                          "arboles_con_variable": int(arboles[i])})
    return pd.DataFrame(filas)


def analizar_errores(
    resultados: list[detector.ResultadoEntrenamiento], informe: temporal.InformeTemporal,
) -> DiagnosticoErrores:
    """Consume evaluación ya calculada; no conecta, transforma, entrena ni puntúa."""
    if not resultados or len(resultados) != len(informe.por_ventana):
        raise ValueError("Se requieren resultados y métricas de las mismas ventanas.")
    comparaciones = []
    for numero, (resultado, evaluacion) in enumerate(zip(resultados, informe.por_ventana), start=1):
        tabla = evaluacion.metricas.comparacion
        if set(tabla.transaccion_id) != set(resultado.predicciones.transaccion_id):
            raise ValueError("Las operaciones de una ventana no corresponden a sus predicciones.")
        comparaciones.append(tabla.assign(ventana=numero))
    datos = clasificar_errores(pd.concat(comparaciones, ignore_index=True))
    agregado = informe.agregado.metricas.comparacion
    if len(datos) != len(agregado) or set(datos.transaccion_id) != set(agregado.transaccion_id):
        raise ValueError("El diagnóstico y el agregado deben contener las mismas operaciones únicas.")

    # El último entrenamiento más su evaluación cubren el dataset. Aquí solo se
    # recuperan IDs, cuentas y fechas originales; sus características de train
    # NO se reutilizan para explicar evaluaciones de modelos anteriores.
    contexto = pd.concat([resultados[-1].entrenamiento[["transaccion_id", "cuenta_id", "fecha_hora"]],
                          resultados[-1].predicciones[["transaccion_id", "cuenta_id", "fecha_hora"]]], ignore_index=True)
    tipos = pd.concat([datos[["transaccion_id", "tipo_anomalia"]], informe.anomalias_no_evaluadas], ignore_index=True)
    ids_concentracion = set(tipos.loc[tipos.tipo_anomalia.eq("concentracion"), "transaccion_id"])
    if contexto.transaccion_id.duplicated().any() or not ids_concentracion.issubset(set(contexto.transaccion_id)):
        raise ValueError("No se pudo recuperar un contexto único para las concentraciones.")
    concentraciones = contexto.loc[contexto.transaccion_id.isin(ids_concentracion)]
    detalle, rafagas = reconstruir_rafagas(concentraciones, datos)
    return DiagnosticoErrores(
        movimientos=datos,
        por_cuenta=resumir_falsas_alertas(datos, ["cuenta_id"]),
        por_ventana=resumir_falsas_alertas(datos, ["ventana"]),
        por_cuenta_ventana=resumir_falsas_alertas(datos, ["ventana", "cuenta_id"]),
        distribuciones=comparar_distribuciones(datos),
        falsos_negativos=datos.loc[datos.resultado_evaluacion.eq("FN")].copy(),
        detalle_rafagas=detalle, resumen_rafagas=rafagas,
        entrenamientos=resumir_entrenamientos(resultados),
    )


def _mostrar(titulo: str, tabla: pd.DataFrame) -> None:
    print(f"\n{titulo}")
    if tabla.empty:
        print("Sin casos.")
        return
    tasas = {c: (lambda v: f"{v:.2%}") for c in tabla if c.startswith(("tasa_", "proporcion_"))}
    print(tabla.to_string(index=False, na_rep="N/D", float_format=lambda v: f"{v:.6f}", formatters=tasas))


def main() -> int:
    """Reproduce las tres ventanas existentes y recién después consulta etiquetas."""
    try:
        engine = detector.crear_engine()
        try:
            transacciones = detector.cargar_transacciones(engine)
        finally:
            engine.dispose()
    except (detector.ErrorConfiguracion, detector.ErrorConexion) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("Error al leer PostgreSQL o liberar la conexión. Revisá la configuración local.", file=sys.stderr)
        return 1
    try:
        resultados = temporal.predecir_ventanas(transacciones)
        etiquetas_reales = detector.cargar_etiquetas(detector.RUTA_ETIQUETAS, transacciones.transaccion_id)
        informe = temporal.evaluar_ventanas(resultados, etiquetas_reales)
        diagnostico = analizar_errores(resultados, informe)
    except ValueError as error:
        print(f"Error de análisis: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("Error inesperado al analizar errores. Revisá datos y dependencias.", file=sys.stderr)
        return 1

    datos = diagnostico.movimientos
    print(f"Análisis de errores — scikit-learn {detector.version_sklearn}")
    print("PostgreSQL leído en solo lectura y cerrado antes del entrenamiento.")
    print("Se reutilizan las tres ventanas de 6; etiquetas leídas después de generar todas sus predicciones.")
    print(f"Hechos observados — {len(datos)} operaciones evaluadas únicas:")
    print(datos.resultado_evaluacion.value_counts().reindex(["TP", "FP", "TN", "FN"], fill_value=0).to_string())
    print(f"Umbral por ventana: {dict(enumerate([-float(r.modelo.offset_) for r in resultados], start=1))}")
    metricas = informe.agregado.metricas
    print(f"Precisión: {metricas.precision:.6f} | Recall: {informe.agregado.recall} | F1: {metricas.f1:.6f}")
    print("Tasa FP = FP/(FP+TN); proporción FP = FP/total; tasa de alertas = (FP+TP)/total. N/D = sin denominador.")
    _mostrar("Falsas alertas por cuenta (incluye cuentas sin FP)", diagnostico.por_cuenta)
    _mostrar("Falsas alertas por ventana", diagnostico.por_ventana.sort_values("ventana"))
    _mostrar("Diez combinaciones cuenta/ventana con más FP", diagnostico.por_cuenta_ventana.head(10))
    _mostrar("Distribuciones FP frente a TN: globales y por ventana", diagnostico.distribuciones)
    columnas = ["transaccion_id", "cuenta_id", "fecha_hora", "tipo_anomalia", *VARIABLES, "ventana", "prediccion_anomalia"]
    _mostrar("Todas las anomalías omitidas (FN)", diagnostico.falsos_negativos[columnas])
    _mostrar("Diez FP con mayor puntaje (no son causas atribuidas)",
             datos.loc[datos.resultado_evaluacion.eq("FP")].sort_values(["puntaje_anomalia", "transaccion_id"],
                                                                       ascending=[False, True]).head(10)[columnas])
    _mostrar("Ráfagas etiquetadas: misma cuenta y gaps consecutivos <=5 min, independientes de las alertas", diagnostico.detalle_rafagas)
    _mostrar("Resumen de ráfagas y primera alerta observada", diagnostico.resumen_rafagas)
    print("Posición/demora cuentan desde el inicio de la ráfaga etiquetada completa. Sin evaluación indica censura; varias ventanas indican cambio de modelo.")
    _mostrar("Variación en entrenamiento y uso de variables en árboles ya ajustados", diagnostico.entrenamientos)
    print("Los cortes de árboles indican uso estructural; no son importancias causales ni atribuciones de un puntaje individual.")
    print("\nHipótesis para investigar, sin ajustes en esta etapa:")
    print("- Comparar cobertura de los comportamientos en entrenamiento, en especial la variación de frecuencia.")
    print("- Revisar referencias por cuenta, ventanas de actividad y falsas alertas en períodos nuevos.")
    print("- Evaluar alertas por episodio y demora; estudiar umbrales con costos y validación separada.")
    print("Contrastes descriptivos: no prueban causalidad. Los puntajes y referencias de ventanas diferentes no están calibrados entre sí.")
    print("Pocos episodios sintéticos, entrenamientos solapados y posible contaminación histórica limitan la generalización.")
    print("No se modificaron parámetros, características, umbral, PostgreSQL, CSV ni Streamlit. Resultados solo en memoria.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
