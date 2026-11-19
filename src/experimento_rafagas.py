"""Comparación experimental IF vs IF OR frecuencia>=3. Sin integración al modelo."""

from dataclasses import dataclass
import sys

import numpy as np
import pandas as pd

if __package__:
    from . import analisis_errores as diagnostico
    from . import entrenar_modelo as detector
    from . import evaluacion_temporal as temporal
    from .generar_escenarios_rafagas import generar_escenarios, SEMILLA_ESCENARIOS
    from .preparar_datos import COLUMNAS_ORIGINALES
else:
    import analisis_errores as diagnostico
    import entrenar_modelo as detector
    import evaluacion_temporal as temporal
    from generar_escenarios_rafagas import generar_escenarios, SEMILLA_ESCENARIOS
    from preparar_datos import COLUMNAS_ORIGINALES


UMBRAL_CONCENTRACION = 3  # Fijado por el protocolo, no se busca ni optimiza.
ENFOQUES = {"Isolation Forest": "prediccion_anomalia", "Híbrido": "prediccion_hibrida"}


@dataclass
class ComparacionEnfoques:
    metricas: pd.DataFrame
    comparacion: pd.DataFrame
    nuevas_alertas: pd.DataFrame


def aplicar_regla(predicciones: pd.DataFrame) -> pd.DataFrame:
    """Agrega decisiones experimentales conservando la predicción y puntaje IF."""
    requeridas = {"transaccion_id", "prediccion_anomalia", "movimientos_5min"}
    if not predicciones.columns.is_unique or not requeridas.issubset(predicciones.columns):
        raise ValueError("Faltan columnas de predicción para aplicar la regla.")
    datos = predicciones.copy(deep=True)
    if (datos.transaccion_id.isna().any() or datos.transaccion_id.duplicated().any()
            or not datos.transaccion_id.map(lambda v: isinstance(v, str) and bool(v.strip())).all()
            or not datos.prediccion_anomalia.isin([0, 1]).all()):
        raise ValueError("Se requieren IDs únicos y predicciones binarias.")
    conteos = datos.movimientos_5min.to_numpy(dtype="float64")
    if (not np.isfinite(conteos).all() or (conteos < 1).any() or (conteos != np.floor(conteos)).any()):
        raise ValueError("Los conteos causales deben ser enteros positivos y finitos.")
    datos["alerta_regla"] = (conteos >= UMBRAL_CONCENTRACION).astype("int64")
    datos["prediccion_hibrida"] = (datos.prediccion_anomalia.eq(1) | datos.alerta_regla.eq(1)).astype("int64")
    datos["alerta_nueva"] = (datos.prediccion_anomalia.eq(0) & datos.alerta_regla.eq(1)).astype("int64")
    return datos


def predecir_congelado(
    transacciones: pd.DataFrame, resultado: detector.ResultadoEntrenamiento, historial: pd.DataFrame,
) -> pd.DataFrame:
    """Solo transforma con referencias existentes y puntúa con el modelo ya ajustado."""
    datos = detector.validar_y_convertir(transacciones).sort_values(
        ["fecha_hora", "transaccion_id"], kind="stable",
    ).reset_index(drop=True)
    pasado = detector.validar_y_convertir(historial)
    if (datos.empty or pasado.empty or datos.fecha_hora.min() <= pasado.fecha_hora.max()
            or not set(resultado.entrenamiento.transaccion_id).issubset(set(pasado.transaccion_id))
            or set(datos.transaccion_id) & set(pasado.transaccion_id)):
        raise ValueError("Las pruebas deben ser posteriores al historial, con IDs nuevos e historial de entrenamiento disponible.")
    preparados = detector.transformar_datos(datos, resultado.referencias, historial=pasado)
    puntajes = -resultado.modelo.score_samples(preparados[detector.CARACTERISTICAS])
    if not np.isfinite(puntajes).all():
        raise ValueError("El modelo produjo puntajes no finitos.")
    preparados["puntaje_anomalia"] = puntajes
    preparados["prediccion_anomalia"] = (puntajes > -float(resultado.modelo.offset_)).astype("int64")
    return preparados


def comparar_enfoques(predicciones: pd.DataFrame, etiquetas: pd.DataFrame) -> ComparacionEnfoques:
    """Une etiquetas por ID después de predecir; OR solo puede sumar alertas."""
    if {"es_anomalia", "tipo_anomalia"} & set(predicciones.columns):
        raise ValueError("Las etiquetas deben estar separadas de las predicciones de entrada.")
    datos = aplicar_regla(predicciones)
    filas, comparacion = [], None
    for enfoque, columna in ENFOQUES.items():
        evaluacion = detector.evaluar_predicciones(datos.assign(prediccion_anomalia=datos[columna]), etiquetas)
        if enfoque == "Isolation Forest":
            comparacion = evaluacion.comparacion
        tn, fp, fn, tp = (int(v) for v in evaluacion.matriz_confusion.ravel())
        concentracion = evaluacion.por_tipo.set_index("tipo_anomalia").loc["concentracion"]
        filas.append({"enfoque": enfoque, "operaciones": len(datos), "reales": tp + fn,
                      "TP": tp, "FP": fp, "TN": tn, "FN": fn, "alertas": tp + fp,
                      "precision": evaluacion.precision, "recall": evaluacion.recall if tp + fn else np.nan,
                      "f1": evaluacion.f1, "tasa_fp": fp / (fp + tn) if fp + tn else np.nan,
                      "concentracion_detectada": int(concentracion.detectadas),
                      "concentracion_real": int(concentracion.reales)})
    return ComparacionEnfoques(pd.DataFrame(filas), comparacion,
                              comparacion.loc[comparacion.alerta_nueva.eq(1)].copy())


def evaluar_episodios(datos: pd.DataFrame, episodios: pd.DataFrame) -> pd.DataFrame:
    """Membresía fijada sin predicciones; alguna alerta y cobertura total son distintas."""
    requeridas = ["transaccion_id", "episodio_id", "escenario", "patron"]
    if (not set(requeridas).issubset(episodios.columns) or episodios[requeridas].isna().any().any()
            or episodios.transaccion_id.duplicated().any() or datos.transaccion_id.duplicated().any()
            or not set(episodios.transaccion_id).issubset(set(datos.transaccion_id))):
        raise ValueError("El mapa de episodios debe tener IDs únicos y cobertura completa en las predicciones.")
    seleccion = episodios.loc[episodios.episodio_id.ne(""), requeridas]
    detalle = seleccion.merge(datos, on="transaccion_id", how="left", validate="one_to_one")
    filas = []
    for episodio_id, grupo in detalle.groupby("episodio_id", sort=True):
        if any(grupo[c].nunique() != 1 for c in ("cuenta_id", "escenario", "patron")):
            raise ValueError("Cada episodio debe corresponder a una cuenta y un escenario únicos.")
        grupo = grupo.sort_values(["fecha_hora", "transaccion_id"], kind="stable")
        for enfoque, columna in ENFOQUES.items():
            posiciones = np.flatnonzero(grupo[columna].to_numpy() == 1)
            primera = grupo.iloc[posiciones[0]] if len(posiciones) else None
            filas.append({"episodio_id": episodio_id, "escenario": grupo.escenario.iloc[0],
                          "patron": grupo.patron.iloc[0], "cuenta_id": grupo.cuenta_id.iloc[0], "enfoque": enfoque,
                          "inicio": grupo.fecha_hora.min(), "fin": grupo.fecha_hora.max(),
                          "operaciones": len(grupo), "alertas": len(posiciones), "alguna_alerta": len(posiciones) > 0,
                          "cobertura_completa": len(posiciones) == len(grupo),
                          "primera_posicion": int(posiciones[0]) + 1 if primera is not None else np.nan,
                          "primera_fecha": primera.fecha_hora if primera is not None else pd.NaT,
                          "demora_segundos": (primera.fecha_hora - grupo.fecha_hora.min()).total_seconds()
                          if primera is not None else np.nan})
    columnas = ["episodio_id", "escenario", "patron", "cuenta_id", "enfoque", "inicio", "fin", "operaciones",
                "alertas", "alguna_alerta", "cobertura_completa", "primera_posicion", "primera_fecha", "demora_segundos"]
    return pd.DataFrame(filas, columns=columnas)


def resumir_episodios(episodios: pd.DataFrame) -> pd.DataFrame:
    """Demoras condicionadas a detección, con su número de casos explícito."""
    filas = []
    for (escenario, enfoque), grupo in episodios.groupby(["escenario", "enfoque"], sort=True):
        demoras = grupo.demora_segundos.dropna()
        filas.append({"escenario": escenario, "enfoque": enfoque, "episodios": len(grupo),
                      "con_alerta": int(grupo.alguna_alerta.sum()),
                      "con_cobertura_completa": int(grupo.cobertura_completa.sum()),
                      "operaciones_alertadas": int(grupo.alertas.sum()), "n_demoras": len(demoras),
                      "mediana_demora_segundos": demoras.median()})
    return pd.DataFrame(filas)


def _mapa_historico(comparacion: pd.DataFrame) -> pd.DataFrame:
    """Reutiliza la reconstrucción de 6B solo para las concentraciones históricas."""
    concentraciones = comparacion.loc[comparacion.tipo_anomalia.eq("concentracion"),
                                      ["transaccion_id", "cuenta_id", "fecha_hora"]]
    detalle, _ = diagnostico.reconstruir_rafagas(concentraciones, comparacion)
    return pd.DataFrame({"transaccion_id": detalle.transaccion_id,
                         "episodio_id": detalle.rafaga.map(lambda v: f"H{int(v):02d}"),
                         "escenario": "concentracion_anomala", "patron": "historica"})


def _mostrar(titulo: str, tabla: pd.DataFrame) -> None:
    print(f"\n{titulo}")
    print(tabla.to_string(index=False, na_rep="N/D", float_format=lambda v: f"{v:.6f}") if not tabla.empty else "Sin casos.")


def _informar(titulo: str, comparacion: ComparacionEnfoques, episodios: pd.DataFrame) -> None:
    _mostrar(titulo, comparacion.metricas)
    nuevas = comparacion.nuevas_alertas
    print(f"Alertas añadidas: {len(nuevas)}; TP nuevos: {int(nuevas.es_anomalia.sum())}; FP nuevos: {int(nuevas.es_anomalia.eq(0).sum())}.")
    _mostrar("Operaciones añadidas por la regla", nuevas[["transaccion_id", "cuenta_id", "fecha_hora", "movimientos_5min",
                                                          "puntaje_anomalia", "es_anomalia", "tipo_anomalia"]])
    _mostrar("Comportamiento por episodio completo", episodios)
    _mostrar("Resumen por episodio: alguna alerta y cobertura total", resumir_episodios(episodios))


def main() -> int:
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
        ultimo = resultados[-1]  # Elegido por orden temporal, nunca por sus métricas.
        escenarios = generar_escenarios(ultimo.entrenamiento[COLUMNAS_ORIGINALES], ultimo.referencias)
        nuevas_predicciones = predecir_congelado(escenarios.transacciones, ultimo, transacciones)
        # También la prueba nueva ya está predicha antes de leer etiquetas históricas.
        etiquetas_historicas = detector.cargar_etiquetas(detector.RUTA_ETIQUETAS, transacciones.transaccion_id)
        historicas = pd.concat([r.predicciones.assign(ventana=i) for i, r in enumerate(resultados, start=1)], ignore_index=True)
        historico = comparar_enfoques(historicas, etiquetas_historicas)
        nuevo = comparar_enfoques(nuevas_predicciones, escenarios.etiquetas)
        episodios_historicos = evaluar_episodios(historico.comparacion, _mapa_historico(historico.comparacion))
        episodios_nuevos = evaluar_episodios(nuevo.comparacion, escenarios.episodios)
    except ValueError as error:
        print(f"Error del experimento: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("Error inesperado del experimento. Revisá datos y dependencias.", file=sys.stderr)
        return 1

    print(f"Experimento 6C — scikit-learn {detector.version_sklearn}; PostgreSQL leído en solo lectura.")
    print(f"Regla fija: movimientos_5min >= {UMBRAL_CONCENTRACION}; híbrido = Isolation Forest OR regla.")
    print(f"Parámetros IF intactos: {detector.PARAMETROS_MODELO}")
    print(f"Prueba nueva: modelo congelado de {len(ultimo.entrenamiento)} movimientos, hasta {ultimo.entrenamiento.fecha_hora.max()}.")
    print(f"Historial disponible sin reajustar: {len(transacciones)} movimientos, hasta {transacciones.fecha_hora.max()}.")
    print(f"Semilla de escenarios {SEMILLA_ESCENARIOS}; {len(escenarios.transacciones)} movimientos de abril, 16 episodios predefinidos.")
    _informar("Histórico exploratorio — operaciones ya conocidas", historico, episodios_historicos)
    filas = []
    for ventana, grupo in historicas.groupby("ventana", sort=True):
        filas.append(comparar_enfoques(grupo, etiquetas_historicas).metricas.assign(ventana=ventana))
    _mostrar("Histórico por ventana (sin volver a entrenar)", pd.concat(filas, ignore_index=True))
    _informar("Escenarios nuevos — prueba sintética controlada con modelo fijo", nuevo, episodios_nuevos)
    detalle = nuevo.comparacion.merge(escenarios.episodios, on="transaccion_id", validate="one_to_one")
    filas = []
    for clase, grupo in detalle.groupby("escenario", sort=True):
        normales = grupo.es_anomalia.eq(0)
        filas.append({"escenario": clase, "operaciones": len(grupo),
                      "FP_IF": int((normales & grupo.prediccion_anomalia.eq(1)).sum()),
                      "FP_hibrido": int((normales & grupo.prediccion_hibrida.eq(1)).sum()),
                      "FP_nuevos": int((normales & grupo.alerta_nueva.eq(1)).sum())})
    _mostrar("Falsas alertas nuevas por clase de actividad", pd.DataFrame(filas))
    print("\nEpisodio detectado = alguna alerta; cobertura completa = todas sus operaciones alertadas. N/D = sin detección o soporte.")
    print("Las demoras solo resumen episodios detectados y muestran su denominador; no convierten omisiones en demora cero.")
    print("La comparación histórica es exploratoria. La prueba de abril es nueva, sintética y condicionada a perfiles del entrenamiento.")
    print("Patrones/semilla fijados antes de medir; anomalías cortas/espaciadas y controles legítimos impiden equiparar verdad a la regla.")
    print("Una regla parecida al generador puede rendir bien sin generalizar. Dos etiquetas semánticas pueden tener características similares.")
    print("El híbrido conserva todas las alertas IF y puede añadir falsas alertas. No se cambió el umbral ni se buscó otro protocolo.")
    print("Resultados en memoria. Sin cambios en módulos anteriores, PostgreSQL, CSV o Streamlit; sin integración automática.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
