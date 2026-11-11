"""Evaluación walk-forward complementaria; datos y resultados solo en memoria."""

from dataclasses import dataclass
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

if __package__:
    from . import entrenar_modelo as detector
else:
    import entrenar_modelo as detector


@dataclass
class MetricasTemporales:
    metricas: detector.ResultadoEvaluacion
    average_precision: float | None

    @property
    def recall(self) -> float | None:
        """Sin positivos reales no hay soporte para medir recuperación."""
        return self.metricas.recall if self.metricas.comparacion["es_anomalia"].any() else None


@dataclass
class InformeTemporal:
    por_ventana: list[MetricasTemporales]
    agregado: MetricasTemporales
    anomalias_no_evaluadas: pd.DataFrame


def crear_ventanas(transacciones: pd.DataFrame) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Prefijos de 40/60/80 % y los tres bloques siguientes, sin mirar etiquetas.

    Reutiliza el corte de 4B: todo timestamp compartido pasa al lado derecho.
    Los límites se fijan una sola vez sobre el conjunto completo. Los bloques de
    evaluación son adyacentes; una evaluación pasada puede ser entrenamiento en
    una ventana posterior. Se rechazan cortes colapsados por falta de datos.
    """
    datos = detector.validar_y_convertir(transacciones).sort_values(
        ["fecha_hora", "transaccion_id"], kind="stable",
    ).reset_index(drop=True)
    cortes = [len(detector.separar_cronologicamente(datos, fraccion)[0])
              for fraccion in (0.4, 0.6, 0.8)]
    cortes.append(len(datos))
    if not all(a < b for a, b in zip([0, *cortes[:-1]], cortes)):
        raise ValueError("No se pueden formar tres ventanas temporales no vacías sin separar timestamps iguales.")
    return [(datos.iloc[:inicio].copy(), datos.iloc[inicio:fin].copy())
            for inicio, fin in zip(cortes[:-1], cortes[1:])]


def predecir_ventanas(transacciones: pd.DataFrame) -> list[detector.ResultadoEntrenamiento]:
    """Ajusta tres referencias/modelos independientes, sin acceder a etiquetas."""
    return [detector.entrenar_particion(train, evaluacion)
            for train, evaluacion in crear_ventanas(transacciones)]


def medir_predicciones(predicciones: pd.DataFrame, etiquetas: pd.DataFrame) -> MetricasTemporales:
    """Métricas de 4B y AP con puntajes continuos, tras vincular etiquetas por ID.

    Por criterio explícito, AP se informa solo cuando hay positivos y negativos
    reales. Con una sola clase no permite evaluar la separación entre ambas.
    """
    metricas = detector.evaluar_predicciones(predicciones, etiquetas)
    comparacion = metricas.comparacion
    if "puntaje_anomalia" not in comparacion:
        raise ValueError("Se requieren puntajes de anomalía para evaluar el ranking.")
    puntajes = comparacion["puntaje_anomalia"].to_numpy(dtype="float64")
    if not np.isfinite(puntajes).all():
        raise ValueError("Los puntajes de anomalía deben ser finitos.")
    ap = (float(average_precision_score(comparacion["es_anomalia"], puntajes))
          if comparacion["es_anomalia"].nunique() == 2 else None)
    return MetricasTemporales(metricas, ap)


def evaluar_ventanas(
    resultados: list[detector.ResultadoEntrenamiento], etiquetas: pd.DataFrame,
) -> InformeTemporal:
    """Mide predicciones ya generadas; el agregado exige IDs únicos, sin promediar."""
    if len(resultados) != 3:
        raise ValueError("Se requieren las predicciones de las tres ventanas.")
    todos_ids = set().union(*(set(r.entrenamiento["transaccion_id"])
                              | set(r.predicciones["transaccion_id"]) for r in resultados))
    if set(etiquetas["transaccion_id"]) != todos_ids:
        raise ValueError("Las etiquetas deben corresponder a todos los movimientos de las ventanas.")
    # evaluar_predicciones valida el join uno a uno y rechaza duplicados: nunca
    # se descartan silenciosamente operaciones repetidas para armar el agregado.
    predicciones = pd.concat([r.predicciones for r in resultados], ignore_index=True)
    agregado = medir_predicciones(predicciones, etiquetas)
    por_ventana = [medir_predicciones(r.predicciones, etiquetas) for r in resultados]
    no_evaluadas = etiquetas.loc[
        etiquetas["es_anomalia"].eq(1) & ~etiquetas["transaccion_id"].isin(predicciones["transaccion_id"]),
        ["transaccion_id", "tipo_anomalia"],
    ].sort_values("transaccion_id").reset_index(drop=True)
    return InformeTemporal(por_ventana, agregado, no_evaluadas)


def _imprimir_metricas(titulo: str, resultado: MetricasTemporales) -> None:
    metricas = resultado.metricas
    tn, fp, fn, tp = (int(v) for v in metricas.matriz_confusion.ravel())
    print(f"\n{titulo}")
    print(f"Transacciones evaluadas: {len(metricas.comparacion)} | Anomalías reales: {tp + fn} | Alertas: {tp + fp}")
    print(f"TP: {tp} | FP: {fp} | TN: {tn} | FN: {fn}")
    print(f"Precisión: {metricas.precision:.6f}")
    print(f"Recall: {resultado.recall:.6f}" if resultado.recall is not None
          else "Recall: N/D (sin anomalías reales)")
    print(f"F1-score: {metricas.f1:.6f}")
    print(f"Average Precision: {resultado.average_precision:.6f}" if resultado.average_precision is not None
          else "Average Precision: N/D (se requieren ambas clases reales)")
    print("Matriz [[TN, FP], [FN, TP]]:")
    print(metricas.matriz_confusion)
    print("Por tipo real; N/D = sin casos, el modelo no predice tipos:")
    print(metricas.por_tipo.to_string(index=False, na_rep="N/D"))


def main() -> int:
    """Lee PostgreSQL una vez; predice todo y comprueba 80/20 antes de leer etiquetas."""
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
        resultados = predecir_ventanas(transacciones)
        # Comprobación independiente del entry point anterior, también sin etiquetas.
        base = detector.entrenar_y_predecir(transacciones)
        if (not resultados[-1].entrenamiento.equals(base.entrenamiento)
                or not resultados[-1].predicciones.equals(base.predicciones)):
            raise ValueError("La última ventana no reproduce el 80/20. Revisá cortes, orden y referencias antes de interpretar métricas.")
        etiquetas = detector.cargar_etiquetas(detector.RUTA_ETIQUETAS, transacciones["transaccion_id"])
        informe = evaluar_ventanas(resultados, etiquetas)
    except ValueError as error:
        print(f"Error de evaluación temporal: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("Error inesperado en la evaluación temporal. Revisá datos y dependencias.", file=sys.stderr)
        return 1

    print(f"Walk-forward — scikit-learn {detector.version_sklearn}")
    print(f"PostgreSQL leído en solo lectura: {len(transacciones)} transacciones, {transacciones.cuenta_id.nunique()} cuentas.")
    print("Cortes globales 40/60/80 %, ajustados por timestamps; períodos elegidos sin etiquetas.")
    print(f"Parámetros fijos: {detector.PARAMETROS_MODELO}")
    print("Puntaje = -score_samples; alerta si puntaje > 0.5, sin ajuste con etiquetas.")
    print("Referencias solo con entrenamiento; conteos de cinco minutos causales, con historial.")
    print("Última ventana frente al 80/20: coincidencia exacta de entrenamiento, características, puntajes y predicciones.")
    print("Todas las predicciones preceden a la lectura de etiquetas; no se seleccionan períodos con ellas.")
    for numero, (resultado, metricas) in enumerate(zip(resultados, informe.por_ventana), start=1):
        train, evaluacion = resultado.entrenamiento, resultado.predicciones
        print(f"\nVentana {numero} — entrenamiento: {len(train)}; evaluación: {len(evaluacion)}")
        print(f"Entrenamiento: {train.fecha_hora.min()} a {train.fecha_hora.max()}")
        print(f"Evaluación: {evaluacion.fecha_hora.min()} a {evaluacion.fecha_hora.max()}")
        _imprimir_metricas(f"Resultados de ventana {numero}", metricas)
    _imprimir_metricas("Agregado — cada transacción de evaluación contada una sola vez", informe.agregado)
    print("AP agregada descriptiva: combina puntajes de modelos distintos, sin calibración entre ventanas.")
    print("Precisión y F1 sin denominador se informan como 0 (convención de 4B); recall sin casos = N/D.")
    print(f"\nAnomalías no evaluadas (solo en entrenamiento inicial): {len(informe.anomalias_no_evaluadas)}")
    if not informe.anomalias_no_evaluadas.empty:
        print(informe.anomalias_no_evaluadas.to_string(index=False))
        print("Cantidad no evaluada por tipo:")
        print(informe.anomalias_no_evaluadas.tipo_anomalia.value_counts().to_string())
    print("\nLimitaciones: datos sintéticos, pocos positivos, períodos de distinta duración y entrenamiento con posibles anomalías.")
    print("Evaluación complementaria: no constituye una prueba independiente para ajustar y volver a medir parámetros.")
    print("Resultados en memoria; no se modificaron archivos de datos, PostgreSQL ni Streamlit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
