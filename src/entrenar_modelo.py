"""Primera evaluación temporal de Isolation Forest; etiquetas solo al finalizar.

Reutiliza la conexión de solo lectura y las características de la etapa 4A.
No guarda modelos ni resultados, ni modifica archivos de datos o PostgreSQL.
"""

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn import __version__ as version_sklearn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

if __package__:
    from .database import ErrorConfiguracion, ErrorConexion, crear_engine
    from .preparar_datos import (
        CARACTERISTICAS, Referencias, ajustar_referencias, transformar_datos, validar_y_convertir,
    )
    from .verificar_conexion import cargar_transacciones
else:
    from database import ErrorConfiguracion, ErrorConexion, crear_engine
    from preparar_datos import (
        CARACTERISTICAS, Referencias, ajustar_referencias, transformar_datos, validar_y_convertir,
    )
    from verificar_conexion import cargar_transacciones


# Elegidos antes de consultar etiquetas; no se ajustan según las métricas.
PARAMETROS_MODELO = {
    "n_estimators": 200, "max_samples": "auto", "contamination": "auto",
    "max_features": 1.0, "bootstrap": False, "random_state": 42, "n_jobs": 1,
}
TIPOS_ANOMALIA = ("importe_inusual", "horario_atipico", "concentracion")
RUTA_ETIQUETAS = Path(__file__).resolve().parents[1] / "data" / "etiquetas_evaluacion.csv"


@dataclass
class ResultadoEntrenamiento:
    modelo: IsolationForest
    referencias: Referencias
    entrenamiento: pd.DataFrame
    predicciones: pd.DataFrame


@dataclass
class ResultadoEvaluacion:
    comparacion: pd.DataFrame
    precision: float
    recall: float
    f1: float
    matriz_confusion: np.ndarray
    por_tipo: pd.DataFrame


def separar_cronologicamente(
    transacciones: pd.DataFrame, fraccion_entrenamiento: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Primer 80% para entrenamiento, resto futuro para evaluación, sin barajar.

    Ordena por timestamp completo e ID. Si el corte nominal comparte timestamp,
    envía todo ese grupo a evaluación. Rechaza particiones vacías; garantiza que
    la última fecha de entrenamiento sea anterior a la primera de evaluación.
    """
    if (isinstance(fraccion_entrenamiento, bool) or not isinstance(fraccion_entrenamiento, Real)
            or not 0 < fraccion_entrenamiento < 1):
        raise ValueError("La fracción de entrenamiento debe estar entre 0 y 1.")
    datos = validar_y_convertir(transacciones).sort_values(
        ["fecha_hora", "transaccion_id"], kind="stable",
    ).reset_index(drop=True)
    corte = int(len(datos) * fraccion_entrenamiento)
    if corte < 1 or corte >= len(datos):
        raise ValueError("Se necesitan transacciones en entrenamiento y evaluación.")
    inicio_evaluacion = datos.iloc[corte]["fecha_hora"]
    entrenamiento = datos.loc[datos["fecha_hora"] < inicio_evaluacion].copy()
    evaluacion = datos.loc[datos["fecha_hora"] >= inicio_evaluacion].copy()
    if entrenamiento.empty or evaluacion.empty:
        raise ValueError("No hay una separación temporal con ambos conjuntos no vacíos.")
    return entrenamiento, evaluacion


def entrenar_y_predecir(transacciones: pd.DataFrame) -> ResultadoEntrenamiento:
    """Ajusta referencias y un único modelo solo con entrenamiento; nunca lee etiquetas.

    Los conteos son causales dentro de cada lote; evaluación recibe además el
    historial de entrenamiento. Las medianas/distribuciones quedan fijas al corte.
    """
    entrenamiento, evaluacion = separar_cronologicamente(transacciones)
    return entrenar_particion(entrenamiento, evaluacion)


def entrenar_particion(
    entrenamiento: pd.DataFrame, evaluacion: pd.DataFrame,
) -> ResultadoEntrenamiento:
    """Mismo detector de 4B sobre una partición temporal explícita, sin etiquetas.

    Compartido con walk-forward: las referencias usan solo entrenamiento y los
    conteos de evaluación incluyen su historial. No cambia parámetros ni umbral.
    """
    entrenamiento = validar_y_convertir(entrenamiento).sort_values(
        ["fecha_hora", "transaccion_id"], kind="stable",
    )
    evaluacion = validar_y_convertir(evaluacion).sort_values(
        ["fecha_hora", "transaccion_id"], kind="stable",
    )
    if (entrenamiento.empty or evaluacion.empty
            or entrenamiento["fecha_hora"].max() >= evaluacion["fecha_hora"].min()
            or set(entrenamiento["transaccion_id"]) & set(evaluacion["transaccion_id"])):
        raise ValueError("Se requieren particiones no vacías, sin IDs compartidos y con entrenamiento anterior a evaluación.")
    referencias = ajustar_referencias(entrenamiento)
    preparados_train = transformar_datos(entrenamiento, referencias)
    preparados_eval = transformar_datos(evaluacion, referencias, historial=entrenamiento)
    modelo = IsolationForest(**PARAMETROS_MODELO)
    modelo.fit(preparados_train[CARACTERISTICAS])
    puntajes = -modelo.score_samples(preparados_eval[CARACTERISTICAS])
    if not np.isfinite(puntajes).all():
        raise ValueError("El modelo produjo puntajes no finitos.")
    preparados_eval["puntaje_anomalia"] = puntajes
    # auto usa offset_=-0.5. Equivale a predict==-1, incluyendo empate normal.
    preparados_eval["prediccion_anomalia"] = (puntajes > -float(modelo.offset_)).astype("int64")
    return ResultadoEntrenamiento(modelo, referencias, preparados_train, preparados_eval)


def _validar_ids(tabla: pd.DataFrame) -> None:
    if (not tabla.columns.is_unique or "transaccion_id" not in tabla
            or tabla["transaccion_id"].isna().any()
            or not tabla["transaccion_id"].map(lambda v: isinstance(v, str) and bool(v.strip())).all()
            or tabla["transaccion_id"].duplicated().any()):
        raise ValueError("Se requieren identificadores de transacción únicos y no vacíos.")


def _validar_etiquetas(etiquetas: pd.DataFrame) -> pd.DataFrame:
    if set(etiquetas.columns) != {"transaccion_id", "es_anomalia", "tipo_anomalia"}:
        raise ValueError("El archivo de etiquetas debe tener las tres columnas esperadas.")
    _validar_ids(etiquetas)
    datos = etiquetas.copy(deep=True)
    if datos["es_anomalia"].isna().any() or not datos["es_anomalia"].isin([0, 1]).all():
        raise ValueError("es_anomalia debe contener únicamente 0 o 1.")
    datos["es_anomalia"] = datos["es_anomalia"].astype("int64")
    normales = datos["es_anomalia"] == 0
    if (not datos.loc[normales, "tipo_anomalia"].eq("").all()
            or not datos.loc[~normales, "tipo_anomalia"].isin(TIPOS_ANOMALIA).all()):
        raise ValueError("El tipo de anomalía no coincide con el indicador o no es reconocido.")
    return datos


def cargar_etiquetas(ruta: Path, ids_esperados) -> pd.DataFrame:
    """Lee solo para evaluación; exige cobertura exacta de los IDs recuperados de DB."""
    try:
        datos = pd.read_csv(ruta, dtype={"transaccion_id": "str"}, keep_default_na=False)
    except (OSError, UnicodeError, pd.errors.ParserError, ValueError):
        raise ValueError("No se pudo leer el CSV de etiquetas. Revisá su ubicación y formato.") from None
    datos = _validar_etiquetas(datos)
    if set(datos["transaccion_id"]) != set(ids_esperados):
        raise ValueError("Las etiquetas no coinciden exactamente con los IDs recuperados de PostgreSQL.")
    return datos


def evaluar_predicciones(predicciones: pd.DataFrame, etiquetas: pd.DataFrame) -> ResultadoEvaluacion:
    """Une por ID y mide el detector binario sin entrenar ni cambiar su umbral.

    Las etiquetas pueden incluir entrenamiento, pero las métricas usan únicamente
    los IDs de predicciones (evaluación). El modelo no clasifica tipos de anomalía.
    """
    _validar_ids(predicciones)
    if (predicciones.empty or "prediccion_anomalia" not in predicciones
            or not predicciones["prediccion_anomalia"].isin([0, 1]).all()):
        raise ValueError("Se necesitan predicciones binarias no vacías para evaluar.")
    datos_etiquetas = _validar_etiquetas(etiquetas)
    if not set(predicciones["transaccion_id"]).issubset(set(datos_etiquetas["transaccion_id"])):
        raise ValueError("Faltan etiquetas para una o más predicciones.")
    comparacion = predicciones.merge(datos_etiquetas, on="transaccion_id", how="left",
                                     validate="one_to_one", sort=False)
    reales = comparacion["es_anomalia"].to_numpy()
    predichas = comparacion["prediccion_anomalia"].to_numpy()
    precision, recall, f1, _ = precision_recall_fscore_support(
        reales, predichas, average="binary", pos_label=1, zero_division=0,
    )
    matriz = confusion_matrix(reales, predichas, labels=[0, 1])
    filas = []
    for tipo in TIPOS_ANOMALIA:
        grupo = comparacion.loc[comparacion["tipo_anomalia"] == tipo]
        cantidad = len(grupo)
        detectadas = int(grupo["prediccion_anomalia"].sum())
        filas.append({"tipo_anomalia": tipo, "reales": cantidad, "detectadas": detectadas,
                      "no_detectadas": cantidad - detectadas,
                      "recall": detectadas / cantidad if cantidad else np.nan})
    return ResultadoEvaluacion(comparacion, float(precision), float(recall), float(f1),
                               matriz, pd.DataFrame(filas))


def principales_anomalias(predicciones: pd.DataFrame, cantidad: int = 10) -> pd.DataFrame:
    """Mayores puntajes entre las operaciones evaluadas; empates ordenados por ID."""
    if isinstance(cantidad, bool) or not isinstance(cantidad, int) or cantidad < 1:
        raise ValueError("La cantidad del ranking debe ser un entero positivo.")
    return predicciones.sort_values(["puntaje_anomalia", "transaccion_id"],
                                    ascending=[False, True], kind="stable").head(cantidad).copy()


def main() -> int:
    """Lectura real, entrenamiento/predicción sin etiquetas y evaluación posterior."""
    try:
        engine = crear_engine()
        try:
            transacciones = cargar_transacciones(engine)
        finally:
            engine.dispose()
    except (ErrorConfiguracion, ErrorConexion) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("Error al leer PostgreSQL o liberar la conexión. Revisá la configuración.", file=sys.stderr)
        return 1

    try:
        resultado = entrenar_y_predecir(transacciones)
        # Esta lectura es posterior a generar TODOS los puntajes y predicciones.
        etiquetas = cargar_etiquetas(RUTA_ETIQUETAS, transacciones["transaccion_id"])
        evaluacion = evaluar_predicciones(resultado.predicciones, etiquetas)
    except ValueError as error:
        print(f"Error de entrenamiento o evaluación: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("Error inesperado al entrenar o evaluar. Revisá dependencias y datos.", file=sys.stderr)
        return 1

    entrenamiento, predicciones = resultado.entrenamiento, resultado.predicciones
    print(f"Isolation Forest — scikit-learn {version_sklearn}")
    print("PostgreSQL leído en modo solo lectura. Referencias y modelo ajustados solo con entrenamiento.")
    print(f"Transacciones: {len(transacciones)} | Cuentas distintas: {transacciones['cuenta_id'].nunique()}")
    print(f"Entrenamiento: {len(entrenamiento)} ({len(entrenamiento) / len(transacciones):.1%})")
    print(f"Evaluación: {len(predicciones)} ({len(predicciones) / len(transacciones):.1%})")
    print(f"Período de entrenamiento: {entrenamiento.fecha_hora.min()} a {entrenamiento.fecha_hora.max()}")
    print(f"Período de evaluación: {predicciones.fecha_hora.min()} a {predicciones.fecha_hora.max()}")
    print(f"Parámetros fijos: {PARAMETROS_MODELO}")
    print(f"Muestras por árbol: {resultado.modelo.max_samples_}")
    print(f"Anomalía si puntaje_anomalia > {-float(resultado.modelo.offset_):.6f}; puntaje = -score_samples.")
    print("Las etiquetas se leyeron después de generar las predicciones; no se ajustaron parámetros con ellas.")
    tn, fp, fn, tp = evaluacion.matriz_confusion.ravel()
    print(f"\nAnomalías reales en evaluación: {tp + fn}")
    print(f"Anomalías detectadas: {tp + fp}")
    print(f"Verdaderos positivos: {tp} | Falsos positivos: {fp} | No detectadas: {fn}")
    print(f"Precisión: {evaluacion.precision:.6f}")
    print(f"Recall: {evaluacion.recall:.6f}")
    print(f"F1-score: {evaluacion.f1:.6f}")
    print("Matriz de confusión: filas=reales [0,1]; columnas=predicciones [0,1]")
    print(evaluacion.matriz_confusion)
    if tp + fn == 0:
        print("No hay anomalías reales en evaluación: el recall no permite medir su detección.")
    print("\nResultados por tipo real (no son clases predichas; N/D = sin casos):")
    print(evaluacion.por_tipo.to_string(index=False, na_rep="N/D"))
    print("\nDiez operaciones evaluadas con mayores puntajes:")
    columnas = ["transaccion_id", "cuenta_id", "fecha_hora", "monto", *CARACTERISTICAS,
                "puntaje_anomalia", "prediccion_anomalia"]
    print(principales_anomalias(predicciones)[columnas].to_string(index=False))
    print("\nModelo y resultados en memoria; no se modificaron archivos ni PostgreSQL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
