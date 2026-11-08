"""Preparación en memoria: referencias estadísticas y tres características numéricas.

No consulta etiquetas ni persiste resultados. La lectura de PostgreSQL se realiza
únicamente al ejecutar main(), reutilizando los módulos de la etapa 3.
"""

from dataclasses import dataclass
from numbers import Number
import sys

import numpy as np
import pandas as pd

if __package__:
    from .database import ErrorConfiguracion, ErrorConexion, crear_engine
    from .verificar_conexion import cargar_transacciones
else:
    from database import ErrorConfiguracion, ErrorConexion, crear_engine
    from verificar_conexion import cargar_transacciones


COLUMNAS_ORIGINALES = [
    "transaccion_id", "cuenta_id", "fecha_hora", "tipo_movimiento",
    "direccion", "monto", "canal", "contraparte_id",
]
CARACTERISTICAS = ["monto_relativo", "rareza_horaria", "movimientos_5min"]


@dataclass(frozen=True)
class Referencias:
    """Parámetros ajustados; transformar no los recalcula ni incorpora historial."""

    medianas_por_cuenta: dict[str, float]
    mediana_global: float
    probabilidades_por_cuenta: dict[str, np.ndarray]
    probabilidades_globales: np.ndarray
    min_observaciones: int


def validar_y_convertir(transacciones: pd.DataFrame) -> pd.DataFrame:
    """Copia las ocho columnas; valida IDs, fechas sin zona e importes positivos.

    Conserva el orden y el índice recibido, incluso si este último está repetido.
    La identidad de una operación depende exclusivamente de transaccion_id.
    """
    if not isinstance(transacciones, pd.DataFrame):
        raise ValueError("Se esperaba un DataFrame de transacciones.")
    if (not transacciones.columns.is_unique
            or set(transacciones.columns) != set(COLUMNAS_ORIGINALES)):
        raise ValueError("Se requieren exactamente las ocho columnas originales.")
    datos = transacciones.loc[:, COLUMNAS_ORIGINALES].copy(deep=True)
    if datos.isna().to_numpy().any():
        raise ValueError("Las columnas obligatorias no pueden contener nulos.")
    columnas_texto = [c for c in COLUMNAS_ORIGINALES if c not in ("monto", "fecha_hora")]
    for columna in columnas_texto:
        if not datos[columna].map(lambda v: isinstance(v, str) and bool(v.strip())).all():
            raise ValueError(f"La columna {columna} debe contener textos no vacíos.")
    if datos["transaccion_id"].duplicated().any():
        raise ValueError("Los identificadores de transacción deben ser únicos.")

    if datos["monto"].map(lambda v: isinstance(v, (bool, np.bool_, complex, np.complexfloating))).any():
        raise ValueError("Los importes deben ser números reales positivos.")
    try:
        datos["monto"] = pd.to_numeric(datos["monto"], errors="raise").astype("float64")
    except (ValueError, TypeError, OverflowError):
        raise ValueError("No se pudieron convertir los importes a float64.") from None
    if not np.isfinite(datos["monto"].to_numpy()).all() or (datos["monto"] <= 0).any():
        raise ValueError("Los importes deben ser positivos y finitos.")

    # Evita interpretar accidentalmente números como fechas desde el epoch.
    if any(isinstance(valor, Number) for valor in datos["fecha_hora"]):
        raise ValueError("Las fechas deben ser fechas válidas, no números.")
    try:
        fechas = pd.to_datetime(datos["fecha_hora"], errors="raise", format="mixed")
    except (ValueError, TypeError, OverflowError):
        raise ValueError("No se pudieron convertir las fechas a datetime.") from None
    if (not pd.api.types.is_datetime64_any_dtype(fechas.dtype)
            or isinstance(fechas.dtype, pd.DatetimeTZDtype) or fechas.isna().any()):
        raise ValueError("Se requieren fechas válidas sin zona horaria, como en PostgreSQL.")
    datos["fecha_hora"] = fechas
    return datos


def _probabilidades_horarias(fechas: pd.Series) -> np.ndarray:
    """Histograma circular de 24 horas y suavizado de Laplace con alpha=1."""
    conteos = np.bincount(fechas.dt.hour.to_numpy(), minlength=24).astype("float64")
    suavizados = 0.25 * np.roll(conteos, 1) + 0.5 * conteos + 0.25 * np.roll(conteos, -1)
    return (suavizados + 1.0) / (len(fechas) + 24.0)


def ajustar_referencias(transacciones: pd.DataFrame, min_observaciones: int = 5) -> Referencias:
    """Estima medianas y horarios exclusivamente con el conjunto de referencia.

    Las cuentas con menos de cinco observaciones (por defecto) y las nuevas usan
    la mediana y distribución globales de ese mismo conjunto. No se usan etiquetas.
    """
    if (isinstance(min_observaciones, bool) or not isinstance(min_observaciones, int)
            or min_observaciones < 1):
        raise ValueError("min_observaciones debe ser un entero positivo.")
    datos = validar_y_convertir(transacciones)
    if datos.empty:
        raise ValueError("Se necesita al menos una transacción de referencia.")
    mediana_global = float(datos["monto"].median())
    medianas = {}
    probabilidades = {}
    for cuenta, grupo in datos.groupby("cuenta_id", sort=True):
        if len(grupo) >= min_observaciones:
            medianas[cuenta] = float(grupo["monto"].median())
            probabilidades[cuenta] = _probabilidades_horarias(grupo["fecha_hora"])
    if not all(np.isfinite(v) and v > 0 for v in [mediana_global, *medianas.values()]):
        raise ValueError("No se pudieron obtener medianas positivas y finitas.")
    return Referencias(medianas, mediana_global, probabilidades,
                       _probabilidades_horarias(datos["fecha_hora"]), min_observaciones)


def calcular_movimientos_5min(
    transacciones: pd.DataFrame, historial: pd.DataFrame | None = None,
) -> pd.Series:
    """Cuenta por cuenta en [t - 5 minutos, t], incluyendo la operación actual.

    A igual fecha_hora se procesan IDs en orden lexicográfico ascendente: solo
    cuentan el ID actual y los anteriores. Un historial explícito permite procesar
    lotes nuevos; sus filas futuras nunca cuentan. Los IDs compartidos se cuentan
    una sola vez si todos sus datos coinciden; si difieren, se rechazan.
    """
    datos = validar_y_convertir(transacciones)
    anteriores = validar_y_convertir(historial) if historial is not None else datos.iloc[:0].copy()
    compartidos = anteriores["transaccion_id"].isin(datos["transaccion_id"])
    if compartidos.any():
        comunes = anteriores.loc[compartidos].set_index("transaccion_id", drop=False)
        actuales = datos.set_index("transaccion_id", drop=False).loc[comunes.index]
        if not (comunes.eq(actuales)).to_numpy().all():
            raise ValueError("Una transacción compartida con el historial tiene datos distintos.")
    if datos.empty:
        return pd.Series([], index=datos.index, dtype="int64", name="movimientos_5min")

    todos = pd.concat([anteriores.loc[~compartidos], datos], ignore_index=True)
    todos = todos.sort_values(["cuenta_id", "fecha_hora", "transaccion_id"], kind="stable")
    conteos_por_id = {}
    for _, grupo in todos.groupby("cuenta_id", sort=False):
        fechas = pd.DatetimeIndex(grupo["fecha_hora"])
        # searchsorted(left) incluye el límite exacto; el prefijo excluye futuros
        # y los IDs posteriores a igual horario, incluso si aparecen en historial.
        inicios = fechas.searchsorted(fechas - pd.Timedelta(minutes=5), side="left")
        conteos = np.arange(len(grupo), dtype="int64") + 1 - inicios
        conteos_por_id.update(zip(grupo["transaccion_id"], conteos))
    valores = datos["transaccion_id"].map(conteos_por_id).to_numpy(dtype="int64")
    return pd.Series(valores, index=datos.index, name="movimientos_5min", dtype="int64")


def transformar_datos(
    transacciones: pd.DataFrame, referencias: Referencias,
    historial: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Aplica referencias fijas y devuelve las ocho columnas más tres variables.

    El historial sirve solo para frecuencia, nunca para volver a ajustar medianas
    ni distribuciones horarias. No altera los datos recibidos ni las referencias.
    """
    datos = validar_y_convertir(transacciones)
    movimientos = calcular_movimientos_5min(datos, historial)
    medianas = datos["cuenta_id"].map(referencias.medianas_por_cuenta).fillna(referencias.mediana_global)
    horas = datos["fecha_hora"].dt.hour.to_numpy()
    probabilidades = np.array([
        referencias.probabilidades_por_cuenta.get(cuenta, referencias.probabilidades_globales)[hora]
        for cuenta, hora in zip(datos["cuenta_id"], horas)
    ], dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore", over="ignore", under="ignore"):
        datos["monto_relativo"] = datos["monto"].to_numpy() / medianas.to_numpy(dtype="float64")
        datos["rareza_horaria"] = -np.log(probabilidades)
    # Asignación posicional tras recuperar por ID, sin usar el índice como identidad.
    datos["movimientos_5min"] = movimientos.to_numpy()
    variables = datos[CARACTERISTICAS].to_numpy(dtype="float64")
    if not np.isfinite(variables).all() or (datos["monto_relativo"] <= 0).any():
        raise ValueError("Las características deben ser finitas y el monto relativo positivo.")
    if (len(datos) != len(transacciones)
            or datos["transaccion_id"].tolist() != transacciones["transaccion_id"].tolist()):
        raise ValueError("No se conservaron las transacciones y sus identificadores.")
    return datos


def main() -> int:
    """Demostración descriptiva sobre PostgreSQL; no constituye una evaluación."""
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
        # Cualquier excepción del controlador o del cierre puede incluir datos
        # sensibles, incluso si es ValueError: nunca imprimir su detalle original.
        print("Error al leer PostgreSQL o liberar la conexión. Revisá la configuración.", file=sys.stderr)
        return 1

    try:
        referencias = ajustar_referencias(transacciones)
        preparados = transformar_datos(transacciones, referencias)
    except ValueError as error:
        print(f"Error de preparación: {error}", file=sys.stderr)
        return 1
    except Exception:
        print("Error inesperado al preparar los datos. Revisá la configuración y la conexión.", file=sys.stderr)
        return 1

    print("Demostración descriptiva: las referencias se ajustan con todo el dataset.")
    print("Esto no constituye una evaluación válida de un modelo.")
    print(f"Registros procesados: {len(preparados)}")
    print(f"Cuentas diferentes: {preparados['cuenta_id'].nunique()}")
    print(f"Total columnas: {len(preparados.columns)}")
    print("\nPrimeras cinco transacciones preparadas:")
    print(preparados.head(5).to_string(index=False))
    print("\nTipos de datos de las características:")
    print(preparados[CARACTERISTICAS].dtypes.to_string())
    print("\nEstadísticas descriptivas:")
    print(preparados[CARACTERISTICAS].describe().to_string())
    print("\nSin nulos ni infinitos en monto ni en las tres características; fechas válidas.")
    print("Filas e identificadores conservados. No se guardaron archivos ni se escribió en PostgreSQL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
