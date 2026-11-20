"""Configuración de la fuente y lectura validada del dataset de demostración."""

import os
from pathlib import Path

from dotenv.parser import parse_stream
import pandas as pd

from src.preparar_datos import validar_y_convertir


RAIZ_PROYECTO = Path(__file__).resolve().parents[1]
RUTA_TRANSACCIONES = RAIZ_PROYECTO / "data" / "transacciones.csv"
RUTA_ENV = RAIZ_PROYECTO / ".env"


class ErrorFuenteDatos(ValueError):
    """Configuración inválida, sin reproducir valores potencialmente sensibles."""


def validar_fuente_datos(fuente: str) -> str:
    if fuente not in ("postgres", "csv"):
        raise ErrorFuenteDatos("DATA_SOURCE debe ser postgres o csv. Revisá la configuración y reiniciá Streamlit.")
    return fuente


def _fuente_desde_env() -> str:
    """Leer el .env opcional sin interpolar ni cargar credenciales en el proceso.

    Se usa el parser de python-dotenv para detectar sintaxis inválida en lugar
    de ignorarla y seleccionar PostgreSQL silenciosamente.
    """
    fuente = "postgres"
    try:
        archivo = RUTA_ENV.open(encoding="utf-8")
    except FileNotFoundError:
        return fuente
    with archivo:
        for entrada in parse_stream(archivo):
            if entrada.error:
                raise ErrorFuenteDatos("No se pudo interpretar .env. Revisá su sintaxis o definí DATA_SOURCE en PowerShell.")
            if entrada.key == "DATA_SOURCE":
                fuente = entrada.value
    return fuente


def obtener_fuente_datos() -> str:
    """El entorno prevalece sobre el .env opcional; PostgreSQL es el valor inicial.

    Una selección explícita (por ejemplo, DATA_SOURCE=csv) no lee .env.
    La lectura opcional no modifica os.environ ni interpola sus valores.
    Se resuelve antes de entrar en la caché del dashboard.
    """
    fuente = os.getenv("DATA_SOURCE")
    if fuente is None:
        try:
            fuente = _fuente_desde_env()
        except (OSError, UnicodeError):
            raise ErrorFuenteDatos("No se pudo leer la configuración local. Revisá .env o definí DATA_SOURCE en PowerShell.") from None
    return validar_fuente_datos(fuente)


def cargar_transacciones_csv() -> pd.DataFrame:
    """Solo las ocho columnas originales, con la validación de la etapa 4A.

    Leer primero como texto conserva identificadores y deja que el validador
    compartido convierta montos y fechas. No lee etiquetas ni abre conexiones.
    """
    datos = pd.read_csv(RUTA_TRANSACCIONES, dtype=str, keep_default_na=False)
    if not isinstance(datos.index, pd.RangeIndex):
        raise ValueError("El CSV contiene campos adicionales sin encabezado.")
    return validar_y_convertir(datos)
