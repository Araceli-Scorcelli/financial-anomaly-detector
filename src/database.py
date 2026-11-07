"""Configuración reutilizable de PostgreSQL y conexiones de solo lectura.

Importar este módulo no lee credenciales ni abre conexiones. El Engine es
perezoso: la conexión física se solicita al entrar en conectar_solo_lectura.
"""

from collections.abc import Iterator
from contextlib import contextmanager
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine, URL
from sqlalchemy.exc import OperationalError, SQLAlchemyError


class ErrorConfiguracion(ValueError):
    """Configuración local incompleta o inválida, sin valores sensibles."""


class ErrorConexion(RuntimeError):
    """Fallo de conexión o lectura, con un mensaje seguro para consola."""


def crear_engine(ruta_env: Path | None = None) -> Engine:
    """Configura un motor sin conectarse y conserva las variables del entorno."""
    archivo_env = Path(ruta_env) if ruta_env is not None else Path(__file__).resolve().parents[1] / ".env"
    try:
        load_dotenv(dotenv_path=archivo_env, override=False, interpolate=False, encoding="utf-8")
    except (OSError, UnicodeError):
        raise ErrorConfiguracion("No se pudo leer .env. Revisá sus permisos y su codificación UTF-8.") from None

    variables = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")
    valores = {variable: os.getenv(variable) for variable in variables}
    faltantes = [variable for variable, valor in valores.items() if valor is None or not valor.strip()]
    if faltantes:
        raise ErrorConfiguracion(
            "Faltan variables requeridas: " + ", ".join(faltantes)
            + ". Copiá .env.example a .env y completá la configuración local."
        )
    try:
        puerto = int(valores["DB_PORT"])
    except ValueError:
        raise ErrorConfiguracion("DB_PORT debe ser un entero entre 1 y 65535.") from None
    if not 1 <= puerto <= 65535:
        raise ErrorConfiguracion("DB_PORT debe ser un entero entre 1 y 65535.")

    # URL.create recibe la contraseña literal y evita concatenarla en una URL.
    url = URL.create(
        "postgresql+psycopg", host=valores["DB_HOST"], port=puerto,
        database=valores["DB_NAME"], username=valores["DB_USER"], password=valores["DB_PASSWORD"],
    )
    try:
        return create_engine(
            url,
            echo=False,
            hide_parameters=True,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 5,
                "options": "-c default_transaction_read_only=on",
            },
            execution_options={"postgresql_readonly": True},
        )
    except ImportError:
        raise ErrorConfiguracion("Falta el controlador PostgreSQL. Instalá requirements.txt en .venv.") from None
    except (SQLAlchemyError, OSError, ValueError):
        raise ErrorConfiguracion("No se pudo configurar el motor PostgreSQL. Revisá las variables y las dependencias.") from None


@contextmanager
def conectar_solo_lectura(engine: Engine) -> Iterator[Connection]:
    """Entrega una conexión con READ ONLY comprobado y la cierra al terminar.

    La característica se aplica antes de la primera consulta. Salir del contexto
    revierte la transacción de lectura y devuelve la conexión al pool. No se emiten
    comandos de creación o modificación de tablas o registros.
    """
    try:
        with engine.connect() as conexion:
            conexion = conexion.execution_options(postgresql_readonly=True)
            if conexion.execute(text("SHOW transaction_read_only")).scalar_one() != "on":
                raise ErrorConexion("La sesión no confirmó el modo de solo lectura. Se canceló la consulta.")
            yield conexion
    except OperationalError:
        raise ErrorConexion(
            "No se pudo conectar o mantener la conexión con PostgreSQL. "
            "Revisá que el servidor esté activo, DB_HOST, DB_PORT y las credenciales locales."
        ) from None
    except (SQLAlchemyError, OSError):
        raise ErrorConexion(
            "No se pudo completar la consulta. Revisá DB_NAME, la tabla "
            "public.transacciones y los permisos de lectura del usuario."
        ) from None
