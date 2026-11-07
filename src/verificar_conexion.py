"""Verificación manual: SELECT 1 y lectura de public.transacciones con Pandas."""

import sys

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

if __package__:
    from .database import ErrorConfiguracion, ErrorConexion, conectar_solo_lectura, crear_engine
else:
    # Permite también: python src/verificar_conexion.py.
    from database import ErrorConfiguracion, ErrorConexion, conectar_solo_lectura, crear_engine


CONSULTA_TRANSACCIONES = text("""
SELECT transaccion_id, cuenta_id, fecha_hora, tipo_movimiento,
       direccion, monto, canal, contraparte_id
FROM public.transacciones
ORDER BY fecha_hora, transaccion_id
""")


def cargar_transacciones(engine: Engine) -> pd.DataFrame:
    """Comprueba la conexión y recupera todas las filas, conservando los Decimal."""
    with conectar_solo_lectura(engine) as conexion:
        if conexion.execute(text("SELECT 1")).scalar_one() != 1:
            raise ErrorConexion("La comprobación SELECT 1 no devolvió el resultado esperado.")
        return pd.read_sql_query(
            CONSULTA_TRANSACCIONES, conexion,
            parse_dates=["fecha_hora"], coerce_float=False,
        )


def main() -> int:
    """Devuelve 0 solo tras leer los datos y liberar los recursos correctamente."""
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
        # Frontera de consola: los errores inesperados, incluso de cierre, pueden
        # contener una URL o datos del controlador. No imprimir el error original.
        print(
            "Error: no se pudo completar la verificación o liberar sus recursos. "
            "Revisá las dependencias, la configuración y la estructura de public.transacciones.",
            file=sys.stderr,
        )
        return 1

    print("Conexión PostgreSQL verificada con SELECT 1 y lectura en modo solo lectura.")
    print(f"Movimientos: {len(transacciones)}")
    print(f"Cuentas distintas: {transacciones['cuenta_id'].nunique()}")
    print("\nTipos de datos:")
    print(transacciones.dtypes.to_string())
    print("\nPrimeras cinco transacciones:")
    print(transacciones.head(5).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
