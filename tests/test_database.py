"""Pruebas de configuración y lectura con el transporte PostgreSQL bloqueado."""

from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
from sqlalchemy.exc import OperationalError, ProgrammingError

from src import database, verificar_conexion


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.carpeta = tempfile.TemporaryDirectory()
        self.addCleanup(self.carpeta.cleanup)
        self.env_path = Path(self.carpeta.name) / ".env"
        self.valores = {
            "DB_HOST": "localhost", "DB_PORT": "5432",
            "DB_NAME": "financial_anomalies", "DB_USER": "usuario_ficticio",
            "DB_PASSWORD": "clave_ficticia_solo_tests",
        }
        self.entorno = patch.dict(os.environ, self.valores, clear=True)
        self.entorno.start()
        self.addCleanup(self.entorno.stop)
        bloqueo = patch("psycopg.connect", side_effect=AssertionError("Conexiones externas prohibidas en pruebas"))
        self.conectar_driver = bloqueo.start()
        self.addCleanup(bloqueo.stop)

    def test_engine_perezoso_y_password_con_caracteres_especiales(self):
        password = "  p@ss:/?#%${NO_EXPANDIR} con espacios  "
        os.environ["DB_PASSWORD"] = password
        engine = database.crear_engine(self.env_path)
        self.addCleanup(engine.dispose)
        self.assertEqual(engine.url.drivername, "postgresql+psycopg")
        self.assertEqual(engine.url.password, password)
        self.assertEqual(engine.url.database, "financial_anomalies")
        self.assertEqual(engine.url.port, 5432)
        self.assertFalse(engine.echo)
        self.assertTrue(engine.hide_parameters)
        self.assertTrue(engine.get_execution_options()["postgresql_readonly"])
        self.assertNotIn(password, str(engine.url))
        self.conectar_driver.assert_not_called()

    def test_opciones_de_inicio_exigen_solo_lectura_y_timeout(self):
        with patch.object(database, "create_engine") as crear:
            database.crear_engine(self.env_path)
        opciones = crear.call_args.kwargs
        self.assertEqual(opciones["connect_args"]["connect_timeout"], 5)
        self.assertEqual(opciones["connect_args"]["options"], "-c default_transaction_read_only=on")
        self.assertTrue(opciones["execution_options"]["postgresql_readonly"])
        self.assertTrue(opciones["hide_parameters"])
        self.assertFalse(opciones["echo"])

    def test_rechaza_variables_faltantes_o_vacias_sin_mostrar_valores(self):
        for variable in self.valores:
            with self.subTest(variable=variable), patch.dict(os.environ, self.valores, clear=True):
                os.environ[variable] = " "
                with self.assertRaises(database.ErrorConfiguracion) as resultado:
                    database.crear_engine(self.env_path)
                self.assertIn(variable, str(resultado.exception))
                self.assertNotIn(self.valores["DB_PASSWORD"], str(resultado.exception))
        self.conectar_driver.assert_not_called()

    def test_rechaza_puertos_invalidos_sin_incluir_su_valor(self):
        for puerto in ("0", "65536", "-1", "5432.5", "texto_clave_ficticia"):
            with self.subTest(puerto=puerto), patch.dict(os.environ, {"DB_PORT": puerto}):
                with self.assertRaises(database.ErrorConfiguracion) as resultado:
                    database.crear_engine(self.env_path)
                self.assertIn("DB_PORT", str(resultado.exception))
                self.assertNotIn(puerto, str(resultado.exception))

    def test_dotenv_explicito_respeta_entorno_y_no_interpola_password(self):
        self.env_path.write_text(
            "DB_HOST=host_del_archivo\nDB_PORT=5432\nDB_NAME=financial_anomalies\n"
            "DB_USER=usuario_del_archivo\nDB_PASSWORD='${NO_EXPANDIR}@/# con espacios'\n",
            encoding="utf-8",
        )
        with patch.dict(os.environ, {"DB_HOST": "host_del_entorno", "NO_EXPANDIR": "valor_ficticio"}, clear=True):
            engine = database.crear_engine(self.env_path)
            self.addCleanup(engine.dispose)
            self.assertEqual(engine.url.host, "host_del_entorno")
            self.assertEqual(engine.url.username, "usuario_del_archivo")
            self.assertEqual(engine.url.password, "${NO_EXPANDIR}@/# con espacios")

    def test_ruta_dotenv_predeterminada_es_la_raiz_del_proyecto(self):
        with patch.object(database, "load_dotenv") as cargar, patch.object(database, "create_engine"):
            database.crear_engine()
        self.assertEqual(cargar.call_args.kwargs["dotenv_path"], Path(database.__file__).resolve().parents[1] / ".env")
        self.assertFalse(cargar.call_args.kwargs["override"])
        self.assertFalse(cargar.call_args.kwargs["interpolate"])

    def test_error_de_lectura_dotenv_no_expone_detalle_original(self):
        with patch.object(database, "load_dotenv", side_effect=OSError(self.valores["DB_PASSWORD"])):
            with self.assertRaises(database.ErrorConfiguracion) as resultado:
                database.crear_engine(self.env_path)
        self.assertNotIn(self.valores["DB_PASSWORD"], str(resultado.exception))

    def preparar_conexion(self):
        engine = MagicMock()
        conexion = engine.connect.return_value.__enter__.return_value
        conexion.execution_options.return_value = conexion
        modo, ping = MagicMock(), MagicMock()
        modo.scalar_one.return_value = "on"
        ping.scalar_one.return_value = 1
        conexion.execute.side_effect = [modo, ping]
        return engine, conexion, modo, ping

    def test_contexto_verifica_solo_lectura_antes_de_entregar_conexion(self):
        engine, conexion, _, _ = self.preparar_conexion()
        with database.conectar_solo_lectura(engine) as entregada:
            self.assertIs(entregada, conexion)
        self.assertEqual(conexion.method_calls[0][0], "execution_options")
        self.assertEqual(conexion.method_calls[0].kwargs, {"postgresql_readonly": True})
        self.assertEqual(str(conexion.execute.call_args.args[0]), "SHOW transaction_read_only")
        engine.connect.return_value.__exit__.assert_called_once()

    def test_contexto_rechaza_sesion_con_escritura_habilitada(self):
        engine, _, modo, _ = self.preparar_conexion()
        modo.scalar_one.return_value = "off"
        with self.assertRaises(database.ErrorConexion):
            with database.conectar_solo_lectura(engine):
                self.fail("No debe entregar una sesión con escritura habilitada")

    def test_error_de_conexion_no_expone_credenciales(self):
        engine = MagicMock()
        secreto = self.valores["DB_PASSWORD"]
        engine.connect.side_effect = OperationalError("conexion", {}, Exception(secreto))
        with self.assertRaises(database.ErrorConexion) as resultado:
            with database.conectar_solo_lectura(engine):
                self.fail("No debe entregar una conexión fallida")
        self.assertNotIn(secreto, str(resultado.exception))
        self.assertIsNone(resultado.exception.__cause__)

    def test_error_de_consulta_o_permisos_tiene_mensaje_seguro(self):
        engine, conexion, _, _ = self.preparar_conexion()
        conexion.execute.side_effect = ProgrammingError("consulta", {}, Exception(self.valores["DB_PASSWORD"]))
        with self.assertRaises(database.ErrorConexion) as resultado:
            with database.conectar_solo_lectura(engine):
                pass
        self.assertIn("permisos", str(resultado.exception))
        self.assertNotIn(self.valores["DB_PASSWORD"], str(resultado.exception))

    def datos_de_prueba(self):
        return pd.DataFrame({
            "transaccion_id": [f"TX{numero:06d}" for numero in range(1, 8)],
            "cuenta_id": ["CTA001", "CTA002", "CTA001", "CTA003", "CTA002", "CTA001", "CTA003"],
            "fecha_hora": pd.date_range("2026-01-02 09:00", periods=7, freq="h"),
            "tipo_movimiento": ["transferencia"] * 7,
            "direccion": ["ingreso"] * 7,
            "monto": [Decimal("100.10")] * 7,
            "canal": ["homebanking"] * 7,
            "contraparte_id": ["CP0001"] * 7,
        })

    def test_lectura_usa_select_1_y_recupera_todas_las_columnas(self):
        engine, conexion, _, _ = self.preparar_conexion()
        esperado = self.datos_de_prueba()
        with patch.object(verificar_conexion.pd, "read_sql_query", return_value=esperado) as leer:
            obtenido = verificar_conexion.cargar_transacciones(engine)
        pd.testing.assert_frame_equal(obtenido, esperado)
        self.assertEqual([str(llamada.args[0]) for llamada in conexion.execute.call_args_list],
                         ["SHOW transaction_read_only", "SELECT 1"])
        consulta = str(leer.call_args.args[0])
        self.assertIn("FROM public.transacciones", consulta)
        self.assertIn("ORDER BY fecha_hora, transaccion_id", consulta)
        self.assertNotIn("LIMIT", consulta.upper())
        self.assertNotIn("es_anomalia", consulta)
        for columna in esperado.columns:
            self.assertIn(columna, consulta)
        self.assertIs(leer.call_args.args[1], conexion)
        self.assertFalse(leer.call_args.kwargs["coerce_float"])
        self.assertEqual(leer.call_args.kwargs["parse_dates"], ["fecha_hora"])
        self.assertIsInstance(obtenido.at[0, "monto"], Decimal)

    def test_no_lee_tabla_si_select_1_no_devuelve_uno(self):
        engine, _, _, ping = self.preparar_conexion()
        ping.scalar_one.return_value = 0
        with patch.object(verificar_conexion.pd, "read_sql_query") as leer:
            with self.assertRaises(database.ErrorConexion):
                verificar_conexion.cargar_transacciones(engine)
        leer.assert_not_called()

    def ejecutar_main(self, engine, datos=None):
        salida, errores = io.StringIO(), io.StringIO()
        with patch.object(verificar_conexion, "crear_engine", return_value=engine), \
                patch.object(verificar_conexion.pd, "read_sql_query", return_value=datos), \
                redirect_stdout(salida), redirect_stderr(errores):
            codigo = verificar_conexion.main()
        return codigo, salida.getvalue(), errores.getvalue()

    def test_consola_muestra_conteos_reales_tipos_y_solo_cinco_filas(self):
        engine, _, _, _ = self.preparar_conexion()
        codigo, salida, errores = self.ejecutar_main(engine, self.datos_de_prueba())
        self.assertEqual(codigo, 0)
        self.assertEqual(errores, "")
        self.assertIn("Movimientos: 7", salida)
        self.assertIn("Cuentas distintas: 3", salida)
        self.assertIn("Tipos de datos", salida)
        self.assertIn("monto", salida)
        self.assertIn("TX000005", salida)
        self.assertNotIn("TX000006", salida)
        self.assertNotIn(self.valores["DB_PASSWORD"], salida)
        engine.dispose.assert_called_once()

    def test_main_error_de_conexion_no_imprime_credenciales_y_libera_engine(self):
        engine = MagicMock()
        engine.connect.side_effect = OperationalError("conexion", {}, Exception(self.valores["DB_PASSWORD"]))
        codigo, salida, errores = self.ejecutar_main(engine)
        self.assertEqual(codigo, 1)
        self.assertEqual(salida, "")
        self.assertIn("PostgreSQL", errores)
        self.assertNotIn(self.valores["DB_PASSWORD"], errores)
        self.assertNotIn("Traceback", errores)
        engine.dispose.assert_called_once()

    def test_main_error_al_liberar_recursos_no_expone_detalles(self):
        engine, _, _, _ = self.preparar_conexion()
        engine.dispose.side_effect = RuntimeError(self.valores["DB_PASSWORD"])
        codigo, salida, errores = self.ejecutar_main(engine, self.datos_de_prueba())
        self.assertEqual(codigo, 1)
        self.assertEqual(salida, "")
        self.assertNotIn(self.valores["DB_PASSWORD"], errores)
        self.assertNotIn("Traceback", errores)

    def test_main_falta_configuracion_no_intenta_leer_tabla(self):
        salida, errores = io.StringIO(), io.StringIO()
        with patch.object(verificar_conexion, "crear_engine", side_effect=database.ErrorConfiguracion("Falta DB_USER")), \
                patch.object(verificar_conexion.pd, "read_sql_query") as leer, \
                redirect_stdout(salida), redirect_stderr(errores):
            self.assertEqual(verificar_conexion.main(), 1)
        self.assertIn("DB_USER", errores.getvalue())
        leer.assert_not_called()


if __name__ == "__main__":
    unittest.main()
