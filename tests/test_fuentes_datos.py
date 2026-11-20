"""Selección explícita y lectura del CSV, sin credenciales ni conexiones."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

from src import fuentes_datos as fuentes
from src.preparar_datos import COLUMNAS_ORIGINALES


def fila_valida():
    return pd.DataFrame([{
        "transaccion_id": "0001", "cuenta_id": "001", "fecha_hora": "2026-01-02 10:30:00",
        "tipo_movimiento": "transferencia", "direccion": "egreso", "monto": "1250.50",
        "canal": "homebanking", "contraparte_id": "0002",
    }])


class ConfiguracionFuenteTests(unittest.TestCase):
    def test_predeterminado_postgres_sin_env_ni_archivo(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(fuentes, "_fuente_desde_env", return_value="postgres"):
            self.assertEqual(fuentes.obtener_fuente_datos(), "postgres")

    def test_fuente_explicita_no_lee_env(self):
        for valor in ("csv", "postgres"):
            with self.subTest(valor=valor), patch.dict(os.environ, {"DATA_SOURCE": valor}, clear=True), \
                    patch.object(fuentes, "_fuente_desde_env", side_effect=AssertionError("No leer .env")):
                self.assertEqual(fuentes.obtener_fuente_datos(), valor)

    def test_env_opcional_y_precedencia_del_proceso(self):
        with TemporaryDirectory() as directorio:
            ruta = Path(directorio) / ".env"
            ruta.write_text("DATA_SOURCE=csv\n", encoding="utf-8")
            with patch.object(fuentes, "RUTA_ENV", ruta), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(fuentes.obtener_fuente_datos(), "csv")
                self.assertNotIn("DATA_SOURCE", os.environ)
                with patch.dict(os.environ, {"DATA_SOURCE": "postgres"}):
                    self.assertEqual(fuentes.obtener_fuente_datos(), "postgres")

    def test_valores_invalidos_no_exponen_su_contenido(self):
        for valor in ("", "CSV", "postgres://usuario:secreto@host"):
            with self.subTest(valor=valor), patch.dict(os.environ, {"DATA_SOURCE": valor}, clear=True):
                with self.assertRaisesRegex(fuentes.ErrorFuenteDatos, "DATA_SOURCE") as error:
                    fuentes.obtener_fuente_datos()
                self.assertNotIn("secreto", str(error.exception))

    def test_error_al_leer_configuracion_no_expone_detalles(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(fuentes, "_fuente_desde_env", side_effect=OSError("secreto")):
            with self.assertRaises(fuentes.ErrorFuenteDatos) as error:
                fuentes.obtener_fuente_datos()
            self.assertNotIn("secreto", str(error.exception))

    def test_env_ausente_conserva_predeterminado_y_malformado_muestra_error(self):
        with TemporaryDirectory() as directorio:
            ruta = Path(directorio) / ".env"
            with patch.object(fuentes, "RUTA_ENV", ruta), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(fuentes.obtener_fuente_datos(), "postgres")
                ruta.write_text('DATA_SOURCE="csv\n', encoding="utf-8")
                with self.assertRaises(fuentes.ErrorFuenteDatos):
                    fuentes.obtener_fuente_datos()


class LecturaCSVTests(unittest.TestCase):
    def test_ruta_del_proyecto_independiente_del_directorio_actual(self):
        ruta = Path(__file__).resolve().parents[1] / "data" / "transacciones.csv"
        self.assertEqual(fuentes.RUTA_TRANSACCIONES, ruta)
        with patch.object(fuentes.pd, "read_csv", return_value=fila_valida()) as lectura:
            datos = fuentes.cargar_transacciones_csv()
        lectura.assert_called_once_with(ruta, dtype=str, keep_default_na=False)
        self.assertEqual(list(datos.columns), list(COLUMNAS_ORIGINALES))

    def test_conversion_real_preserva_identificadores_y_valida_tipos(self):
        with TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "transacciones.csv"
            fila_valida().to_csv(ruta, index=False)
            with patch.object(fuentes, "RUTA_TRANSACCIONES", ruta):
                datos = fuentes.cargar_transacciones_csv()
        self.assertEqual(datos.transaccion_id.tolist(), ["0001"])
        self.assertEqual(datos.cuenta_id.tolist(), ["001"])
        self.assertEqual(datos.monto.iloc[0], 1250.5)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(datos.fecha_hora))

    def test_validacion_rechaza_errores_y_columnas_de_etiquetas(self):
        original = fila_valida()
        casos = {
            "columna_faltante": original.drop(columns="canal"),
            "etiqueta_infiltrada": original.assign(es_anomalia=1),
            "duplicado": pd.concat([original, original], ignore_index=True),
            "identificador_vacio": original.assign(transaccion_id=""),
            "nulo": original.assign(cuenta_id=None),
            "monto_negativo": original.assign(monto="-1"),
            "fecha_invalida": original.assign(fecha_hora="2026-02-30"),
        }
        for nombre, datos in casos.items():
            with self.subTest(nombre=nombre), patch.object(fuentes.pd, "read_csv", return_value=datos):
                with self.assertRaises(ValueError):
                    fuentes.cargar_transacciones_csv()

    def test_archivo_ausente_no_dispara_otra_fuente(self):
        with TemporaryDirectory() as directorio, \
                patch.object(fuentes, "RUTA_TRANSACCIONES", Path(directorio) / "ausente.csv"):
            with self.assertRaises(FileNotFoundError):
                fuentes.cargar_transacciones_csv()

    def test_fila_con_campo_extra_sin_encabezado_no_se_acepta_como_indice(self):
        with TemporaryDirectory() as directorio:
            ruta = Path(directorio) / "transacciones.csv"
            contenido = fila_valida().to_csv(index=False)
            encabezado, fila = contenido.splitlines()
            ruta.write_text(f"{encabezado}\nbasura,{fila}\n", encoding="utf-8")
            with patch.object(fuentes, "RUTA_TRANSACCIONES", ruta):
                with self.assertRaises(ValueError):
                    fuentes.cargar_transacciones_csv()


if __name__ == "__main__":
    unittest.main()
