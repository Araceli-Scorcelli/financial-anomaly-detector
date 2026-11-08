"""Características sobre ejemplos pequeños, sin PostgreSQL ni etiquetas reales."""

from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src import preparar_datos as preparacion


def movimientos(filas):
    """Cada fila especifica ID, cuenta, fecha e importe; las demás son constantes."""
    return pd.DataFrame([
        {"transaccion_id": identificador, "cuenta_id": cuenta, "fecha_hora": fecha,
         "tipo_movimiento": "transferencia", "direccion": "ingreso", "monto": monto,
         "canal": "homebanking", "contraparte_id": "CP_FICTICIA"}
        for identificador, cuenta, fecha, monto in filas
    ], columns=["transaccion_id", "cuenta_id", "fecha_hora", "tipo_movimiento",
                "direccion", "monto", "canal", "contraparte_id"])


class PreparacionTests(unittest.TestCase):
    def setUp(self):
        bloqueo = patch("psycopg.connect", side_effect=AssertionError("No se permiten conexiones reales"))
        bloqueo.start()
        self.addCleanup(bloqueo.stop)
        self.referencia = movimientos([
            (f"R{i}", "A", f"2026-01-{i + 1:02d} 00:15:00", Decimal(str(monto)))
            for i, monto in enumerate([10, 20, 30, 40, 50])
        ])

    def test_conversion_copia_importes_decimal_y_fecha(self):
        datos = movimientos([("T1", "A", "2026-01-01 09:00:00", Decimal("123.45"))])
        original = datos.copy(deep=True)
        resultado = preparacion.validar_y_convertir(datos)
        self.assertEqual(resultado["monto"].dtype, np.dtype("float64"))
        self.assertAlmostEqual(resultado.at[0, "monto"], 123.45)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(resultado["fecha_hora"]))
        self.assertEqual(list(resultado.columns), list(datos.columns))
        pd.testing.assert_frame_equal(datos, original)

    def test_rechaza_importes_invalidos(self):
        for valor in (None, "no-numero", np.nan, np.inf, -1, 0, True, 1 + 2j):
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                preparacion.validar_y_convertir(self.referencia.assign(monto=valor))

    def test_rechaza_fechas_invalidas_o_con_zona(self):
        for valor in (None, "no-fecha", "2026-13-01", 123, pd.Timestamp("2026-01-01", tz="UTC")):
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                preparacion.validar_y_convertir(self.referencia.assign(fecha_hora=valor))

    def test_rechaza_ids_duplicados_y_campos_obligatorios(self):
        casos = [self.referencia.assign(transaccion_id="duplicado"),
                 self.referencia.assign(cuenta_id=None), self.referencia.assign(transaccion_id=" "),
                 self.referencia.drop(columns="canal"), self.referencia.assign(extra=1)]
        for datos in casos:
            with self.subTest(columnas=list(datos.columns)), self.assertRaises(ValueError):
                preparacion.validar_y_convertir(datos)

    def test_medianas_individuales_conocidas(self):
        referencia_b = movimientos([
            (f"B{i}", "B", f"2026-01-{i + 1:02d} 10:00:00", monto)
            for i, monto in enumerate([100, 200, 300, 400, 500])
        ])
        referencias = preparacion.ajustar_referencias(pd.concat([self.referencia, referencia_b]))
        objetivos = movimientos([("T1", "A", "2026-02-01 09:00:00", 60),
                                 ("T2", "B", "2026-02-01 09:00:00", 150)])
        resultado = preparacion.transformar_datos(objetivos, referencias)
        np.testing.assert_allclose(resultado["monto_relativo"], [2, 0.5])

    def test_cuentas_escasas_y_nuevas_usan_referencia_global(self):
        escasa = movimientos([("R5", "ESCASA", "2026-01-06 00:15:00", 1000)])
        referencias = preparacion.ajustar_referencias(pd.concat([self.referencia, escasa]))
        # La mediana global de [10,20,30,40,50,1000] es 35.
        objetivos = movimientos([("T1", "ESCASA", "2026-02-01 12:00:00", 70),
                                 ("T2", "NUEVA", "2026-02-01 12:00:00", 70)])
        resultado = preparacion.transformar_datos(objetivos, referencias)
        np.testing.assert_allclose(resultado["monto_relativo"], [2, 2])
        self.assertEqual(resultado.at[0, "rareza_horaria"], resultado.at[1, "rareza_horaria"])

    def test_rareza_laplace_circular_con_valores_conocidos(self):
        referencias = preparacion.ajustar_referencias(self.referencia)
        objetivos = movimientos([
            ("T0", "A", "2026-02-01 00:01:00", 30),
            ("T1", "A", "2026-02-01 01:01:00", 30),
            ("T2", "A", "2026-02-01 23:59:00", 30),
            ("T3", "A", "2026-02-01 12:01:00", 30),
        ])
        resultado = preparacion.transformar_datos(objetivos, referencias)
        # Cinco observaciones a las 00: c suavizado = 2.5 en 00, 1.25 en 01/23.
        esperado = -np.log(np.array([3.5, 2.25, 2.25, 1]) / 29)
        np.testing.assert_allclose(resultado["rareza_horaria"], esperado)
        self.assertLess(esperado[0], esperado[1])
        self.assertLess(esperado[2], esperado[3])

    def test_distribucion_uniforme_permanece_normalizada(self):
        referencia = movimientos([
            (f"R{hora:02d}", "A", f"2026-01-01 {hora:02d}:00:00", 10)
            for hora in range(24)
        ])
        resultado = preparacion.transformar_datos(referencia, preparacion.ajustar_referencias(referencia))
        np.testing.assert_allclose(resultado["rareza_horaria"], np.log(24))
        self.assertAlmostEqual(np.exp(-resultado["rareza_horaria"]).sum(), 1)

    def test_transformar_no_reajusta_referencias_ni_las_modifica(self):
        referencias = preparacion.ajustar_referencias(self.referencia)
        objetivo = movimientos([("T1", "A", "2026-02-01 12:00:00", 60)])
        anterior = preparacion.transformar_datos(objetivo, referencias)
        extremo = preparacion.transformar_datos(objetivo.assign(monto=6000), referencias)
        self.assertEqual(extremo.at[0, "monto_relativo"], 200)
        pd.testing.assert_frame_equal(anterior, preparacion.transformar_datos(objetivo, referencias))
        with patch.object(preparacion, "ajustar_referencias", side_effect=AssertionError("No reajustar")):
            preparacion.transformar_datos(objetivo, referencias)

    def test_rechaza_referencia_vacia_y_umbral_invalido(self):
        with self.assertRaises(ValueError):
            preparacion.ajustar_referencias(self.referencia.iloc[:0])
        for minimo in (0, -1, True, 1.5):
            with self.subTest(minimo=minimo), self.assertRaises(ValueError):
                preparacion.ajustar_referencias(self.referencia, min_observaciones=minimo)

    def test_ventana_inclusiva_cuentas_separadas_y_orden_restablecido(self):
        datos = movimientos([
            ("T5", "A", "2026-01-01 09:10:00", 10),
            ("T2", "A", "2026-01-01 09:04:00", 10),
            ("T6", "B", "2026-01-01 09:04:00", 10),
            ("T4", "A", "2026-01-01 09:05:00.000001", 10),
            ("T1", "A", "2026-01-01 09:00:00", 10),
            ("T3", "A", "2026-01-01 09:05:00", 10),
        ])
        conteos = preparacion.calcular_movimientos_5min(datos)
        self.assertEqual(conteos.tolist(), [3, 2, 1, 3, 1, 3])

    def test_simultaneos_se_ordenan_por_id_sin_contar_los_posteriores(self):
        datos = movimientos([(identificador, "A", "2026-01-01 09:00:00", 10)
                             for identificador in ("T3", "T1", "T2")])
        self.assertEqual(preparacion.calcular_movimientos_5min(datos).tolist(), [3, 1, 2])

    def test_conteo_cruza_medianoche_y_respeta_limite_exacto(self):
        datos = movimientos([
            ("T1", "A", "2026-01-01 23:58:00", 10),
            ("T2", "A", "2026-01-02 00:01:00", 10),
            ("T3", "A", "2026-01-02 00:03:00", 10),
            ("T4", "A", "2026-01-02 00:03:00.000001", 10),
        ])
        self.assertEqual(preparacion.calcular_movimientos_5min(datos).tolist(), [1, 2, 3, 3])

    def test_historial_explicito_ignora_futuro_y_otras_cuentas(self):
        historial = movimientos([
            ("H1", "A", "2026-01-01 08:58:00", 10),
            ("H2", "A", "2026-01-01 09:00:00", 10),
            ("HF", "A", "2026-01-01 10:00:00", 10),
            ("HB", "B", "2026-01-01 09:01:00", 10),
        ])
        objetivos = movimientos([("N1", "A", "2026-01-01 09:02:00", 10),
                                 ("N2", "A", "2026-01-01 09:07:00", 10)])
        self.assertEqual(preparacion.calcular_movimientos_5min(objetivos, historial).tolist(), [3, 2])
        primero = preparacion.calcular_movimientos_5min(objetivos.iloc[:1], historial)
        segundo = preparacion.calcular_movimientos_5min(objetivos.iloc[1:], pd.concat([historial, objetivos.iloc[:1]]))
        self.assertEqual(primero.tolist() + segundo.tolist(), [3, 2])

    def test_historial_simultaneo_usa_el_mismo_criterio_de_id(self):
        objetivos = movimientos([("T1", "A", "2026-01-01 09:00:00", 10),
                                 ("T2", "A", "2026-01-01 09:00:00", 10)])
        historial = movimientos([("T0", "A", "2026-01-01 09:00:00", 10),
                                 ("T3", "A", "2026-01-01 09:00:00", 10)])
        self.assertEqual(preparacion.calcular_movimientos_5min(objetivos, historial).tolist(), [2, 3])

    def test_historial_solapado_se_deduplica_y_conflictos_se_rechazan(self):
        datos = movimientos([("T1", "A", "2026-01-01 09:00:00", 10),
                             ("T2", "A", "2026-01-01 09:02:00", 10)])
        self.assertEqual(preparacion.calcular_movimientos_5min(datos, datos.iloc[:1]).tolist(), [1, 2])
        with self.assertRaises(ValueError):
            preparacion.calcular_movimientos_5min(datos, datos.iloc[:1].assign(monto=999))

    def test_ajuste_no_incorpora_historial_implicitamente(self):
        referencias = preparacion.ajustar_referencias(self.referencia)
        objetivos = movimientos([("T1", "A", "2026-01-01 00:17:00", 10)])
        sin_historial = preparacion.transformar_datos(objetivos, referencias)
        con_historial = preparacion.transformar_datos(objetivos, referencias, self.referencia)
        self.assertEqual(sin_historial.at[0, "movimientos_5min"], 1)
        self.assertEqual(con_historial.at[0, "movimientos_5min"], 2)

    def test_preserva_ids_orden_indice_duplicado_y_once_columnas_sin_mutar(self):
        datos = movimientos([("T3", "A", "2026-02-01 09:06:00", Decimal("60")),
                             ("T1", "A", "2026-02-01 09:00:00", Decimal("30")),
                             ("T2", "A", "2026-02-01 09:05:00", Decimal("45"))])
        datos.index = [99, 99, -4]
        original = datos.copy(deep=True)
        resultado = preparacion.transformar_datos(datos, preparacion.ajustar_referencias(self.referencia))
        self.assertEqual(len(resultado), 3)
        self.assertEqual(resultado.index.tolist(), [99, 99, -4])
        self.assertEqual(resultado["transaccion_id"].tolist(), ["T3", "T1", "T2"])
        self.assertEqual(resultado["movimientos_5min"].tolist(), [2, 1, 2])
        self.assertEqual(list(resultado.columns), list(datos.columns) + ["monto_relativo", "rareza_horaria", "movimientos_5min"])
        self.assertFalse(resultado.isna().any().any())
        self.assertTrue(np.isfinite(resultado[["monto_relativo", "rareza_horaria", "movimientos_5min"]].to_numpy()).all())
        pd.testing.assert_frame_equal(datos, original)

    def test_transformacion_vacia_conserva_esquema(self):
        resultado = preparacion.transformar_datos(self.referencia.iloc[:0], preparacion.ajustar_referencias(self.referencia))
        self.assertEqual(resultado.shape, (0, 11))
        self.assertEqual(resultado["movimientos_5min"].dtype, np.dtype("int64"))

    def test_main_reutiliza_lector_existente_y_muestra_resumen(self):
        engine = MagicMock()
        salida = io.StringIO()
        with patch.object(preparacion, "crear_engine", return_value=engine), \
                patch.object(preparacion, "cargar_transacciones", return_value=self.referencia) as leer, \
                redirect_stdout(salida):
            self.assertEqual(preparacion.main(), 0)
        leer.assert_called_once_with(engine)
        engine.dispose.assert_called_once()
        self.assertIn("Registros procesados: 5", salida.getvalue())
        self.assertIn("Cuentas diferentes: 1", salida.getvalue())
        self.assertIn("Total columnas: 11", salida.getvalue())
        self.assertIn("Sin nulos ni infinitos", salida.getvalue())

    def test_main_falla_sin_exponer_detalles_ni_fingir_datos(self):
        salida, errores = io.StringIO(), io.StringIO()
        engine = MagicMock()
        with patch.object(preparacion, "crear_engine", return_value=engine), \
                patch.object(preparacion, "cargar_transacciones", side_effect=RuntimeError("secreto_ficticio")), \
                redirect_stdout(salida), redirect_stderr(errores):
            self.assertEqual(preparacion.main(), 1)
        engine.dispose.assert_called_once()
        self.assertEqual(salida.getvalue(), "")
        self.assertNotIn("secreto_ficticio", errores.getvalue())

    def test_main_oculta_valueerror_del_lector_y_del_cierre(self):
        for origen in ("lectura", "cierre"):
            with self.subTest(origen=origen):
                salida, errores = io.StringIO(), io.StringIO()
                engine = MagicMock()
                if origen == "cierre":
                    engine.dispose.side_effect = ValueError("secreto_ficticio")
                fallo_lectura = ValueError("secreto_ficticio") if origen == "lectura" else None
                with patch.object(preparacion, "crear_engine", return_value=engine), \
                        patch.object(preparacion, "cargar_transacciones", return_value=self.referencia,
                                     side_effect=fallo_lectura), \
                        redirect_stdout(salida), redirect_stderr(errores):
                    self.assertEqual(preparacion.main(), 1)
                engine.dispose.assert_called_once()
                self.assertEqual(salida.getvalue(), "")
                self.assertNotIn("secreto_ficticio", errores.getvalue())


if __name__ == "__main__":
    unittest.main()
