"""Streamlit y filtros con resultados ficticios, sin PostgreSQL ni etiquetas reales."""

from datetime import date
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

import pandas as pd
from streamlit.testing.v1 import AppTest

import app


def tearDownModule():
    # AppTest mantiene un directorio global: cerrarlo evita un ResourceWarning
    # de su finalizador al ejecutar la suite con -W error en Python 3.14.
    from streamlit.testing.v1.app_test import TMP_DIR
    TMP_DIR.cleanup()


def datos_ficticios():
    predicciones = pd.DataFrame({
        "transaccion_id": ["E1", "E2", "E3", "E4"], "cuenta_id": ["A", "B", "A", "B"],
        "fecha_hora": pd.to_datetime(["2026-03-13 10:00", "2026-03-14 23:59:59",
                                      "2026-03-14 08:00", "2026-03-31 12:00"], format="mixed"),
        "tipo_movimiento": ["transferencia"] * 4, "direccion": ["egreso"] * 4,
        "monto": [100.0, 200.0, 50.0, 75.0], "canal": ["homebanking"] * 4,
        "contraparte_id": ["CP1"] * 4, "monto_relativo": [3.0, 2.0, 0.5, 0.75],
        "rareza_horaria": [2.0, 3.0, 1.0, 1.0], "movimientos_5min": [1] * 4,
        "puntaje_anomalia": [0.8, 0.7, 0.3, 0.2], "prediccion_anomalia": [1, 1, 0, 0],
    })
    referencias = pd.DataFrame({"transaccion_id": ["E4", "E3", "E2", "E1"],
                               "es_anomalia": [0, 0, 0, 1],
                               "tipo_anomalia": ["", "", "", "importe_inusual"]})
    entrenamiento = predicciones.iloc[:2].drop(columns=["puntaje_anomalia", "prediccion_anomalia"]).copy()
    entrenamiento["transaccion_id"] = ["T1", "T2"]
    entrenamiento["fecha_hora"] -= pd.Timedelta(days=30)
    resultado = SimpleNamespace(predicciones=predicciones, entrenamiento=entrenamiento,
                                modelo=SimpleNamespace(offset_=-0.5))
    transacciones = pd.concat([entrenamiento, predicciones[entrenamiento.columns]], ignore_index=True)
    return transacciones, resultado, referencias


class FiltrosTests(unittest.TestCase):
    def setUp(self):
        _, resultado, _ = datos_ficticios()
        self.predicciones = resultado.predicciones

    def test_filtros_cuenta_fecha_y_resultado_conservan_scores_ids_y_entrada(self):
        original = self.predicciones.copy(deep=True)
        filtradas = app.filtrar_transacciones(self.predicciones, "B", (date(2026, 3, 14), date(2026, 3, 14)), "Alertas")
        self.assertEqual(filtradas.transaccion_id.tolist(), ["E2"])
        self.assertEqual(filtradas.puntaje_anomalia.tolist(), [0.7])
        pd.testing.assert_frame_equal(self.predicciones, original)

    def test_fin_de_dia_inclusivo_y_separacion_de_normales(self):
        fechas = (date(2026, 3, 13), date(2026, 3, 31))
        normales = app.filtrar_transacciones(self.predicciones, None, fechas, "Normales")
        self.assertEqual(normales.transaccion_id.tolist(), ["E3", "E4"])
        todas = app.filtrar_transacciones(self.predicciones, None, fechas, "Todas")
        self.assertEqual(len(todas), 4)

    def test_combinacion_vacia_y_rangos_parciales_o_invertidos(self):
        vacias = app.filtrar_transacciones(self.predicciones, "A", (date(2026, 3, 31), date(2026, 3, 31)), "Todas")
        self.assertTrue(vacias.empty)
        for fechas in ((), (date(2026, 3, 14),), (date(2026, 3, 31), date(2026, 3, 13))):
            with self.subTest(fechas=fechas), self.assertRaises(ValueError):
                app.filtrar_transacciones(self.predicciones, None, fechas, "Todas")

    def test_tabla_destaca_alertas_y_excluye_etiquetas_historicas(self):
        entrada = self.predicciones.assign(es_anomalia=1, tipo_anomalia="importe_inusual")
        tabla = app.preparar_tabla(entrada.iloc[::-1])
        self.assertEqual(tabla.transaccion_id.tolist(), ["E1", "E2", "E3", "E4"])
        self.assertEqual(tabla.estado.tolist(), ["Alerta potencial", "Alerta potencial", "Normal", "Normal"])
        self.assertEqual(tabla.columns.tolist(), [
            "transaccion_id", "cuenta_id", "fecha_hora", "monto", "estado", "puntaje_anomalia",
            "monto_relativo", "rareza_horaria", "movimientos_5min",
        ])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(tabla.fecha_hora))
        self.assertTrue(pd.api.types.is_float_dtype(tabla.monto))
        self.assertEqual(tabla.puntaje_anomalia.tolist(), [0.8, 0.7, 0.3, 0.2])
        self.assertNotIn("es_anomalia", tabla)
        self.assertNotIn("tipo_anomalia", tabla)


class CargaYAplicacionTests(unittest.TestCase):
    def setUp(self):
        app.cargar_resultados.clear()
        self.addCleanup(app.cargar_resultados.clear)
        self.transacciones, self.resultado, self.etiquetas = datos_ficticios()
        self.engine = MagicMock()
        self.eventos = []

        def predecir(datos):
            self.eventos.append("predicciones")
            return self.resultado

        def etiquetas(ruta, ids):
            self.assertEqual(self.eventos[-1], "predicciones")
            self.assertEqual(set(ids), set(self.transacciones.transaccion_id))
            self.eventos.append("etiquetas")
            return self.etiquetas

        parches = [patch("psycopg.connect", side_effect=AssertionError("No usar DB real")),
                   patch.object(app.pipeline, "crear_engine", return_value=self.engine),
                   patch.object(app.pipeline, "cargar_transacciones", return_value=self.transacciones),
                   patch.object(app.pipeline, "entrenar_y_predecir", side_effect=predecir),
                   patch.object(app.pipeline, "cargar_etiquetas", side_effect=etiquetas)]
        self.mocks = [parche.start() for parche in parches]
        for parche in parches:
            self.addCleanup(parche.stop)

    def iniciar(self):
        aplicacion = AppTest.from_string("import app\napp.main()", default_timeout=90).run()
        self.assertEqual(len(aplicacion.exception), 0)
        self.assertEqual(len(aplicacion.error), 0)
        return aplicacion

    def test_carga_reutiliza_pipeline_cierra_engine_y_no_expone_etiquetas_individuales(self):
        datos = app.cargar_resultados.__wrapped__()
        self.assertEqual(self.eventos, ["predicciones", "etiquetas"])
        self.engine.dispose.assert_called_once()
        self.assertEqual(datos["total_transacciones"], 6)
        self.assertEqual(datos["precision"], 0.5)
        self.assertEqual(datos["recall"], 1)
        self.assertEqual(datos["predicciones"].transaccion_id.tolist(), ["E1", "E2", "E3", "E4"])
        self.assertNotIn("es_anomalia", datos["predicciones"])
        self.assertNotIn("modelo", datos)
        self.assertNotIn("engine", datos)

    def test_filtros_no_reentrenan_ni_cambian_metricas_globales(self):
        aplicacion = self.iniciar()
        globales = [(m.label, m.value) for m in aplicacion.metric[:6]]
        self.assertEqual(globales[0], ("Transacciones evaluadas", ":red[4]"))
        aplicacion.selectbox(key="filtro_cuenta").select("B").run()
        aplicacion.date_input(key="filtro_fechas").set_value((date(2026, 3, 14), date(2026, 3, 14))).run()
        aplicacion.radio(key="filtro_resultado").set_value("Alertas").run()
        self.assertEqual(len(aplicacion.exception), 0)
        self.assertEqual([(m.label, m.value) for m in aplicacion.metric[:6]], globales)
        tabla = aplicacion.dataframe[0].value
        self.assertEqual(tabla.transaccion_id.tolist(), ["E2"])
        self.assertEqual(tabla.puntaje_anomalia.tolist(), [0.7])
        self.mocks[2].assert_called_once()
        self.mocks[3].assert_called_once()
        self.mocks[4].assert_called_once()
        self.engine.dispose.assert_called_once()

    def test_presentacion_metricas_y_detalle_tecnico_no_cambian_resultados(self):
        aplicacion = self.iniciar()
        globales = [(m.label, m.value) for m in aplicacion.metric[:6]]
        grilla = aplicacion.get_by_key("indicadores_globales")
        self.assertEqual(len(grilla.metric), 6)
        self.assertEqual(globales[4], ("Recall", ":red[100,00%]"))
        self.assertTrue(all(m.proto.help for m in grilla.metric))
        principales = ["transaccion_id", "cuenta_id", "fecha_hora", "monto", "estado", "puntaje_anomalia"]
        tabla_inicial = aplicacion.dataframe[0].value.copy()
        self.assertEqual(tabla_inicial.columns.tolist(), principales)
        aplicacion.toggle(key="mostrar_caracteristicas").set_value(True).run()
        self.assertEqual(len(aplicacion.exception), 0)
        tabla_detallada = aplicacion.dataframe[0].value
        self.assertEqual(tabla_detallada.columns.tolist(), principales + [
            "monto_relativo", "rareza_horaria", "movimientos_5min",
        ])
        pd.testing.assert_frame_equal(tabla_inicial, tabla_detallada[principales])
        self.assertEqual([(m.label, m.value) for m in aplicacion.metric[:6]], globales)
        self.mocks[2].assert_called_once()
        self.mocks[3].assert_called_once()
        self.mocks[4].assert_called_once()

    def test_actualizar_recarga_una_vez_y_restablece_filtros(self):
        aplicacion = self.iniciar()
        aplicacion.selectbox(key="filtro_cuenta").select("A").run()
        aplicacion.button(key="actualizar_resultados").click().run()
        self.assertEqual(len(aplicacion.exception), 0)
        self.assertIsNone(aplicacion.selectbox(key="filtro_cuenta").value)
        self.assertEqual(self.mocks[2].call_count, 2)
        self.assertEqual(self.mocks[3].call_count, 2)
        self.assertEqual(self.engine.dispose.call_count, 2)

    def test_vista_sin_resultados_y_rango_incompleto_no_afectan_globales(self):
        aplicacion = self.iniciar()
        globales = [(m.label, m.value) for m in aplicacion.metric[:6]]
        aplicacion.selectbox(key="filtro_cuenta").select("A").run()
        aplicacion.date_input(key="filtro_fechas").set_value((date(2026, 3, 31), date(2026, 3, 31))).run()
        self.assertEqual(len(aplicacion.exception), 0)
        self.assertTrue(any("No hay movimientos" in aviso.value for aviso in aplicacion.info))
        self.assertEqual(len(aplicacion.dataframe), 0)
        aplicacion.date_input(key="filtro_fechas").set_value((date(2026, 3, 14),)).run()
        self.assertEqual(len(aplicacion.exception), 0)
        self.assertEqual([(m.label, m.value) for m in aplicacion.metric[:6]], globales)
        self.mocks[3].assert_called_once()

    def test_errores_de_lectura_cierre_modelo_y_etiquetas_ocultan_detalles(self):
        for origen in ("lectura", "cierre", "modelo", "etiquetas"):
            with self.subTest(origen=origen):
                app.cargar_resultados.clear()
                with patch.object(self.engine, "dispose", side_effect=RuntimeError("secreto_ficticio")
                                  if origen == "cierre" else None), \
                        patch.object(app.pipeline, "cargar_transacciones", return_value=self.transacciones,
                                     side_effect=RuntimeError("secreto_ficticio") if origen == "lectura" else None), \
                        patch.object(app.pipeline, "entrenar_y_predecir", return_value=self.resultado,
                                     side_effect=RuntimeError("secreto_ficticio") if origen == "modelo" else None), \
                        patch.object(app.pipeline, "cargar_etiquetas", return_value=self.etiquetas,
                                     side_effect=RuntimeError("secreto_ficticio") if origen == "etiquetas" else None):
                    aplicacion = AppTest.from_string("import app\napp.main()", default_timeout=90).run()
                self.assertEqual(len(aplicacion.exception), 0)
                self.assertEqual(len(aplicacion.error), 1)
                self.assertNotIn("secreto_ficticio", aplicacion.error[0].value)
                self.assertEqual(len(aplicacion.metric), 0)


if __name__ == "__main__":
    unittest.main()
