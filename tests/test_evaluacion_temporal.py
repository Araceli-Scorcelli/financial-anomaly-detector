"""Walk-forward con ejemplos ficticios, sin PostgreSQL ni CSV reales."""

from contextlib import redirect_stderr, redirect_stdout
import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src import entrenar_modelo as detector
from src import evaluacion_temporal as temporal


def movimientos(cantidad=20):
    return pd.DataFrame({
        "transaccion_id": [f"T{i:04d}" for i in range(cantidad)],
        "cuenta_id": ["A"] * cantidad,
        "fecha_hora": pd.date_range("2026-01-01 09:00", periods=cantidad, freq="min"),
        "tipo_movimiento": ["transferencia"] * cantidad,
        "direccion": ["ingreso"] * cantidad,
        "monto": [10.0 + i % 4 for i in range(cantidad)],
        "canal": ["homebanking"] * cantidad,
        "contraparte_id": ["CP_FICTICIA"] * cantidad,
    })


def etiquetas(ids, anomalias=None):
    anomalias = anomalias or {}
    return pd.DataFrame({
        "transaccion_id": list(ids),
        "es_anomalia": [int(i in anomalias) for i in ids],
        "tipo_anomalia": [anomalias.get(i, "") for i in ids],
    })


class VentanasTests(unittest.TestCase):
    def test_tamanios_crecientes_evaluaciones_disjuntas_y_ultimo_corte_80_20(self):
        datos = movimientos(1500).iloc[::-1].copy()
        datos.index = [7] * len(datos)
        original = datos.copy(deep=True)
        ventanas = temporal.crear_ventanas(datos)
        self.assertEqual([(len(t), len(e)) for t, e in ventanas],
                         [(600, 300), (900, 300), (1200, 300)])
        ids = []
        for train, evaluacion in ventanas:
            self.assertLess(train.fecha_hora.max(), evaluacion.fecha_hora.min())
            self.assertFalse(set(train.transaccion_id) & set(evaluacion.transaccion_id))
            ids.extend(evaluacion.transaccion_id)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, [f"T{i:04d}" for i in range(600, 1500)])
        train, evaluacion = detector.separar_cronologicamente(datos)
        pd.testing.assert_frame_equal(ventanas[-1][0], train)
        pd.testing.assert_frame_equal(ventanas[-1][1], evaluacion)
        pd.testing.assert_frame_equal(datos, original)

    def test_empates_en_los_tres_cortes_quedan_juntos_sin_perder_movimientos(self):
        datos = movimientos()
        for corte in (8, 12, 16):
            datos.loc[corte, "fecha_hora"] = datos.loc[corte - 1, "fecha_hora"]
        ventanas = temporal.crear_ventanas(datos.iloc[::-1])
        self.assertEqual([(len(t), len(e)) for t, e in ventanas], [(7, 4), (11, 4), (15, 5)])
        for train, evaluacion in ventanas:
            self.assertLess(train.fecha_hora.max(), evaluacion.fecha_hora.min())
        self.assertEqual(ventanas[0][1].transaccion_id.tolist(), ["T0007", "T0008", "T0009", "T0010"])
        self.assertEqual(ventanas[1][1].transaccion_id.tolist(), ["T0011", "T0012", "T0013", "T0014"])
        self.assertEqual(ventanas[2][1].transaccion_id.tolist(), [f"T{i:04d}" for i in range(15, 20)])

    def test_rechaza_ventanas_vacias_o_colapsadas_y_ids_duplicados(self):
        colapsadas = movimientos()
        colapsadas.loc[7:12, "fecha_hora"] = colapsadas.loc[7, "fecha_hora"]
        for datos in (movimientos(0), movimientos(1), colapsadas,
                      movimientos().assign(fecha_hora=pd.Timestamp("2026-01-01")),
                      movimientos().assign(transaccion_id="duplicado")):
            with self.subTest(filas=len(datos)), self.assertRaises(ValueError):
                temporal.crear_ventanas(datos)


class PrediccionesTests(unittest.TestCase):
    def setUp(self):
        bloqueo = patch("psycopg.connect", side_effect=AssertionError("No conectar a PostgreSQL"))
        bloqueo.start()
        self.addCleanup(bloqueo.stop)

    def test_referencias_solo_train_e_historial_causal_sin_lectura_de_etiquetas(self):
        datos = movimientos()
        with patch("pandas.read_csv", side_effect=AssertionError("No consultar etiquetas")), \
                patch.object(detector, "ajustar_referencias", wraps=detector.ajustar_referencias) as ajustar:
            resultados = temporal.predecir_ventanas(datos)
        self.assertEqual([len(llamada.args[0]) for llamada in ajustar.call_args_list], [8, 12, 16])
        for numero, resultado in enumerate(resultados):
            self.assertEqual(resultado.referencias.mediana_global, 11.5)
            self.assertEqual(resultado.entrenamiento.movimientos_5min.iloc[:6].tolist(), [1, 2, 3, 4, 5, 6])
            self.assertEqual(resultado.predicciones.movimientos_5min.tolist(), [6] * 4)
            self.assertEqual(resultado.predicciones.transaccion_id.iloc[0], f"T{8 + 4 * numero:04d}")

    def test_cambiar_futuro_no_altera_referencias_train_ni_predicciones_anteriores(self):
        datos = movimientos()
        antes = temporal.predecir_ventanas(datos)
        datos.loc[19, "monto"] = 1e9
        datos.loc[19, "fecha_hora"] += pd.Timedelta(seconds=1)
        despues = temporal.predecir_ventanas(datos)
        for i, (a, d) in enumerate(zip(antes, despues)):
            pd.testing.assert_frame_equal(a.entrenamiento, d.entrenamiento)
            self.assertEqual(a.referencias.medianas_por_cuenta, d.referencias.medianas_por_cuenta)
            np.testing.assert_array_equal(a.referencias.probabilidades_globales,
                                          d.referencias.probabilidades_globales)
            pd.testing.assert_frame_equal(a.predicciones.iloc[:-1] if i == 2 else a.predicciones,
                                          d.predicciones.iloc[:-1] if i == 2 else d.predicciones)

    def test_ultima_ventana_reproduce_predicciones_80_20_y_parametros_actuales(self):
        datos = movimientos().iloc[::-1]
        resultados = temporal.predecir_ventanas(datos)
        base = detector.entrenar_y_predecir(datos)
        pd.testing.assert_frame_equal(resultados[-1].entrenamiento, base.entrenamiento)
        pd.testing.assert_frame_equal(resultados[-1].predicciones, base.predicciones)
        for resultado in resultados:
            for nombre, valor in detector.PARAMETROS_MODELO.items():
                self.assertEqual(resultado.modelo.get_params()[nombre], valor)
            self.assertEqual(resultado.modelo.feature_names_in_.tolist(), detector.CARACTERISTICAS)
            self.assertEqual(resultado.modelo.offset_, -0.5)
            np.testing.assert_array_equal(resultado.predicciones.prediccion_anomalia,
                                          (resultado.predicciones.puntaje_anomalia > 0.5).astype(int))

    def test_particion_compartida_rechaza_vacios_futuro_y_ids_compartidos(self):
        datos = movimientos()
        for train, evaluacion in ((datos.iloc[:0], datos.iloc[8:12]),
                                  (datos.iloc[:8], datos.iloc[:0]),
                                  (datos.iloc[:8], datos.iloc[7:12]),
                                  (datos.iloc[:8], datos.iloc[8:12].assign(transaccion_id=["T0000", "X", "Y", "Z"]))):
            with self.subTest(filas=(len(train), len(evaluacion))), self.assertRaises(ValueError):
                detector.entrenar_particion(train, evaluacion)


class MetricasTests(unittest.TestCase):
    def setUp(self):
        self.predicciones = pd.DataFrame({
            "transaccion_id": ["A", "B", "C", "D"],
            "puntaje_anomalia": [0.8, 0.9, 0.2, 0.1],
            "prediccion_anomalia": [1, 1, 0, 0],
        })

    def test_average_precision_usa_scores_continuos_y_etiquetas_unidas_por_id(self):
        reales = etiquetas(["D", "B", "A", "C"], {"B": "importe_inusual", "C": "horario_atipico"})
        resultado = temporal.medir_predicciones(self.predicciones, reales)
        np.testing.assert_array_equal(resultado.metricas.matriz_confusion, [[1, 1], [1, 1]])
        self.assertEqual((resultado.metricas.precision, resultado.recall, resultado.metricas.f1), (0.5, 0.5, 0.5))
        self.assertAlmostEqual(resultado.average_precision, 5 / 6)
        self.assertEqual(resultado.metricas.comparacion.es_anomalia.tolist(), [0, 1, 1, 0])
        tipos = resultado.metricas.por_tipo.set_index("tipo_anomalia")
        self.assertTrue(pd.isna(tipos.loc["concentracion", "recall"]))

    def test_recall_no_evaluable_sin_anomalias_y_ap_no_evaluable_con_una_clase(self):
        normales = temporal.medir_predicciones(self.predicciones, etiquetas(["D", "C", "B", "A"]))
        self.assertIsNone(normales.recall)
        self.assertIsNone(normales.average_precision)
        self.assertEqual(normales.metricas.precision, 0)
        todas = etiquetas(["A", "B", "C", "D"], dict.fromkeys("ABCD", "importe_inusual"))
        anomalias = temporal.medir_predicciones(self.predicciones, todas)
        self.assertIsNone(anomalias.average_precision)
        self.assertEqual(anomalias.recall, 0.5)

    def test_scores_invalidos_no_generan_average_precision(self):
        for valor in (np.nan, np.inf, -np.inf):
            datos = self.predicciones.copy()
            datos.loc[0, "puntaje_anomalia"] = valor
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                temporal.medir_predicciones(datos, etiquetas("ABCD", {"A": "importe_inusual"}))

    def test_agregado_cuenta_cada_id_una_vez_y_separa_anomalias_no_evaluadas(self):
        datos = movimientos()
        resultados = temporal.predecir_ventanas(datos)
        reales = etiquetas(datos.transaccion_id, {"T0001": "importe_inusual", "T0008": "importe_inusual",
                                                 "T0013": "horario_atipico", "T0018": "concentracion"})
        informe = temporal.evaluar_ventanas(resultados, reales.iloc[::-1])
        agregado = informe.agregado.metricas
        self.assertEqual(len(agregado.comparacion), 12)
        self.assertTrue(agregado.comparacion.transaccion_id.is_unique)
        self.assertEqual(int(agregado.comparacion.es_anomalia.sum()), 3)
        np.testing.assert_array_equal(agregado.matriz_confusion,
                                      sum(m.metricas.matriz_confusion for m in informe.por_ventana))
        self.assertEqual(informe.anomalias_no_evaluadas.transaccion_id.tolist(), ["T0001"])
        self.assertEqual(agregado.por_tipo.reales.tolist(), [1, 1, 1])
        with self.assertRaises(ValueError):
            temporal.evaluar_ventanas([resultados[0], resultados[0], resultados[2]], reales)


class ConsolaTests(unittest.TestCase):
    def test_todas_las_predicciones_y_comprobacion_80_20_preceden_la_lectura_de_etiquetas(self):
        datos = movimientos()
        eventos = []
        entrenar_real = detector.entrenar_particion

        def predecir(train, evaluacion):
            resultado = entrenar_real(train, evaluacion)
            eventos.append("predicciones")
            return resultado

        def leer(ruta, ids):
            self.assertEqual(eventos, ["predicciones"] * 4)
            self.assertEqual(set(ids), set(datos.transaccion_id))
            eventos.append("etiquetas")
            return etiquetas(datos.transaccion_id)

        engine, salida = MagicMock(), io.StringIO()
        with patch.object(detector, "crear_engine", return_value=engine), \
                patch.object(detector, "cargar_transacciones", return_value=datos), \
                patch.object(detector, "entrenar_particion", side_effect=predecir), \
                patch.object(detector, "cargar_etiquetas", side_effect=leer), redirect_stdout(salida):
            self.assertEqual(temporal.main(), 0)
        engine.dispose.assert_called_once()
        self.assertEqual(eventos, ["predicciones"] * 4 + ["etiquetas"])
        for texto in ("Ventana 1", "Ventana 2", "Ventana 3", "Agregado", "80/20: coincidencia exacta",
                      "Recall: N/D (sin anomalías reales)", "Average Precision: N/D"):
            self.assertIn(texto, salida.getvalue())

    def test_error_de_lectura_o_cierre_no_expone_credenciales_ni_consulta_etiquetas(self):
        for origen in ("lectura", "cierre"):
            with self.subTest(origen=origen):
                engine, errores = MagicMock(), io.StringIO()
                if origen == "cierre":
                    engine.dispose.side_effect = ValueError("secreto_ficticio")
                with patch.object(detector, "crear_engine", return_value=engine), \
                        patch.object(detector, "cargar_transacciones", return_value=movimientos(),
                                     side_effect=ValueError("secreto_ficticio") if origen == "lectura" else None), \
                        patch.object(detector, "cargar_etiquetas") as leer, redirect_stderr(errores):
                    self.assertEqual(temporal.main(), 1)
                leer.assert_not_called()
                engine.dispose.assert_called_once()
                self.assertNotIn("secreto_ficticio", errores.getvalue())


if __name__ == "__main__":
    unittest.main()
