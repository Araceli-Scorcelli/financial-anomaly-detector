"""Etapa 4B sobre ejemplos ficticios: sin DB ni archivo real de etiquetas."""

from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src import entrenar_modelo as detector


def movimientos(cantidad=10):
    return pd.DataFrame({
        "transaccion_id": [f"T{i:02d}" for i in range(cantidad)],
        "cuenta_id": ["A"] * cantidad,
        "fecha_hora": pd.date_range("2026-01-01 09:00:00", periods=cantidad, freq="min"),
        "tipo_movimiento": ["transferencia"] * cantidad,
        "direccion": ["ingreso"] * cantidad,
        "monto": [10.0 + i % 5 for i in range(cantidad)],
        "canal": ["homebanking"] * cantidad,
        "contraparte_id": ["CP_FICTICIA"] * cantidad,
    })


def etiquetas(ids, valores=None, tipos=None):
    return pd.DataFrame({"transaccion_id": list(ids),
                         "es_anomalia": valores if valores is not None else [0] * len(ids),
                         "tipo_anomalia": tipos if tipos is not None else [""] * len(ids)})


class EntrenamientoTests(unittest.TestCase):
    def setUp(self):
        bloqueo = patch("psycopg.connect", side_effect=AssertionError("No conectar a PostgreSQL"))
        bloqueo.start()
        self.addCleanup(bloqueo.stop)

    def test_corte_80_20_ordena_y_conserva_ids_sin_usar_indice(self):
        datos = movimientos().iloc[::-1].copy()
        datos.index = [7] * 10
        original = datos.copy(deep=True)
        entrenamiento, evaluacion = detector.separar_cronologicamente(datos)
        self.assertEqual(entrenamiento["transaccion_id"].tolist(), [f"T{i:02d}" for i in range(8)])
        self.assertEqual(evaluacion["transaccion_id"].tolist(), ["T08", "T09"])
        self.assertLess(entrenamiento["fecha_hora"].max(), evaluacion["fecha_hora"].min())
        self.assertEqual(set(entrenamiento.transaccion_id) | set(evaluacion.transaccion_id), set(datos.transaccion_id))
        self.assertFalse(set(entrenamiento.transaccion_id) & set(evaluacion.transaccion_id))
        pd.testing.assert_frame_equal(datos, original)

    def test_corte_no_separa_timestamps_empatados(self):
        datos = movimientos()
        datos.loc[8, "fecha_hora"] = datos.loc[7, "fecha_hora"]
        entrenamiento, evaluacion = detector.separar_cronologicamente(datos.iloc[::-1])
        self.assertEqual(len(entrenamiento), 7)
        self.assertEqual(evaluacion.transaccion_id.tolist(), ["T07", "T08", "T09"])
        self.assertLess(entrenamiento.fecha_hora.max(), evaluacion.fecha_hora.min())

    def test_rechaza_particiones_invalidas_y_ids_duplicados(self):
        for fraccion in (0, 1, -0.1, np.nan, True):
            with self.subTest(fraccion=fraccion), self.assertRaises(ValueError):
                detector.separar_cronologicamente(movimientos(), fraccion)
        for datos in (movimientos(0), movimientos(1),
                      movimientos().assign(fecha_hora=pd.Timestamp("2026-01-01")),
                      movimientos().assign(transaccion_id="duplicado")):
            with self.subTest(filas=len(datos)), self.assertRaises(ValueError):
                detector.separar_cronologicamente(datos)

    def test_entrena_solo_tres_features_y_referencias_solo_train_sin_leer_etiquetas(self):
        datos = movimientos()
        original = datos.copy(deep=True)
        with patch.object(detector, "ajustar_referencias", wraps=detector.ajustar_referencias) as ajustar, \
                patch.object(detector, "transformar_datos", wraps=detector.transformar_datos) as transformar, \
                patch("pandas.read_csv", side_effect=AssertionError("No leer etiquetas en entrenamiento")):
            resultado = detector.entrenar_y_predecir(datos)
        self.assertEqual(ajustar.call_count, 1)
        self.assertEqual(ajustar.call_args.args[0].transaccion_id.tolist(), [f"T{i:02d}" for i in range(8)])
        self.assertEqual(transformar.call_args.kwargs["historial"].transaccion_id.tolist(),
                         [f"T{i:02d}" for i in range(8)])
        self.assertEqual(resultado.modelo.feature_names_in_.tolist(),
                         ["monto_relativo", "rareza_horaria", "movimientos_5min"])
        self.assertEqual(resultado.predicciones.transaccion_id.tolist(), ["T08", "T09"])
        self.assertEqual(resultado.entrenamiento.shape, (8, 11))
        self.assertAlmostEqual(resultado.referencias.mediana_global, 11.5)
        pd.testing.assert_frame_equal(datos, original)

    def test_futuro_no_altera_referencias_entrenamiento_ni_objetivos_anteriores(self):
        datos = movimientos()
        antes = detector.entrenar_y_predecir(datos)
        datos.loc[9, "monto"] = 1e8
        datos.loc[9, "fecha_hora"] += pd.Timedelta(days=30)
        despues = detector.entrenar_y_predecir(datos)
        pd.testing.assert_frame_equal(antes.entrenamiento, despues.entrenamiento)
        self.assertEqual(antes.referencias.medianas_por_cuenta, despues.referencias.medianas_por_cuenta)
        np.testing.assert_array_equal(antes.referencias.probabilidades_globales,
                                      despues.referencias.probabilidades_globales)
        pd.testing.assert_frame_equal(antes.predicciones.iloc[:1], despues.predicciones.iloc[:1])

    def test_frecuencia_eval_incluye_historial_y_prefijo_del_lote(self):
        resultado = detector.entrenar_y_predecir(movimientos())
        self.assertEqual(resultado.entrenamiento.movimientos_5min.tolist(), [1, 2, 3, 4, 5, 6, 6, 6])
        self.assertEqual(resultado.predicciones.movimientos_5min.tolist(), [6, 6])

    def test_modelo_y_puntajes_reproducibles_y_equivalentes_a_predict(self):
        primero = detector.entrenar_y_predecir(movimientos())
        segundo = detector.entrenar_y_predecir(movimientos().iloc[::-1])
        pd.testing.assert_frame_equal(primero.predicciones, segundo.predicciones)
        parametros = primero.modelo.get_params()
        self.assertEqual(parametros["random_state"], 42)
        self.assertEqual(parametros["n_estimators"], 200)
        self.assertEqual(parametros["contamination"], "auto")
        self.assertEqual(parametros["max_samples"], "auto")
        self.assertEqual(primero.modelo.offset_, -0.5)
        x = primero.predicciones[detector.CARACTERISTICAS]
        np.testing.assert_allclose(primero.predicciones.puntaje_anomalia, -primero.modelo.score_samples(x))
        np.testing.assert_array_equal(primero.predicciones.prediccion_anomalia,
                                      (primero.modelo.predict(x) == -1).astype(int))

    def test_umbral_mayor_a_medio_y_empate_normal(self):
        modelo = MagicMock()
        modelo.offset_ = -0.5
        modelo.score_samples.return_value = np.array([-0.5, -0.50001])
        with patch.object(detector, "IsolationForest", return_value=modelo):
            resultado = detector.entrenar_y_predecir(movimientos())
        self.assertEqual(resultado.predicciones.prediccion_anomalia.tolist(), [0, 1])

    def test_top_limitado_a_eval_con_orden_por_puntaje_e_id(self):
        datos = pd.DataFrame({"transaccion_id": ["T09", "T08", "T10"],
                              "puntaje_anomalia": [0.8, 0.8, 0.2]})
        top = detector.principales_anomalias(datos, cantidad=2)
        self.assertEqual(top.transaccion_id.tolist(), ["T08", "T09"])
        self.assertEqual(len(datos), 3)


class EvaluacionTests(unittest.TestCase):
    def setUp(self):
        self.predicciones = pd.DataFrame({
            "transaccion_id": ["A", "B", "C", "D"],
            "puntaje_anomalia": [0.8, 0.9, 0.2, 0.1],
            "prediccion_anomalia": [1, 1, 0, 0],
        })
        self.etiquetas = etiquetas(["D", "B", "A", "C"], [0, 1, 0, 1],
                                   ["", "importe_inusual", "", "horario_atipico"])

    def test_metricas_conocidas_y_union_por_id_con_etiquetas_desordenadas(self):
        resultado = detector.evaluar_predicciones(self.predicciones, self.etiquetas)
        self.assertEqual(resultado.comparacion.transaccion_id.tolist(), ["A", "B", "C", "D"])
        self.assertEqual(resultado.comparacion.es_anomalia.tolist(), [0, 1, 1, 0])
        self.assertEqual(resultado.precision, 0.5)
        self.assertEqual(resultado.recall, 0.5)
        self.assertEqual(resultado.f1, 0.5)
        np.testing.assert_array_equal(resultado.matriz_confusion, [[1, 1], [1, 1]])

    def test_soporte_y_detectadas_por_tipo_sin_inventar_tipos_ausentes(self):
        tipos = detector.evaluar_predicciones(self.predicciones, self.etiquetas).por_tipo.set_index("tipo_anomalia")
        self.assertEqual(tipos.loc["importe_inusual", "reales"], 1)
        self.assertEqual(tipos.loc["importe_inusual", "detectadas"], 1)
        self.assertEqual(tipos.loc["horario_atipico", "no_detectadas"], 1)
        self.assertEqual(tipos.loc["concentracion", "reales"], 0)
        self.assertTrue(pd.isna(tipos.loc["concentracion", "recall"]))

    def test_sin_predicciones_positivas_y_sin_anomalias_reales(self):
        predicciones = self.predicciones.assign(prediccion_anomalia=0)
        resultado = detector.evaluar_predicciones(predicciones, self.etiquetas)
        self.assertEqual((resultado.precision, resultado.recall, resultado.f1), (0, 0, 0))
        normales = etiquetas(self.predicciones.transaccion_id)
        resultado = detector.evaluar_predicciones(predicciones, normales)
        np.testing.assert_array_equal(resultado.matriz_confusion, [[4, 0], [0, 0]])

    def test_rechaza_etiquetas_invalidas_y_predicciones_ambiguas(self):
        invalidas = [pd.concat([self.etiquetas, self.etiquetas.iloc[:1]]),
                     self.etiquetas.iloc[:3], self.etiquetas.assign(es_anomalia=2),
                     self.etiquetas.assign(tipo_anomalia="desconocido"),
                     self.etiquetas.assign(es_anomalia=None), self.etiquetas.assign(transaccion_id=" "),
                     self.etiquetas.assign(tipo_anomalia="importe_inusual")]
        for tabla in invalidas:
            with self.subTest(filas=len(tabla)), self.assertRaises(ValueError):
                detector.evaluar_predicciones(self.predicciones, tabla)
        for tabla in (pd.concat([self.predicciones, self.predicciones.iloc[:1]]),
                      self.predicciones.assign(prediccion_anomalia=-1)):
            with self.subTest(filas=len(tabla)), self.assertRaises(ValueError):
                detector.evaluar_predicciones(tabla, self.etiquetas)

    def test_lector_valida_cobertura_total_y_no_modifica_csv_ficticio(self):
        with tempfile.TemporaryDirectory() as temporal:
            ruta = Path(temporal) / "etiquetas.csv"
            self.etiquetas.to_csv(ruta, index=False)
            antes = ruta.read_bytes()
            cargadas = detector.cargar_etiquetas(ruta, ["A", "B", "C", "D"])
            self.assertEqual(set(cargadas.transaccion_id), {"A", "B", "C", "D"})
            self.assertEqual(ruta.read_bytes(), antes)
            for ids in (["A", "B", "C"], ["A", "B", "C", "D", "E"]):
                with self.subTest(ids=ids), self.assertRaises(ValueError):
                    detector.cargar_etiquetas(ruta, ids)

    def test_evaluar_no_modifica_predicciones_y_no_entrena(self):
        original = self.predicciones.copy(deep=True)
        with patch.object(detector, "IsolationForest", side_effect=AssertionError("No entrenar en evaluación")):
            detector.evaluar_predicciones(self.predicciones, self.etiquetas)
        pd.testing.assert_frame_equal(self.predicciones, original)


class ConsolaTests(unittest.TestCase):
    def test_etiquetas_se_leen_despues_de_generar_todas_las_predicciones(self):
        datos = movimientos()
        eventos = []
        entrenar_real = detector.entrenar_y_predecir

        def predecir(tabla):
            resultado = entrenar_real(tabla)
            self.assertEqual(len(resultado.predicciones), 2)
            eventos.append("predicciones_listas")
            return resultado

        def leer_etiquetas(ruta, ids):
            self.assertEqual(eventos, ["predicciones_listas"])
            self.assertEqual(set(ids), set(datos.transaccion_id))
            eventos.append("etiquetas")
            return etiquetas(datos.transaccion_id)

        engine, salida = MagicMock(), io.StringIO()
        with patch.object(detector, "crear_engine", return_value=engine), \
                patch.object(detector, "cargar_transacciones", return_value=datos), \
                patch.object(detector, "entrenar_y_predecir", side_effect=predecir), \
                patch.object(detector, "cargar_etiquetas", side_effect=leer_etiquetas), \
                redirect_stdout(salida):
            self.assertEqual(detector.main(), 0)
        engine.dispose.assert_called_once()
        self.assertEqual(eventos, ["predicciones_listas", "etiquetas"])
        for texto in ("Entrenamiento: 8", "Evaluación: 2", "Anomalías reales en evaluación: 0", "N/D"):
            self.assertIn(texto, salida.getvalue())

    def test_error_de_lectura_o_cierre_no_expone_detalles_ni_lee_etiquetas(self):
        for origen in ("lectura", "cierre"):
            with self.subTest(origen=origen):
                engine, errores = MagicMock(), io.StringIO()
                if origen == "cierre":
                    engine.dispose.side_effect = ValueError("secreto_ficticio")
                with patch.object(detector, "crear_engine", return_value=engine), \
                        patch.object(detector, "cargar_transacciones", return_value=movimientos(),
                                     side_effect=ValueError("secreto_ficticio") if origen == "lectura" else None), \
                        patch.object(detector, "cargar_etiquetas") as leer, redirect_stderr(errores):
                    self.assertEqual(detector.main(), 1)
                leer.assert_not_called()
                engine.dispose.assert_called_once()
                self.assertNotIn("secreto_ficticio", errores.getvalue())


if __name__ == "__main__":
    unittest.main()
