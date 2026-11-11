"""Diagnósticos sobre fixtures: sin PostgreSQL, etiquetas reales ni escrituras."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src import analisis_errores as analisis
from src import entrenar_modelo as detector
from src import evaluacion_temporal as temporal


def comparacion():
    return pd.DataFrame({
        "transaccion_id": ["A1", "A2", "A3", "A4", "B1", "C1", "C2", "D1"],
        "cuenta_id": ["A", "A", "A", "A", "B", "C", "C", "D"],
        "fecha_hora": pd.date_range("2026-02-01 09:00", periods=8, freq="min"),
        "es_anomalia": [0, 0, 0, 0, 0, 1, 1, 0],
        "tipo_anomalia": ["", "", "", "", "", "concentracion", "concentracion", ""],
        "prediccion_anomalia": [1, 1, 0, 0, 1, 1, 0, 0],
        "monto_relativo": [2., 3., 1., 1., 4., 1., 1., 1.],
        "rareza_horaria": [3., 4., 2., 2., 5., 2., 2., 2.],
        "movimientos_5min": [1, 1, 1, 1, 1, 2, 1, 1],
        "puntaje_anomalia": [.6, .7, .3, .4, .8, .65, .45, .2],
        "ventana": [1, 1, 1, 1, 2, 2, 2, 3],
    })


def movimientos():
    return pd.DataFrame({
        "transaccion_id": [f"T{i:02d}" for i in range(20)], "cuenta_id": ["A"] * 20,
        "fecha_hora": pd.date_range("2026-01-01 09:00", periods=20, freq="h"),
        "tipo_movimiento": ["transferencia"] * 20, "direccion": ["ingreso"] * 20,
        "monto": [10. + i % 4 for i in range(20)], "canal": ["homebanking"] * 20,
        "contraparte_id": ["CP_FICTICIA"] * 20,
    })


def etiquetas(ids):
    return pd.DataFrame({"transaccion_id": list(ids), "es_anomalia": [0] * len(ids),
                         "tipo_anomalia": [""] * len(ids)})


class DiagnosticoTests(unittest.TestCase):
    def test_clasifica_las_cuatro_respuestas_sin_alterar_ids_valores_ni_entrada(self):
        datos = comparacion().iloc[::-1].copy()
        datos.index = [4] * len(datos)
        original = datos.copy(deep=True)
        resultado = analisis.clasificar_errores(datos).set_index("transaccion_id")
        self.assertEqual(resultado.loc[["A1", "A2", "A3", "A4", "B1", "C1", "C2", "D1"],
                                       "resultado_evaluacion"].tolist(),
                         ["FP", "FP", "TN", "TN", "FP", "TP", "FN", "TN"])
        self.assertEqual(resultado.loc["C2", "puntaje_anomalia"], .45)
        pd.testing.assert_frame_equal(datos, original)

    def test_rechaza_duplicados_nulos_predicciones_invalidas_y_scores_no_finitos(self):
        datos = comparacion()
        for invalida in (pd.concat([datos, datos.iloc[:1]]), datos.assign(transaccion_id=None),
                         datos.assign(es_anomalia=2), datos.assign(prediccion_anomalia=-1),
                         datos.assign(puntaje_anomalia=np.inf), datos.assign(monto_relativo=np.nan),
                         datos.drop(columns="ventana")):
            with self.subTest(columnas=len(invalida.columns)), self.assertRaises(ValueError):
                analisis.clasificar_errores(invalida)

    def test_tasas_por_cuenta_distinguen_volumen_y_negativos_reales(self):
        tabla = analisis.resumir_falsas_alertas(analisis.clasificar_errores(comparacion()), ["cuenta_id"])
        cuentas = tabla.set_index("cuenta_id")
        self.assertEqual(cuentas.loc["A", "FP"], 2)
        self.assertEqual(cuentas.loc["B", "FP"], 1)
        self.assertEqual(cuentas.loc["A", "tasa_fp_normales"], .5)
        self.assertEqual(cuentas.loc["B", "tasa_fp_normales"], 1.)
        self.assertEqual(cuentas.loc["C", "total"], 2)
        self.assertEqual(cuentas.loc["C", "normales_reales"], 0)
        self.assertTrue(pd.isna(cuentas.loc["C", "tasa_fp_normales"]))
        self.assertEqual(cuentas.loc["C", "tasa_alertas_total"], .5)
        self.assertEqual(cuentas.loc["D", "FP"], 0)
        self.assertEqual(cuentas.loc["D", "tasa_fp_normales"], 0)

    def test_tasas_por_ventana_y_distribuciones_fp_tn_con_valores_conocidos(self):
        datos = analisis.clasificar_errores(comparacion())
        ventanas = analisis.resumir_falsas_alertas(datos, ["ventana"]).set_index("ventana")
        self.assertEqual(ventanas.loc[2, "FP"], 1)
        self.assertEqual(ventanas.loc[2, "tasa_fp_normales"], 1)
        self.assertAlmostEqual(ventanas.loc[2, "proporcion_fp_total"], 1 / 3)
        resumen = analisis.comparar_distribuciones(datos).set_index(["ventana", "grupo", "variable"])
        self.assertEqual(resumen.loc[("agregado", "FP", "monto_relativo"), "mediana"], 3)
        self.assertEqual(resumen.loc[("agregado", "FP", "monto_relativo"), "q25"], 2.5)
        self.assertEqual(resumen.loc[("agregado", "TN", "puntaje_anomalia"), "mediana"], .3)
        self.assertEqual(resumen.loc[(1, "FP", "monto_relativo"), "mediana"], 2.5)


class RafagasTests(unittest.TestCase):
    def setUp(self):
        self.concentraciones = pd.DataFrame({
            "transaccion_id": ["R0", "R1", "R2", "R3", "R4", "R5"],
            "cuenta_id": ["A", "A", "A", "A", "A", "B"],
            "fecha_hora": pd.to_datetime(["2026-02-01 09:00", "2026-02-01 09:01", "2026-02-01 09:06",
                                           "2026-02-01 09:11:01", "2026-02-01 09:11:01", "2026-02-01 09:02"], format="mixed"),
        })
        self.datos = comparacion().iloc[:5].copy()
        self.datos["transaccion_id"] = ["R1", "R2", "R3", "R4", "R5"]
        self.datos["prediccion_anomalia"] = [0, 1, 0, 0, 1]
        self.datos["ventana"] = [1, 2, 2, 2, 1]

    def test_agrupa_por_cuenta_y_gaps_con_limite_inclusivo_y_ordena_empates_por_id(self):
        detalle, resumen = analisis.reconstruir_rafagas(self.concentraciones.iloc[::-1], self.datos)
        grupos = [grupo.transaccion_id.tolist() for _, grupo in detalle.groupby("rafaga")]
        self.assertEqual(grupos, [["R0", "R1", "R2"], ["R5"], ["R3", "R4"]])
        self.assertEqual(resumen.operaciones.tolist(), [3, 1, 2])
        self.assertEqual(detalle.set_index("transaccion_id").loc["R4", "posicion"], 2)

    def test_primera_alerta_demora_censura_y_rafaga_sin_alertas(self):
        detalle, resumen = analisis.reconstruir_rafagas(self.concentraciones, self.datos)
        primera = resumen.iloc[0]
        self.assertEqual(primera.primera_alerta_posicion, 3)
        self.assertEqual(primera.demora_segundos, 360)
        self.assertEqual(primera.sin_evaluacion, 1)
        self.assertEqual(primera.ventanas, "1,2")
        self.assertTrue(primera.cambio_modelo)
        self.assertEqual(detalle.set_index("transaccion_id").loc["R0", "resultado_correcto"], "Sin evaluación")
        self.assertTrue(pd.isna(resumen.iloc[2].primera_alerta_posicion))
        self.assertTrue(pd.isna(resumen.iloc[2].demora_segundos))
        self.assertEqual(resumen.iloc[2].detectadas, 0)

    def test_cambiar_alertas_no_cambia_pertenencia_ni_posiciones_de_las_rafagas(self):
        antes, _ = analisis.reconstruir_rafagas(self.concentraciones, self.datos)
        despues, _ = analisis.reconstruir_rafagas(self.concentraciones, self.datos.assign(prediccion_anomalia=0))
        columnas = ["transaccion_id", "rafaga", "posicion"]
        pd.testing.assert_frame_equal(antes[columnas], despues[columnas])

    def test_sin_concentraciones_devuelve_tablas_vacias(self):
        detalle, resumen = analisis.reconstruir_rafagas(self.concentraciones.iloc[:0], self.datos)
        self.assertTrue(detalle.empty)
        self.assertTrue(resumen.empty)


class IntegracionTests(unittest.TestCase):
    def test_analisis_no_ajusta_no_recalcula_no_lee_y_conserva_las_predicciones(self):
        datos = movimientos()
        resultados = temporal.predecir_ventanas(datos)
        reales = etiquetas(datos.transaccion_id)
        reales.loc[reales.transaccion_id.isin(["T00", "T08", "T09"]), ["es_anomalia", "tipo_anomalia"]] = [1, "concentracion"]
        informe = temporal.evaluar_ventanas(resultados, reales)
        originales = [r.predicciones.copy(deep=True) for r in resultados]
        with ExitStack() as pila:
            for nombre in ("entrenar_particion", "ajustar_referencias", "transformar_datos", "crear_engine", "cargar_etiquetas"):
                pila.enter_context(patch.object(detector, nombre, side_effect=AssertionError("No ajustar ni leer durante diagnóstico")))
            pila.enter_context(patch("pandas.read_csv", side_effect=AssertionError("No leer")))
            for r in resultados:
                pila.enter_context(patch.object(r.modelo, "score_samples", side_effect=AssertionError("No recalcular puntajes")))
            diagnostico = analisis.analizar_errores(resultados, informe)
        self.assertEqual(len(diagnostico.movimientos), 12)
        self.assertTrue(diagnostico.movimientos.transaccion_id.is_unique)
        self.assertEqual(int(diagnostico.resumen_rafagas.sin_evaluacion.sum()), 1)
        esperados = diagnostico.movimientos.loc[diagnostico.movimientos.resultado_evaluacion.eq("FN"), "transaccion_id"].tolist()
        self.assertEqual(diagnostico.falsos_negativos.transaccion_id.tolist(), esperados)
        for original, resultado in zip(originales, resultados):
            pd.testing.assert_frame_equal(original, resultado.predicciones)

    def test_variacion_y_uso_de_variables_inspeccionan_el_modelo_ya_entrenado(self):
        resultado = detector.entrenar_y_predecir(movimientos())
        parametros = resultado.modelo.get_params().copy()
        resumen = analisis.resumir_entrenamientos([resultado]).set_index("variable")
        fila = resumen.loc["movimientos_5min"]
        self.assertEqual((fila.minimo, fila.maximo, fila.distintos), (1, 1, 1))
        self.assertEqual((fila.arboles_con_variable, fila.cortes), (0, 0))
        self.assertEqual(resultado.modelo.get_params(), parametros)

    def test_cli_lee_etiquetas_despues_de_las_tres_predicciones_y_cierra_motor(self):
        datos, eventos, engine, salida = movimientos(), [], MagicMock(), io.StringIO()
        entrenar_real = detector.entrenar_particion

        def entrenar(train, evaluacion):
            resultado = entrenar_real(train, evaluacion)
            eventos.append("prediccion")
            return resultado

        def leer(ruta, ids):
            self.assertEqual(eventos, ["prediccion"] * 3)
            return etiquetas(list(ids))

        with patch.object(detector, "crear_engine", return_value=engine), \
                patch.object(detector, "cargar_transacciones", return_value=datos), \
                patch.object(detector, "entrenar_particion", side_effect=entrenar), \
                patch.object(detector, "cargar_etiquetas", side_effect=leer), redirect_stdout(salida):
            self.assertEqual(analisis.main(), 0)
        engine.dispose.assert_called_once()
        self.assertIn("12 operaciones evaluadas únicas", salida.getvalue())
        self.assertIn("Hechos observados", salida.getvalue())
        self.assertIn("Hipótesis", salida.getvalue())

    def test_errores_de_lectura_o_cierre_no_exponen_detalles_ni_consultan_etiquetas(self):
        for origen in ("lectura", "cierre"):
            with self.subTest(origen=origen):
                engine, errores = MagicMock(), io.StringIO()
                if origen == "cierre":
                    engine.dispose.side_effect = ValueError("secreto_ficticio")
                with patch.object(detector, "crear_engine", return_value=engine), \
                        patch.object(detector, "cargar_transacciones", return_value=movimientos(),
                                     side_effect=ValueError("secreto_ficticio") if origen == "lectura" else None), \
                        patch.object(detector, "cargar_etiquetas") as leer, redirect_stderr(errores):
                    self.assertEqual(analisis.main(), 1)
                leer.assert_not_called()
                engine.dispose.assert_called_once()
                self.assertNotIn("secreto_ficticio", errores.getvalue())


if __name__ == "__main__":
    unittest.main()
