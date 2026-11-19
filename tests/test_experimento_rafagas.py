"""Experimento aislado: fixtures, modelos pequeños de prueba y ninguna DB real."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src import entrenar_modelo as detector
from src import evaluacion_temporal as temporal
from src import experimento_rafagas as experimento
from src import generar_escenarios_rafagas as generador
from src.preparar_datos import COLUMNAS_ORIGINALES


def catalogo():
    filas = []
    for dia in range(6):
        for numero in range(1, 31):
            filas.append({"transaccion_id": f"H{dia * 30 + numero:04d}", "cuenta_id": f"CTA{numero:03d}",
                          "fecha_hora": pd.Timestamp("2026-01-01 09:00") + pd.Timedelta(days=dia, minutes=numero),
                          "tipo_movimiento": "transferencia", "direccion": "egreso",
                          "monto": float(1000 + numero * 100 + dia * 10), "canal": "homebanking",
                          "contraparte_id": "CP_FICTICIA"})
    return pd.DataFrame(filas)


def predicciones():
    return pd.DataFrame({"transaccion_id": list("ABCDE"), "movimientos_5min": [1, 2, 3, 4, 5],
                         "prediccion_anomalia": [1, 0, 0, 1, 0], "puntaje_anomalia": [.7, .4, .3, .6, .2]})


def etiquetas(ids, positivos=()):
    return pd.DataFrame({"transaccion_id": list(ids), "es_anomalia": [int(i in positivos) for i in ids],
                         "tipo_anomalia": ["concentracion" if i in positivos else "" for i in ids]})


class ReglaYMetricasTests(unittest.TestCase):
    def test_or_inclusivo_desde_tres_sin_perder_alertas_ni_modificar_original(self):
        datos = predicciones()
        original = datos.copy(deep=True)
        hibridas = experimento.aplicar_regla(datos)
        self.assertEqual(hibridas.alerta_regla.tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(hibridas.prediccion_hibrida.tolist(), [1, 0, 1, 1, 1])
        self.assertEqual(hibridas.alerta_nueva.tolist(), [0, 0, 1, 0, 1])
        pd.testing.assert_frame_equal(hibridas[original.columns], original)
        pd.testing.assert_frame_equal(datos, original)

    def test_rechaza_ids_duplicados_y_conteos_invalidos(self):
        datos = predicciones()
        for invalida in (pd.concat([datos, datos.iloc[:1]]), datos.assign(transaccion_id=None),
                         datos.assign(movimientos_5min=np.inf), datos.assign(movimientos_5min=-1),
                         datos.assign(movimientos_5min=2.5), datos.assign(prediccion_anomalia=2)):
            with self.subTest(filas=len(invalida)), self.assertRaises(ValueError):
                experimento.aplicar_regla(invalida)

    def test_metricas_y_nuevas_alertas_identifican_tp_y_fp_por_id(self):
        resultado = experimento.comparar_enfoques(predicciones(), etiquetas("EDCBA", "BCD"))
        metricas = resultado.metricas.set_index("enfoque")
        self.assertEqual(metricas.loc["Isolation Forest", ["TP", "FP", "TN", "FN"]].tolist(), [1, 1, 1, 2])
        self.assertEqual(metricas.loc["Híbrido", ["TP", "FP", "TN", "FN"]].tolist(), [2, 2, 0, 1])
        self.assertAlmostEqual(metricas.loc["Híbrido", "recall"], 2 / 3)
        self.assertEqual(metricas.loc["Híbrido", "tasa_fp"], 1)
        self.assertEqual(resultado.nuevas_alertas.transaccion_id.tolist(), ["C", "E"])
        self.assertEqual(resultado.nuevas_alertas.es_anomalia.tolist(), [1, 0])
        self.assertEqual(resultado.comparacion.transaccion_id.tolist(), list("ABCDE"))

    def test_recall_sin_positivos_es_no_evaluable(self):
        resultado = experimento.comparar_enfoques(predicciones(), etiquetas("ABCDE"))
        self.assertTrue(resultado.metricas.recall.isna().all())


class EpisodiosTests(unittest.TestCase):
    def setUp(self):
        self.datos = experimento.aplicar_regla(pd.DataFrame({
            "transaccion_id": ["A1", "A2", "A3", "B1", "B2", "C1", "C2"],
            "cuenta_id": ["A"] * 3 + ["B"] * 2 + ["C"] * 2,
            "fecha_hora": pd.to_datetime(["2026-04-01 09:00", "2026-04-01 09:01", "2026-04-01 09:02",
                                           "2026-04-02 09:00", "2026-04-02 09:01", "2026-04-03 09:00", "2026-04-03 09:02"]),
            "movimientos_5min": [1, 2, 3, 3, 4, 1, 2],
            "prediccion_anomalia": [0, 0, 0, 0, 1, 0, 0],
        }))
        self.mapa = pd.DataFrame({"transaccion_id": self.datos.transaccion_id,
                                  "episodio_id": ["E1"] * 3 + ["E2"] * 2 + ["E3"] * 2,
                                  "escenario": ["concentracion_anomala"] * 3 + ["concentracion_legitima"] * 2
                                  + ["concentracion_anomala"] * 2,
                                  "patron": ["densa"] * 3 + ["lote"] * 2 + ["corta"] * 2})

    def test_detectar_episodio_difiere_de_alertar_todas_y_mide_demoras(self):
        resultado = experimento.evaluar_episodios(self.datos.iloc[::-1], self.mapa.iloc[::-1])
        tabla = resultado.set_index(["episodio_id", "enfoque"])
        primero = tabla.loc[("E1", "Híbrido")]
        self.assertTrue(primero.alguna_alerta)
        self.assertFalse(primero.cobertura_completa)
        self.assertEqual((primero.alertas, primero.primera_posicion, primero.demora_segundos), (1, 3, 120))
        self.assertFalse(tabla.loc[("E1", "Isolation Forest"), "alguna_alerta"])
        self.assertTrue(tabla.loc[("E2", "Híbrido"), "cobertura_completa"])
        self.assertEqual(tabla.loc[("E2", "Híbrido"), "demora_segundos"], 0)
        self.assertTrue(pd.isna(tabla.loc[("E3", "Híbrido"), "demora_segundos"]))
        resumen = experimento.resumir_episodios(resultado).set_index(["escenario", "enfoque"])
        fila = resumen.loc[("concentracion_anomala", "Híbrido")]
        self.assertEqual((fila.episodios, fila.con_alerta, fila.con_cobertura_completa), (2, 1, 0))
        self.assertEqual((fila.n_demoras, fila.mediana_demora_segundos), (1, 120))

    def test_mapa_no_admite_duplicados_ni_episodios_incompletos(self):
        for mapa, datos in ((pd.concat([self.mapa, self.mapa.iloc[:1]]), self.datos),
                            (self.mapa, self.datos.iloc[:-1])):
            with self.assertRaises(ValueError):
                experimento.evaluar_episodios(datos, mapa)

    def test_cambiar_predicciones_no_cambia_el_manifesto_de_episodios(self):
        antes = experimento.evaluar_episodios(self.datos, self.mapa)
        despues = experimento.evaluar_episodios(self.datos.assign(prediccion_anomalia=0, prediccion_hibrida=0), self.mapa)
        campos = ["episodio_id", "enfoque", "escenario", "patron", "operaciones", "inicio", "fin"]
        pd.testing.assert_frame_equal(antes[campos], despues[campos])


class GeneracionYPrediccionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.historial = catalogo()
        cls.resultado = detector.entrenar_y_predecir(cls.historial)
        cls.train = cls.resultado.entrenamiento[COLUMNAS_ORIGINALES]

    def generar(self, semilla=2026):
        return generador.generar_escenarios(self.train, self.resultado.referencias, semilla=semilla)

    def test_generacion_reproducible_con_esquemas_separados_y_fechas_posteriores(self):
        primero, segundo, distinto = self.generar(), self.generar(), self.generar(2027)
        for campo in ("transacciones", "etiquetas", "episodios"):
            pd.testing.assert_frame_equal(getattr(primero, campo), getattr(segundo, campo))
        self.assertFalse(primero.transacciones.equals(distinto.transacciones))
        self.assertEqual(primero.transacciones.columns.tolist(), COLUMNAS_ORIGINALES)
        self.assertEqual((len(primero.transacciones), int(primero.etiquetas.es_anomalia.sum())), (180, 30))
        self.assertTrue(primero.transacciones.transaccion_id.is_unique)
        self.assertEqual(set(primero.transacciones.transaccion_id), set(primero.etiquetas.transaccion_id))
        self.assertEqual(set(primero.transacciones.transaccion_id), set(primero.episodios.transaccion_id))
        self.assertGreater(primero.transacciones.fecha_hora.min(), pd.Timestamp("2026-03-31 23:59:59"))
        self.assertLess(primero.transacciones.fecha_hora.max(), pd.Timestamp("2026-05-01"))
        self.assertTrue((primero.transacciones.monto > 0).all())
        self.assertFalse(primero.transacciones.isna().any().any())

    def test_protocolo_equilibra_patrones_perfiles_y_etiquetas_sin_densidad_como_verdad(self):
        escenarios = self.generar()
        episodios = escenarios.episodios.loc[escenarios.episodios.episodio_id.ne("")]
        manifiesto = episodios.drop_duplicates("episodio_id")
        self.assertEqual(manifiesto.groupby(["patron", "escenario"]).size().tolist(), [2] * 8)
        self.assertEqual(manifiesto.groupby(["patron", "escenario", "perfil"]).size().tolist(), [1] * 16)
        cuentas = episodios.merge(escenarios.transacciones[["transaccion_id", "cuenta_id", "fecha_hora"]], on="transaccion_id")
        self.assertEqual(cuentas.groupby("episodio_id").cuenta_id.nunique().tolist(), [1] * 16)
        self.assertEqual(cuentas.drop_duplicates("episodio_id").cuenta_id.nunique(), 16)
        preparados = detector.transformar_datos(escenarios.transacciones, self.resultado.referencias, self.historial)
        detalle = preparados.merge(episodios, on="transaccion_id")
        maximos = detalle.groupby(["escenario", "patron"]).movimientos_5min.max()
        for clase in ("concentracion_anomala", "concentracion_legitima"):
            self.assertEqual(maximos.loc[(clase, "corta")], 2)
            self.assertEqual(maximos.loc[(clase, "espaciada")], 2)
            self.assertEqual(maximos.loc[(clase, "borde")], 3)
            self.assertEqual(maximos.loc[(clase, "densa")], 5)

    def test_prediccion_congelada_no_ajusta_y_preserva_parametros_y_referencias(self):
        escenarios = self.generar()
        parametros = self.resultado.modelo.get_params().copy()
        medianas = self.resultado.referencias.medianas_por_cuenta.copy()
        probabilidades = self.resultado.referencias.probabilidades_globales.copy()
        train_antes = self.resultado.entrenamiento.copy(deep=True)
        with ExitStack() as pila:
            for objeto, nombre in ((self.resultado.modelo, "fit"), (detector, "ajustar_referencias"),
                                   (detector, "entrenar_particion")):
                pila.enter_context(patch.object(objeto, nombre, side_effect=AssertionError("No reajustar")))
            predichas = experimento.predecir_congelado(escenarios.transacciones, self.resultado, self.historial)
        self.assertEqual(len(predichas), 180)
        self.assertEqual(self.resultado.modelo.get_params(), parametros)
        self.assertEqual(self.resultado.referencias.medianas_por_cuenta, medianas)
        np.testing.assert_array_equal(self.resultado.referencias.probabilidades_globales, probabilidades)
        pd.testing.assert_frame_equal(self.resultado.entrenamiento, train_antes)
        np.testing.assert_array_equal(predichas.prediccion_anomalia, (predichas.puntaje_anomalia > .5).astype(int))

    def test_prefijos_causales_y_cambiar_futuro_no_altera_operaciones_anteriores(self):
        escenarios = self.generar()
        antes = experimento.predecir_congelado(escenarios.transacciones, self.resultado, self.historial)
        ids = escenarios.episodios.loc[escenarios.episodios.patron.eq("densa"), "transaccion_id"].iloc[:3]
        corte = int(escenarios.transacciones.index[escenarios.transacciones.transaccion_id.eq(ids.iloc[-1])][0]) + 1
        prefijo = experimento.predecir_congelado(escenarios.transacciones.iloc[:corte], self.resultado, self.historial)
        pd.testing.assert_frame_equal(antes.iloc[:corte], prefijo)
        futuro = escenarios.transacciones.copy(deep=True)
        futuro.loc[futuro.index[-1], "monto"] = 1e9
        futuro.loc[futuro.index[-1], "fecha_hora"] += pd.Timedelta(days=1)
        despues = experimento.predecir_congelado(futuro, self.resultado, self.historial)
        pd.testing.assert_frame_equal(antes.iloc[:-1], despues.iloc[:-1])

    def test_rechaza_solapamientos_temporales_con_historial(self):
        nuevas = self.generar().transacciones.copy()
        nuevas.loc[0, "fecha_hora"] = self.historial.fecha_hora.max()
        with self.assertRaises(ValueError):
            experimento.predecir_congelado(nuevas, self.resultado, self.historial)


class ConsolaTests(unittest.TestCase):
    def test_completa_predicciones_historicas_y_nuevas_antes_del_csv_de_etiquetas(self):
        datos, eventos, engine, salida = catalogo(), [], MagicMock(), io.StringIO()
        entrenar_real, predecir_real = detector.entrenar_particion, experimento.predecir_congelado

        def entrenar(train, evaluacion):
            resultado = entrenar_real(train, evaluacion)
            eventos.append("historica")
            return resultado

        def predecir(*args):
            resultado = predecir_real(*args)
            eventos.append("nueva")
            return resultado

        def leer(ruta, ids):
            self.assertEqual(eventos, ["historica"] * 3 + ["nueva"])
            return etiquetas(list(ids))

        with patch.object(detector, "crear_engine", return_value=engine), \
                patch.object(detector, "cargar_transacciones", return_value=datos), \
                patch.object(detector, "entrenar_particion", side_effect=entrenar), \
                patch.object(experimento, "predecir_congelado", side_effect=predecir), \
                patch.object(detector, "cargar_etiquetas", side_effect=leer), redirect_stdout(salida):
            self.assertEqual(experimento.main(), 0)
        engine.dispose.assert_called_once()
        self.assertIn("Histórico exploratorio", salida.getvalue())
        self.assertIn("Escenarios nuevos", salida.getvalue())

    def test_error_lectura_o_cierre_no_expone_credenciales(self):
        for origen in ("lectura", "cierre"):
            with self.subTest(origen=origen):
                engine, errores = MagicMock(), io.StringIO()
                if origen == "cierre":
                    engine.dispose.side_effect = ValueError("secreto_ficticio")
                with patch.object(detector, "crear_engine", return_value=engine), \
                        patch.object(detector, "cargar_transacciones", return_value=catalogo(),
                                     side_effect=ValueError("secreto_ficticio") if origen == "lectura" else None), \
                        patch.object(detector, "cargar_etiquetas") as leer, redirect_stderr(errores):
                    self.assertEqual(experimento.main(), 1)
                leer.assert_not_called()
                self.assertNotIn("secreto_ficticio", errores.getvalue())


if __name__ == "__main__":
    unittest.main()
