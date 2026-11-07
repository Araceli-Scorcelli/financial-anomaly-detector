"""Pruebas del dataset sintético; ejecutar con unittest discover -s tests."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.generar_dataset import generar_dataset, guardar_dataset, validar_dataset


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transacciones, cls.etiquetas = generar_dataset()

    def test_columnas_y_cardinalidades(self):
        self.assertEqual(list(self.transacciones.columns), [
            "transaccion_id", "cuenta_id", "fecha_hora", "tipo_movimiento",
            "direccion", "monto", "canal", "contraparte_id",
        ])
        self.assertEqual(len(self.transacciones), 1500)
        self.assertEqual(self.transacciones["cuenta_id"].nunique(), 30)
        self.assertEqual(len(self.etiquetas), 1500)
        self.assertEqual(self.etiquetas["es_anomalia"].sum(), 30)
        tipos = self.etiquetas.loc[self.etiquetas["es_anomalia"] == 1, "tipo_anomalia"]
        self.assertEqual(tipos.value_counts().to_dict(), {
            "importe_inusual": 10, "horario_atipico": 10, "concentracion": 10,
        })

    def test_identificadores_y_campos_obligatorios(self):
        self.assertTrue(self.transacciones["transaccion_id"].is_unique)
        self.assertTrue(self.etiquetas["transaccion_id"].is_unique)
        self.assertEqual(set(self.transacciones["transaccion_id"]),
                         set(self.etiquetas["transaccion_id"]))
        self.assertFalse(self.transacciones.isna().any().any())
        self.assertFalse(self.etiquetas[["transaccion_id", "es_anomalia"]].isna().any().any())

    def test_montos_y_fechas(self):
        montos = self.transacciones["monto"]
        self.assertTrue(np.isfinite(montos).all())
        self.assertTrue((montos > 0).all())
        np.testing.assert_allclose(montos * 100, np.round(montos * 100), atol=1e-5, rtol=0)
        fechas = self.transacciones["fecha_hora"]
        self.assertTrue((fechas >= pd.Timestamp("2026-01-01")).all())
        self.assertTrue((fechas < pd.Timestamp("2026-04-01")).all())
        self.assertTrue(fechas.is_monotonic_increasing)

    def test_operaciones_consistentes(self):
        combinaciones = {
            ("transferencia", "ingreso", "homebanking"),
            ("transferencia", "egreso", "homebanking"),
            ("pago_tarjeta", "egreso", "pos"),
            ("cobro_tarjeta", "ingreso", "pos"),
            ("pago_servicio", "egreso", "homebanking"),
            ("retiro_efectivo", "egreso", "cajero"),
            ("deposito_efectivo", "ingreso", "cajero"),
        }
        observadas = set(self.transacciones[
            ["tipo_movimiento", "direccion", "canal"]
        ].itertuples(index=False, name=None))
        self.assertTrue(observadas <= combinaciones)
        frecuencias = self.transacciones.groupby("cuenta_id").size()
        self.assertGreater(frecuencias.nunique(), 1)
        self.assertLess(frecuencias.loc["CTA020"], frecuencias.loc["CTA021"])

    def test_anomalias_respecto_de_su_cuenta(self):
        datos = self.transacciones.merge(self.etiquetas, on="transaccion_id", validate="one_to_one")
        normales = datos.loc[datos["es_anomalia"] == 0]
        self.assertTrue(normales["fecha_hora"].dt.hour.between(8, 21).all())
        concentradas = datos.loc[datos["tipo_anomalia"] == "concentracion"]
        self.assertEqual(concentradas["cuenta_id"].nunique(), 2)
        for cuenta, grupo in concentradas.groupby("cuenta_id"):
            self.assertEqual(len(grupo), 5)
            self.assertEqual(grupo["fecha_hora"].max() - grupo["fecha_hora"].min(),
                             pd.Timedelta(minutes=4))
            habituales = normales.loc[normales["cuenta_id"] == cuenta]
            self.assertTrue(grupo["fecha_hora"].dt.hour.between(
                habituales["fecha_hora"].dt.hour.min(), habituales["fecha_hora"].dt.hour.max()
            ).all())
            # Un horario entre el mínimo y máximo podría caer en el hueco entre
            # las dos ventanas habituales. La ráfaga debe ocupar una de ellas.
            inicio_minutos = habituales["fecha_hora"].dt.hour.min() * 60
            desfase = grupo["fecha_hora"].dt.hour * 60 + grupo["fecha_hora"].dt.minute - inicio_minutos
            self.assertTrue((desfase.between(0, 89) | desfase.between(360, 449)).all())
            self.assertLessEqual(grupo["monto"].max(), habituales["monto"].max() * 2)
            self.assertFalse((habituales["fecha_hora"].dt.normalize() ==
                              grupo["fecha_hora"].iloc[0].normalize()).any())
            frecuencia_normal = pd.Series(1, index=habituales["fecha_hora"]).sort_index().rolling(
                "5min", closed="both"
            ).sum().max()
            frecuencia_rafaga = pd.Series(1, index=grupo["fecha_hora"]).sort_index().rolling(
                "5min", closed="both"
            ).sum().max()
            self.assertEqual(frecuencia_normal, 1)
            self.assertEqual(frecuencia_rafaga, 5)
            self.assertGreater(frecuencia_rafaga, frecuencia_normal)
        tipos_por_cuenta = datos.loc[datos["es_anomalia"] == 1].groupby("cuenta_id")["tipo_anomalia"].nunique()
        self.assertTrue((tipos_por_cuenta == 1).all())

    def test_importes_con_distintos_niveles_de_desviacion(self):
        datos = self.transacciones.merge(self.etiquetas, on="transaccion_id")
        normales = datos.loc[datos["es_anomalia"] == 0]
        importes = datos.loc[datos["tipo_anomalia"] == "importe_inusual"]
        niveles = []
        relativos_al_maximo = []
        for fila in importes.itertuples():
            habituales = normales.loc[normales["cuenta_id"] == fila.cuenta_id, "monto"]
            referencia = habituales.median()
            nivel = min((3, 5, 10), key=lambda valor: abs(fila.monto / referencia - valor))
            self.assertAlmostEqual(fila.monto, round(referencia * nivel, 2), places=2)
            niveles.append(nivel)
            relativos_al_maximo.append(fila.monto / habituales.max())
        self.assertEqual(pd.Series(niveles).value_counts().to_dict(), {3: 4, 5: 3, 10: 3})
        # Al menos un caso moderado queda por debajo de dos máximos habituales.
        self.assertLess(min(relativos_al_maximo), 2)

    def test_horarios_variados_fuera_de_las_ventanas_de_su_cuenta(self):
        datos = self.transacciones.merge(self.etiquetas, on="transaccion_id")
        horarios = datos.loc[datos["tipo_anomalia"] == "horario_atipico"]
        self.assertGreaterEqual(horarios["fecha_hora"].dt.hour.nunique(), 4)
        cercanos = lejanos = antes = despues = intermedios = 0
        for fila in horarios.itertuples():
            # Ventanas documentadas: inicio variable y dos bloques de 90 minutos.
            numero = int(fila.cuenta_id.removeprefix("CTA"))
            inicio = (8 + numero % 3) * 3600
            fin = inicio + (8 if numero > 20 else 10) * 3600
            ventanas = [(inicio, inicio + 90 * 60), (fin - 2 * 3600, fin - 30 * 60)]
            segundo = fila.fecha_hora.hour * 3600 + fila.fecha_hora.minute * 60 + fila.fecha_hora.second
            self.assertFalse(any(desde <= segundo < hasta for desde, hasta in ventanas))
            distancia = min(abs(segundo - borde) for ventana in ventanas for borde in ventana)
            cercano = 20 * 60 <= distancia < 41 * 60
            lejano = 90 * 60 <= distancia < 121 * 60
            self.assertTrue(cercano or lejano)
            cercanos += cercano
            lejanos += lejano
            antes += segundo < ventanas[0][0]
            despues += segundo >= ventanas[1][1]
            intermedios += ventanas[0][1] <= segundo < ventanas[1][0]
        self.assertEqual((cercanos, lejanos), (5, 5))
        self.assertGreater(antes, 0)
        self.assertGreater(despues, 0)
        self.assertGreater(intermedios, 0)

    def test_frecuencia_normal_sin_concentraciones_accidentales(self):
        normales = self.transacciones.merge(self.etiquetas, on="transaccion_id")
        normales = normales.loc[normales["es_anomalia"] == 0].copy()
        normales["dia"] = normales["fecha_hora"].dt.normalize()
        self.assertLessEqual(normales.groupby(["cuenta_id", "dia"]).size().max(), 2)
        for _, grupo in normales.groupby(["cuenta_id", "dia"]):
            intervalos = grupo["fecha_hora"].sort_values().diff().dropna()
            self.assertTrue((intervalos >= pd.Timedelta(hours=4)).all())

    def test_reproducibilidad(self):
        transacciones, etiquetas = generar_dataset(semilla=42)
        pd.testing.assert_frame_equal(self.transacciones, transacciones)
        pd.testing.assert_frame_equal(self.etiquetas, etiquetas)

    def test_validador_acepta_dataset_y_rechaza_corrupciones(self):
        validar_dataset(self.transacciones, self.etiquetas)

        def dispersar_rafagas(transacciones, etiquetas):
            ids = etiquetas.loc[etiquetas["tipo_anomalia"] == "concentracion", "transaccion_id"]
            mascara = transacciones["transaccion_id"].isin(ids)
            transacciones.loc[mascara, "fecha_hora"] = pd.date_range("2026-02-02 09:00", periods=10, freq="D")
            return transacciones, etiquetas

        casos = [
            ("cantidad_transacciones", lambda t, e: (t.iloc[:-1].copy(), e)),
            ("cuentas", lambda t, e: (t.assign(cuenta_id="CTA001"), e)),
            ("id_transaccion_duplicado", lambda t, e: (t.assign(transaccion_id="TX000001"), e)),
            ("id_etiqueta_duplicado", lambda t, e: (t, e.assign(transaccion_id="TX000001"))),
            ("campo_nulo", lambda t, e: (t.assign(canal=None), e)),
            ("id_vacio", lambda t, e: (t.assign(contraparte_id=" "), e)),
            ("monto_cero", lambda t, e: (t.assign(monto=0), e)),
            ("monto_negativo", lambda t, e: (t.assign(monto=-1), e)),
            ("monto_infinito", lambda t, e: (t.assign(monto=np.inf), e)),
            ("fecha_fuera", lambda t, e: (t.assign(fecha_hora="2026-04-01"), e)),
            ("fecha_invalida", lambda t, e: (t.assign(fecha_hora="fecha-invalida"), e)),
            ("cantidad_anomalias", lambda t, e: (t, e.assign(es_anomalia=0))),
            ("etiqueta_sin_transaccion", lambda t, e: (t, e.assign(transaccion_id=e["transaccion_id"] + "X"))),
            ("etiqueta_faltante", lambda t, e: (t, e.iloc[:-1].copy())),
            ("indicador_invalido", lambda t, e: (t, e.assign(es_anomalia=2))),
            ("tipo_anomalia_invalido", lambda t, e: (t, e.assign(tipo_anomalia="desconocida"))),
            ("operacion_inconsistente", lambda t, e: (t.assign(direccion="otra"), e)),
            ("rafagas_dispersas", dispersar_rafagas),
        ]
        for nombre, corromper in casos:
            with self.subTest(caso=nombre):
                transacciones, etiquetas = corromper(self.transacciones.copy(), self.etiquetas.copy())
                with self.assertRaises(ValueError):
                    validar_dataset(transacciones, etiquetas)

    def test_exportacion_y_recarga_sin_perdida(self):
        with tempfile.TemporaryDirectory() as carpeta:
            destino = Path(carpeta)
            guardar_dataset(destino)
            transacciones = pd.read_csv(destino / "transacciones.csv", parse_dates=["fecha_hora"])
            etiquetas = pd.read_csv(destino / "etiquetas_evaluacion.csv")
            validar_dataset(transacciones, etiquetas)
            pd.testing.assert_frame_equal(self.transacciones, transacciones)
            pd.testing.assert_frame_equal(self.etiquetas, etiquetas.fillna(""))
            self.assertTrue((destino / "diccionario_datos.md").is_file())
            contenidos = {archivo.name: archivo.read_bytes() for archivo in destino.iterdir()}
            guardar_dataset(destino)
            self.assertEqual(contenidos, {
                archivo.name: archivo.read_bytes() for archivo in destino.iterdir()
            })


if __name__ == "__main__":
    unittest.main()
