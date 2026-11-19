"""Protocolo de abril fijado antes de medir: datos nuevos en memoria, sin CSV."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

if __package__:
    from .preparar_datos import COLUMNAS_ORIGINALES, Referencias, validar_y_convertir
else:
    from preparar_datos import COLUMNAS_ORIGINALES, Referencias, validar_y_convertir


SEMILLA_ESCENARIOS = 2026
# Segundos desde el comienzo; incluye casos que nunca alcanzan tres en 5 min.
PATRONES = {
    "densa": (0, 60, 120, 180, 240),
    "borde": (0, 120, 300),
    "corta": (0, 120),
    "espaciada": (0, 180, 360, 540, 720),
}


@dataclass
class EscenariosRafagas:
    transacciones: pd.DataFrame
    etiquetas: pd.DataFrame
    episodios: pd.DataFrame


def generar_escenarios(
    entrenamiento: pd.DataFrame, referencias: Referencias, semilla: int = SEMILLA_ESCENARIOS,
) -> EscenariosRafagas:
    """180 operaciones: 120 aisladas, 30 anómalas y 30 legítimas en episodios.

    Hora modal y mediana de importe proceden solo del pasado de entrenamiento.
    Ambos tipos de episodio usan las mismas distribuciones y patrones. Su verdad
    es semántica y predefinida, no se calcula con la regla de tres movimientos.
    Cada episodio tiene una cuenta distinta y un subgenerador independiente.
    """
    if isinstance(semilla, bool) or not isinstance(semilla, (int, np.integer)) or semilla < 0:
        raise ValueError("La semilla debe ser un entero no negativo.")
    datos = validar_y_convertir(entrenamiento)
    cuentas = sorted(datos.cuenta_id.unique())
    personales = [f"CTA{i:03d}" for i in range(1, 21)]
    comerciales = [f"CTA{i:03d}" for i in range(21, 31)]
    if set(cuentas) != set(personales + comerciales) or datos.fecha_hora.max() >= pd.Timestamp("2026-04-01"):
        raise ValueError("El protocolo requiere las 30 cuentas del catálogo y entrenamiento anterior a abril de 2026.")
    horas = {cuenta: int(grupo.fecha_hora.dt.hour.mode().iloc[0]) for cuenta, grupo in datos.groupby("cuenta_id")}
    perfiles = {c: "personal" if c in personales else "comercial" for c in cuentas}
    semillas = np.random.SeedSequence(int(semilla)).spawn(18)
    asignacion, normales = np.random.default_rng(semillas[0]), np.random.default_rng(semillas[1])
    seleccion = {"personal": list(asignacion.permutation(personales)[:8]),
                 "comercial": list(asignacion.permutation(comerciales)[:8])}
    filas, etiquetas, episodios = [], [], []

    def agregar(cuenta, fecha, rng, episodio_id="", escenario="normal_aislada", patron="aislada"):
        identificador = f"EXP{len(filas) + 1:06d}"
        mediana = referencias.medianas_por_cuenta.get(cuenta, referencias.mediana_global)
        monto = max(.01, round(float(mediana * rng.lognormal(0, .2)), 2))
        filas.append({"transaccion_id": identificador, "cuenta_id": cuenta, "fecha_hora": fecha,
                      "tipo_movimiento": "transferencia", "direccion": "egreso", "monto": monto,
                      "canal": "homebanking", "contraparte_id": f"CP_EXP{int(rng.integers(1, 100)):03d}"})
        anomala = escenario == "concentracion_anomala"
        etiquetas.append({"transaccion_id": identificador, "es_anomalia": int(anomala),
                          "tipo_anomalia": "concentracion" if anomala else ""})
        episodios.append({"transaccion_id": identificador, "episodio_id": episodio_id,
                          "escenario": escenario, "patron": patron, "perfil": perfiles[cuenta]})

    for dia in ("2026-04-01", "2026-04-02", "2026-04-29", "2026-04-30"):
        for cuenta in cuentas:
            fecha = (pd.Timestamp(dia) + pd.Timedelta(hours=horas[cuenta], minutes=int(normales.integers(0, 50)),
                                                       seconds=int(normales.integers(0, 60))))
            agregar(cuenta, fecha, normales)
    dias = pd.bdate_range("2026-04-06", periods=16)
    numero = 0
    for patron, desplazamientos in PATRONES.items():
        for escenario in ("concentracion_anomala", "concentracion_legitima"):
            for perfil in ("personal", "comercial"):
                cuenta = seleccion[perfil].pop(0)
                rng = np.random.default_rng(semillas[numero + 2])
                inicio = dias[numero] + pd.Timedelta(hours=horas[cuenta], minutes=int(rng.integers(10, 40)))
                numero += 1
                for segundos in desplazamientos:
                    agregar(cuenta, inicio + pd.Timedelta(seconds=segundos), rng, f"E{numero:02d}", escenario, patron)

    transacciones = validar_y_convertir(pd.DataFrame(filas, columns=COLUMNAS_ORIGINALES)).sort_values(
        ["fecha_hora", "transaccion_id"], kind="stable",
    ).reset_index(drop=True)
    orden = transacciones.transaccion_id
    etiquetas = pd.DataFrame(etiquetas).set_index("transaccion_id").loc[orden].reset_index()
    episodios = pd.DataFrame(episodios).set_index("transaccion_id").loc[orden].reset_index()
    return EscenariosRafagas(transacciones, etiquetas, episodios)
