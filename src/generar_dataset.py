"""Generación reproducible de movimientos ficticios y etiquetas de evaluación."""

from pathlib import Path

import numpy as np
import pandas as pd


INICIO = pd.Timestamp("2026-01-01")
FIN = pd.Timestamp("2026-04-01")  # Límite exclusivo.
COLUMNAS = [
    "transaccion_id", "cuenta_id", "fecha_hora", "tipo_movimiento",
    "direccion", "monto", "canal", "contraparte_id",
]
COLUMNAS_ETIQUETAS = ["transaccion_id", "es_anomalia", "tipo_anomalia"]
# Tipo, dirección, canal y factor sobre el monto base de cada cuenta.
OPERACIONES = [
    ("transferencia", "ingreso", "homebanking", 1.3),
    ("transferencia", "egreso", "homebanking", 1.0),
    ("pago_tarjeta", "egreso", "pos", 0.8),
    ("cobro_tarjeta", "ingreso", "pos", 0.9),
    ("pago_servicio", "egreso", "homebanking", 0.6),
    ("retiro_efectivo", "egreso", "cajero", 0.5),
    ("deposito_efectivo", "ingreso", "cajero", 1.1),
]


def _perfiles_cuentas() -> list[dict]:
    """Perfiles fijos: 600 operaciones personales y 900 comerciales."""
    cantidades = (20, 25, 30, 35, 40) * 4 + (70, 80, 90, 100, 110) * 2
    perfiles = []
    for numero, cantidad in enumerate(cantidades, start=1):
        comercial = numero > 20
        inicio = 8 + (numero % 3)
        perfiles.append({
            "cuenta_id": f"CTA{numero:03d}",
            "categoria": "comercial" if comercial else "personal",
            "cantidad": cantidad,
            "monto_base": (120_000 + 30_000 * (numero % 5)) if comercial
            else (20_000 + 5_000 * (numero % 5)),
            "hora_inicio": inicio,
            "hora_fin": inicio + (8 if comercial else 10),
        })
    return perfiles


def generar_dataset(semilla: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crea 1500 operaciones; modifica 30 existentes y valida el resultado."""
    rng = np.random.default_rng(semilla)
    perfiles = _perfiles_cuentas()
    # Reservar días completos evita mezclar movimientos normales con las ráfagas.
    cuentas_concentracion = rng.choice(
        [perfil["cuenta_id"] for perfil in perfiles[20:]], size=2, replace=False
    ).tolist()
    dias_reservados = dict(zip(cuentas_concentracion, [
        pd.Timestamp("2026-02-17"), pd.Timestamp("2026-03-12"),
    ]))
    disponibles = [p["cuenta_id"] for p in perfiles if p["cuenta_id"] not in dias_reservados]
    cuentas_anomalas = rng.choice(disponibles, size=20, replace=False).tolist()
    registros = []

    for numero, perfil in enumerate(perfiles, start=1):
        cuenta = perfil["cuenta_id"]
        comercial = perfil["categoria"] == "comercial"
        candidatos = []
        for dia in pd.date_range(INICIO, FIN, inclusive="left", freq="D"):
            if (comercial and dia.day_of_week >= 5) or dia == dias_reservados.get(cuenta):
                continue
            # Hasta dos operaciones diarias, separadas por al menos 4,5 horas.
            for hora in (perfil["hora_inicio"], perfil["hora_fin"] - 2):
                candidatos.append(dia + pd.Timedelta(
                    hours=hora, minutes=int(rng.integers(0, 90)),
                    seconds=int(rng.integers(0, 60)),
                ))
        seleccion = rng.choice(len(candidatos), size=perfil["cantidad"], replace=False)
        opciones = [0, 1, 3, 4, 6] if comercial else [0, 1, 2, 4, 5]
        probabilidades = [0.35, 0.25, 0.25, 0.10, 0.05] if comercial else [0.18, 0.22, 0.35, 0.15, 0.10]
        cantidad_contrapartes = 12 if comercial else 6
        pesos = [0.45] + [0.55 / (cantidad_contrapartes - 1)] * (cantidad_contrapartes - 1)
        for indice in seleccion:
            tipo, direccion, canal, factor = OPERACIONES[int(rng.choice(opciones, p=probabilidades))]
            variacion = float(np.clip(rng.lognormal(0, 0.2), 0.7, 1.4))
            contraparte = (numero - 1) * 12 + int(rng.choice(cantidad_contrapartes, p=pesos)) + 1
            registros.append({
                "transaccion_id": f"TX{len(registros) + 1:06d}",
                "cuenta_id": cuenta,
                "fecha_hora": candidatos[int(indice)],
                "tipo_movimiento": tipo,
                "direccion": direccion,
                "monto": round(perfil["monto_base"] * factor * variacion, 2),
                "canal": canal,
                "contraparte_id": f"CP{contraparte:04d}",
            })

    transacciones = pd.DataFrame(registros, columns=COLUMNAS)
    etiquetas = pd.DataFrame({
        "transaccion_id": transacciones["transaccion_id"],
        "es_anomalia": 0,
        "tipo_anomalia": "",
    })
    perfiles_por_cuenta = {perfil["cuenta_id"]: perfil for perfil in perfiles}
    # Las cuentas ya se eligieron al azar. Cuatro casos leves, tres medios y tres altos.
    niveles_importe = (3, 5, 10, 3, 5, 10, 3, 5, 10, 3)
    # Ventana (mañana/tarde) y lado del límite (antes/después).
    patrones_horario = ((0, -1), (0, 1), (1, -1), (1, 1))
    for posicion, cuenta in enumerate(cuentas_anomalas):
        indices = transacciones.index[transacciones["cuenta_id"] == cuenta].to_numpy()
        indice = int(rng.choice(indices))
        if posicion < 10:
            # Una sola anomalía de importe por cuenta: las demás filas son normales.
            habituales = transacciones.loc[
                (transacciones["cuenta_id"] == cuenta) & (transacciones.index != indice), "monto"
            ]
            referencia = float(habituales.median())
            transacciones.at[indice, "monto"] = round(referencia * niveles_importe[posicion], 2)
            tipo_anomalia = "importe_inusual"
        else:
            perfil = perfiles_por_cuenta[cuenta]
            ventanas = (
                (perfil["hora_inicio"] * 60, perfil["hora_inicio"] * 60 + 90),
                ((perfil["hora_fin"] - 2) * 60, (perfil["hora_fin"] - 2) * 60 + 90),
            )
            numero_horario = posicion - 10
            ventana, lado = patrones_horario[numero_horario % 4]
            cercano = (numero_horario // 4 + numero_horario) % 2 == 0
            minimo, limite = (20, 41) if cercano else (90, 121)
            desplazamiento = pd.Timedelta(
                minutes=int(rng.integers(minimo, limite)), seconds=int(rng.integers(0, 60))
            )
            borde = ventanas[ventana][0 if lado == -1 else 1]
            dia = transacciones.at[indice, "fecha_hora"].normalize()
            transacciones.at[indice, "fecha_hora"] = (
                dia + pd.Timedelta(minutes=borde) + lado * desplazamiento
            )
            tipo_anomalia = "horario_atipico"
        etiquetas.at[indice, "es_anomalia"] = 1
        etiquetas.at[indice, "tipo_anomalia"] = tipo_anomalia

    for perfil in perfiles:
        cuenta = perfil["cuenta_id"]
        if cuenta not in dias_reservados:
            continue
        indices = rng.choice(
            transacciones.index[transacciones["cuenta_id"] == cuenta].to_numpy(),
            size=5, replace=False,
        )
        comienzo = dias_reservados[cuenta] + pd.Timedelta(hours=perfil["hora_inicio"], minutes=30)
        for minuto, indice in enumerate(indices):
            transacciones.at[indice, "fecha_hora"] = comienzo + pd.Timedelta(minutes=minuto)
            etiquetas.at[indice, "es_anomalia"] = 1
            etiquetas.at[indice, "tipo_anomalia"] = "concentracion"

    transacciones = transacciones.sort_values(
        ["fecha_hora", "transaccion_id"], kind="stable"
    ).reset_index(drop=True)
    # La relación se conserva por ID, independientemente del orden de las filas.
    etiquetas = etiquetas.set_index("transaccion_id").loc[
        transacciones["transaccion_id"]
    ].reset_index()
    validar_dataset(transacciones, etiquetas)
    return transacciones, etiquetas


def validar_dataset(transacciones: pd.DataFrame, etiquetas: pd.DataFrame) -> None:
    """Rechaza datos incompletos o inconsistentes mediante ValueError."""
    if list(transacciones.columns) != COLUMNAS or list(etiquetas.columns) != COLUMNAS_ETIQUETAS:
        raise ValueError("Las columnas no coinciden con el esquema del dataset.")
    if len(transacciones) != 1500 or transacciones["cuenta_id"].nunique() != 30:
        raise ValueError("Se requieren exactamente 1500 transacciones y 30 cuentas.")
    if len(etiquetas) != 1500:
        raise ValueError("Se requiere una etiqueta por cada una de las 1500 transacciones.")
    for datos, obligatorios in (
        (transacciones, COLUMNAS), (etiquetas, ["transaccion_id", "es_anomalia"])
    ):
        if datos[obligatorios].isna().any().any():
            raise ValueError("Hay valores nulos en campos obligatorios.")
        if any(datos[columna].astype("string").str.strip().eq("").any() for columna in obligatorios):
            raise ValueError("Hay campos obligatorios vacíos.")
        if not datos["transaccion_id"].is_unique:
            raise ValueError("Los identificadores de transacción deben ser únicos en cada archivo.")
    if set(transacciones["transaccion_id"]) != set(etiquetas["transaccion_id"]):
        raise ValueError("Las etiquetas y transacciones deben corresponderse uno a uno.")
    montos = pd.to_numeric(transacciones["monto"], errors="coerce")
    if not (np.isfinite(montos) & (montos > 0)).all():
        raise ValueError("Los montos deben ser números finitos y positivos.")
    if not np.isclose(montos * 100, np.round(montos * 100), rtol=0, atol=1e-5).all():
        raise ValueError("Los montos deben tener como máximo dos decimales.")
    fechas = pd.to_datetime(transacciones["fecha_hora"], format="ISO8601", errors="coerce")
    if fechas.dt.tz is not None or not ((fechas >= INICIO) & (fechas < FIN)).all():
        raise ValueError("Las fechas deben ser horas locales válidas entre enero y marzo de 2026.")
    if not etiquetas["es_anomalia"].isin([0, 1]).all() or etiquetas["es_anomalia"].sum() != 30:
        raise ValueError("Se requieren exactamente 30 anomalías con indicador 0/1.")
    tipos = etiquetas["tipo_anomalia"].fillna("")
    if not tipos.loc[etiquetas["es_anomalia"] == 0].eq("").all():
        raise ValueError("Las operaciones normales no deben tener un tipo de anomalía.")
    if tipos.loc[etiquetas["es_anomalia"] == 1].value_counts().to_dict() != {
        "importe_inusual": 10, "horario_atipico": 10, "concentracion": 10,
    }:
        raise ValueError("Se requieren 10 anomalías de cada tipo.")
    permitidas = {operacion[:3] for operacion in OPERACIONES}
    observadas = set(transacciones[
        ["tipo_movimiento", "direccion", "canal"]
    ].itertuples(index=False, name=None))
    if not observadas <= permitidas:
        raise ValueError("Hay combinaciones de movimiento, dirección y canal inconsistentes.")
    datos = transacciones.assign(fecha_hora=fechas).merge(
        etiquetas, on="transaccion_id", validate="one_to_one"
    )
    concentradas = datos.loc[datos["tipo_anomalia"] == "concentracion"]
    if concentradas["cuenta_id"].nunique() != 2:
        raise ValueError("Las concentraciones deben ocupar dos cuentas diferentes.")
    for cuenta, grupo in concentradas.groupby("cuenta_id"):
        if (len(grupo) != 5 or grupo["fecha_hora"].nunique() != 5
                or grupo["fecha_hora"].max() - grupo["fecha_hora"].min() != pd.Timedelta(minutes=4)):
            raise ValueError("Cada ráfaga debe contener cinco movimientos distintos en cuatro minutos.")
        habituales = datos.loc[(datos["cuenta_id"] == cuenta) & (datos["es_anomalia"] == 0)]
        picos = [
            pd.Series(1, index=subconjunto["fecha_hora"]).sort_index().rolling(
                "5min", closed="both"
            ).sum().max()
            for subconjunto in (habituales, grupo)
        ]
        if picos != [1, 5]:
            raise ValueError("La ráfaga debe elevar de uno a cinco el máximo de movimientos en cinco minutos.")


def _diccionario_datos() -> str:
    texto = """# Diccionario de datos

Dataset exclusivamente ficticio para un proyecto académico. Contiene 1500
movimientos de 30 cuentas entre 2026-01-01 y 2026-03-31, con importes positivos
en pesos argentinos (ARS). No contiene datos bancarios reales ni simula saldos.

## Formato e importación futura a PostgreSQL

CSV UTF-8 con encabezado, separador coma, punto decimal y sin índice de Pandas.
Las fechas tienen formato `YYYY-MM-DD HH:MM:SS`: hora local de Argentina,
sin desplazamiento ni conversión a UTC. Los tipos siguientes son sugeridos;
esta etapa no crea tablas ni conexiones.

| Columna de transacciones.csv | Tipo PostgreSQL sugerido | Criterio |
| --- | --- | --- |
| transaccion_id | TEXT | Obligatorio y único; TX seguido de seis dígitos. |
| cuenta_id | TEXT | Obligatorio; CTA001 a CTA030, se repite entre operaciones. |
| fecha_hora | TIMESTAMP WITHOUT TIME ZONE | Obligatoria; enero-marzo de 2026. |
| tipo_movimiento | TEXT | transferencia, pago_tarjeta, cobro_tarjeta, pago_servicio, retiro_efectivo o deposito_efectivo. |
| direccion | TEXT | ingreso o egreso; el monto nunca lleva signo negativo. |
| monto | NUMERIC(16,2) | Obligatorio; ARS positivos, redondeados a centavos. |
| canal | TEXT | homebanking, pos o cajero, según el tipo de operación. |
| contraparte_id | TEXT | Obligatorio; CP seguido de cuatro dígitos, identificador externo ficticio recurrente. |

Todas las columnas del archivo principal son obligatorias. Las contrapartes son
externas a las 30 cuentas; no se generan movimientos espejo ni se comprueban
saldos. Solo transaccion_id es único: cuenta_id y contraparte_id se repiten
intencionalmente. Las filas se ordenan por fecha y luego por transaccion_id;
los identificadores se asignan antes de introducir anomalías y no las señalan.

| Columna de etiquetas_evaluacion.csv | Tipo PostgreSQL sugerido | Criterio |
| --- | --- | --- |
| transaccion_id | TEXT | Obligatorio y único; corresponde exactamente a una operación del archivo principal. |
| es_anomalia | SMALLINT | Obligatorio; 0 para 1470 operaciones normales, 1 para 30 anomalías controladas. |
| tipo_anomalia | TEXT, admite NULL | Vacío para normales; importe_inusual, horario_atipico o concentracion para anomalías. |

Hay una etiqueta por transacción, incluidas las normales. Al importar, interpretar
el campo vacío de tipo_anomalia como NULL. Las etiquetas son exclusivamente para
evaluación: no forman parte del archivo principal ni de las variables del futuro
modelo. Una anomalía sintética no implica fraude.

## Comportamiento habitual

Por defecto se usa exclusivamente `numpy.random.default_rng(42)`. Las cuentas CTA001-CTA020
son personales (600 operaciones) y CTA021-CTA030 comerciales (900 operaciones).
Cada perfil fija una frecuencia trimestral, un monto base y una franja horaria.
Los importes normales son monto_base × factor_operación × variación lognormal
(media logarítmica 0, sigma 0,2), acotada a [0,7; 1,4] y redondeada a centavos.

Las transferencias de ingreso/egreso usan homebanking (factores 1,3/1,0);
los pagos/cobros con tarjeta usan pos (0,8/0,9); los pagos de servicios usan
homebanking (0,6); retiros y depósitos de efectivo usan cajero (0,5/1,1).
Pagos y retiros son egresos; cobros y depósitos son ingresos.

Las personales eligen transferencia ingreso 18%, transferencia egreso 22%,
pago con tarjeta 35%, servicios 15% y retiro 10%. Las comerciales eligen
transferencia ingreso 35%, transferencia egreso 25%, cobro con tarjeta 25%,
servicios 10% y depósito 5%. Son probabilidades, no cuotas exactas. Cada cuenta
reutiliza seis contrapartes personales o doce comerciales; una concentra el
45% de la probabilidad y el resto se reparte entre las otras.

Las fechas normales se muestrean sin reemplazo entre dos franjas de 90 minutos
por día: desde hora_inicio y desde hora_fin menos dos horas, con minutos y
segundos aleatorios. Hay como máximo dos movimientos por día, separados por más
de cuatro horas. Las personales operan cualquier día y las comerciales de lunes
a viernes; no se modelan feriados. Se excluyen los días reservados para ráfagas.
Esto permite más actividad comercial sin generar concentraciones accidentales.

## Anomalías introducidas

Se generan primero 1500 movimientos habituales y se transforman exactamente 30,
sin agregar filas. Los tres tipos utilizan cuentas y transacciones distintas:

- 10 importes inusuales: se toma como referencia la mediana de las demás
  operaciones normales de esa cuenta, excluyendo la fila transformada. Cuatro
  casos pasan a valer 3 veces esa referencia, tres casos 5 veces y tres casos
  10 veces, redondeados a centavos. Las cuentas se eligen aleatoriamente y los
  niveles se reparten entre ellas. Se conservan fecha, tipo y canal. Usar la
  mediana en lugar del máximo permite casos moderados: algunos importes quedan
  por debajo de dos máximos habituales de su propia cuenta. La referencia agrupa
  los tipos de operación de la cuenta; no representa un umbral de fraude.
- 10 horarios atípicos: se consideran las dos ventanas habituales de 90 minutos
  de cada cuenta, no una hora universal ni la franja diurna amplia. Cinco casos
  se desplazan entre 20 y 40 minutos y cinco entre 90 y 120 minutos respecto del
  límite de una ventana, con segundos aleatorios adicionales. Hay casos antes
  de la primera ventana, después de la última y en el hueco entre ambas; todos
  quedan fuera de las dos ventanas reales. Se conservan día, monto, tipo y canal.
- 10 por concentración: cinco operaciones de cada una de otras dos cuentas
  comerciales pasan a minutos consecutivos (cinco operaciones en cuatro minutos).
  Se usan 17 de febrero y 12 de marzo, respectivamente, a hora_inicio + 30 minutos.
  Se reservan esos días sin normales para esas cuentas. Montos, tipos y canales
  permanecen habituales; los grupos se evalúan por cuenta. Se comparan ventanas
  móviles de cinco minutos, incluidos ambos extremos: el máximo normal de cada
  cuenta seleccionada es uno y el de la ráfaga es cinco. El validador rechaza
  ráfagas dispersas o cuentas sin ese contraste con su frecuencia habitual.
  Estas ráfagas siguen siendo deliberadamente claras en este dataset pequeño.

## Validaciones y reproducibilidad

El script comprueba el esquema, 1500 filas, 30 cuentas, IDs únicos, campos
obligatorios completos, montos finitos positivos con dos decimales, fechas
válidas en el trimestre, combinaciones operación/dirección/canal, 1500 etiquetas
correspondientes uno a uno y exactamente 30 anomalías (10 de cada tipo).
También comprueba dos ráfagas de cinco y el contraste de frecuencia por cuenta.
Las pruebas verifican los tres niveles de importe respecto de la mediana normal,
la variedad y distancia de horarios respecto de las ventanas reales, ausencia
de concentraciones normales, archivos recargados y repetibilidad binaria de los
tres archivos con semilla 42. Se prueban además datos alterados inválidos.

Ejecutar el script vuelve a generar y sobrescribe los tres archivos de data/.
La semilla fija reproduce los mismos archivos con el mismo código y versiones
de bibliotecas; no se garantiza igualdad binaria entre versiones distintas.

## Perfiles por cuenta

El tipo de cuenta se documenta aquí para conservar las ocho columnas solicitadas.
La franja es [hora_inicio, hora_fin); no se añaden columnas de perfil al CSV.

| Cuenta | Categoría | Operaciones totales | Monto base ARS | Franja habitual |
| --- | --- | --- | --- | --- |
"""
    for perfil in _perfiles_cuentas():
        texto += (
            f"| {perfil['cuenta_id']} | {perfil['categoria']} | {perfil['cantidad']} "
            f"| {perfil['monto_base']:.2f} | {perfil['hora_inicio']:02d}:00–"
            f"{perfil['hora_fin']:02d}:00 |\n"
        )
    return texto


def guardar_dataset(directorio: Path | None = None, semilla: int = 42) -> None:
    """Valida y exporta los dos CSV y su diccionario, sin depender del cwd."""
    destino = Path(directorio) if directorio is not None else Path(__file__).resolve().parents[1] / "data"
    transacciones, etiquetas = generar_dataset(semilla)
    destino.mkdir(parents=True, exist_ok=True)
    transacciones.to_csv(
        destino / "transacciones.csv", index=False, encoding="utf-8",
        date_format="%Y-%m-%d %H:%M:%S", float_format="%.2f", lineterminator="\n",
    )
    etiquetas.to_csv(
        destino / "etiquetas_evaluacion.csv", index=False, encoding="utf-8", lineterminator="\n"
    )
    (destino / "diccionario_datos.md").write_text(_diccionario_datos(), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    guardar_dataset()
    print("Dataset validado: 1500 transacciones, 30 cuentas y 30 anomalías (10 por tipo).")
    print("Archivos generados en data/: transacciones.csv, etiquetas_evaluacion.csv y diccionario_datos.md.")
