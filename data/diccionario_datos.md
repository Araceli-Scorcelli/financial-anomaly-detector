# Diccionario de datos

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
| CTA001 | personal | 20 | 25000.00 | 09:00–19:00 |
| CTA002 | personal | 25 | 30000.00 | 10:00–20:00 |
| CTA003 | personal | 30 | 35000.00 | 08:00–18:00 |
| CTA004 | personal | 35 | 40000.00 | 09:00–19:00 |
| CTA005 | personal | 40 | 20000.00 | 10:00–20:00 |
| CTA006 | personal | 20 | 25000.00 | 08:00–18:00 |
| CTA007 | personal | 25 | 30000.00 | 09:00–19:00 |
| CTA008 | personal | 30 | 35000.00 | 10:00–20:00 |
| CTA009 | personal | 35 | 40000.00 | 08:00–18:00 |
| CTA010 | personal | 40 | 20000.00 | 09:00–19:00 |
| CTA011 | personal | 20 | 25000.00 | 10:00–20:00 |
| CTA012 | personal | 25 | 30000.00 | 08:00–18:00 |
| CTA013 | personal | 30 | 35000.00 | 09:00–19:00 |
| CTA014 | personal | 35 | 40000.00 | 10:00–20:00 |
| CTA015 | personal | 40 | 20000.00 | 08:00–18:00 |
| CTA016 | personal | 20 | 25000.00 | 09:00–19:00 |
| CTA017 | personal | 25 | 30000.00 | 10:00–20:00 |
| CTA018 | personal | 30 | 35000.00 | 08:00–18:00 |
| CTA019 | personal | 35 | 40000.00 | 09:00–19:00 |
| CTA020 | personal | 40 | 20000.00 | 10:00–20:00 |
| CTA021 | comercial | 70 | 150000.00 | 08:00–16:00 |
| CTA022 | comercial | 80 | 180000.00 | 09:00–17:00 |
| CTA023 | comercial | 90 | 210000.00 | 10:00–18:00 |
| CTA024 | comercial | 100 | 240000.00 | 08:00–16:00 |
| CTA025 | comercial | 110 | 120000.00 | 09:00–17:00 |
| CTA026 | comercial | 70 | 150000.00 | 10:00–18:00 |
| CTA027 | comercial | 80 | 180000.00 | 08:00–16:00 |
| CTA028 | comercial | 90 | 210000.00 | 09:00–17:00 |
| CTA029 | comercial | 100 | 240000.00 | 10:00–18:00 |
| CTA030 | comercial | 110 | 120000.00 | 08:00–16:00 |
