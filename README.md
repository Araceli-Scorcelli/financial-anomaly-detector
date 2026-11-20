# Financial Anomaly Detector

Proyecto académico en Python para explorar la detección de anomalías en
transacciones financieras. El alcance previsto es pequeño y realizable en
2 o 3 semanas: trabajar con un único conjunto de datos CSV, un modelo de
detección de anomalías y una visualización sencilla.

## Estado actual

La primera etapa genera un dataset bancario sintético con Pandas y NumPy,
reproducible con semilla 42: 1500 movimientos de 30 cuentas entre enero y marzo
de 2026. Incluye 20 cuentas personales (600 movimientos) y 10 comerciales
(900 movimientos), con frecuencias, montos y horarios habituales por cuenta.

Se transforman 30 registros existentes: 10 con importes de 3, 5 y 10 veces la
mediana de las operaciones normales restantes de su cuenta (cuatro, tres y tres
casos, respectivamente). Otros 10 cambian a horarios fuera de las dos ventanas
habituales de 90 minutos de su cuenta: cinco quedan cerca, a 20–40 minutos más
segundos del límite más cercano, y cinco lejos, a 90–120 minutos más segundos.
Se distribuyen antes y después de las ventanas y en el hueco entre ellas.
Los últimos 10 se concentran en dos grupos de cinco movimientos durante cuatro
minutos, dentro de una ventana habitual. Estos grupos usan dos cuentas comerciales
distintas de las usadas para las otras anomalías y se comparan por cuenta con un
máximo normal de un movimiento en cualquier ventana móvil de cinco minutos.
Una anomalía representa un comportamiento inusual; no implica necesariamente fraude.

La segunda etapa incorpora los archivos SQL para crear `public.transacciones`
y explorar los datos. En el entorno local ya se crearon la base
`financial_anomalies` y la tabla, y se importaron los 1500 movimientos de 30 cuentas.
La tercera etapa agrega la lectura desde Python con SQLAlchemy, Psycopg 3 y
Pandas, usando una conexión de solo lectura y configuración local en `.env`.
La conexión real desde Python ya se verificó: `SELECT 1` devolvió `1` y se
recuperaron 1500 movimientos de 30 cuentas.

La etapa 4A prepara esos movimientos y agrega tres características numéricas:
`monto_relativo`, `rareza_horaria` y `movimientos_5min`. Separa el ajuste de
referencias de su aplicación a nuevas transacciones, sin utilizar etiquetas.

La etapa 4B incorpora Isolation Forest y una evaluación sobre transacciones
posteriores al entrenamiento. Las etiquetas se consultan después de generar las
predicciones, únicamente para medir resultados.

La etapa 5 agrega una aplicación local de Streamlit para explorar las predicciones
y revisar las métricas históricas. Reutiliza las etapas anteriores y admite la
lectura de PostgreSQL o un modo demostración CSV, sin modificar datos. Es una demostración académica con datos
sintéticos; una alerta no confirma fraude. No se conecta con bancos reales.

La etapa 6 agrega una evaluación temporal complementaria con tres ventanas de
entrenamiento creciente y bloques posteriores de evaluación sin solapamientos.
Conserva el detector, sus parámetros y la evaluación 80/20 de las etapas anteriores.
Se ejecuta desde consola y no cambia la aplicación Streamlit.

La etapa 6B analiza los errores de esas predicciones: compara falsas alertas con
normales correctamente clasificadas, distingue conteos de tasas por cuenta y
ventana, y reconstruye las ráfagas etiquetadas para revisar omisiones y demoras.
Es un diagnóstico posterior; no ajusta el detector a partir de los resultados.

La etapa 6C compara, en un experimento separado, Isolation Forest con la unión
de sus alertas y una regla fija de concentración: `movimientos_5min >= 3`.
Incluye una comparación histórica exploratoria y escenarios nuevos de abril,
sin incorporar el enfoque híbrido al modelo principal ni a Streamlit.

## Estructura

```text
financial-anomaly-detector/
├── .streamlit/
│   └── config.toml                # Tema visual de la aplicación local.
├── data/
│   ├── transacciones.csv          # Dataset principal sin etiquetas.
│   ├── etiquetas_evaluacion.csv   # Etiquetas separadas para evaluación.
│   └── diccionario_datos.md       # Campos, formatos y reglas.
├── src/
│   ├── generar_dataset.py         # Generador reproducible.
│   ├── database.py                # Configuración y conexión de solo lectura.
│   ├── fuentes_datos.py           # Fuente explícita y lectura validada del CSV.
│   ├── verificar_conexion.py      # Verificación manual y lectura con Pandas.
│   ├── preparar_datos.py          # Validación, referencias y características.
│   ├── entrenar_modelo.py         # Isolation Forest y evaluación cronológica.
│   ├── evaluacion_temporal.py     # Walk-forward, métricas y cobertura temporal.
│   ├── analisis_errores.py        # Diagnóstico post hoc de errores y ráfagas.
│   ├── generar_escenarios_rafagas.py # Escenarios nuevos solo en memoria.
│   └── experimento_rafagas.py     # Comparación experimental IF y regla.
├── sql/
│   ├── crear_tablas.sql           # Estructura e indicaciones de importación.
│   └── consultas_exploratorias.sql # Consultas de lectura en pgAdmin.
├── tests/
│   ├── test_generar_dataset.py    # Validaciones del dataset con unittest.
│   ├── test_database.py           # Pruebas de conexión sin una base real.
│   ├── test_fuentes_datos.py      # Configuración y validación del modo CSV.
│   ├── test_preparar_datos.py     # Características con datos ficticios, sin DB.
│   ├── test_entrenar_modelo.py    # Separación, modelo y métricas sin DB.
│   ├── test_app.py               # AppTest con datos ficticios y CSV real, sin DB.
│   ├── test_evaluacion_temporal.py # Ventanas, causalidad y métricas sin DB.
│   ├── test_analisis_errores.py   # Tasas, errores, ráfagas y diagnóstico sin DB.
│   └── test_experimento_rafagas.py # Regla, escenarios y modelo fijo sin DB.
├── app.py              # Dashboard local de Streamlit.
├── .env.example        # Plantilla versionada, sin credenciales reales.
├── README.md           # Descripción, instalación y alcance.
├── requirements.txt    # Dependencias de Python.
└── .gitignore          # Archivos excluidos de Git.
```

`transacciones.csv` contiene ocho columnas: `transaccion_id`, `cuenta_id`,
`fecha_hora`, `tipo_movimiento`, `direccion`, `monto`, `canal` y `contraparte_id`.
`etiquetas_evaluacion.csv` contiene 1500 filas con `transaccion_id`, `es_anomalia`
(0 o 1) y `tipo_anomalia` (vacío para movimientos normales). Las etiquetas se
usan únicamente para evaluar resultados; no deben usarse como entrada del modelo.

## Preparación del entorno

Requiere Python 3 y pip. Abrir la carpeta raíz del proyecto en VS Code y abrir
una terminal PowerShell. Si todavía no existe el entorno virtual, crearlo:

```powershell
python -m venv .venv
```

En la paleta de comandos de VS Code, elegir **Python: Select Interpreter** y
seleccionar `.venv\Scripts\python.exe`. Instalar las dependencias con ese mismo
intérprete, desde la raíz:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Pandas y NumPy se usan para generar los datos. SQLAlchemy, `psycopg[binary]`
y `python-dotenv` permiten leer PostgreSQL desde Python. Scikit-learn se usa
para Isolation Forest y sus métricas. Streamlit presenta el dashboard local.

## Lectura de PostgreSQL desde Python

Esta etapa usa PostgreSQL 18 local, con el servicio encendido, la base
`financial_anomalies` y la tabla `public.transacciones` ya creadas e importadas.
No hace falta volver a crear la tabla ni importar el CSV para verificar la conexión.
Los archivos de `sql/` conservan las instrucciones de las etapas anteriores.

Usar un usuario existente que tenga permiso `CONNECT` sobre la base, `USAGE`
sobre el esquema `public` y `SELECT` sobre `public.transacciones`. Se recomienda
un usuario sin privilegios de superusuario. El código no crea roles ni modifica
permisos.

Para reforzar la protección, usar un rol que solo tenga permisos de lectura:
el modo `READ ONLY` del cliente no reemplaza los permisos del servidor.

1. Desde la raíz, copiar la plantilla únicamente si todavía no existe `.env`:

   ```powershell
   if (-not (Test-Path -LiteralPath .env)) {
       Copy-Item -LiteralPath .env.example -Destination .env
   }
   ```

2. Abrir `.env` en VS Code y completar las cinco variables con la configuración
   local. La plantilla contiene:

   ```dotenv
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=financial_anomalies
   DB_USER=
   DB_PASSWORD=
   ```

   Escribir el usuario y la contraseña reales solamente en `.env`. Si un valor
   contiene `#` o espacios, encerrarlo entre comillas simples para conservarlo.
   La contraseña se pasa tal como es, sin codificar sus caracteres para una URL:
   la conexión se construye con [SQLAlchemy `URL.create`](https://docs.sqlalchemy.org/en/20/core/engines.html#creating-urls-programmatically).
   La lectura del archivo usa `interpolate=False`, de modo que `${...}` se
   conserva literalmente. Las variables de entorno que ya existan en Windows o
   en la terminal tienen prioridad sobre `.env` (`override=False`). Si un cambio
   en el archivo no se aplica, revisar esa configuración previa.

3. Comprobar que Git excluya el archivo local:

   ```powershell
   git check-ignore -v .env
   ```

   La salida debe mostrar la regla que excluye `.env`. `.env.example` se versiona
   como plantilla; el archivo `.env` y sus credenciales no deben incorporarse a Git.

4. Ejecutar la verificación desde la raíz con el intérprete seleccionado:

   ```powershell
   .\.venv\Scripts\python.exe -m src.verificar_conexion
   ```

   También se admite la ejecución directa:

   ```powershell
   .\.venv\Scripts\python.exe src\verificar_conexion.py
   ```

El verificador comprueba que `SELECT 1` devuelva `1` y lee las ocho columnas de
`public.transacciones`, ordenadas por `fecha_hora` y `transaccion_id`. Muestra la
cantidad de filas cargadas, las cuentas únicas, los tipos de datos y las primeras
cinco filas del DataFrame. La verificación real confirmó **1500 movimientos
y 30 cuentas**; los valores mostrados se calculan a partir de los datos leídos.

El importe `monto` conserva los valores `Decimal` de PostgreSQL mediante
[`pandas.read_sql_query(..., coerce_float=False)`](https://pandas.pydata.org/docs/reference/api/pandas.read_sql_query.html),
por lo que puede aparecer con tipo `object` en el DataFrame. Esto evita convertir
automáticamente los importes exactos a números de punto flotante.

`src/database.py` expone `crear_engine(ruta_env=None)` y
`conectar_solo_lectura(engine)`. Importar el módulo no carga `.env` ni abre una
conexión. Al crear el motor se busca `.env` en la raíz del proyecto, se validan
los datos y que `DB_PORT` sea un entero entre 1 y 65535, y se configura un tiempo
de espera de conexión de cinco segundos. La conexión usa opciones de sesión y
[transacciones de solo lectura de SQLAlchemy](https://docs.sqlalchemy.org/en/20/dialects/postgresql.html#setting-read-only-deferrable);
antes de consultar los datos se verifica que `SHOW transaction_read_only` devuelva
`on`. Al finalizar se cierran la conexión y el motor.

Una ejecución correcta termina con código de salida `0`; un error termina con
`1` y un mensaje sin mostrar la URL ni las credenciales. Ante un error, revisar
que PostgreSQL esté encendido, las cinco variables locales, el puerto y los
permisos del usuario sobre la base, el esquema y la tabla.

## Etapa 4A: preparación y características

La ingeniería de características transforma datos originales en medidas útiles
para un modelo. Estas tres variables representan un importe inusual, un horario
poco frecuente y una concentración de operaciones de la misma cuenta. Un valor
alto indica mayor intensidad de ese comportamiento; por sí solo no demuestra
fraude ni determina que una transacción sea anómala.

`validar_y_convertir(df)` devuelve una copia de las ocho columnas originales,
convierte `monto` de `Decimal` a `float64` y comprueba que `fecha_hora` sea
datetime. Rechaza identificadores duplicados, datos faltantes e importes no
positivos o no finitos. No usa el índice como identificador ni elimina filas.
La conversión a punto flotante permite el procesamiento numérico, aunque pierde
la representación decimal exacta usada para almacenar dinero.

El procesamiento distingue dos pasos: `ajustar_referencias(referencia)` estima
las medianas y distribuciones horarias y devuelve un objeto `Referencias`;
`transformar_datos(transacciones, referencias, historial=None)` aplica esos
valores fijos, sin volver a ajustarlos. Devuelve las ocho columnas originales y
las tres nuevas, conservando el orden, los identificadores y el índice de entrada,
incluso si ese índice tiene valores repetidos.

- **`monto_relativo`**: divide el importe por la mediana de su cuenta en la
  referencia. Un resultado de `1` equivale a esa mediana; `3` significa tres
  veces ese importe. La mediana reduce la influencia de extremos frente al
  promedio, pero las anomalías de la referencia también pueden afectarla.
  Cuentas con menos de cinco observaciones y cuentas nuevas usan la mediana
  global de la referencia. El mínimo se configura con `min_observaciones` al
  ajustar. Una referencia vacía se rechaza.
- **`rareza_horaria`**: agrupa las horas en 24 intervalos (`0` a `23`). Para cada
  cuenta, si `c_h` es la cantidad de operaciones en la hora `h`, suaviza los
  conteos con `s_h = 0.25*c_(h-1) + 0.5*c_h + 0.25*c_(h+1)`. Las horas son
  circulares: `23` y `0` son vecinas. Luego aplica
  `p_h = (s_h + 1) / (n + 24)`, donde `n` es la cantidad de operaciones de
  referencia de esa cuenta, y calcula `rareza_horaria = -ln(p_h)`. Las
  probabilidades suman `1` y nunca son cero. Una hora más frecuente tiene menor
  rareza; una menos frecuente tiene mayor rareza. Cuentas con pocas observaciones
  o nuevas usan la distribución global calculada del mismo modo.
- **`movimientos_5min`**: cuenta operaciones de la misma cuenta en el intervalo
  inclusivo `[t - 5 minutos, t]`, incluida la actual. Ordena por cuenta, fecha
  y `transaccion_id` en orden lexicográfico. Para operaciones simultáneas cuenta
  la actual y los identificadores anteriores, sin anticipar los posteriores.
  `calcular_movimientos_5min(transacciones, historial=None)` también permite
  obtener este conteo por separado; su resultado es `int64`.

El suavizado horario reparte información entre horas vecinas y agrega una
observación por hora (Laplace), evitando asignar probabilidad cero a horarios
sin antecedentes. Sus 24 observaciones adicionales tienen bastante peso con
cuentas pequeñas. El cálculo ignora los minutos dentro de cada hora, y el
respaldo global mezcla perfiles personales y comerciales. Las distintas
cantidades de observaciones por cuenta también afectan la estimación; estas
probabilidades describen horarios, **no son probabilidades de fraude**.

El historial para los cinco minutos se pasa explícitamente: ajustar referencias
no lo guarda ni lo incorpora automáticamente al conteo. Para cada objetivo solo
se consideran movimientos anteriores según el orden indicado; registros futuros
del historial no influyen en objetivos pasados. Un ID presente tanto en historial
como en objetivos se cuenta una sola vez si sus datos originales coinciden; si
hay diferencias, se rechaza para evitar resultados ambiguos. Para procesar lotes
nuevos, pasar el historial observado junto con las transacciones del lote. Las
referencias permanecen fijas hasta un nuevo ajuste explícito.

Antes de evaluar un modelo hay que separar cronológicamente las transacciones:
ajustar referencias con el período anterior y transformar el período futuro,
aportando el historial disponible para los conteos. Por ejemplo, con esos dos
DataFrames ya separados:

```python
from src.preparar_datos import ajustar_referencias, transformar_datos

referencias = ajustar_referencias(transacciones_anteriores)
preparadas = transformar_datos(
    transacciones_nuevas, referencias, historial=transacciones_anteriores
)
```

Para ejecutar la demostración con PostgreSQL desde la raíz:

```powershell
.\.venv\Scripts\python.exe -m src.preparar_datos
```

También se admite `.\.venv\Scripts\python.exe src\preparar_datos.py`. Reutiliza
`crear_engine` y `cargar_transacciones` de los módulos existentes y la misma
configuración local de solo lectura. Muestra filas procesadas, cuentas distintas,
las primeras cinco filas, tipos y estadísticas descriptivas de las tres variables,
y comprueba la ausencia de nulos e infinitos. El DataFrame resultante tiene once
columnas. No guarda CSV ni modifica PostgreSQL.

La demostración ajusta y transforma los mismos 1500 movimientos únicamente para
explorar las características: **no constituye una evaluación válida**. No lee
etiquetas ni usa los horarios o reglas del generador como patrones predefinidos.
Este comando no entrena un modelo ni aplica umbrales; la etapa 4B lo hace por separado.

Las pruebas de `tests/test_preparar_datos.py` usan pequeños ejemplos ficticios sin
PostgreSQL ni etiquetas reales. Cubren validación, medianas, distribución horaria,
medianoche, cuentas con pocos datos, conteos causales, empates, historial y la
separación entre ajuste y transformación. Ejecutar la suite desde la raíz:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Etapa 4B: Isolation Forest y evaluación

Isolation Forest separa observaciones mediante cortes aleatorios en sus variables.
Una operación que se aísla con pocos cortes resulta más inusual para el modelo.
Se entrena sin etiquetas usando solo `monto_relativo`, `rareza_horaria` y
`movimientos_5min`, reutilizadas de 4A, sin escalador adicional. Los identificadores,
las fechas y los demás campos originales se conservan para revisar resultados,
pero no son entradas del modelo.

`separar_cronologicamente(df, fraccion_entrenamiento=0.8)` ordena por
`fecha_hora` y `transaccion_id`: usa el primer 80 % para entrenamiento y el resto
para evaluación. Si el corte divide operaciones con la misma fecha y hora, todo
ese grupo pasa a evaluación; por eso la proporción es aproximada. Rechaza cortes
que dejan un conjunto vacío. No distribuye las anomalías entre conjuntos usando
sus etiquetas.

`entrenar_y_predecir(transacciones)` realiza la separación, ajusta las referencias
exclusivamente con entrenamiento y transforma ambos conjuntos con esas referencias
fijas. Para los cinco minutos, entrenamiento usa su propio lote; evaluación agrega
el historial de entrenamiento y las operaciones previas de su mismo lote,
respetando el orden causal de 4A. Ninguna operación futura entra en un conteo.
Las anomalías que pueda contener entrenamiento no se eliminan con etiquetas.

La configuración se fija antes de consultar las etiquetas:

```python
IsolationForest(
    n_estimators=200, max_samples="auto", contamination="auto",
    max_features=1.0, bootstrap=False, random_state=42, n_jobs=1,
)
```

Son 200 árboles; cada uno usa hasta `min(256, n_entrenamiento)` observaciones y
las tres variables. La semilla permite reproducir el entrenamiento. No se ajustan
parámetros mirando los resultados de evaluación.

El puntaje es `puntaje_anomalia = -modelo.score_samples(caracteristicas)`:
cuanto mayor, más inusual considera el modelo una operación. Con
`contamination="auto"`, el criterio es `prediccion_anomalia = 1` si el puntaje
es estrictamente mayor que `0.5`; en igualdad, es `0`. El puntaje no es una
probabilidad de fraude y la configuración no obliga a detectar un 2 % de casos.
El umbral determina las alertas; no usa el porcentaje conocido de anomalías
sintéticas. El signo del puntaje y el umbral se corresponden con la
[documentación de IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

La función devuelve `ResultadoEntrenamiento`, que contiene el modelo, las
referencias, el entrenamiento preparado con once columnas y las predicciones
del período de evaluación con esas once columnas más `puntaje_anomalia` y
`prediccion_anomalia`. No guarda el modelo ni los resultados en archivos.

Para ejecutar desde la raíz, con PostgreSQL disponible y la configuración local
de la etapa 3:

```powershell
.\.venv\Scripts\python.exe -m src.entrenar_modelo
```

También se admite `.\.venv\Scripts\python.exe src\entrenar_modelo.py`. El comando
reutiliza la lectura protegida de PostgreSQL y cierra sus recursos antes de
entrenar. Genera todas las predicciones y recién entonces lee
`data/etiquetas_evaluacion.csv` con `cargar_etiquetas`. Valida identificadores
únicos y su correspondencia con todos los movimientos recuperados, y vincula
las etiquetas con las predicciones uno a uno mediante `transaccion_id`.
Las métricas se calculan únicamente sobre evaluación; las etiquetas no ajustan
referencias, modelo ni umbral. No modifica PostgreSQL, los CSV ni el generador.

El informe muestra cantidades y períodos de ambos conjuntos, las cuentas distintas
recuperadas, parámetros y umbral, anomalías reales y alertas, y los diez puntajes más altos del período de
evaluación con sus identificadores, fechas, cuentas, características y predicción.
Para interpretar sus métricas:

- **Verdaderos positivos (TP)**: anomalías reales que recibieron alerta.
- **Falsos positivos (FP)**: movimientos normales que recibieron alerta y
  requerirían una revisión innecesaria.
- **Falsos negativos (FN)**: anomalías reales que el modelo omitió.
- **Precisión**, `TP / (TP + FP)`: qué proporción de las alertas era anómala.
- **Recall**, `TP / (TP + FN)`: qué proporción de las anomalías fue detectada.
- **F1**: media armónica de precisión y recall; resume ambas medidas.

La matriz de confusión tiene filas reales `[normal, anomalía]` y columnas
predichas `[normal, anomalía]`, es decir, `[[TN, FP], [FN, TP]]`; `TN` son los
movimientos normales correctamente descartados. Si una métrica global tiene
denominador cero, se informa `0` (`zero_division=0`) y debe leerse junto con las
cantidades de casos.

Para `importe_inusual`, `horario_atipico` y `concentracion` también se muestran
casos reales, detectados, omitidos y recall. El modelo solo decide normal o
anómalo: este desglose compara esa decisión con cada tipo real, no clasifica el
tipo de anomalía. Si un tipo no aparece en evaluación, su recall se informa
como **N/D**: la ausencia de ejemplos no demuestra éxito ni fracaso.

Los resultados describen un único corte temporal de datos sintéticos. Puede haber
muy pocos casos por tipo o tipos ausentes, y las anomalías presentes en
entrenamiento pueden influir en las referencias y en el detector. Un buen
resultado aquí no garantiza el mismo rendimiento en transacciones reales.
Cambiar características, parámetros o umbral a partir de este informe requerirá
otro conjunto de validación independiente; no debe usarse repetidamente este
período como si siguiera siendo una prueba nueva.

La primera ejecución real, verificada el 8 de noviembre de 2026 con Scikit-learn
1.9.1, recuperó 1500 movimientos de 30 cuentas: 1200 para entrenamiento, hasta
el 13 de marzo a las 09:25:21, y 300 para evaluación, desde el 13 de marzo a las
09:30:51 hasta el 31 de marzo de 2026. Produjo 28 alertas: 3 verdaderos positivos
y 25 falsos positivos, sin falsos negativos. La precisión fue 0.107143, el recall
1.000000 y F1 0.193548; la matriz fue `[[272, 25], [0, 3]]`.

En evaluación quedaron dos anomalías de importe y una de horario, todas
detectadas, y ninguna de concentración. Por tanto, este corte no permite medir
la detección de concentraciones. El recall del 100 % se apoya en solo tres casos
y convive con baja precisión: 25 de las 28 alertas fueron falsas. Estos resultados
no se utilizaron para cambiar los parámetros ni el umbral. Son un registro de
esa ejecución; el comando calcula las métricas con los datos disponibles.

Las pruebas de `tests/test_entrenar_modelo.py` usan ejemplos ficticios sin
PostgreSQL ni etiquetas reales. Para ejecutar toda la suite tratando las
advertencias como errores:

```powershell
.\.venv\Scripts\python.exe -W error -m unittest discover -s tests -v
```

## Etapa 5: aplicación Streamlit con PostgreSQL o CSV

Ejecutar desde la raíz del proyecto, en PowerShell. El diseño y el pipeline son
los mismos en ambos modos; `DATA_SOURCE` selecciona explícitamente la fuente.

**Modo demostración CSV**, sin servidor PostgreSQL, credenciales ni archivo `.env`:

```powershell
$env:DATA_SOURCE = "csv"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

**Modo PostgreSQL**, con el servicio activo y la configuración local de la etapa 3:

```powershell
$env:DATA_SOURCE = "postgres"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

PostgreSQL sigue siendo el valor predeterminado. La variable del proceso tiene
prioridad sobre `DATA_SOURCE` en el `.env` opcional de la raíz. Si no se define en
ninguno, se utiliza `postgres`. Solo se aceptan los valores exactos `postgres` y
`csv`; un valor inválido o un `.env` con errores de sintaxis muestra un error sin
reproducir su contenido.
Para volver al valor del archivo o al predeterminado en esta terminal:

```powershell
Remove-Item Env:DATA_SOURCE -ErrorAction SilentlyContinue
```

Detener y volver a iniciar Streamlit para cambiar la configuración. Seleccionar
`csv` en PowerShell omite por completo la lectura de `.env` y la creación de
motores/conexiones a PostgreSQL. No existe un cambio automático de fuente ante
errores: PostgreSQL indisponible continúa mostrando su error de conexión.
`DATA_SOURCE` controla el dashboard; los comandos de las etapas anteriores
conservan su funcionamiento original con PostgreSQL.

Abrir [http://localhost:8501](http://localhost:8501) en el navegador. Para detener
el servidor, presionar **Ctrl+C** en la terminal.

El dashboard **Financial Anomaly Detector** muestra un resumen de los movimientos
recuperados, cuentas y tamaños de entrenamiento y evaluación. Todas las cifras
se calculan con los datos cargados. Sus seis indicadores globales corresponden
al conjunto completo de evaluación: transacciones, alertas, anomalías reales,
precisión, recall y F1. Se distribuyen en una grilla de hasta tres columnas que
se adapta al ancho disponible, manteniendo visibles los títulos y porcentajes.
También advierte sobre la cantidad de casos reales, los
falsos positivos y los tipos sin ejemplos de evaluación; el significado y las
limitaciones de estas métricas son los de la etapa 4B.

Los filtros permiten elegir cuenta, fechas y resultado: **Todas**, **Alertas** o
**Normales**. Las fechas incluyen los días seleccionados completos; «Normales»
significa sin alerta del modelo. La vista muestra sus propios contadores, una
tabla de transacciones de evaluación con identificadores, estado destacado,
características y puntaje, y un gráfico de barras de alertas frente a normales.
La tabla prioriza transacción, cuenta, fecha y hora, importe en ARS, resultado y
puntaje. El interruptor **Mostrar características técnicas** agrega las tres
variables al final, sin cambiar filas, predicciones ni métricas. Los importes
muestran separadores de miles y dos decimales; los puntajes, cuatro decimales.
La sección **Cómo leer las características** explica su significado y aclara que
describen los movimientos, sin ofrecer una explicación causal de cada alerta.
Los filtros afectan esa tabla, sus contadores y el gráfico; los seis indicadores
globales permanecen referidos a toda la evaluación. Las etiquetas reales solo
intervienen en las métricas históricas: no se usan para filtrar, entrenar ni
mostrar la tabla operativa.

Al cargar por primera vez, `cargar_resultados(fuente)` consulta PostgreSQL en modo
de solo lectura y cierra sus recursos, o lee `data/transacciones.csv` con Pandas.
El lector CSV reutiliza `validar_y_convertir`: exige las ocho columnas originales,
identificadores únicos y no vacíos, campos obligatorios completos, importes
positivos y finitos y fechas válidas sin zona horaria. Conserva los identificadores
como texto y convierte los importes y fechas con las validaciones compartidas.
Las rutas del CSV, las etiquetas y el `.env` opcional se basan en la ubicación del
proyecto, no en el directorio actual de la terminal. Si se inicia desde otra
carpeta, pasar las rutas completas del intérprete y de `app.py` al comando.

En ambos modos se ejecutan las mismas funciones `pipeline.entrenar_y_predecir()`
y `pipeline.evaluar_predicciones()`. Se conservan la separación cronológica 80/20,
las referencias ajustadas con entrenamiento, las tres características, el cálculo
causal de frecuencia, los 200 árboles, `contamination="auto"`, `random_state=42`
y el umbral original. Solo después de generar las predicciones se leen las
etiquetas separadas para evaluar; nunca se incorporan al entrenamiento.
La fuente aparece en el resumen y en la indicación existente de la barra lateral.

La caché distingue la fuente mediante un argumento explícito y conserva una
única entrada con resultados en memoria
mediante `st.cache_data(max_entries=1)`, sin vencimiento automático. La caché es
compartida por las sesiones del mismo servidor y cada ejecución recibe una copia;
no conserva el modelo ni conexiones a la base y no persiste resultados en disco.
Cambiar filtros reutiliza esos resultados, sin reajustar referencias ni reentrenar.

El botón **Actualizar resultados** invalida esa caché, vuelve a consultar los
datos y repite el entrenamiento y la evaluación. Usarlo cuando cambien los datos
de la fuente seleccionada o las etiquetas. Reiniciar el servidor también borra la caché;
si cambia `.env`, reiniciar el proceso para aplicar la configuración de conexión
de la etapa 3.

La aplicación es local y no publica el proyecto ni modifica PostgreSQL o los
CSV. Las pruebas de `tests/test_app.py` usan `AppTest` de Streamlit con datos
ficticios y el CSV real, bloqueando los accesos a PostgreSQL. Cubren las métricas
calculadas, separación de etiquetas, caché por fuente, filtros, actualización,
ausencia de respaldo automático y errores sin detalles sensibles. Las de
`tests/test_fuentes_datos.py` cubren configuración y validación de la lectura CSV.
Para ejecutar toda la suite secuencialmente:

```powershell
.\.venv\Scripts\python.exe -W error -m unittest discover -s tests -v
```

La verificación local tras la mejora visual aprobó **77 pruebas**: las 67 anteriores y
10 de la aplicación. También se comprobó la interfaz con PostgreSQL real: mostró
300 transacciones de evaluación y, al filtrar, conservó los puntajes y las seis
métricas globales con una sola lectura y un solo entrenamiento. El servidor
local inició correctamente y respondió con HTTP 200; se detuvo al terminar la
prueba. Estas comprobaciones no requieren abrir otro navegador durante la suite.

### Verificación del modo demostración CSV

Pasaron **133 pruebas** con `-W error`: las 116 existentes y 17 adicionales.
Se ejecutaron secuencialmente. La prueba del CSV real conserva las variables de
Windows necesarias para Streamlit, elimina las cinco variables `DB_*`, bloquea
la lectura de `.env` y los accesos a PostgreSQL, y comprueba las métricas, los
300 movimientos y la caché al filtrar. También se leyó el CSV desde otro directorio.
Los casos de configuración y datos mal formados se prueban con archivos temporales.

Se compararon las fuentes usando la conexión de solo lectura existente: ambas
contenían las mismas 1500 transacciones y 30 cuentas. Las predicciones y los
puntajes resultaron idénticos, calculados mediante el pipeline original:

| Fuente | Entrenamiento | Evaluación | Alertas | Precisión | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CSV | 1200 | 300 | 28 | 10,71 % | 100,00 % | 19,35 % |
| PostgreSQL | 1200 | 300 | 28 | 10,71 % | 100,00 % | 19,35 % |

En ambos casos: TN=272, FP=25, FN=0 y TP=3; matriz de confusión `[[272,25],[0,3]]`
(filas: clase real normal/anómala; columnas: predicción normal/alerta).
Estos valores documentan esta ejecución; la interfaz sigue calculándolos con los
datos cargados. No se agregaron dependencias ni se modificaron los CSV, el tema,
la conexión, el entrenamiento ni los experimentos anteriores.

También se inició el servidor Streamlit real en modo CSV sin las cinco variables
de conexión. Respondió HTTP 200 y la comprobación de la aplicación mediante su
API local mostró los 300 movimientos, las 28 alertas, los filtros y el detalle
técnico. Las métricas y el momento de carga se conservaron al filtrar; actualizar
resultados volvió a mostrar los 300 movimientos y restableció los filtros.
La API se habilitó solo mediante una opción temporal de arranque para esta
verificación; no se agregó a la configuración del proyecto.

### Preparación para Streamlit Community Cloud

Esta etapa no publica la aplicación. Un futuro despliegue usará `app.py` como
punto de entrada, las dependencias existentes de `requirements.txt` y los dos CSV
versionados. No necesita credenciales de PostgreSQL. Para seleccionar el demo,
configurar `DATA_SOURCE = "csv"` como clave raíz en **Advanced settings → Secrets**;
Streamlit expone las claves raíz también como variables de entorno. No subir
`.env` ni contraseñas al repositorio. Ver la [documentación de secretos de
Streamlit](https://docs.streamlit.io/develop/concepts/connections/secrets-management).

Se revisó `.streamlit/config.toml` y se conserva sin cambios, incluido el tema
aprobado y `browser.gatherUsageStats = false`. Su `server.address = "127.0.0.1"`
limita el servidor al equipo local: antes de desplegar, revisar el enlace de red
de la plataforma y quitar o sobrescribir esa dirección (por ejemplo, con
`STREAMLIT_SERVER_ADDRESS=0.0.0.0`) y usar inicio sin navegador (`server.headless`).
Las [variables de entorno y opciones de arranque de Streamlit](https://docs.streamlit.io/develop/concepts/configuration/options)
permiten sobrescribir la configuración sin tocar el tema. No hace falta desactivar
las protecciones CORS/XSRF ni habilitar interfaces de prueba en un despliegue.

## Etapa 6: evaluación temporal progresiva

Con PostgreSQL activo y el `.env` local ya configurado, ejecutar desde la raíz:

```powershell
.\.venv\Scripts\python.exe -X utf8 -W error -m src.evaluacion_temporal
```

El comando lee PostgreSQL una sola vez mediante la conexión de solo lectura y
libera sus recursos antes de entrenar. Los resultados quedan en memoria y se
imprimen en consola; no guarda modelos ni cambia los CSV o la interfaz.

Se ordenan los movimientos por `fecha_hora` y `transaccion_id`. Los límites se
fijan en el 40 %, 60 % y 80 % de todos los registros, sin consultar etiquetas.
Si un límite comparte timestamp, se desplaza al comienzo de ese grupo, igual
que el corte de 4B. Con 1500 movimientos sin empates en los cortes, las ventanas
son 600/300, 900/300 y 1200/300. Si los empates impiden formar tres evaluaciones
no vacías, se informa un error en vez de separar el grupo o repetir filas.

Cada entrenamiento es el prefijo completo anterior a su evaluación. Las
evaluaciones son bloques adyacentes y disjuntos; una operación evaluada puede
entrar en el entrenamiento de una ventana posterior como parte del pasado
disponible, sin utilizar su etiqueta. Los primeros 600 movimientos son solo
entrenamiento inicial: no reciben predicciones de evaluación en este esquema.
Son ventanas por cantidad de registros, con duraciones de calendario diferentes.

`entrenar_particion` comparte el bloque de preparación, entrenamiento y
predicción de 4B con la nueva evaluación. En cada ventana se reajustan las
referencias solo con entrenamiento y se conservan fijas durante la evaluación.
`movimientos_5min` reutiliza el cálculo causal de 4A: incluye el historial de
entrenamiento, las operaciones previas del bloque y la operación actual, sin
contar movimientos futuros. A igual timestamp se conserva el orden por ID.

Se mantienen las tres características y todos los parámetros de 4B: 200 árboles,
`max_samples="auto"`, `contamination="auto"`, `max_features=1.0`,
`bootstrap=False`, `random_state=42` y `n_jobs=1`. El puntaje sigue siendo
`-score_samples` y la alerta exige superar estrictamente 0.5.

Primero se generan las predicciones de las tres ventanas. Además, se ejecuta
el punto de entrada 80/20 de 4B para comprobar que su entrenamiento, características,
puntajes y predicciones coincidan exactamente con la última ventana. Esa cuarta
ejecución es solo una comprobación y no agrega filas al resumen. Recién después
se lee el CSV de etiquetas, se valida su cobertura total y se vincula por ID.

Cada informe incluye períodos, cantidades, TP/FP/TN/FN, matriz de confusión,
precisión, recall, F1 y resultados por tipo. El agregado concatena las predicciones
de evaluación y exige IDs únicos; calcula sus métricas sobre esas filas, sin
promediar porcentajes entre ventanas. También identifica las anomalías que
quedaron fuera de evaluación en el entrenamiento inicial.

**Average Precision (AP)** resume el ordenamiento por los puntajes continuos;
se calcula con `average_precision_score`, sin usar las decisiones binarias ni
optimizar el umbral. Su definición está en la
[documentación de Scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).
Por criterio explícito, se informa **N/D** si falta cualquiera de las dos clases
reales, porque no se puede medir su separación. El recall global o por tipo es
**N/D** cuando no hay anomalías reales del grupo. Precisión y F1 conservan la
convención `zero_division=0` de 4B; deben interpretarse junto con los conteos.

### Resultados reales de esta etapa

Ejecución verificada el **10 de noviembre de 2026**, con Scikit-learn **1.9.1**:
1500 movimientos de 30 cuentas. No hubo ajustes de tamaño por empates en estos
cortes. Todos los entrenamientos comienzan el **01/01/2026 a las 08:01:05**.

| Ventana | Entrenamiento | Hasta (entrenamiento) | Evaluación | Desde (evaluación) | Hasta (evaluación) |
| --- | ---: | --- | ---: | --- | --- |
| 1 | 600 | 04/02/2026 14:42:24 | 300 | 04/02/2026 14:54:42 | 20/02/2026 15:16:20 |
| 2 | 900 | 20/02/2026 15:16:20 | 300 | 20/02/2026 15:18:47 | 13/03/2026 09:25:21 |
| 3 | 1200 | 13/03/2026 09:25:21 | 300 | 13/03/2026 09:30:51 | 31/03/2026 19:13:50 |

| Evaluación | Movimientos | Anomalías reales | Alertas | TP | FP | TN | FN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ventana 1 | 300 | 9 | 63 | 4 | 59 | 232 | 5 |
| Ventana 2 | 300 | 8 | 43 | 7 | 36 | 256 | 1 |
| Ventana 3 | 300 | 3 | 28 | 3 | 25 | 272 | 0 |
| Agregado | 900 | 20 | 134 | 14 | 120 | 760 | 6 |

| Evaluación | Precisión | Recall | F1 | AP |
| --- | ---: | ---: | ---: | ---: |
| Ventana 1 | 6,35 % | 44,44 % | 11,11 % | 0,308107 |
| Ventana 2 | 16,28 % | 87,50 % | 27,45 % | 0,602596 |
| Ventana 3 | 10,71 % | 100,00 % | 19,35 % | 1,000000 |
| Agregado | 10,45 % | 70,00 % | 18,18 % | 0,517434 |

Matrices `[[TN, FP], [FN, TP]]`: ventana 1 `[[232, 59], [5, 4]]`,
ventana 2 `[[256, 36], [1, 7]]`, ventana 3 `[[272, 25], [0, 3]]` y agregado
`[[760, 120], [6, 14]]`. La ventana 3 reproduce **exactamente** las características,
los puntajes y las decisiones del 80/20, además de sus métricas originales.

Resultados por tipo, expresados como **detectadas / reales**:

| Evaluación | Importe inusual | Horario atípico | Concentración |
| --- | ---: | ---: | ---: |
| Ventana 1 | 1/1 (100 %) | 3/3 (100 %) | 0/5 (0 %) |
| Ventana 2 | 2/2 (100 %) | 1/1 (100 %) | 4/5 (80 %) |
| Ventana 3 | 2/2 (100 %) | 1/1 (100 %) | 0/0 (N/D) |
| Agregado | 5/5 (100 %) | 5/5 (100 %) | 4/10 (40 %) |

Quedaron **10 anomalías sin evaluar**, todas dentro del entrenamiento inicial:
cinco de importe (`TX000039`, `TX000108`, `TX000370`, `TX000399`, `TX000727`) y
cinco de horario (`TX000127`, `TX000287`, `TX000341`, `TX000486`, `TX000587`).
No se movieron los cortes para incluirlas. Las diez operaciones de concentración
sí quedaron en evaluación: seis fueron falsos negativos, no casos sin evaluar.

### Interpretación y limitaciones

El recall del 100 % del último bloque no describe todo el período: el agregado
es 70 %, con 120 falsas alertas entre 134 alertas. El comportamiento más débil
observado corresponde a las concentraciones. Estas diferencias entre ventanas
no prueban que aumentar el entrenamiento cause una mejora: cambian las cuentas,
las operaciones y los casos anómalos de cada evaluación.

AP mide el ranking a distintos niveles de corte, mientras las alertas usan el
umbral fijo de 0.5. Por eso una AP de 1 en la última ventana puede coexistir con
25 falsos positivos: sus tres anomalías encabezan el ranking, pero otras
operaciones también superan el umbral. No se utilizó ese resultado para ajustarlo.
La AP agregada se ofrece como descripción del ranking combinado, con la
limitación de mezclar puntajes de tres modelos sin calibración entre ventanas;
conviene priorizar las AP de cada ventana para interpretar el ranking.

Solo se evaluaron 20 anomalías sintéticas; las diez de concentración pertenecen
a **dos ráfagas**, no a diez episodios independientes. Además, con un conteo
causal la primera operación de una ráfaga aún no revela la concentración futura.
No se puede generalizar este rendimiento a transacciones reales, interpretar
una alerta como fraude ni atribuirle una causa específica. Las referencias y
los modelos pueden estar afectados por anomalías no identificadas del pasado:
no se quitaron usando las etiquetas. El prefijo inicial no tiene una evaluación
fuera de muestra en este diseño, y los entrenamientos de ventanas sucesivas se
solapan, de modo que las métricas entre ventanas tampoco son independientes.

Como trabajo posterior conviene reunir más períodos y episodios independientes,
analizar falsas alertas por cuenta y horario, y medir también la detección y
demora por ráfaga. Una eventual revisión de características o del umbral debería
usar períodos de validación separados y una evaluación final nueva, con un
criterio explícito de costo de falsas alertas y omisiones. Esta etapa no cambia
características, parámetros ni umbral para mejorar las métricas observadas.

La suite completa aprobó **90 pruebas** (77 anteriores y 13 nuevas) sin una base
real. Las nuevas cubren cortes y empates, IDs, causalidad, referencias sin futuro,
equivalencia con 80/20, lectura de etiquetas posterior a todas las predicciones,
AP con un ejemplo conocido, clases ausentes, agregado sin duplicados y errores
de conexión seguros. Se ejecutaron con:

```powershell
.\.venv\Scripts\python.exe -X utf8 -W error -m unittest discover -s tests -v
```

Después se ejecutó el comando real de esta etapa, confirmado en modo de solo
lectura. Los resultados anteriores son el registro de esa ejecución; el comando
los vuelve a calcular con los movimientos disponibles.

## Etapa 6B: análisis de errores

Para reproducir el diagnóstico desde la raíz del proyecto, con PostgreSQL activo
y el `.env` local configurado:

```powershell
.\.venv\Scripts\python.exe -X utf8 -W error -m src.analisis_errores
```

Como las etapas anteriores no persisten modelos o predicciones, el comando
reproduce una sola vez las tres ventanas mediante `predecir_ventanas`. Después
de completar sus predicciones lee las etiquetas, llama a `evaluar_ventanas` y
consume esas mismas tablas en `analizar_errores`. El diagnóstico no vuelve a
ajustar referencias, transformar características, entrenar ni calcular puntajes.
La conexión de solo lectura se cierra antes de entrenar. Todos los resultados
quedan en memoria y se imprimen en consola; no se modifican archivos de datos.

El informe clasifica cada operación evaluada como TP, FP, TN o FN por su ID y
rechaza duplicados. Las tablas por cuenta, ventana y cuenta/ventana distinguen:

- **FP:** cantidad absoluta de falsas alertas.
- **Tasa de falsos positivos:** `FP / (FP + TN)`, sobre operaciones realmente
  normales. Si no hay normales reales, se informa N/D.
- **Proporción de falsas alertas:** `FP / total evaluado` del grupo.
- **Tasa de alertas:** `(TP + FP) / total evaluado` del grupo.

Se incluyen los denominadores y los grupos con cero FP. Una cuenta con más FP
no necesariamente tiene una tasa mayor; tampoco una tasa alta con muy pocos
casos demuestra que su comportamiento sea problemático. Las tasas describen
solo el período evaluado, no toda la actividad histórica de la cuenta.

Para FP y TN se muestran cantidad, media, mediana, cuartiles y extremos de las
tres características y del puntaje, tanto agregados como por ventana. Son
contrastes descriptivos, sin pruebas de significación ni atribuciones causales.
Los puntajes y las referencias cambian entre modelos; por eso el desglose por
ventana acompaña al resumen agregado. Se imprimen todas las omisiones y las
diez falsas alertas de mayor puntaje, conservando sus IDs.

La reconstrucción usa **solo las operaciones etiquetadas como concentración**:
misma cuenta y distancia entre operaciones consecutivas de hasta cinco minutos,
ordenadas por timestamp e ID. Es una definición post hoc fijada sin mirar las
alertas. El encadenamiento puede durar más de cinco minutos y no equivale al
conteo móvil `movimientos_5min`, que sigue incluyendo todas las operaciones
previas de la cuenta según la regla causal original. Las posiciones dentro de
la ráfaga cuentan sus operaciones etiquetadas, no todas las transacciones de
la cuenta ni solo las que generaron alertas.

Las concentraciones fuera de evaluación se conservan con puntajes y decisiones
ausentes, sin imputar una operación normal. El informe señala las partes sin
evaluación y los episodios que cruzan ventanas/modelos. La primera alerta se
informa como posición, timestamp y segundos desde el inicio de la ráfaga completa;
si no hay alertas, se informa N/D. En episodios incompletos es solamente la
primera alerta **observada** durante evaluación.

También se inspeccionan las características del entrenamiento ya preparado y
los cortes de los árboles ya ajustados: mínimo, mediana, máximo, valores distintos,
número de cortes y árboles que usaron cada variable. La lectura de la estructura
se basa en los atributos descritos por
[Scikit-learn](https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html).
No se calculan importancias causales ni se explica un puntaje individual por la
cantidad de veces que una variable aparece en los árboles.

### Resultados reales del diagnóstico

La ejecución de esta etapa, con Scikit-learn **1.9.1**, reprodujo las **900
operaciones únicas**, 14 TP, 120 FP, 760 TN y 6 FN: precisión 10,45 %, recall
70 % y F1 18,18 %. El umbral permaneció en 0.5 en las tres ventanas. A continuación
se distinguen hechos medidos de hipótesis; las cifras están redondeadas solo
para presentación y no intervienen en las decisiones.

**Falsas alertas por cuenta y ventana.** Hubo FP en 26 de las 30 cuentas. Las
tres cuentas con más FP reunieron 31 de los 120 casos (25,83 %). Los conteos y
tasas muestran diferencias importantes de volumen:

| Cuenta | FP | Normales reales (FP+TN) | Tasa FP | Total evaluado |
| --- | ---: | ---: | ---: | ---: |
| CTA030 | 12 | 67 | 17,91 % | 67 |
| CTA007 | 10 | 19 | 52,63 % | 19 |
| CTA020 | 9 | 25 | 36,00 % | 25 |
| CTA011 | 7 | 12 | 58,33 % | 13 |
| CTA016 | 7 | 13 | 53,85 % | 13 |
| CTA029 | 7 | 61 | 11,48 % | 61 |

CTA030 encabezó el conteo absoluto. CTA011 y CTA016 tuvieron tasas mayores,
calculadas sobre denominadores pequeños; esto merece revisión, sin calificar
las cuentas como problemáticas. Para CTA011, FP/total fue 53,85 %, diferente de
su tasa FP del 58,33 %, porque una de sus 13 operaciones era una anomalía real.

| Ventana | FP | Normales reales | Tasa FP | FP/total evaluado |
| --- | ---: | ---: | ---: | ---: |
| 1 | 59 | 291 | 20,27 % | 19,67 % |
| 2 | 36 | 292 | 12,33 % | 12,00 % |
| 3 | 25 | 297 | 8,42 % | 8,33 % |

La primera ventana concentró el 49,17 % de las falsas alertas y tuvo la tasa FP
más alta. Las ventanas cambian en operaciones, cuentas, referencias y bosque;
esta comparación no aísla un efecto causal del tamaño de entrenamiento.

**Características y puntajes de FP frente a TN.** Los siguientes son hechos
descriptivos del agregado; el comando también muestra cada ventana por separado:

| Variable | Mediana FP | Q1–Q3 FP | Mediana TN | Q1–Q3 TN |
| --- | ---: | --- | ---: | --- |
| `monto_relativo` | 0,978219 | 0,674775–1,595154 | 0,995920 | 0,816585–1,202918 |
| `rareza_horaria` | 2,439444 | 2,116034–2,584509 | 1,998096 | 1,884541–2,112231 |
| `movimientos_5min` | 1 | 1–1 | 1 | 1–1 |
| `puntaje_anomalia` | 0,547098 | 0,516162–0,600941 | 0,408268 | 0,389162–0,441162 |

La rareza horaria mediana fue mayor en FP también dentro de cada ventana:
2,436116 frente a 2,098490; 2,484907 frente a 1,965031; y 2,384823 frente a
1,911561. Los importes relativos FP presentaron mayor dispersión agregada,
con valores tanto menores como mayores que la mediana de referencia: su mediana
global cercana a 1 no resume esa dispersión. El patrón de importes varía entre
ventanas, por lo que no se atribuyen todas las falsas alertas a importes altos.
Todas las operaciones FP y TN tuvieron `movimientos_5min=1`; esa variable no
distingue estos dos grupos en los datos evaluados.

Los puntajes FP estuvieron entre 0,500679 y 0,678763; los TN, entre 0,363546 y
0,499834. Esta separación responde a la regla de alerta: por sí sola no explica
por qué se asignaron esos puntajes. La falsa alerta de mayor puntaje fue
`TX000150` (CTA005, ventana 3): monto relativo 2,225725, rareza 2,523373 y
puntaje 0,678763. FP significa discrepancia respecto de las etiquetas sintéticas,
sin implicar una conclusión de fraude.

**Las seis omisiones.** Todas fueron de concentración y tuvieron predicción 0:

| ID | Cuenta | Fecha y hora | Ventana | Monto relativo | Rareza horaria | Movimientos/5 min | Puntaje |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| TX000643 | CTA021 | 17/02/2026 08:30:00 | 1 | 0,580196 | 2,313056 | 1 | 0,494996 |
| TX000626 | CTA021 | 17/02/2026 08:31:00 | 1 | 1,207209 | 2,313056 | 2 | 0,431074 |
| TX000638 | CTA021 | 17/02/2026 08:32:00 | 1 | 1,202440 | 2,313056 | 3 | 0,431736 |
| TX000664 | CTA021 | 17/02/2026 08:33:00 | 1 | 0,885090 | 2,313056 | 4 | 0,420430 |
| TX000621 | CTA021 | 17/02/2026 08:34:00 | 1 | 1,171568 | 2,313056 | 5 | 0,429521 |
| TX001258 | CTA028 | 12/03/2026 09:30:00 | 2 | 0,724449 | 1,834319 | 1 | 0,413823 |

En la primera ráfaga, la frecuencia subió de 1 a 5 sin producir alertas. Sus
importes relativos se mantuvieron entre 0,580196 y 1,207209, y la rareza fue
constante. El primer movimiento quedó a 0,005004 del umbral. En la segunda
ráfaga se omitió el primer movimiento, cuyo conteo causal todavía era 1 y cuyo
horario tenía rareza 1,834319. Son descripciones de las señales disponibles,
sin atribuir una causa específica a cada decisión.

**Comparación de ráfagas.** Ambas tuvieron cinco operaciones separadas por un
minuto, duraron cuatro minutos, quedaron completamente en evaluación y no
cruzaron modelos:

| Ráfaga | Cuenta / ventana | Período | Detectadas | Primera alerta | Demora |
| --- | --- | --- | ---: | --- | --- |
| 1 | CTA021 / 1 | 17/02/2026 08:30–08:34 | 0/5 | N/D | N/D |
| 2 | CTA028 / 2 | 12/03/2026 09:30–09:34 | 4/5 | 2.ª operación, TX001266, 09:31:00 | 60 segundos |

En CTA028, los puntajes cronológicos fueron 0,413823; 0,516967; 0,655514;
0,619951; 0,662827. Sus montos relativos estuvieron entre 0,724449 y 1,774522 y
su rareza horaria fue 1,834319. El comando imprime los diez movimientos de ambas
ráfagas con IDs, características, puntajes y resultado correcto/incorrecto.

La inspección del entrenamiento y de la estructura ya ajustada aportó este
hecho adicional para `movimientos_5min`:

| Ventana | Entrenamiento | Valores observados | Cortes con frecuencia | Árboles que la usaron |
| --- | ---: | --- | ---: | ---: |
| 1 | 600 | Solo 1 | 0 | 0/200 |
| 2 | 900 | 1, 2, 3, 4, 5 | 176 | 130/200 |
| 3 | 1200 | 1, 2, 3, 4, 5 | 261 | 169/200 |

En el primer bosque la frecuencia no interviene en ningún corte: no existía
variación de esa característica en su entrenamiento. En el segundo ya había
variación histórica, incluida la primera ráfaga, y la variable apareció en
130 árboles. Esto permite identificar una limitación estructural del primer
modelo; no equivale a una explicación causal completa de cada puntaje ni a una
medida de importancia relativa de las variables.

### Hipótesis y trabajo posterior

- La ausencia de variación histórica de frecuencia es compatible con la
  dificultad del primer bosque para responder a la concentración. Conviene
  estudiar cobertura y diversidad del entrenamiento en nuevos episodios.
- La asociación de falsas alertas con mayor rareza horaria sugiere revisar la
  estabilidad de las referencias por cuenta y la cantidad de historial
  disponible. Este análisis no prueba que la hora sea su causa individual.
- Conviene medir detección y demora por episodio, además de por transacción:
  una primera operación con conteo causal 1 aún no revela una ráfaga futura.
- Cualquier cambio de características o eventual calibración del umbral debe
  usar validación separada, costos explícitos de revisión/omisión y una prueba
  final nueva. No corresponde optimizar y volver a reportar estas mismas ventanas
  como evidencia independiente.

Solo hay dos episodios sintéticos de concentración, con cuentas, importes,
referencias y modelos diferentes. Sus operaciones están relacionadas; los
entrenamientos crecientes también se solapan. Estas limitaciones impiden aislar
la contribución de cada diferencia observada y generalizar los resultados al
sistema bancario real. La reconstrucción con etiquetas tampoco constituye un
detector de ráfagas listo para producción.

La verificación aprobó **102 pruebas**: las 90 anteriores y 12 nuevas. Estas
últimas cubren clasificación e IDs, tasas y denominadores, distribuciones,
ráfagas y empates, primera alerta, casos sin evaluación, independencia respecto
de las predicciones, lectura posterior de etiquetas, ausencia de recálculos y
errores de conexión seguros. Se ejecutó la suite completa con `-W error` y luego
el análisis real, secuencialmente. Se verificaron por SHA-256 los módulos de
las etapas anteriores, CSV, conexión, interfaz y dependencias: permanecen intactos.

## Etapa 6C: experimento controlado de concentración

Este comando ejecuta un experimento separado. No reemplaza el detector ni
incorpora reglas a Streamlit. Desde PowerShell, con PostgreSQL activo y el
`.env` local configurado:

```powershell
.\.venv\Scripts\python.exe -X utf8 -W error -m src.experimento_rafagas
```

Se comparan las predicciones originales de **Isolation Forest** con una
predicción **híbrida**, definida como `IF OR movimientos_5min >= 3`. La regla
queda fijada por el protocolo; no se buscan otros valores. Se conservan los
200 árboles, `contamination="auto"`, `random_state=42`, las tres características
y el umbral original. Los puntajes siguen siendo los de IF: la regla añade
una decisión binaria, sin inventar un puntaje híbrido ni una probabilidad.
Una alerta adicional puede ser un acierto o una falsa alerta.

El comando lee PostgreSQL en modo de solo lectura y cierra la conexión antes
de entrenar. Como las etapas anteriores no guardan modelos, reproduce una sola
vez las tres ventanas con `predecir_ventanas` y reutiliza sus predicciones.
Cada una de las 900 operaciones históricas se cuenta una sola vez. Esta parte
es **retrospectiva y exploratoria**: la regla se propuso después de conocer sus
errores, por lo que no constituye una validación independiente.

### Protocolo nuevo fijado antes de medir

`src/generar_escenarios_rafagas.py` crea datos solo en memoria, con semilla
**2026**, sin escribir ni modificar los CSV existentes. El protocolo genera
**180 operaciones de abril de 2026**, posteriores a todo el histórico:

- 120 operaciones normales aisladas: una por cuenta los días 1, 2, 29 y 30.
- Ocho episodios etiquetados como concentración anómala, con 30 operaciones.
- Ocho episodios de actividad concentrada legítima, con otras 30 operaciones.

Los 16 episodios usan cuentas distintas, un generador aleatorio independiente
por episodio y fechas laborables entre el 6 y el 27 de abril. Cada patrón se
repite en una cuenta personal y una comercial para **cada** clase de episodio.
Las etiquetas y la pertenencia a episodios se definen al generar los casos,
separadas de las ocho columnas originales y sin consultar predicciones:

| Patrón | Segundos desde el inicio | Operaciones/episodio | Máximo causal en 5 min |
| --- | --- | ---: | ---: |
| Densa | 0, 60, 120, 180, 240 | 5 | 5 |
| Borde de ventana | 0, 120, 300 | 3 | 3 |
| Corta | 0, 120 | 2 | 2 |
| Espaciada | 0, 180, 360, 540, 720 | 5 | 2 |

La ventana móvil incluye el límite de cinco minutos y la operación actual.
Los casos cortos y espaciados no alcanzan el umbral de la regla. En las
secuencias densas la regla empieza en la tercera operación, a los 120 segundos;
en las de borde, en la tercera, a los 300 segundos. Esto no implica que IF
espere esos tiempos: puede alertar antes por su propio puntaje.

Los montos siguen una distribución lognormal alrededor de la mediana de
referencia de la cuenta, con dispersión 0,2, redondeo a centavos y mínimo
positivo. Las horas proceden de la hora modal de entrenamiento de cada cuenta.
Todos son egresos por transferencia/homebanking. Los dos tipos de episodio
usan los mismos patrones y distribuciones; su diferencia de etiqueta es una
condición semántica del escenario, no una propiedad que estas tres variables
permitan necesariamente distinguir. No se define la verdad como `conteo >= 3`.

Para abril se conserva **el último modelo cronológico**, entrenado con 1200
operaciones, y sus referencias ya ajustadas. Se elige por orden temporal,
sin comparar rendimientos. Las 1500 operaciones históricas están disponibles
como historial sin etiquetas para los conteos, pero las últimas 300 no se usan
para reajustar referencias ni el bosque. `transformar_datos` conserva el
cálculo causal, incluyendo solo movimientos anteriores disponibles y el actual,
con su criterio original para empates. No hay ningún `fit` sobre abril.

Se generan todas las predicciones históricas y nuevas antes de leer el archivo
de etiquetas histórico; después se vinculan las etiquetas por `transaccion_id`.
No se modifican escenarios, semilla, parámetros ni umbrales según los resultados.

### Lectura de los resultados

La consola presenta por separado la comparación histórica, su desglose por
ventana y la prueba nueva; incluye matrices de confusión, precisión, recall,
F1, tasa FP y operaciones de concentración detectadas. Lista cada alerta
añadida, indicando si fue TP o FP, y distingue falsas alertas sobre operaciones
aisladas de las producidas sobre actividad concentrada legítima.

**Episodio detectado** significa al menos una alerta; **cobertura completa**
significa que todas sus operaciones generaron alertas. No son equivalentes.
Se muestran posición y timestamp de la primera alerta, y segundos desde el
inicio. Sin alertas, la demora es N/D. Las medianas de demora incluyen solo
episodios con alerta y muestran ese denominador; no imputan cero a omisiones.
En actividad legítima, un episodio con alerta representa una falsa alarma.
Los dos episodios históricos del dataset actual están completamente evaluados;
si se cambiara el dataset y uno cruzara el corte inicial de entrenamiento,
habría que revisar su cobertura antes de interpretar métricas de episodio.

### Resultados de la ejecución real

Se ejecutó el comando con PostgreSQL en solo lectura y Scikit-learn **1.9.1**.
El modelo congelado de abril usó las 1200 operaciones hasta el **13/03/2026
09:25:21**; el historial disponible llegó hasta el **31/03/2026 19:13:50**.
Los resultados siguientes no motivaron ningún cambio del protocolo.

**Histórico exploratorio: 900 operaciones, 20 anomalías reales.**

| Enfoque | Alertas | TP | FP | TN | FN | Precisión | Recall | F1 | Concentración detectada |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Isolation Forest | 134 | 14 | 120 | 760 | 6 | 10,45 % | 70,00 % | 18,18 % | 4/10 |
| Híbrido | 137 | 17 | 120 | 760 | 3 | 12,41 % | 85,00 % | 21,66 % | 7/10 |

La regla añadió **tres TP y cero FP**: `TX000638`, `TX000664` y `TX000621`,
de CTA021, con conteos 3, 4 y 5. La tasa FP quedó en 13,64 % en ambos casos.
Solo cambió la primera ventana: TP 4→7, FN 5→2 y alertas 63→66, con 59 FP.
La segunda conservó 7 TP, 36 FP y una omisión; la tercera mantuvo las 28
alertas originales, tres TP, 25 FP y ninguna omisión.

| Ráfaga histórica | IF: operaciones detectadas | Híbrido: operaciones detectadas | Primera alerta IF | Primera alerta híbrida |
| --- | ---: | ---: | --- | --- |
| H01, CTA021, 17/02 | 0/5 | 3/5 | N/D | 3.ª operación, 08:32:00, a los 120 s |
| H02, CTA028, 12/03 | 4/5 | 4/5 | 2.ª operación, 09:31:00, a los 60 s | Igual: 2.ª operación, a los 60 s |

La detección de episodios pasó de **1/2 a 2/2**, pero la cobertura de todas
sus operaciones siguió en **0/2**. La mediana de demora fue 60 s sobre un
episodio con IF y 90 s sobre dos con el híbrido: no describe un retraso en el
mismo caso, sino la incorporación de una detección antes ausente. El beneficio
se observa en el bosque cuyo entrenamiento tenía frecuencia constante; sigue
siendo evidencia exploratoria sobre casos ya conocidos.

**Escenarios nuevos: 180 operaciones, 30 anomalías reales.**

| Enfoque | Alertas | TP | FP | TN | FN | Precisión | Recall | F1 | Concentración detectada |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Isolation Forest | 44 | 22 | 22 | 128 | 8 | 50,00 % | 73,33 % | 59,46 % | 22/30 |
| Híbrido | 44 | 22 | 22 | 128 | 8 | 50,00 % | 73,33 % | 59,46 % | 22/30 |

La regla añadió **cero TP y cero FP**: todas las operaciones que cumplían
su condición ya estaban alertadas por el bosque congelado. Por tanto, tampoco
adelantó ninguna primera alerta. La tasa FP fue 14,67 % para ambos enfoques.

El detalle de los 16 episodios aparece en consola. Ambos enfoques tuvieron
este mismo comportamiento por patrón; cada fila agrupa dos episodios anómalos
y dos legítimos, con una cuenta personal y una comercial por clase:

| Patrón | TP / operaciones anómalas | FP / operaciones legítimas | Episodios anómalos con alerta | Episodios legítimos con falsa alarma | Primera alerta en cada episodio |
| --- | ---: | ---: | ---: | ---: | --- |
| Densa | 8/10 | 8/10 | 2/2 | 2/2 | 2.ª operación, 60 s |
| Borde | 4/6 | 4/6 | 2/2 | 2/2 | 2.ª operación, 120 s |
| Corta | 2/4 | 2/4 | 2/2 | 2/2 | 2.ª operación, 120 s |
| Espaciada | 8/10 | 8/10 | 2/2 | 2/2 | 2.ª operación, 180 s |

Se detectaron **8/8 episodios anómalos**, con **0/8** de cobertura completa.
Las ocho omisiones fueron sus primeras operaciones, con conteo causal 1.
Los ocho episodios legítimos también recibieron falsas alarmas; ninguna
operación normal aislada fue alertada (**0/120**). Los **22 FP** se concentran
en las 30 operaciones legítimas agrupadas: tasa FP 73,33 % en ese subgrupo.
No son falsas alertas añadidas por la regla; ya estaban presentes en IF.
La mediana de primera alerta fue 120 s, sobre ocho episodios, tanto para las
concentraciones anómalas como para las legítimas y en ambos enfoques.

### Interpretación y decisión experimental

La comparación histórica muestra que una regla explícita puede cubrir una
característica sin variación en el entrenamiento del primer bosque. En la
prueba nueva, el último bosque ya respondió a esas concentraciones y la regla
fue redundante. **Estos resultados no justifican incorporar automáticamente
el híbrido al detector o a la aplicación.** Permanece como experimento separado.

La ausencia de FP adicionales en esta ejecución no demuestra que la regla sea
inocua en otros datos: también se activa sobre actividad legítima concentrada.
El control legítimo muestra un problema común a ambos enfoques: detectar un
patrón de actividad no basta para decidir si merece revisión. Las etiquetas
semánticas de estos casos usan las mismas distribuciones observables, así que
no se espera inferir su significado solo con estas tres características.

La prueba de abril es nueva respecto del histórico y se fijó antes de medir,
pero sigue siendo **sintética y condicionada a perfiles del entrenamiento**.
Hay pocos episodios, montos cercanos a sus medianas, horarios modales y un solo
tipo/dirección/canal. Sus operaciones están relacionadas; no son 180 pruebas
independientes de generalización. Tampoco se probó en abril el bosque inicial
con frecuencia constante: se eligió de antemano el último modelo cronológico.
La proporción de anomalías cambia de 20/900 a 30/180, por lo que no corresponde
interpretar la diferencia de precisión entre poblaciones como una mejora.

Una regla parecida al mecanismo que construye las anomalías sintéticas puede
obtener buenos resultados sin generalizar a movimientos reales. Antes de
integrarla convendría investigar, con un protocolo y datos nuevos, su aporte
cuando el entrenamiento carece de variación en frecuencia; incluir más lotes
legítimos y perfiles de cuenta, y acordar el costo de alertas y omisiones.
También conviene medir carga de revisión por episodio. Esas son hipótesis para
trabajo posterior; no se ajustaron escenarios, características ni umbrales aquí.

### Verificación del experimento

Pasaron **116 pruebas**: las 102 existentes y 14 nuevas, con `-W error`.
Se ejecutaron secuencialmente, antes de la evaluación real. Las pruebas nuevas
comprueban la unión inclusiva desde tres, conservación de predicciones y
puntajes originales, correspondencia por ID, conteos y métricas conocidas,
recall no evaluable sin anomalías, episodios completos y demoras, reproducibilidad
y balance de escenarios, límites de cinco minutos, causalidad por prefijos,
independencia del futuro, ausencia de reajuste y lectura posterior de etiquetas.
También verifican errores de conexión seguros, sin una base real.

```powershell
.\.venv\Scripts\python.exe -X utf8 -W error -m unittest discover -s tests -v
```

Los módulos anteriores, la conexión, CSV, dependencias y Streamlit conservaron
sus hashes SHA-256. No se incorporaron dependencias, no se modificó PostgreSQL
y no se realizaron commits. Los datos nuevos y los resultados quedan en memoria
y consola; ejecutar el comando no escribe archivos del dataset.

## Generación y validación

Desde la raíz del proyecto, en PowerShell:

```powershell
.\.venv\Scripts\python.exe src\generar_dataset.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

El generador resuelve la carpeta `data/` respecto de su propio archivo y
sobrescribe los tres archivos de datos y documentación en cada ejecución.
Las pruebas usan `unittest` de la biblioteca estándar, sin `pytest`, y verifican
cantidades, identificadores, nulos permitidos, montos, fechas, correspondencia
de etiquetas y coherencia entre movimiento, dirección y canal. También cubren
los niveles de importe respecto de la mediana normal de cada cuenta, las
distancias y la variedad de los horarios atípicos respecto de sus ventanas
habituales, las dos ráfagas frente a la frecuencia normal por cuenta, la
exportación y la reproducibilidad.

Las pruebas de `tests/test_database.py` usan dobles de conexión y bloquean las
conexiones reales de Psycopg. Comprueban la configuración, la protección de solo
lectura, la lectura con Pandas y la liberación de recursos sin requerir PostgreSQL
ni credenciales. La suite no sustituye la verificación manual de la conexión local.

## Alcance y cronograma orientativo

- Semana 1: generar y validar el dataset sintético; explorarlo y preparar los datos con Pandas.
- Semana 2: implementar un único modelo con Scikit-learn y revisar las anomalías detectadas.
- Semana 3: crear una vista sencilla con Streamlit, agregar pruebas básicas y completar la documentación.

Para terminar en 2 semanas, combinar la visualización y la documentación con la
segunda semana. Quedan fuera del alcance las conexiones bancarias, el análisis
en tiempo real y el despliegue en producción.
