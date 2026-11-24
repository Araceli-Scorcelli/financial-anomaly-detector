# Financial Anomaly Detector

Proyecto académico de Ciencia de Datos e Inteligencia Artificial para estudiar cómo un modelo de **aprendizaje no supervisado** identifica comportamientos inusuales en transacciones financieras sintéticas. Se implementó con Python, Pandas, PostgreSQL, scikit-learn y Streamlit.

El objetivo es comprender el funcionamiento y los límites de **Isolation Forest**, no construir un sistema bancario ni clasificar operaciones como fraudulentas. Todos los movimientos son sintéticos y las etiquetas de anomalía se reservan para la evaluación.

## Aplicación online

La demostración interactiva está publicada mediante Streamlit Community Cloud:

**[Abrir Financial Anomaly Detector](https://financial-anomaly-detector-5wzvcxhptf9mubphaiabrg.streamlit.app/)**

La aplicación permite explorar el análisis de transacciones sintéticas. No se conecta con bancos reales ni confirma fraudes. Quien prefiera ejecutar una copia en su computadora puede utilizar el modo CSV descrito más abajo, sin disponer de PostgreSQL.

## Ejecutar la aplicación localmente con CSV

Con Python instalado, desde la carpeta raíz del repositorio (PowerShell en Windows):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:DATA_SOURCE = "csv"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

El modo `csv` utiliza los archivos sintéticos incluidos en `data/`: no requiere instalar PostgreSQL ni configurar credenciales. Es la alternativa más sencilla para probar el dashboard en una computadora propia. Una vez iniciado Streamlit, abrir [http://localhost:8501](http://localhost:8501) en esa misma computadora.

**Importante:** el código conserva `postgres` como fuente predeterminada cuando no se define `DATA_SOURCE`. Por eso, para usar los CSV localmente es necesario establecer `DATA_SOURCE=csv` como indica el comando anterior.

### ¿Cuándo se necesita PostgreSQL?

| Uso | Fuente de datos | ¿Requiere PostgreSQL local? |
| --- | --- | --- |
| Abrir la [aplicación online](https://financial-anomaly-detector-5wzvcxhptf9mubphaiabrg.streamlit.app/) | CSV sintéticos | No |
| Ejecutar el dashboard en la computadora | CSV con `DATA_SOURCE=csv` | No |
| Ejecutar el dashboard con una base de datos | PostgreSQL con `DATA_SOURCE=postgres` | Sí |
| Repetir los entrenamientos y las evaluaciones desde los scripts de consola documentados | PostgreSQL | Sí |

El uso de PostgreSQL se mantiene como parte del desarrollo original y de la reproducción técnica de los experimentos, pero **no es un requisito para explorar la aplicación**, ni online ni con el modo CSV local. Las instrucciones de configuración y los comandos de los experimentos aparecen más adelante. Las pruebas automatizadas pueden ejecutarse con `python -m unittest discover -s tests -v`.

## Experiencia y resultados principales

- **Datos:** 1.500 transacciones sintéticas de enero a marzo de 2026, distribuidas entre 30 cuentas. Se incorporaron 30 anomalías conocidas: 10 de importe, 10 de horario y 10 de concentración temporal.
- **Características del modelo:** `monto_relativo`, `rareza_horaria` (24 franjas de una hora, con suavizado) y `movimientos_5min` (conteo causal por cuenta).
- **Entrenamiento:** Isolation Forest con 200 árboles, sin etiquetas de anomalía y con un umbral de alerta de 0,5.
- **Evaluación:** separación cronológica inicial de 1.200 operaciones para entrenamiento y 300 para evaluación; después, tres ventanas temporales con 600, 900 y 1.200 operaciones de entrenamiento, respectivamente.
- **Exploración adicional:** análisis de falsos positivos y anomalías no detectadas, más una comparación experimental con una regla de concentración. La regla **no se incorporó** al detector principal.

| Evaluación | Operaciones evaluadas | Anomalías conocidas | Detectadas | Falsas alertas | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inicial (1.200/300) | 300 | 3 | 3 | 25 | 10,71 % | 100 % | 19,35 % |
| Tres ventanas progresivas | 900 | 20 | 14 | 120 | 10,45 % | 70 % | 18,18 % |

La evaluación inicial coincide con la tercera ventana y **no se contabiliza dos veces**. La regla experimental elevó las detecciones de 14 a 17 en las 900 operaciones sintéticas evaluadas inicialmente, pero en 180 escenarios sintéticos nuevos de abril de 2026 ambos enfoques obtuvieron los mismos resultados. Esto no demuestra desempeño en operaciones financieras reales.

## Cómo se desarrolló

El proyecto avanzó desde la construcción de datos de prueba hasta el análisis de los errores del modelo:

1. **Datos sintéticos:** se generaron 1.500 transacciones de 30 cuentas, con comportamientos habituales y 30 anomalías conocidas de importe, horario y concentración. Las etiquetas se guardaron por separado para evaluar las predicciones.
2. **Almacenamiento y preparación:** los movimientos se guardaron en CSV y PostgreSQL. Con Python se construyeron las tres características que recibe Isolation Forest: `monto_relativo`, `rareza_horaria` y `movimientos_5min`.
3. **Primer entrenamiento:** se entrenó Isolation Forest sin etiquetas y se evaluó sobre las últimas 300 operaciones, posteriores al período de entrenamiento.
4. **Aplicación Streamlit:** se creó un dashboard para explorar las alertas y sus métricas. Puede ejecutarse con PostgreSQL o directamente con los CSV; también hay una [versión online](https://financial-anomaly-detector-5wzvcxhptf9mubphaiabrg.streamlit.app/).
5. **Evaluación ampliada:** se repitió el entrenamiento con cantidades crecientes de datos anteriores y se revisaron los errores del modelo.
6. **Experimento complementario:** se probó combinar las alertas de Isolation Forest con una regla de concentración. Los resultados no justificaron incorporarla al detector principal.

Las etapas detalladas del código, las comprobaciones y los criterios utilizados están documentados más abajo y en el [detalle técnico de los experimentos](docs/detalle_experimentos.md).

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
para Isolation Forest y sus métricas. Streamlit presenta el dashboard tanto de forma local como en la demostración online.

## PostgreSQL local: almacenamiento y reproducción de experimentos

**Sección opcional para quienes solo quieren probar el dashboard:** la aplicación online y el modo CSV local no requieren instalar PostgreSQL. Estas indicaciones documentan la base utilizada durante el desarrollo y permiten ejecutar la aplicación con base de datos y reproducir los scripts de entrenamiento y evaluación descritos más adelante.


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
recuperadas, parámetros y umbral, anomalías conocidas y alertas, y los diez puntajes más altos del período de
evaluación con sus identificadores, fechas, cuentas, características y predicción.
Para interpretar sus métricas:

- **Verdaderos positivos (TP)**: anomalías conocidas que recibieron alerta.
- **Falsos positivos (FP)**: movimientos normales que recibieron alerta y
  requerirían una revisión innecesaria.
- **Falsos negativos (FN)**: anomalías conocidas que el modelo omitió.
- **Precisión**, `TP / (TP + FP)`: qué proporción de las alertas era anómala.
- **Recall**, `TP / (TP + FN)`: qué proporción de las anomalías fue detectada.
- **F1**: media armónica de precisión y recall; resume ambas medidas.

La matriz de confusión tiene filas reales `[normal, anomalía]` y columnas
predichas `[normal, anomalía]`, es decir, `[[TN, FP], [FN, TP]]`; `TN` son los
movimientos normales correctamente descartados. Si una métrica global tiene
denominador cero, se informa `0` (`zero_division=0`) y debe leerse junto con las
cantidades de casos.

Para cada tipo de anomalía incorporada al conjunto sintético (`importe_inusual`,
`horario_atipico` y `concentracion`), se informa cuántas operaciones estaban
presentes en los datos de evaluación, cuántas fueron detectadas, cuántas
quedaron sin identificar y el recall correspondiente.

Isolation Forest no identifica el tipo de anomalía: únicamente señala operaciones
como potencialmente anómalas. La comparación por tipos se realiza después,
utilizando las etiquetas conocidas para analizar los resultados.

Si un tipo de anomalía no aparece en el período evaluado, su recall se informa
como **N/D**, ya que no hay casos de ese tipo para calcularlo.

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

## Etapa 5: aplicación Streamlit con CSV o PostgreSQL

Para **probar la aplicación**, se recomienda el modo CSV explicado al principio del README; no es necesario configurar PostgreSQL. La segunda alternativa conserva la conexión de solo lectura utilizada durante el desarrollo.


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

Una vez iniciado Streamlit, la aplicación puede abrirse desde el navegador en
[http://localhost:8501](http://localhost:8501). Esta dirección corresponde a una
ejecución local, por lo que solo estará disponible en la computadora donde se
haya iniciado el servidor.

Para detener la aplicación, presionar **Ctrl+C** en la terminal.

El dashboard **Financial Anomaly Detector** muestra un resumen de los movimientos
recuperados, cuentas y tamaños de entrenamiento y evaluación. Todas las cifras
se calculan con los datos cargados. Sus seis indicadores globales corresponden
al conjunto completo de evaluación: transacciones, alertas, «Anomalías reales»
(etiquetas sintéticas conocidas), precisión, recall y F1. Se distribuyen en una grilla de hasta tres columnas que
se adapta al ancho disponible, manteniendo visibles los títulos y porcentajes.
También advierte sobre la cantidad de anomalías conocidas, los
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
globales permanecen referidos a toda la evaluación. Las etiquetas conocidas solo
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

### Aplicación publicada en Streamlit Community Cloud

Enlace público de la demostración: **[Financial Anomaly Detector](https://financial-anomaly-detector-5wzvcxhptf9mubphaiabrg.streamlit.app/)**.

Esta sección conserva las decisiones técnicas de preparación del despliegue y las instrucciones para reproducirlo o actualizarlo. La dirección pública es distinta de `http://localhost:8501`, que solo funciona al ejecutar la aplicación localmente.

La revisión de despliegue quitó únicamente `[server] address = "127.0.0.1"` de
`.streamlit/config.toml`, para que la plataforma gestione la dirección y el puerto.
La paleta, tipografías, espacios, bordes y estilos del tema se conservaron íntegros.
No se fijó un puerto ni se habilitó la API de pruebas. Tampoco se añadieron cambios
de CORS/XSRF. `browser.gatherUsageStats = false` se mantiene en el archivo local;
Community Cloud puede sobrescribir esta y otras opciones según su
[configuración de plataforma](https://docs.streamlit.io/deploy/streamlit-community-cloud/status).

`requirements.txt` fija las siete dependencias directas a las versiones instaladas
y probadas, sin agregar paquetes ni actualizar el entorno: Streamlit 1.67.0,
Pandas 3.1.1, NumPy 2.5.4, Scikit-learn 1.9.1, SQLAlchemy 2.1.8,
psycopg[binary] 3.3.8 y python-dotenv 1.2.4. Así se conservan las APIs del diseño
aprobado y las versiones principales del entorno científico. SQLAlchemy, Psycopg y
dotenv se mantienen porque los módulos compartidos los importan incluso en modo
CSV; no se crean conexiones al seleccionarlo. Las dependencias transitivas no
constituyen un bloqueo completo del entorno y pueden variar al resolverlo.

Seleccionar **Python 3.14** en Cloud para coincidir con la versión menor probada
localmente (3.14.10). Todas estas dependencias admiten esa versión; se verificaron
en PyPI distribuciones Linux/CPython 3.14 para las dependencias binarias principales.
Community Cloud admite versiones publicadas de Python con soporte de seguridad y
permite elegirlas en [Advanced settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).
No depender de la versión predeterminada de la plataforma ni usar `runtime.txt`
como sustituto de esa selección. El parche concreto de Python lo gestiona Cloud.

Guía para reproducir o actualizar el despliegue:

1. Guardar y sincronizar los cambios con GitHub; Streamlit Community Cloud usa el contenido remoto y no el de VS Code.
2. Para un nuevo despliegue, elegir el repositorio, la rama que contenga la versión deseada y **Main file path: `app.py`**. En **Advanced settings**, seleccionar Python 3.14. Para actualizar la aplicación existente, comprobar primero qué rama tiene configurada en Community Cloud.
3. En **Secrets**, pegar únicamente esta configuración TOML, en la raíz, sin una
   sección `[database]` o `[DATA_SOURCE]`:

   ```toml
   DATA_SOURCE = "csv"
   ```

4. No configurar `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER` ni `DB_PASSWORD` para
   el demo. El valor raíz se expone como variable de entorno antes de ejecutar la
   aplicación, según la [documentación de Secrets](https://docs.streamlit.io/develop/concepts/connections/secrets-management).
   Sin esa selección explícita, el proyecto sigue usando PostgreSQL por defecto;
   no hay respaldo automático hacia CSV.
5. Tras un despliegue o actualización, revisar los logs de instalación y comprobar
   300 movimientos evaluados, 28 alertas y las métricas documentadas arriba, además
   de filtros y actualización. Las pruebas locales no sustituyen una verificación de la aplicación en Community Cloud.

Los archivos necesarios están versionados: `app.py`, los módulos de `src/`,
`.streamlit/config.toml`, `requirements.txt`, `data/transacciones.csv` y
`data/etiquetas_evaluacion.csv`. El modelo se entrena al cargar, sin requerir un
archivo de modelo local. El demo no depende de `.venv`, `.env`, un servidor
PostgreSQL ni credenciales. Las etiquetas continúan leyéndose después de predecir.

### Verificación y consideraciones de seguridad

Durante el desarrollo se realizaron pruebas automatizadas para comprobar la
preparación de los datos, el entrenamiento, la evaluación del modelo y el
funcionamiento de la aplicación.

La versión publicada utiliza los archivos sintéticos del repositorio y no
requiere acceso a una base de datos bancaria ni credenciales de PostgreSQL.
Los archivos de configuración local que pueden contener información sensible,
como `.env`, están excluidos del control de versiones.

Estas verificaciones contribuyen a la reproducibilidad del proyecto, aunque
no sustituyen la comprobación de su funcionamiento en el entorno de despliegue.

## Etapa 6: evaluación temporal progresiva

### ¿Por qué ampliar la evaluación?

En el primer experimento, el modelo detectó las tres anomalías conocidas presentes en las últimas 300 transacciones, pero también generó 25 falsas alertas. Como eran pocos casos, ese resultado no bastaba para comprender su comportamiento en otros períodos.

Por eso se realizó una **evaluación temporal progresiva**: en cada prueba se entrenó un nuevo modelo con los movimientos anteriores y se lo evaluó sobre las 300 operaciones siguientes. Se respetó el orden de las fechas y no se utilizaron las etiquetas de anomalía para entrenar.

| Prueba | Operaciones de entrenamiento | Operaciones de evaluación | Anomalías conocidas | Detectadas | Falsas alertas |
| --- | ---: | ---: | ---: | ---: | ---: |
| Primera ventana | 600 | 300 | 9 | 4 | 59 |
| Segunda ventana | 900 | 300 | 8 | 7 | 36 |
| Tercera ventana | 1.200 | 300 | 3 | 3 | 25 |
| **Total evaluado** | — | **900 distintas** | **20** | **14** | **120** |

La tercera ventana corresponde al mismo corte que el primer experimento de 1.200/300: **no se cuenta dos veces**.

En conjunto, el modelo identificó 14 de las 20 anomalías conocidas (**recall del 70 %**), pero la mayoría de las alertas correspondió a movimientos etiquetados como habituales (**precision del 10,45 %** y **F1 del 18,18 %**).

La cantidad de información anterior aumentó en cada entrenamiento, pero **también cambiaron las transacciones evaluadas**. Por eso no podemos atribuir las diferencias de resultados únicamente al tamaño del entrenamiento.

### Cómo reproducir la prueba

Este script de consola utiliza la conexión local de PostgreSQL documentada anteriormente:

```powershell
.\.venv\Scripts\python.exe -X utf8 -W error -m src.evaluacion_temporal
```

El código de la evaluación está en `src/evaluacion_temporal.py`. Para consultar fechas exactas, matrices de confusión, *Average Precision* (AP), parámetros y comprobaciones, ver el [detalle técnico de la etapa 6](docs/detalle_experimentos.md#etapa-6-evaluación-temporal-progresiva).

## Etapa 6B: análisis de errores

### ¿Qué operaciones resultaron difíciles de detectar?

La evaluación temporal dejó **120 falsas alertas** y **6 anomalías conocidas sin detectar**. Ambos resultados importan: las falsas alertas pueden aumentar el trabajo de revisión, mientras que una anomalía omitida no genera ninguna señal.

Al comparar las alertas con las operaciones habituales que no fueron señaladas, se observó una **rareza horaria mediana mayor** en el primer grupo (aproximadamente 2,44 frente a 2,00). Es una pista para investigar, no una explicación automática de cada decisión: Isolation Forest considera las tres características.

Las seis anomalías omitidas correspondían a **concentraciones de movimientos**, conocidas en el proyecto como *ráfagas*. Los resultados por tipo fueron:

| Tipo de anomalía conocida | Detectadas | Total en evaluación |
| --- | ---: | ---: |
| Importe inusual | 5 | 5 |
| Horario atípico | 5 | 5 |
| Concentración de movimientos | 4 | 10 |

Las diez operaciones de concentración pertenecían a **dos ráfagas de cinco movimientos**, no a diez situaciones independientes. En la primera no se detectó ninguna operación; en la segunda se detectaron cuatro de cinco.

Una observación relevante fue que, en el primer entrenamiento, `movimientos_5min` tenía el mismo valor para todas las operaciones. Por lo tanto, los árboles no podían utilizar esa característica para dividir los datos. En entrenamientos posteriores sí existía variación. Esto ayuda a comprender una limitación del primer modelo, aunque no explica por sí solo cada alerta u omisión.

### Cómo reproducir el análisis

Con PostgreSQL local configurado:

```powershell
.\.venv\Scripts\python.exe -X utf8 -W error -m src.analisis_errores
```

El análisis se implementó en `src/analisis_errores.py`. El [detalle técnico de la etapa 6B](docs/detalle_experimentos.md#etapa-6b-análisis-de-errores) conserva las tablas por cuenta, los identificadores de las seis omisiones, los puntajes y la revisión de los árboles.

## Etapa 6C: experimento complementario con una regla

### ¿Una regla sencilla podía ayudar a detectar las ráfagas?

Como varias anomalías de concentración habían pasado inadvertidas, se exploró una alternativa: **mantener Isolation Forest y agregar una alerta cuando una cuenta registrara tres o más movimientos en cinco minutos** (`movimientos_5min >= 3`).

Primero se compararon ambos enfoques sobre las **900 transacciones sintéticas que ya se habían evaluado**. Como sus errores eran conocidos cuando se propuso la regla, esta primera comparación fue exploratoria: no podía considerarse una validación independiente.

| Comparación en las 900 operaciones ya evaluadas | Isolation Forest | Modelo + regla |
| --- | ---: | ---: |
| Anomalías detectadas | 14 de 20 | 17 de 20 |
| Falsas alertas | 120 | 120 |
| Recall | 70 % | 85 % |
| F1-score | 18,18 % | 21,66 % |

La regla recuperó tres operaciones de concentración que el modelo no había señalado. Sin embargo, **mejorar sobre un conjunto ya conocido no garantiza que la mejora se mantenga**.

### Una segunda prueba con escenarios sintéticos nuevos

Para comprobarlo, se generaron **180 transacciones nuevas de abril de 2026**, posteriores al conjunto inicial: 120 operaciones aisladas, 30 concentradas etiquetadas como anómalas y 30 concentradas etiquetadas como legítimas. Esta última distinción permitió observar si la concentración también podía producir falsas alertas.

Se mantuvo fijo el último modelo entrenado, sin reajustar sus parámetros a partir de los nuevos escenarios.

| Resultados sobre las 180 operaciones nuevas | Isolation Forest | Modelo + regla |
| --- | ---: | ---: |
| Anomalías detectadas | 22 de 30 | 22 de 30 |
| Falsas alertas | 22 | 22 |
| Recall | 73,33 % | 73,33 % |
| F1-score | 59,46 % | 59,46 % |

**La regla no agregó detecciones en esta segunda prueba.** Por eso se decidió mantener Isolation Forest como detector principal y conservar la estrategia combinada únicamente como experimento. No se incorporó al dashboard de Streamlit.

La prueba de abril también fue sintética: sus resultados permiten comparar ambos enfoques en nuevos escenarios controlados, pero no validar un sistema financiero real. Además, muestran que una concentración de movimientos no es necesariamente una anomalía relevante.

### Cómo reproducir el experimento

Con PostgreSQL local configurado:

```powershell
.\.venv\Scripts\python.exe -X utf8 -W error -m src.experimento_rafagas
```

El código se encuentra en `src/experimento_rafagas.py` y `src/generar_escenarios_rafagas.py`. El [detalle técnico de la etapa 6C](docs/detalle_experimentos.md#etapa-6c-experimento-controlado-de-concentración) conserva los patrones de escenarios, resultados por episodio, tiempos de detección y controles de reproducción.

> **Sobre los datos:** tanto las 1.500 transacciones de enero a marzo como las 180 de abril son **simuladas**. En la documentación técnica, «historial» se refiere únicamente a los movimientos sintéticos anteriores a la prueba de abril, no a operaciones bancarias reales.

## Generación de datos y pruebas

Para regenerar el dataset sintético o ejecutar las pruebas automatizadas desde la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe src\generar_dataset.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

**Atención:** volver a ejecutar el generador sobrescribe los archivos de transacciones, etiquetas y diccionario de datos en `data/`. Si solo querés explorar la aplicación, no necesitás regenerarlos: los CSV ya están incluidos.

Las pruebas utilizan `unittest` para comprobar la generación de datos, las características, el entrenamiento, las métricas, el dashboard y los experimentos. Pueden ejecutarse sin disponer de una base PostgreSQL local. El detalle de las verificaciones de cada etapa se conserva en [`docs/detalle_experimentos.md`](docs/detalle_experimentos.md).

## Alcance y limitaciones

Este repositorio documenta una experiencia académica reproducible con datos sintéticos. Incluye la generación y preparación de transacciones, el entrenamiento, la evaluación temporal, el análisis de errores y la comparación experimental. La aplicación de exploración puede utilizarse localmente o mediante la versión publicada en Streamlit Community Cloud.

No se conecta con bancos reales, no valida fraude ni está preparado para producción o análisis en tiempo real. Las etiquetas conocidas sirven para medir resultados y no son entradas del modelo. La interpretación de una alerta requiere considerar el contexto de cada operación y las limitaciones de los datos.

El código principal, los datos de ejemplo, las pruebas y la documentación técnica se conservan en el repositorio. El archivo `.env`, las contraseñas locales y `.streamlit/secrets.toml` no deben versionarse.
