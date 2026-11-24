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

El modo `csv` utiliza los archivos sintéticos incluidos en `data/`, sin necesidad de instalar PostgreSQL ni configurar credenciales. Es la alternativa recomendada para ejecutar el dashboard localmente. Una vez iniciado Streamlit, abrir [http://localhost:8501](http://localhost:8501) en esa misma computadora.

**Importante:** el código conserva `postgres` como fuente predeterminada cuando no se define `DATA_SOURCE`. Por eso, para usar los CSV localmente es necesario establecer `DATA_SOURCE=csv` como indica el comando anterior.

### ¿Cuándo se necesita PostgreSQL?

| Uso | Fuente de datos | ¿Requiere PostgreSQL local? |
| --- | --- | --- |
| Abrir la [aplicación online](https://financial-anomaly-detector-5wzvcxhptf9mubphaiabrg.streamlit.app/) | CSV sintéticos | No |
| Ejecutar el dashboard en la computadora | CSV con `DATA_SOURCE=csv` | No |
| Ejecutar el dashboard con una base de datos | PostgreSQL con `DATA_SOURCE=postgres` | Sí |
| Repetir los entrenamientos y las evaluaciones desde los scripts de consola documentados | PostgreSQL | Sí |

El uso de PostgreSQL se mantiene como parte del desarrollo original y de la reproducción técnica de los experimentos, pero **no es un requisito para explorar la aplicación**, ni online ni con el modo CSV local. Las instrucciones de configuración y los comandos de los experimentos aparecen más adelante. Las pruebas automatizadas pueden ejecutarse con `python -m unittest discover -s tests -v`.

## Datos y resultados principales

- **Datos:** 1.500 transacciones sintéticas de enero a marzo de 2026, distribuidas entre 30 cuentas. Se incorporaron 30 anomalías conocidas: 10 de importe, 10 de horario y 10 de concentración temporal.
- **Características del modelo:** `monto_relativo`, `rareza_horaria` (24 franjas de una hora, con suavizado) y `movimientos_5min` (conteo causal por cuenta).
- **Entrenamiento:** Isolation Forest con 200 árboles, sin etiquetas de anomalía y con un umbral de alerta de 0,5.
- **Evaluación:** separación cronológica inicial de 1.200 operaciones para entrenamiento y 300 para evaluación; después, tres ventanas temporales con 600, 900 y 1.200 operaciones de entrenamiento, respectivamente.
- **Exploración adicional:** análisis de falsos positivos y anomalías no detectadas, más una comparación experimental con una regla de concentración. La regla **no se incorporó** al detector principal.

| Evaluación | Operaciones evaluadas | Anomalías conocidas | Detectadas | Falsas alertas | Precisión | Recall | F1-score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Inicial (1.200/300) | 300 | 3 | 3 | 25 | 10,71 % | 100 % | 19,35 % |
| Tres ventanas progresivas | 900 | 20 | 14 | 120 | 10,45 % | 70 % | 18,18 % |

La evaluación inicial coincide con la tercera ventana y **no se contabiliza dos veces**. La regla experimental elevó las detecciones de 14 a 17 en las 900 operaciones sintéticas evaluadas inicialmente, pero en 180 escenarios sintéticos nuevos de abril de 2026 ambos enfoques obtuvieron los mismos resultados. Esto no demuestra desempeño en operaciones financieras reales.

## Cómo se desarrolló

El proyecto avanzó desde la construcción de datos de prueba hasta el análisis de los errores del modelo:

1. **Datos sintéticos:** se generaron 1.500 transacciones de 30 cuentas, con comportamientos habituales y 30 anomalías conocidas de importe, horario y concentración. Las etiquetas se conservaron por separado para evaluar las predicciones.
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

## Etapa 4A: preparación de los datos

### ¿Qué información recibe el modelo?

Las transacciones originales describen montos, fechas, cuentas y otras características de cada operación. Para identificar comportamientos poco habituales, se construyeron **tres variables** que permiten comparar cada movimiento con la actividad anterior de su cuenta.

| Característica | ¿Qué representa? | Ejemplo de interpretación |
| --- | --- | --- |
| `monto_relativo` | Relación entre el importe y la mediana de la cuenta | Un valor de 3 indica un importe tres veces mayor que la mediana utilizada como referencia |
| `rareza_horaria` | Qué tan poco frecuente es la hora de una operación | Las horas con menos antecedentes obtienen valores más altos |
| `movimientos_5min` | Cantidad de movimientos de la misma cuenta en una ventana de cinco minutos | Permite observar operaciones cercanas en el tiempo |

Para la rareza horaria se utilizan **24 franjas de una hora**, con un suavizado que tiene en cuenta las horas vecinas y evita asignar probabilidades nulas a horarios sin registros. En el conteo de cinco minutos se consideran únicamente las operaciones disponibles hasta el momento de cada transacción: **no se utilizan movimientos futuros**.

Las referencias de montos y horarios se calculan con el conjunto de entrenamiento y después se aplican sin reajustarse al período de evaluación. Esta separación es importante para no incorporar información del futuro al análisis.

Ninguna de las tres variables demuestra por sí sola que una transacción sea fraudulenta o anómala. Su función es describir comportamientos que el modelo analizará conjuntamente.

### Cómo reproducir la preparación

El siguiente comando permite revisar las características desde la consola con PostgreSQL local configurado:

```powershell
.\.venv\Scripts\python.exe -m src.preparar_datos
```

Este procedimiento explora las características de los datos; **no constituye por sí mismo una evaluación del modelo**. La implementación se encuentra en `src/preparar_datos.py`. Las fórmulas de suavizado, validaciones, casos de cuentas nuevas y ejemplos de uso están en el [detalle técnico de la etapa 4A](docs/detalle_implementacion.md#etapa-4a-preparación-y-características).

## Etapa 4B: entrenamiento y primera evaluación

### ¿Cómo identifica las operaciones inusuales Isolation Forest?

**Isolation Forest** es un algoritmo de aprendizaje no supervisado que intenta aislar observaciones mediante divisiones aleatorias de sus características. Una operación que se separa del resto con pocos cortes suele recibir un puntaje de anomalía más alto.

El modelo se entrenó con las tres características construidas en la etapa anterior, **sin utilizar las etiquetas de anomalía**. Para realizar la primera evaluación se respetó el orden temporal de los datos:

- **Entrenamiento:** las primeras 1.200 transacciones sintéticas.
- **Evaluación:** las 300 transacciones posteriores.
- **Configuración:** 200 árboles, `contamination="auto"`, `random_state=42` y un umbral de alerta de 0,5 sobre el puntaje `-score_samples`.

El puntaje indica qué tan inusual resulta una operación para el modelo; **no representa una probabilidad de fraude**. Las etiquetas conocidas se consultan después de generar las predicciones, exclusivamente para medir los resultados.

### ¿Qué se obtuvo?

| Indicador | Primera evaluación |
| --- | ---: |
| Transacciones evaluadas | 300 |
| Anomalías conocidas presentes | 3 |
| Anomalías detectadas | 3 |
| Falsas alertas | 25 |
| Precisión | 10,71 % |
| Recall | 100 % |
| F1-score | 19,35 % |

El modelo identificó las tres anomalías conocidas de ese período, pero **25 de sus 28 alertas correspondieron a movimientos etiquetados como habituales**. Además, en esta evaluación no había anomalías de concentración, por lo que no era posible medir su detección.

El recall del 100 % debe interpretarse con cautela: se basa en **solo tres casos conocidos**. Precisamente por esa limitación se amplió después la evaluación a tres ventanas temporales, como se explica en la etapa 6.

### Cómo reproducir el entrenamiento

Con PostgreSQL local configurado:

```powershell
.\.venv\Scripts\python.exe -m src.entrenar_modelo
```

El módulo está en `src/entrenar_modelo.py`. Los parámetros completos, la definición de las métricas, las matrices de confusión y los controles metodológicos se conservan en el [detalle técnico de la etapa 4B](docs/detalle_implementacion.md#etapa-4b-isolation-forest-y-evaluación).

## Etapa 5: visualización interactiva con Streamlit

### ¿Qué permite hacer la aplicación?

Se desarrolló un dashboard para explorar las transacciones evaluadas y observar cuáles fueron señaladas por Isolation Forest. Incluye indicadores generales, filtros por cuenta, período y resultado, una tabla de operaciones y un gráfico de alertas frente a movimientos sin alerta.

También permite consultar las tres características técnicas y sus puntajes para comprender **qué información recibió el modelo**. Estas medidas describen las operaciones, pero no explican de forma causal por qué se generó cada alerta.

Los filtros modifican las operaciones visibles y sus contadores, mientras que las métricas generales permanecen referidas al conjunto de evaluación completo. Las etiquetas sintéticas conocidas sirven para calcular esas métricas; **no se utilizan durante el entrenamiento**.

### Cómo acceder

La versión publicada está disponible en **[Financial Anomaly Detector — Streamlit](https://financial-anomaly-detector-5wzvcxhptf9mubphaiabrg.streamlit.app/)**. Para ejecutarla en una computadora propia, se recomienda el modo CSV, que utiliza los archivos del repositorio y no requiere PostgreSQL:

```powershell
$env:DATA_SOURCE = "csv"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Una vez iniciado el servidor, abrir [http://localhost:8501](http://localhost:8501) en esa misma computadora. Para detenerlo, presionar **Ctrl+C** en la terminal. Este comando presupone que Python y las dependencias ya se instalaron, como se explica al inicio del README.

La aplicación también admite PostgreSQL local como fuente de datos:

```powershell
$env:DATA_SOURCE = "postgres"
.\.venv\Scripts\python.exe -m streamlit run app.py
```

En ambos casos se utiliza el mismo procedimiento de preparación, entrenamiento y evaluación. Con los archivos sintéticos incluidos, se verificó que ambas fuentes produjeran **las mismas 28 alertas y las mismas métricas de evaluación**.

**Importante:** el código conserva PostgreSQL como fuente predeterminada cuando no se especifica `DATA_SOURCE`. La demostración online y el comando local anterior seleccionan CSV expresamente.

### Verificación y publicación

Se realizaron pruebas automatizadas de lectura de datos, métricas, filtros y actualización de resultados. Los CSV permiten utilizar la demostración sin compartir contraseñas ni configurar una base de datos. La aplicación publicada utiliza transacciones sintéticas y **no se conecta con servicios bancarios reales**.

La configuración detallada de las dos fuentes, el funcionamiento de la caché, las comprobaciones y las instrucciones para reproducir o actualizar el despliegue se encuentran en el [detalle técnico de la etapa 5](docs/detalle_implementacion.md#etapa-5-aplicación-streamlit-con-csv-o-postgresql).

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

En conjunto, el modelo identificó 14 de las 20 anomalías conocidas (**recall del 70 %**), pero la mayoría de las alertas correspondió a movimientos etiquetados como habituales (**precisión del 10,45 %** y **F1 del 18,18 %**).

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

**Importante:** volver a ejecutar el generador sobrescribe los archivos de transacciones, etiquetas y diccionario de datos en `data/`. Para explorar la aplicación no es necesario regenerarlos: los CSV ya están incluidos.

Las pruebas utilizan `unittest` para comprobar la generación de datos, las características, el entrenamiento, las métricas, el dashboard y los experimentos. Pueden ejecutarse sin disponer de una base PostgreSQL local. El detalle de las verificaciones de cada etapa se conserva en [`docs/detalle_experimentos.md`](docs/detalle_experimentos.md).

## Alcance y limitaciones

Este repositorio documenta una experiencia académica reproducible con datos sintéticos. Incluye la generación y preparación de transacciones, el entrenamiento, la evaluación temporal, el análisis de errores y la comparación experimental. La aplicación de exploración puede utilizarse localmente o mediante la versión publicada en Streamlit Community Cloud.

No se conecta con bancos reales, no valida fraude ni está preparado para producción o análisis en tiempo real. Las etiquetas conocidas sirven para medir resultados y no son entradas del modelo. La interpretación de una alerta requiere considerar el contexto de cada operación y las limitaciones de los datos.

El código principal, los datos de ejemplo, las pruebas y la documentación técnica se conservan en el repositorio. El archivo `.env`, las contraseñas locales y `.streamlit/secrets.toml` no deben versionarse.
