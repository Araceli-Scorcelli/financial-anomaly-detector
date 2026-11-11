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
y revisar las métricas históricas. Reutiliza las etapas anteriores y mantiene la
lectura de PostgreSQL sin modificar datos. Es una demostración académica con datos
sintéticos; una alerta no confirma fraude. No se conecta con bancos reales.

La etapa 6 agrega una evaluación temporal complementaria con tres ventanas de
entrenamiento creciente y bloques posteriores de evaluación sin solapamientos.
Conserva el detector, sus parámetros y la evaluación 80/20 de las etapas anteriores.
Se ejecuta desde consola y no cambia la aplicación Streamlit.

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
│   ├── verificar_conexion.py      # Verificación manual y lectura con Pandas.
│   ├── preparar_datos.py          # Validación, referencias y características.
│   ├── entrenar_modelo.py         # Isolation Forest y evaluación cronológica.
│   └── evaluacion_temporal.py     # Walk-forward, métricas y cobertura temporal.
├── sql/
│   ├── crear_tablas.sql           # Estructura e indicaciones de importación.
│   └── consultas_exploratorias.sql # Consultas de lectura en pgAdmin.
├── tests/
│   ├── test_generar_dataset.py    # Validaciones del dataset con unittest.
│   ├── test_database.py           # Pruebas de conexión sin una base real.
│   ├── test_preparar_datos.py     # Características con datos ficticios, sin DB.
│   ├── test_entrenar_modelo.py    # Separación, modelo y métricas sin DB.
│   ├── test_app.py               # Interfaz con AppTest y datos ficticios.
│   └── test_evaluacion_temporal.py # Ventanas, causalidad y métricas sin DB.
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

## Etapa 5: aplicación local con Streamlit

Con PostgreSQL encendido y la configuración local de la etapa 3, ejecutar desde
la raíz del proyecto:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

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

Al cargar por primera vez, `cargar_resultados()` consulta PostgreSQL en modo de
solo lectura, cierra sus recursos, ejecuta `entrenar_y_predecir` y luego carga las
etiquetas para evaluar. Conserva una única entrada con resultados en memoria
mediante `st.cache_data(max_entries=1)`, sin vencimiento automático. La caché es
compartida por las sesiones del mismo servidor y cada ejecución recibe una copia;
no conserva el modelo ni conexiones a la base y no persiste resultados en disco.
Cambiar filtros reutiliza esos resultados, sin reajustar referencias ni reentrenar.

El botón **Actualizar resultados** invalida esa caché, vuelve a consultar los
datos y repite el entrenamiento y la evaluación. Usarlo cuando cambien los datos
de PostgreSQL o las etiquetas. Reiniciar el servidor también borra la caché;
si cambia `.env`, reiniciar el proceso para aplicar la configuración de conexión
de la etapa 3.

La aplicación es local y no publica el proyecto ni modifica PostgreSQL o los
CSV. Las pruebas de `tests/test_app.py` usan `AppTest` de Streamlit con datos
ficticios, sin conectarse a PostgreSQL. Para ejecutar toda la suite:

```powershell
.\.venv\Scripts\python.exe -W error -m unittest discover -s tests -v
```

La verificación local tras la mejora visual aprobó **77 pruebas**: las 67 anteriores y
10 de la aplicación. También se comprobó la interfaz con PostgreSQL real: mostró
300 transacciones de evaluación y, al filtrar, conservó los puntajes y las seis
métricas globales con una sola lectura y un solo entrenamiento. El servidor
local inició correctamente y respondió con HTTP 200; se detuvo al terminar la
prueba. Estas comprobaciones no requieren abrir otro navegador durante la suite.

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
