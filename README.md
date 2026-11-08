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

Todavía no se implementan modelos de ML, interfaz Streamlit ni conexiones bancarias.

## Estructura

```text
financial-anomaly-detector/
├── data/
│   ├── transacciones.csv          # Dataset principal sin etiquetas.
│   ├── etiquetas_evaluacion.csv   # Etiquetas separadas para evaluación.
│   └── diccionario_datos.md       # Campos, formatos y reglas.
├── src/
│   ├── generar_dataset.py         # Generador reproducible.
│   ├── database.py                # Configuración y conexión de solo lectura.
│   ├── verificar_conexion.py      # Verificación manual y lectura con Pandas.
│   └── preparar_datos.py          # Validación, referencias y características.
├── sql/
│   ├── crear_tablas.sql           # Estructura e indicaciones de importación.
│   └── consultas_exploratorias.sql # Consultas de lectura en pgAdmin.
├── tests/
│   ├── test_generar_dataset.py    # Validaciones del dataset con unittest.
│   ├── test_database.py           # Pruebas de conexión sin una base real.
│   └── test_preparar_datos.py     # Características con datos ficticios, sin DB.
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
y `python-dotenv` permiten leer PostgreSQL desde Python. Scikit-learn y
Streamlit se reservan para las siguientes etapas.

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

Antes de evaluar un modelo habrá que separar cronológicamente las transacciones:
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
Todavía no hay entrenamiento de Isolation Forest ni umbrales de detección.

Las pruebas de `tests/test_preparar_datos.py` usan pequeños ejemplos ficticios sin
PostgreSQL ni etiquetas reales. Cubren validación, medianas, distribución horaria,
medianoche, cuentas con pocos datos, conteos causales, empates, historial y la
separación entre ajuste y transformación. Ejecutar la suite desde la raíz:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

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
