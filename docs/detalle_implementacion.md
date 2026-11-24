# Detalle técnico de la preparación, el modelo y la aplicación

Este documento reúne la descripción técnica completa de las etapas **4A, 4B y 5**. Conserva las fórmulas, funciones, parámetros, comprobaciones y configuraciones que pueden ser útiles para estudiar o reproducir la implementación.

Para entender el objetivo del proyecto, probar la aplicación o consultar sus resultados principales, es preferible comenzar por el [README](../README.md).

**Aclaración:** todos los movimientos usados para el entrenamiento y la evaluación son sintéticos. Las expresiones de comprobación y los números de pruebas a continuación documentan momentos específicos del desarrollo. El dashboard ya cuenta con una versión publicada en Streamlit Community Cloud; la ejecución local descrita aquí es una alternativa.

**Origen de datos:** el dashboard admite CSV o PostgreSQL a través de `DATA_SOURCE`. Los comandos independientes de las primeras etapas siguen documentados con PostgreSQL local y su configuración de solo lectura.

---

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
PostgreSQL ni etiquetas conocidas. Cubren validación, medianas, distribución horaria,
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

La primera ejecución del experimento, verificada el 8 de noviembre de 2026 con Scikit-learn
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

La ejecución local no publica por sí misma el proyecto ni modifica PostgreSQL o los
CSV. La aplicación también cuenta con una versión online, enlazada desde el README. Las pruebas de `tests/test_app.py` usan `AppTest` de Streamlit con datos
ficticios y el CSV sintético incluido, bloqueando los accesos a PostgreSQL. Cubren las métricas
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

### Comprobaciones del modo demostración CSV

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
