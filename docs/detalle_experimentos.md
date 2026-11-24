# Detalle técnico de los experimentos

Este documento conserva el desarrollo técnico completo de las etapas 6, 6B y 6C del proyecto. Incluye los protocolos, los resultados por ventana y cuenta, los casos individuales, las comprobaciones y las instrucciones de ejecución.

El [README principal](../README.md) presenta una explicación más breve de qué se investigó, qué se obtuvo y qué se aprendió. Para una lectura inicial, conviene comenzar allí.

**Cómo interpretar los términos:** todas las transacciones utilizadas en el proyecto son **sintéticas**. Cuando se habla de «historial», «período histórico» o «comparación histórica», se hace referencia a datos simulados de enero a marzo de 2026 disponibles antes de los escenarios sintéticos de abril; **no se trata de registros bancarios auténticos**. Las anomalías conocidas son las que se etiquetaron durante la generación de los datos. Los resultados y el número de pruebas documentan ejecuciones realizadas en diferentes momentos del desarrollo, no necesariamente el estado de la batería de pruebas más reciente.

**Reproducción:** las entradas de consola documentadas en estas etapas utilizan PostgreSQL local configurado en `.env`, a diferencia del dashboard Streamlit, que también puede utilizar los CSV incluidos en el repositorio.

---

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
**N/D** cuando no hay anomalías conocidas del grupo. Precisión y F1 conservan la
convención `zero_division=0` de 4B; deben interpretarse junto con los conteos.

### Resultados obtenidos en esta etapa

Ejecución verificada el **10 de noviembre de 2026**, con Scikit-learn **1.9.1**:
1500 movimientos de 30 cuentas. No hubo ajustes de tamaño por empates en estos
cortes. Todos los entrenamientos comienzan el **01/01/2026 a las 08:01:05**.

| Ventana | Entrenamiento | Hasta (entrenamiento) | Evaluación | Desde (evaluación) | Hasta (evaluación) |
| --- | ---: | --- | ---: | --- | --- |
| 1 | 600 | 04/02/2026 14:42:24 | 300 | 04/02/2026 14:54:42 | 20/02/2026 15:16:20 |
| 2 | 900 | 20/02/2026 15:16:20 | 300 | 20/02/2026 15:18:47 | 13/03/2026 09:25:21 |
| 3 | 1200 | 13/03/2026 09:25:21 | 300 | 13/03/2026 09:30:51 | 31/03/2026 19:13:50 |

| Evaluación | Movimientos | Anomalías conocidas | Alertas | TP | FP | TN | FN |
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

Resultados por tipo, expresados como **detectadas / conocidas**:

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
  normales. Si no hay operaciones etiquetadas como habituales, se informa N/D.
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

### Resultados del diagnóstico

La ejecución de esta etapa, con Scikit-learn **1.9.1**, reprodujo las **900
operaciones únicas**, 14 TP, 120 FP, 760 TN y 6 FN: precisión 10,45 %, recall
70 % y F1 18,18 %. El umbral permaneció en 0.5 en las tres ventanas. A continuación
se distinguen hechos medidos de hipótesis; las cifras están redondeadas solo
para presentación y no intervienen en las decisiones.

**Falsas alertas por cuenta y ventana.** Hubo FP en 26 de las 30 cuentas. Las
tres cuentas con más FP reunieron 31 de los 120 casos (25,83 %). Los conteos y
tasas muestran diferencias importantes de volumen:

| Cuenta | FP | Operaciones habituales conocidas (FP+TN) | Tasa FP | Total evaluado |
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
su tasa FP del 58,33 %, porque una de sus 13 operaciones era una anomalía conocida.

| Ventana | FP | Operaciones habituales conocidas | Tasa FP | FP/total evaluado |
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

### Resultados de la ejecución

Se ejecutó el comando con PostgreSQL en solo lectura y Scikit-learn **1.9.1**.
El modelo congelado de abril usó las 1200 operaciones hasta el **13/03/2026
09:25:21**; el historial disponible llegó hasta el **31/03/2026 19:13:50**.
Los resultados siguientes no motivaron ningún cambio del protocolo.

**Comparación exploratoria sobre el conjunto sintético inicial: 900 operaciones, 20 anomalías conocidas.**

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

**Escenarios nuevos: 180 operaciones, 30 anomalías conocidas.**

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

En la ejecución de esta etapa se verificó que los módulos anteriores, la conexión,
los CSV, las dependencias y Streamlit conservaran sus hashes SHA-256.
El experimento no incorpora dependencias ni modifica PostgreSQL. Los datos nuevos y los resultados quedan en memoria
y consola; ejecutar el comando no escribe archivos del dataset.
