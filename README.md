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

Todavía no se implementan modelos de ML, interfaz Streamlit, conexiones bancarias
ni tablas SQL.

## Estructura

```text
financial-anomaly-detector/
├── data/
│   ├── transacciones.csv          # Dataset principal sin etiquetas.
│   ├── etiquetas_evaluacion.csv   # Etiquetas separadas para evaluación.
│   └── diccionario_datos.md       # Campos, formatos y reglas.
├── src/
│   └── generar_dataset.py         # Generador reproducible.
├── tests/
│   └── test_generar_dataset.py    # Validaciones con unittest.
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

Requiere Python 3 y pip. Desde la raíz del proyecto, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Pandas y NumPy se usan para generar los datos. Scikit-learn y Streamlit se
reservan para las siguientes etapas.

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

## Alcance y cronograma orientativo

- Semana 1: generar y validar el dataset sintético; explorarlo y preparar los datos con Pandas.
- Semana 2: implementar un único modelo con Scikit-learn y revisar las anomalías detectadas.
- Semana 3: crear una vista sencilla con Streamlit, agregar pruebas básicas y completar la documentación.

Para terminar en 2 semanas, combinar la visualización y la documentación con la
segunda semana. Quedan fuera del alcance las conexiones bancarias, el análisis
en tiempo real y el despliegue en producción.
