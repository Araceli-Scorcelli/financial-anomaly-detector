# Financial Anomaly Detector

Proyecto académico en Python para explorar la detección de anomalías en
transacciones financieras. El alcance previsto es pequeño y realizable en
2 o 3 semanas: trabajar con un único conjunto de datos CSV, un modelo de
detección de anomalías y una visualización sencilla.

## Estado inicial

Este repositorio contiene solamente la estructura del proyecto. El procesamiento,
el modelo, la interfaz y las pruebas se implementarán más adelante.

## Estructura

```text
financial-anomaly-detector/
├── data/               # Datos de entrada y resultados del procesamiento.
│   └── .gitkeep
├── src/                # Futuro código de procesamiento, modelo y visualización.
│   └── .gitkeep
├── tests/              # Futuras pruebas del proyecto.
│   └── .gitkeep
├── README.md           # Descripción, instalación y alcance.
├── requirements.txt    # Dependencias de Python.
└── .gitignore          # Archivos excluidos de Git.
```

Los archivos `.gitkeep` permiten conservar las carpetas vacías en Git.

## Preparación del entorno

Requiere Python 3 y pip. Desde la raíz del proyecto, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Las dependencias previstas son Pandas para procesar datos, Scikit-learn para
detectar anomalías y Streamlit para visualizar los resultados. Todavía no hay
una aplicación que ejecutar.

## Alcance y cronograma orientativo

- Semana 1: elegir un CSV sintético o anonimizado, explorarlo y preparar los datos con Pandas.
- Semana 2: implementar un único modelo con Scikit-learn y revisar las anomalías detectadas.
- Semana 3: crear una vista sencilla con Streamlit, agregar pruebas básicas y completar la documentación.

Para terminar en 2 semanas, combinar la visualización y la documentación con la
segunda semana. Quedan fuera del alcance las conexiones bancarias, el análisis
en tiempo real y el despliegue en producción.
