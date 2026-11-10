"""Exploración local de predicciones de 4B; filtros separados del entrenamiento."""

from datetime import datetime

import pandas as pd
import streamlit as st

from src import entrenar_modelo as pipeline


class ErrorAplicacion(RuntimeError):
    """Mensaje preparado para la interfaz, sin detalles internos ni credenciales."""


@st.cache_data(show_spinner=False, max_entries=1)
def cargar_resultados() -> dict:
    """Una ejecución de 4B por caché; no conserva motores, conexiones ni modelos.

    Sin TTL: interactuar con filtros nunca provoca una recarga por vencimiento.
    La actualización es explícita. Streamlit entrega una copia de los datos.
    """
    try:
        engine = pipeline.crear_engine()
        try:
            transacciones = pipeline.cargar_transacciones(engine)
        finally:
            engine.dispose()
    except Exception:
        raise ErrorAplicacion(
            "No se pudo leer PostgreSQL o cerrar la conexión. Revisá que el servicio esté activo, "
            "la configuración local de .env y los permisos de lectura."
        ) from None
    try:
        resultado = pipeline.entrenar_y_predecir(transacciones)
    except Exception:
        raise ErrorAplicacion(
            "No se pudieron preparar los datos o entrenar el detector. "
            "Revisá el formato de public.transacciones y las dependencias del entorno."
        ) from None
    try:
        # Predicciones ya generadas: etiquetas exclusivamente para métricas históricas.
        etiquetas = pipeline.cargar_etiquetas(pipeline.RUTA_ETIQUETAS, transacciones["transaccion_id"])
        evaluacion = pipeline.evaluar_predicciones(resultado.predicciones, etiquetas)
    except Exception:
        raise ErrorAplicacion(
            "No se pudo completar la evaluación histórica. Revisá data/etiquetas_evaluacion.csv "
            "y su correspondencia con los identificadores de PostgreSQL."
        ) from None
    return {
        "predicciones": resultado.predicciones,
        "total_transacciones": len(transacciones),
        "cuentas": int(transacciones["cuenta_id"].nunique()),
        "entrenamiento": len(resultado.entrenamiento),
        "inicio_entrenamiento": resultado.entrenamiento["fecha_hora"].min(),
        "fin_entrenamiento": resultado.entrenamiento["fecha_hora"].max(),
        "precision": evaluacion.precision, "recall": evaluacion.recall, "f1": evaluacion.f1,
        "matriz_confusion": evaluacion.matriz_confusion, "por_tipo": evaluacion.por_tipo,
        "umbral": -float(resultado.modelo.offset_), "actualizado": datetime.now().astimezone(),
    }


def filtrar_transacciones(
    predicciones: pd.DataFrame, cuenta: str | None, fechas: tuple, resultado: str,
) -> pd.DataFrame:
    """Solo selecciona filas; incluye todo el día final y conserva puntajes e IDs."""
    if len(fechas) != 2:
        raise ValueError("Seleccioná una fecha inicial y una fecha final para explorar los movimientos.")
    inicio, fin = (pd.Timestamp(fecha).normalize() for fecha in fechas)
    if inicio > fin:
        raise ValueError("La fecha inicial debe ser anterior o igual a la fecha final.")
    if resultado not in ("Todas", "Alertas", "Normales"):
        raise ValueError("Seleccioná un resultado válido.")
    mascara = (predicciones["fecha_hora"] >= inicio) & (predicciones["fecha_hora"] < fin + pd.Timedelta(days=1))
    if cuenta is not None:
        mascara &= predicciones["cuenta_id"] == cuenta
    if resultado != "Todas":
        mascara &= predicciones["prediccion_anomalia"] == (1 if resultado == "Alertas" else 0)
    return predicciones.loc[mascara].copy()


def preparar_tabla(predicciones: pd.DataFrame) -> pd.DataFrame:
    """Presentación de predicciones, sin incorporar etiquetas individuales."""
    datos = predicciones.sort_values(["puntaje_anomalia", "transaccion_id"], ascending=[False, True]).copy()
    datos["estado"] = datos["prediccion_anomalia"].map({1: "Alerta potencial", 0: "Normal"})
    return datos[["transaccion_id", "cuenta_id", "fecha_hora", "estado", "monto", "puntaje_anomalia",
                  "monto_relativo", "rareza_horaria", "movimientos_5min"]]


def main() -> None:
    st.set_page_config(page_title="Financial Anomaly Detector", page_icon="📊", layout="wide")
    st.markdown("""
    <style>
    .block-container {max-width: 1500px; padding-top: 2rem; padding-bottom: 2rem;}
    h1 {letter-spacing: -0.045em; font-weight: 750;}
    h2, h3 {letter-spacing: -0.02em;}
    .eyebrow {color: #0f766e; font-size: .76rem; font-weight: 700; letter-spacing: .13em; margin-bottom: .2rem;}
    [data-testid="stMetric"] {background: #fff; border-radius: 12px;}
    [data-testid="stMetricValue"] {font-size: 1.8rem; font-weight: 650;}
    [data-testid="stMetricLabel"] {color: #526176; font-size: .85rem;}
    [data-testid="stSidebar"] {border-right: 1px solid #dce4ed;}
    [data-testid="stSidebar"] h2 {font-size: 1.15rem;}
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="eyebrow">DATA SCIENCE · ANÁLISIS EXPLORATORIO</p>', unsafe_allow_html=True)
    st.title("Financial Anomaly Detector")
    st.markdown("Explorá movimientos bancarios sintéticos y revisá **anomalías potenciales** con Isolation Forest.")

    with st.sidebar:
        st.markdown("## Panel de exploración")
        st.caption("Proyecto académico · Datos ficticios")
        actualizar = st.button("Actualizar resultados", key="actualizar_resultados", width="stretch",
                               help="Vuelve a leer PostgreSQL, entrenar y evaluar con la configuración fija de 4B.")
        if actualizar:
            cargar_resultados.clear()
            for clave in ("filtro_cuenta", "filtro_fechas", "filtro_resultado"):
                st.session_state.pop(clave, None)
    try:
        with st.spinner("Cargando el dataset y los resultados del detector…"):
            datos = cargar_resultados()
    except ErrorAplicacion as error:
        st.error(str(error))
        st.caption("Después de corregir el problema, pulsá Actualizar resultados.")
        st.stop()
    except Exception:
        st.error("No se pudieron cargar los resultados. Revisá la configuración local y volvé a actualizar.")
        st.stop()

    predicciones = datos["predicciones"]
    primera, ultima = predicciones["fecha_hora"].min().date(), predicciones["fecha_hora"].max().date()
    tn, fp, fn, tp = (int(valor) for valor in datos["matriz_confusion"].ravel())
    reales = tp + fn
    st.caption(
        f"Dataset completo: {datos['total_transacciones']:,} movimientos · {datos['cuentas']} cuentas · "
        f"Entrenamiento: {datos['entrenamiento']:,} · Evaluación: {len(predicciones):,} · PostgreSQL en solo lectura"
    )
    st.subheader("Evaluación global")
    st.caption(f"Del {primera:%d/%m/%Y} al {ultima:%d/%m/%Y}. Estos indicadores no cambian con los filtros.")
    indicadores = [
        ("Transacciones evaluadas", str(len(predicciones)), "Operaciones del bloque temporal posterior al entrenamiento."),
        ("Alertas generadas", str(int(predicciones["prediccion_anomalia"].sum())), "Operaciones que superan el umbral del detector."),
        ("Anomalías reales", str(reales), "Etiquetas sintéticas del período de evaluación; no son fraudes confirmados."),
        ("Precisión", f"{datos['precision']:.2%}".replace(".", ","), "Proporción de alertas que coinciden con una anomalía etiquetada."),
        ("Recall", f"{datos['recall']:.2%}".replace(".", ","), "Proporción de anomalías etiquetadas que el modelo detectó."),
        ("F1-score", f"{datos['f1']:.2%}".replace(".", ","), "Media armónica de precisión y recall."),
    ]
    for columna, (etiqueta, valor, ayuda) in zip(st.columns(6), indicadores):
        columna.metric(etiqueta, valor, help=ayuda, border=True)
    ausentes = datos["por_tipo"].loc[datos["por_tipo"]["reales"] == 0, "tipo_anomalia"].tolist()
    aviso = (f"Evaluación histórica sobre {reales} anomalías sintéticas: {tp} detectadas, {fn} no detectadas "
             f"y {fp} falsas alertas. ")
    if ausentes:
        aviso += f"Sin casos para evaluar: {', '.join(ausentes)}. "
    aviso += "Una alerta requiere revisión; no confirma fraude ni garantiza rendimiento en datos reales."
    st.info(aviso)

    with st.sidebar:
        st.divider()
        st.subheader("Filtros de visualización")
        cuenta = st.selectbox("Cuenta", [None, *sorted(predicciones["cuenta_id"].unique())],
                              format_func=lambda valor: "Todas las cuentas" if valor is None else valor,
                              key="filtro_cuenta")
        fechas = st.date_input("Rango de fechas", value=(primera, ultima), min_value=primera,
                               max_value=ultima, format="DD/MM/YYYY", key="filtro_fechas")
        resultado = st.radio("Resultado", ["Todas", "Alertas", "Normales"], key="filtro_resultado")
        st.caption("Los filtros solo cambian la tabla y el gráfico. El modelo y sus referencias permanecen fijos.")
        st.divider()
        st.caption(f"Resultados cargados: {datos['actualizado']:%d/%m/%Y %H:%M %Z}")
        st.caption("Para incorporar cambios en los datos, usá Actualizar resultados.")

    st.subheader("Exploración de movimientos")
    st.caption("Vista filtrada del conjunto de evaluación. Cantidades independientes de los indicadores globales.")
    try:
        filtradas = filtrar_transacciones(predicciones, cuenta, fechas, resultado)
    except ValueError as error:
        st.info(str(error))
        return
    cantidad_alertas = int(filtradas["prediccion_anomalia"].sum())
    grafico, conteos = st.columns([3, 1])
    with conteos:
        st.metric("Movimientos visibles", str(len(filtradas)))
        st.metric("Alertas visibles", str(cantidad_alertas))
    if filtradas.empty:
        st.info("No hay movimientos para esta combinación de filtros. Probá otra cuenta, fechas o resultado.")
        return
    with grafico:
        conteo = pd.DataFrame({"Normales": [len(filtradas) - cantidad_alertas], "Alertas": [cantidad_alertas]},
                              index=pd.Index(["Vista filtrada"], name="Selección"))
        st.bar_chart(conteo, y=["Normales", "Alertas"], color=["#0f766e", "#d97706"],
                     horizontal=True, stack=False, height=180, width="stretch",
                     x_label="Movimientos", alt="Cantidad de operaciones normales y alertas de la vista filtrada.")
    st.caption(f"Puntaje mayor = operación más inusual · Alerta si puntaje > {datos['umbral']:.2f} · El puntaje no es una probabilidad")
    tabla = preparar_tabla(filtradas)
    estilo = tabla.style.map(lambda estado: "background-color: #fff3d6; color: #8a4b08; font-weight: 600;"
                            if estado == "Alerta potencial" else "color: #526176;", subset=["estado"])
    st.dataframe(estilo, hide_index=True, width="stretch", height=410, row_height=36,
                 column_config={
                     "transaccion_id": st.column_config.TextColumn("Transacción"),
                     "cuenta_id": st.column_config.TextColumn("Cuenta"),
                     "fecha_hora": st.column_config.DatetimeColumn("Fecha y hora", format="DD/MM/YYYY HH:mm:ss"),
                     "estado": st.column_config.TextColumn("Resultado", width="medium"),
                     "monto": st.column_config.NumberColumn("Monto (ARS)", format="$ %.2f"),
                     "puntaje_anomalia": st.column_config.NumberColumn("Puntaje", format="%.4f"),
                     "monto_relativo": st.column_config.NumberColumn("Monto relativo", format="%.2f"),
                     "rareza_horaria": st.column_config.NumberColumn("Rareza horaria", format="%.2f"),
                     "movimientos_5min": st.column_config.NumberColumn("Movimientos / 5 min", format="%d"),
                 })
    with st.expander("Cómo interpretar esta evaluación"):
        st.markdown(
            "**Precisión:** cuántas alertas coinciden con las etiquetas. **Recall:** cuántas anomalías "
            "etiquetadas fueron detectadas. **F1:** equilibrio entre ambas. Las etiquetas se consultan "
            "después de predecir y se usan únicamente para esta evaluación histórica."
        )
        st.caption(f"Matriz global: normales bien identificadas {tn}, falsos positivos {fp}, "
                   f"falsos negativos {fn}, anomalías detectadas {tp}.")
        st.dataframe(datos["por_tipo"].rename(columns={"tipo_anomalia": "Tipo real", "reales": "Casos",
                                                      "detectadas": "Detectadas", "no_detectadas": "Omitidas",
                                                      "recall": "Recall"}), hide_index=True, width="stretch",
                     column_config={"Recall": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Sin casos de un tipo, su recall no es evaluable. Un único corte temporal y pocos casos "
                   "no permiten asegurar generalización. Las anomalías en entrenamiento pueden influir en "
                   "las referencias y el modelo; no se ajustaron parámetros con las etiquetas.")


if __name__ == "__main__":
    main()
