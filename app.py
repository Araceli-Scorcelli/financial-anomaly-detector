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
    return datos[["transaccion_id", "cuenta_id", "fecha_hora", "monto", "estado", "puntaje_anomalia",
                  "monto_relativo", "rareza_horaria", "movimientos_5min"]]


def main() -> None:
    st.set_page_config(page_title="Financial Anomaly Detector", page_icon=":material/analytics:", layout="wide")
    with st.container(gap="xsmall"):
        st.caption("ANÁLISIS FINANCIERO · PROYECTO ACADÉMICO")
        st.title("Financial Anomaly Detector")
        st.markdown("**Detección de anomalías mediante aprendizaje no supervisado.**")
        st.caption("Explorá movimientos bancarios sintéticos, revisá las alertas de Isolation Forest "
                   "y conocé cómo se evalúan sus resultados.")

    with st.sidebar:
        st.markdown("### Panel de exploración")
        st.caption("Una mirada a lo inusual en los datos.")
        actualizar = st.button("Actualizar resultados", key="actualizar_resultados", width="stretch",
                               type="primary", icon=":material/refresh:",
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
    st.caption(f"Del {primera:%d/%m/%Y} al {ultima:%d/%m/%Y} · Evaluación completa, independiente de los filtros.")
    indicadores = [
        ("Transacciones evaluadas", str(len(predicciones)), "Operaciones del bloque temporal posterior al entrenamiento."),
        ("Alertas generadas", str(int(predicciones["prediccion_anomalia"].sum())), "Operaciones que superan el umbral del detector."),
        ("Anomalías reales", str(reales), "Etiquetas sintéticas del período de evaluación; no son fraudes confirmados."),
        ("Precisión", f"{datos['precision']:.2%}".replace(".", ","), "Proporción de alertas que coinciden con una anomalía etiquetada."),
        ("Recall", f"{datos['recall']:.2%}".replace(".", ","), "Proporción de anomalías etiquetadas que el modelo detectó."),
        ("F1-score", f"{datos['f1']:.2%}".replace(".", ","), "Media armónica de precisión y recall."),
    ]
    with st.grid(3, min_column_width=260, gap="small", key="indicadores_globales"):
        for etiqueta, valor, ayuda in indicadores:
            # Markdown nativo: Bordeaux del tema, sin cambiar el valor calculado.
            st.metric(etiqueta, f":red[{valor}]", help=ayuda, border=True, height="stretch")
    ausentes = datos["por_tipo"].loc[datos["por_tipo"]["reales"] == 0, "tipo_anomalia"].tolist()
    aviso = (f"Evaluación histórica sobre {reales} anomalías sintéticas: {tp} detectadas, {fn} no detectadas "
             f"y {fp} falsas alertas. ")
    if ausentes:
        aviso += f"Sin casos para evaluar: {', '.join(ausentes)}. "
    aviso += "Una anomalía no implica fraude. Estas cifras describen datos sintéticos."
    st.info(aviso, icon=":material/info:")

    with st.sidebar:
        st.space("small")
        st.markdown("### Filtros de visualización")
        st.caption("Elegí qué movimientos querés explorar.")
        cuenta = st.selectbox("Cuenta", [None, *sorted(predicciones["cuenta_id"].unique())],
                              format_func=lambda valor: "Todas las cuentas" if valor is None else valor,
                              key="filtro_cuenta")
        fechas = st.date_input("Rango de fechas", value=(primera, ultima), min_value=primera,
                               max_value=ultima, format="DD/MM/YYYY", key="filtro_fechas")
        resultado = st.radio("Resultado", ["Todas", "Alertas", "Normales"], key="filtro_resultado",
                             help="Normales significa sin alerta del modelo; no es una validación de la operación.")
        st.caption("Los filtros cambian la vista, no la evaluación global. No vuelven a entrenar el modelo.")
        st.space("small")
        st.badge("PostgreSQL · solo lectura", icon=":material/database:", color="blue")
        st.caption(f"Resultados cargados: {datos['actualizado']:%d/%m/%Y %H:%M %Z}")
        st.caption("Para incorporar cambios en los datos, usá Actualizar resultados.")

    st.subheader("Exploración de movimientos")
    st.caption("Usá los filtros del panel lateral para recorrer el conjunto de evaluación.")
    try:
        filtradas = filtrar_transacciones(predicciones, cuenta, fechas, resultado)
    except ValueError as error:
        st.info(str(error))
        return
    cantidad_alertas = int(filtradas["prediccion_anomalia"].sum())
    with st.container(border=True, key="resumen_seleccion"):
        st.markdown("**Distribución de los movimientos seleccionados**")
        grafico, conteos = st.columns([3, 2], vertical_alignment="center", gap="small")
        with conteos:
            with st.grid(2, min_column_width=150, gap="small"):
                st.metric("Movimientos visibles", str(len(filtradas)))
                st.metric("Alertas visibles", str(cantidad_alertas))
        if not filtradas.empty:
            with grafico:
                conteo = pd.DataFrame({"Normales": [len(filtradas) - cantidad_alertas], "Alertas": [cantidad_alertas]},
                                      index=pd.Index(["Vista filtrada"], name="Selección"))
                st.bar_chart(conteo, y=["Normales", "Alertas"],
                             color=st.get_option("theme.chartCategoricalColors")[:2],
                             horizontal=True, stack=False, height=190, width="stretch",
                             x_label="", y_label="Movimientos",
                             alt="Cantidad de operaciones normales y alertas de la vista filtrada.")
        st.caption("Aqua: operaciones sin alerta · Coral: alertas potenciales para revisar.")
    if filtradas.empty:
        st.info("No hay movimientos para esta combinación de filtros. Probá otra cuenta, fechas o resultado.")
        return
    st.markdown("#### Detalle de transacciones")
    st.caption(f"Ordenadas de mayor a menor puntaje · Alerta si puntaje > {datos['umbral']:.2f} · "
               "El puntaje no representa una probabilidad de fraude.")
    mostrar_caracteristicas = st.toggle(
        "Mostrar características técnicas", key="mostrar_caracteristicas",
        help="Agrega monto relativo, rareza horaria y movimientos en cinco minutos al final de la tabla. No recalcula el modelo.",
    )
    tabla = preparar_tabla(filtradas)
    if not mostrar_caracteristicas:
        tabla = tabla.iloc[:, :6]
    estilo = tabla.style.map(lambda estado: "background-color: #FBE4D8; color: #361E1C; font-weight: 600;"
                            if estado == "Alerta potencial" else "background-color: #E4EEEA; color: #11363E;",
                            subset=["estado"])
    st.dataframe(estilo, hide_index=True, width="stretch", height=410, row_height=38,
                 alt="Transacciones del conjunto de evaluación, ordenadas por puntaje de anomalía, con su resultado.",
                 column_config={
                     "transaccion_id": st.column_config.TextColumn("Transacción", width=110, pinned=True),
                     "cuenta_id": st.column_config.TextColumn("Cuenta", width=90),
                     "fecha_hora": st.column_config.DatetimeColumn("Fecha y hora", format="DD/MM/YYYY HH:mm:ss", width=185),
                     "monto": st.column_config.NumberColumn("Importe (ARS)", format="ARS %,.2f", width=155),
                     "estado": st.column_config.TextColumn("Resultado", width=165,
                                                          help="Decisión del detector: alerta potencial o sin alerta. No es una confirmación de fraude."),
                     "puntaje_anomalia": st.column_config.NumberColumn("Puntaje", format="%.4f", width=100,
                                                                      help="Mayor puntaje indica mayor rareza. Se muestran cuatro decimales; la decisión utiliza la precisión completa."),
                     "monto_relativo": st.column_config.NumberColumn("Monto relativo", format="%.2f",
                                                                    help="Importe dividido por la mediana de referencia estimada con entrenamiento."),
                     "rareza_horaria": st.column_config.NumberColumn("Rareza horaria", format="%.2f",
                                                                    help="Un valor mayor describe un horario menos frecuente en la referencia."),
                     "movimientos_5min": st.column_config.NumberColumn("Movimientos / 5 min", format="%d",
                                                                       help="Operaciones de la cuenta en los cinco minutos anteriores, incluida la actual."),
                 })
    with st.expander("Cómo leer las características", icon=":material/query_stats:"):
        st.markdown(
            "- **Monto relativo:** compara el importe con su mediana de referencia. "
            "Un valor de 1 equivale a esa mediana; 3 significa tres veces ese importe.\n"
            "- **Rareza horaria:** aumenta cuanto menos frecuente es la hora en los movimientos de referencia.\n"
            "- **Movimientos en cinco minutos:** cuenta las operaciones de la misma cuenta durante "
            "los cinco minutos anteriores, incluida la actual. Un valor mayor describe más concentración."
        )
        st.caption("Estas características describen el movimiento. No son una explicación causal de cada alerta "
                   "ni permiten atribuirla a una variable específica. Isolation Forest combina las tres "
                   "para asignar un puntaje; una alerta potencial no confirma fraude.")
    with st.expander("Cómo interpretar esta evaluación", icon=":material/school:"):
        st.markdown(
            "**Precisión:** cuántas alertas coinciden con las etiquetas. **Recall:** cuántas anomalías "
            "etiquetadas fueron detectadas. **F1:** equilibrio entre ambas. Las etiquetas se consultan "
            "después de predecir y se usan únicamente para evaluar, no para entrenar."
        )
        st.caption(f"Matriz global: normales bien identificadas {tn}, falsos positivos {fp}, "
                   f"falsos negativos {fn}, anomalías detectadas {tp}.")
        st.dataframe(datos["por_tipo"].rename(columns={"tipo_anomalia": "Tipo real", "reales": "Casos",
                                                      "detectadas": "Detectadas", "no_detectadas": "Omitidas",
                                                      "recall": "Recall"}), hide_index=True, width="stretch",
                     alt="Resultados históricos de evaluación por tipo real de anomalía, con su cantidad de casos.",
                     column_config={"Recall": st.column_config.NumberColumn(format="%.2f")})
        st.caption("Sin casos de un tipo, su recall no es evaluable. Un único corte temporal y pocos casos "
                   "no permiten asegurar generalización. Las anomalías en entrenamiento pueden influir en "
                   "las referencias y el modelo; no se ajustaron parámetros con las etiquetas.")


if __name__ == "__main__":
    main()
