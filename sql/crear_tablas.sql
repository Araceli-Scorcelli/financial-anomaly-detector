-- Financial Anomaly Detector: estructura para el dataset de enero-marzo de 2026.
-- Referencia: data/diccionario_datos.md. Todos los importes están en ARS.
--
-- Ejecución manual en pgAdmin:
-- 1. Seleccionar la base ya existente financial_anomalies y abrir Query Tool.
-- 2. Abrir este archivo y ejecutar el contenido completo una sola vez.
-- 3. Actualizar Schemas > public > Tables para ver public.transacciones.
-- El archivo crea una tabla nueva; si ya existe, PostgreSQL informará el error.
--
-- Importación manual posterior: clic derecho en public.transacciones >
-- Import/Export Data > Import. Seleccionar data/transacciones.csv y configurar:
-- Format: csv; Encoding: UTF8; Header: Yes; Delimiter: ,; Quote: "; Escape: ".
-- Dejar NULL String vacío y seleccionar las ocho columnas en el orden siguiente:
-- transaccion_id, cuenta_id, fecha_hora, tipo_movimiento, direccion, monto,
-- canal, contraparte_id. El encabezado se omite; los valores se asignan por orden.
-- Las fechas ISO YYYY-MM-DD HH:MM:SS y el punto decimal ya están preparados.
-- Importar una sola vez: la clave primaria rechaza identificadores repetidos.
-- Las etiquetas de evaluación permanecen en su archivo separado.

CREATE TABLE public.transacciones (
    -- Identificador del CSV; no se genera una secuencia ni una columna adicional.
    transaccion_id TEXT NOT NULL,
    cuenta_id TEXT NOT NULL,
    -- Hora local de Argentina, tal como aparece en el CSV, sin conversión a UTC.
    fecha_hora TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    tipo_movimiento TEXT NOT NULL,
    direccion TEXT NOT NULL,
    -- Decimal exacto con dos posiciones para los centavos; el signo va en direccion.
    monto NUMERIC(16, 2) NOT NULL,
    canal TEXT NOT NULL,
    contraparte_id TEXT NOT NULL,

    CONSTRAINT pk_transacciones PRIMARY KEY (transaccion_id),

    -- Los patrones también rechazan cadenas vacías o con espacios adicionales.
    CONSTRAINT chk_transaccion_id_formato
        CHECK (transaccion_id ~ '^TX[0-9]{6}$'),
    -- Las 30 cuentas ficticias del dataset son CTA001 a CTA030.
    CONSTRAINT chk_cuenta_id_formato
        CHECK (cuenta_id ~ '^CTA(00[1-9]|0[12][0-9]|030)$'),
    CONSTRAINT chk_contraparte_id_formato
        CHECK (contraparte_id ~ '^CP[0-9]{4}$'),

    -- Límite superior exclusivo: incluye todo el 31 de marzo, con cualquier hora.
    CONSTRAINT chk_fecha_hora_periodo
        CHECK (
            fecha_hora >= TIMESTAMP '2026-01-01 00:00:00'
            AND fecha_hora < TIMESTAMP '2026-04-01 00:00:00'
        ),

    -- NaN es un valor especial de NUMERIC; se excluye además de cero y negativos.
    CONSTRAINT chk_monto_positivo
        CHECK (monto > 0 AND monto <> 'NaN'::NUMERIC),

    -- Este CHECK define los valores permitidos y sus combinaciones coherentes.
    -- Transferencias: ingreso/egreso por homebanking; pagos: egreso; cobros: ingreso.
    CONSTRAINT chk_movimiento_direccion_canal
        CHECK (
            (tipo_movimiento, direccion, canal) IN (
                ('transferencia', 'ingreso', 'homebanking'),
                ('transferencia', 'egreso', 'homebanking'),
                ('pago_tarjeta', 'egreso', 'pos'),
                ('cobro_tarjeta', 'ingreso', 'pos'),
                ('pago_servicio', 'egreso', 'homebanking'),
                ('retiro_efectivo', 'egreso', 'cajero'),
                ('deposito_efectivo', 'ingreso', 'cajero')
            )
        )
);

-- cuenta_id y contraparte_id se repiten intencionalmente y no son claves únicas.
-- En esta etapa no existen otras tablas a las que declarar claves foráneas.
-- Los límites de cuentas y fechas corresponden a este dataset concreto.
-- Se admiten montos altos, horarios atípicos y ráfagas: son datos válidos para
-- explorar anomalías. Las 1500 filas y 30 cuentas se comprueban después de la
-- importación mediante las consultas, no mediante restricciones por fila.
