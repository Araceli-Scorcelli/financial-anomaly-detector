-- CONSULTAS EXPLORATORIAS: transacciones en pesos argentinos (ARS).
-- Antes de empezar en pgAdmin:
-- 1. Seleccionar la base financial_anomalies, ya creada manualmente.
-- 2. Abrir Query Tool sobre esa base y cargar este archivo SQL.
-- 3. Haber creado public.transacciones e importado el CSV de transacciones.
-- 4. Seleccionar y ejecutar UNA consulta a la vez, hasta su punto y coma.
-- Este archivo solo lee datos: no crea ni modifica la base, la tabla o el CSV.
-- Las fechas representan hora local de Argentina, sin conversión de zona horaria.
-- Los montos son positivos para ingresos y egresos: sumarlos mide volumen,
-- no el saldo de una cuenta. Aquí no se clasifican anomalías ni se usan etiquetas.


-- 1. Resumen general y primera comprobación de la importación.
-- Se esperan 1500 movimientos y 30 cuentas, entre enero y marzo de 2026.
SELECT
    COUNT(*) AS cantidad_movimientos,
    COUNT(DISTINCT cuenta_id) AS cantidad_cuentas,
    MIN(fecha_hora) AS primera_fecha_hora,
    MAX(fecha_hora) AS ultima_fecha_hora,
    ROUND(AVG(monto), 2) AS monto_promedio_ars,
    MAX(monto) AS monto_maximo_ars,
    SUM(monto) AS volumen_total_ars
FROM public.transacciones;


-- 2. Cantidad de movimientos por tipo, dirección y canal.
-- La columna dimension indica qué característica se está contando.
SELECT dimension, valor, cantidad_movimientos
FROM (
    SELECT 'tipo_movimiento' AS dimension, tipo_movimiento AS valor,
           COUNT(*) AS cantidad_movimientos
    FROM public.transacciones
    GROUP BY tipo_movimiento
    UNION ALL
    SELECT 'direccion', direccion, COUNT(*)
    FROM public.transacciones
    GROUP BY direccion
    UNION ALL
    SELECT 'canal', canal, COUNT(*)
    FROM public.transacciones
    GROUP BY canal
) AS distribuciones
ORDER BY dimension, cantidad_movimientos DESC, valor;


-- 3. Actividad e importes por cuenta.
-- El volumen suma ingresos y egresos positivos; no representa un saldo.
SELECT
    cuenta_id,
    COUNT(*) AS cantidad_movimientos,
    COUNT(*) FILTER (WHERE direccion = 'ingreso') AS cantidad_ingresos,
    COUNT(*) FILTER (WHERE direccion = 'egreso') AS cantidad_egresos,
    ROUND(AVG(monto), 2) AS monto_promedio_ars,
    MAX(monto) AS monto_maximo_ars,
    SUM(monto) AS volumen_total_ars
FROM public.transacciones
GROUP BY cuenta_id
ORDER BY cantidad_movimientos DESC, cuenta_id;


-- 4. Cantidad de movimientos por mes para explorar el trimestre.
-- Cada mes se muestra con la fecha de su primer día.
SELECT
    DATE_TRUNC('month', fecha_hora)::date AS mes,
    COUNT(*) AS cantidad_movimientos
FROM public.transacciones
GROUP BY DATE_TRUNC('month', fecha_hora)::date
ORDER BY mes;


-- 5. Cantidad de movimientos por fecha, sin separar las horas.
-- Solo aparecen las fechas que tienen al menos un movimiento.
SELECT
    fecha_hora::date AS fecha,
    COUNT(*) AS cantidad_movimientos
FROM public.transacciones
GROUP BY fecha_hora::date
ORDER BY fecha;


-- 6. Distribución por hora local, de 00 a 23, incluyendo horas sin movimientos.
-- generate_series genera las 24 horas; LEFT JOIN conserva las horas vacías.
-- COUNT(t.transaccion_id) devuelve cero en esas horas; COUNT(*) contaría la fila vacía.
SELECT
    horas.hora,
    COUNT(t.transaccion_id) AS cantidad_movimientos
FROM generate_series(0, 23) AS horas(hora)
LEFT JOIN public.transacciones AS t
    ON EXTRACT(HOUR FROM t.fecha_hora) = horas.hora
GROUP BY horas.hora
ORDER BY horas.hora;


-- 7. Distribución por hora local dentro de cada cuenta.
-- Solo aparecen las combinaciones de cuenta y hora que tienen movimientos.
SELECT
    cuenta_id,
    EXTRACT(HOUR FROM fecha_hora)::integer AS hora,
    COUNT(*) AS cantidad_movimientos
FROM public.transacciones
GROUP BY cuenta_id, EXTRACT(HOUR FROM fecha_hora)::integer
ORDER BY cuenta_id, hora;


-- 8. Los 10 movimientos de mayor importe, para observar sus detalles.
-- Ver un importe alto no implica que sea una anomalía.
-- cuenta_id y transaccion_id resuelven los empates de importe de forma estable.
SELECT
    transaccion_id,
    cuenta_id,
    fecha_hora,
    tipo_movimiento,
    direccion,
    monto AS monto_ars,
    canal
FROM public.transacciones
ORDER BY monto DESC, cuenta_id, transaccion_id
LIMIT 10;
