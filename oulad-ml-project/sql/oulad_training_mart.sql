-- Version 1: enrollment-level OULAD mart. The only bound parameter is cutoff_day.
-- Neon is attached by the adapter as the read-only `oulad` catalog.
WITH parameters AS (
    SELECT ?::INTEGER AS cutoff_day
),
cohort AS (
    SELECT
        ec.id_estudiante AS id_student,
        ec.cd_curso AS code_module,
        ec.cd_semestre AS code_presentation,
        e.genero AS gender,
        e.region,
        e.nivel_educativo AS highest_education,
        e.imd_band,
        e.tiene_discapacidad AS disability,
        e.grupo_edad_inicio AS age_band,
        ec.cant_intentos AS num_of_prev_attempts,
        ec.cant_creditos AS studied_credits,
        ec.fecha_registro_dias AS date_registration,
        c.duracion_dias AS course_duration_days
    FROM oulad.public.estudiante_curso AS ec
    JOIN oulad.public.estudiante AS e ON e.id_estudiante = ec.id_estudiante
    JOIN oulad.public.curso AS c
        ON c.cd_curso = ec.cd_curso AND c.cd_semestre = ec.cd_semestre
    -- An unknown registration date cannot prove the student was available at cutoff.
    WHERE ec.fecha_registro_dias IS NOT NULL
      AND ec.fecha_registro_dias <= (SELECT cutoff_day FROM parameters)
),
vle_features AS (
    SELECT
        er.id_estudiante AS id_student,
        er.cd_curso AS code_module,
        er.cd_semestre AS code_presentation,
        SUM(er.cant_clicks) AS total_clicks,
        COUNT(DISTINCT er.fecha_interaccion) AS active_days,
        COUNT(*) AS vle_events,
        COUNT(DISTINCT er.id_recurso) AS vle_sites
    FROM oulad.public.estudiante_recurso AS er
    CROSS JOIN parameters AS p
    WHERE er.fecha_interaccion <= p.cutoff_day
    GROUP BY er.id_estudiante, er.cd_curso, er.cd_semestre
),
assessment_targets AS (
    SELECT
        ec.id_estudiante AS id_student,
        ec.cd_curso AS code_module,
        ec.cd_semestre AS code_presentation,
        SUM(ae.puntuacion * a.peso_evaluacion) / NULLIF(SUM(a.peso_evaluacion), 0)
            AS weighted_assessment_score
    FROM oulad.public.estudiante_curso AS ec
    JOIN oulad.public.asignaciones AS a
        ON a.cd_curso = ec.cd_curso AND a.cd_semestre = ec.cd_semestre
    JOIN oulad.public.asignacion_estudiante AS ae
        ON ae.id_asignacion = a.id_asignacion AND ae.id_estudiante = ec.id_estudiante
    WHERE ae.puntuacion IS NOT NULL
    GROUP BY ec.id_estudiante, ec.cd_curso, ec.cd_semestre
),
outcomes AS (
    SELECT
        ec.id_estudiante AS id_student,
        ec.cd_curso AS code_module,
        ec.cd_semestre AS code_presentation,
        CASE WHEN UPPER(ec.resultado_final) IN ('PASS', 'DISTINCTION') THEN 1 ELSE 0 END AS passed,
        CASE UPPER(ec.resultado_final)
            WHEN 'FAIL' THEN 0
            WHEN 'WITHDRAWN' THEN 1
            WHEN 'PASS' THEN 2
            WHEN 'DISTINCTION' THEN 3
        END AS performance_tier
    FROM oulad.public.estudiante_curso AS ec
)
SELECT
    cohort.*,
    COALESCE(vle_features.total_clicks, 0) AS total_clicks,
    COALESCE(vle_features.active_days, 0) AS active_days,
    COALESCE(vle_features.vle_events, 0) AS vle_events,
    COALESCE(vle_features.vle_sites, 0) AS vle_sites,
    CASE WHEN vle_features.id_student IS NULL THEN 0 ELSE 1 END AS has_vle_activity,
    outcomes.passed,
    outcomes.performance_tier,
    assessment_targets.weighted_assessment_score
FROM cohort
JOIN outcomes USING (id_student, code_module, code_presentation)
LEFT JOIN vle_features USING (id_student, code_module, code_presentation)
LEFT JOIN assessment_targets USING (id_student, code_module, code_presentation)
ORDER BY id_student, code_module, code_presentation;
