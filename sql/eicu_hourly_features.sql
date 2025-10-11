-- Constructs the eICU hourly feature table expected by AIClinician_core_160219.m
-- The query aggregates raw eICU tables into 4-hour blocs per patient.
-- Review the CASE/FILTER clauses to ensure the labname mappings match your local dataset.

CREATE SCHEMA IF NOT EXISTS ai_clinician;

DROP MATERIALIZED VIEW IF EXISTS ai_clinician.eicu_hourly_features;

CREATE MATERIALIZED VIEW ai_clinician.eicu_hourly_features AS
WITH patient_base AS (
    SELECT
        p.patientunitstayid,
        -- gender encoded as 0/1 like the MIMIC dataset (female -> 0, male -> 1)
        CASE LOWER(p.gender)
            WHEN 'female' THEN 0
            WHEN 'male' THEN 1
            ELSE NULL
        END AS gender,
        -- age stored as text; convert to numeric years (use 91.4 for > 89 as in MIMIC preprocessing)
        CASE
            WHEN p.age ~ '^[0-9]+$' THEN p.age::numeric
            WHEN p.age IN ('> 89', '>89', '>= 90') THEN 91.4
            ELSE NULL
        END AS age,
        p.admissionweight::numeric AS admissionweight,
        CASE WHEN COALESCE(p.unitvisitnumber, 1) > 1 THEN 1 ELSE 0 END AS re_admission,
        CASE WHEN LOWER(p.unitdischargestatus) = 'expired' THEN 1 ELSE 0 END AS hospmortality
    FROM eicu.patient p
),

vitals AS (
    SELECT
        patientunitstayid,
        FLOOR(observationoffset / 240.0)::int + 1 AS bloc,
        AVG(heartrate) AS hr,
        AVG(systemicsystolic) AS sysbp,
        AVG(systemicmean) AS meanbp,
        AVG(systemicdiastolic) AS diabp,
        AVG(respiration) AS rr,
        AVG(temperature) AS temp_c,
        AVG(sao2) AS spo2,
        AVG(CASE WHEN systemicsystolic > 0 THEN heartrate::numeric / systemicsystolic END) AS shock_index
    FROM eicu.vitalperiodic
    GROUP BY 1, 2
),

fio2 AS (
    SELECT
        patientunitstayid,
        FLOOR(respchartoffset / 240.0)::int + 1 AS bloc,
        AVG(
            CASE
                WHEN respchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                    AND LOWER(respchartvaluelabel) IN ('fio2', 'fio2 (%)', 'fio2 percent', 'fio2 (set)')
                THEN respchartvalue::numeric
                WHEN respchartvalue LIKE '%"%' -- values recorded as 0.5" etc.
                THEN NULL
                ELSE NULL
            END
        ) AS fio2both
    FROM eicu.respiratorycharting
    GROUP BY 1, 2
),

mechvent AS (
    SELECT
        patientunitstayid,
        generate_series(
            GREATEST(0, FLOOR(COALESCE(ventstartoffset, 0) / 240.0)::int + 1),
            GREATEST(0, FLOOR(COALESCE(NULLIF(ventendoffset, 0), ventstartoffset) / 240.0)::int + 1)
        ) AS bloc
    FROM eicu.respiratorycare
    WHERE ventstartoffset IS NOT NULL
),

-- vasopressors (norepinephrine, epinephrine, vasopressin etc.). Adjust drugname matching as required.
vaso AS (
    SELECT
        patientunitstayid,
        FLOOR(infusionoffset / 240.0)::int + 1 AS bloc,
        MAX(
            CASE
                WHEN LOWER(drugname) SIMILAR TO '%(norepinephrine|epinephrine|phenylephrine|vasopressin)%'
                    AND infusionrate ~ '^[0-9]+(\.[0-9]+)?$'
                THEN infusionrate::numeric
            END
        ) AS max_dose_vaso
    FROM eicu.infusiondrug
    GROUP BY 1, 2
),

-- intake/output per 4h window
fluids AS (
    SELECT
        patientunitstayid,
        FLOOR(intakeoutputoffset / 240.0)::int + 1 AS bloc,
        SUM(COALESCE(intaketotal, 0)) AS input_4hourly_tev,
        SUM(COALESCE(outputtotal, 0)) AS output_4hourly
    FROM eicu.intakeoutput
    GROUP BY 1, 2
),

fluids_with_cum AS (
    SELECT
        patientunitstayid,
        bloc,
        input_4hourly_tev,
        output_4hourly,
        SUM(input_4hourly_tev) OVER (PARTITION BY patientunitstayid ORDER BY bloc
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS input_total_tev,
        SUM(output_4hourly) OVER (PARTITION BY patientunitstayid ORDER BY bloc
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS output_total,
        SUM(input_4hourly_tev - output_4hourly) OVER (PARTITION BY patientunitstayid ORDER BY bloc
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulated_balance_tev
    FROM fluids
),

-- Glasgow Coma Scale
nurse_gcs AS (
    SELECT
        patientunitstayid,
        FLOOR(nursingchartoffset / 240.0)::int + 1 AS bloc,
        AVG(
            CASE
                WHEN LOWER(nursingchartcelltypevallabel) = 'gcs'
                     AND nursingchartvalue ~ '^[0-9]+(\.[0-9]+)?$'
                THEN nursingchartvalue::numeric
            END
        ) AS gcs
    FROM eicu.nursecharting
    GROUP BY 1, 2
),

-- simple SOFA/SIRS proxies; adjust mappings as required
apache AS (
    SELECT
        patientunitstayid,
        MAX(CASE WHEN lower(apacheversion) IN ('iv', 'ivb', 'custom') THEN apachescore END) AS sofa,
        MAX(CASE WHEN lower(apacheversion) IN ('iv', 'ivb', 'custom') THEN acutephysiologyscore END) AS sirs
    FROM eicu.apachepatientresult
    GROUP BY 1
),

-- laboratory values grouped by 4h bloc
labs AS (
    SELECT
        patientunitstayid,
        FLOOR(labresultoffset / 240.0)::int + 1 AS bloc,
        MAX(labresult) FILTER (WHERE LOWER(labname) IN ('potassium', 'potassium, serum', 'potassium, whole blood')) AS potassium,
        MAX(labresult) FILTER (WHERE LOWER(labname) IN ('sodium', 'sodium, serum', 'sodium (na)')) AS sodium,
        MAX(labresult) FILTER (WHERE LOWER(labname) IN ('chloride', 'chloride, serum')) AS chloride,
        MAX(labresult) FILTER (WHERE LOWER(labname) LIKE 'glucose%') AS glucose,
        MAX(labresult) FILTER (WHERE LOWER(labname) LIKE 'magnesium%') AS magnesium,
        MAX(labresult) FILTER (WHERE LOWER(labname) IN ('calcium', 'calcium, serum', 'calcium total')) AS calcium,
        MAX(labresult) FILTER (WHERE LOWER(labname) IN ('hemoglobin', 'hgb')) AS hb,
        MAX(labresult) FILTER (WHERE LOWER(labname) IN ('wbc', 'white blood cell count')) AS wbc_count,
        MAX(labresult) FILTER (WHERE LOWER(labname) IN ('platelets', 'platelet count')) AS platelets_count,
        MAX(labresult) FILTER (WHERE LOWER(labname) = 'ptt') AS ptt,
        MAX(labresult) FILTER (WHERE LOWER(labname) = 'pt') AS pt,
        MAX(labresult) FILTER (WHERE LOWER(labname) = 'inr') AS inr,
        MAX(labresult) FILTER (WHERE LOWER(labname) IN ('ph (arterial)', 'arterial ph', 'ph arterial')) AS arterial_ph,
        MAX(labresult) FILTER (WHERE LOWER(labname) IN ('pao2', 'pa o2')) AS pao2,
        MAX(labresult) FILTER (WHERE LOWER(labname) IN ('paco2', 'pa co2')) AS paco2,
        MAX(labresult) FILTER (WHERE LOWER(labname) IN ('base excess', 'base excess, arterial')) AS arterial_be,
        MAX(labresult) FILTER (WHERE LOWER(labname) IN ('hco3', 'bicarbonate')) AS hco3,
        MAX(labresult) FILTER (WHERE LOWER(labname) LIKE 'lactate%') AS arterial_lactate,
        MAX(labresult) FILTER (WHERE LOWER(labname) = 'bun') AS bun,
        MAX(labresult) FILTER (WHERE LOWER(labname) = 'creatinine') AS creatinine,
        MAX(labresult) FILTER (WHERE LOWER(labname) IN ('sgot', 'ast')) AS sgot,
        MAX(labresult) FILTER (WHERE LOWER(labname) IN ('sgpt', 'alt')) AS sgpt,
        MAX(labresult) FILTER (WHERE LOWER(labname) IN ('total bilirubin', 'bilirubin, total')) AS total_bili
    FROM eicu.lab
    GROUP BY 1, 2
),

all_blocs AS (
    SELECT DISTINCT patientunitstayid, bloc FROM vitals
    UNION
    SELECT DISTINCT patientunitstayid, bloc FROM fio2
    UNION
    SELECT DISTINCT patientunitstayid, bloc FROM fluids
    UNION
    SELECT DISTINCT patientunitstayid, bloc FROM mechvent
    UNION
    SELECT DISTINCT patientunitstayid, bloc FROM labs
    UNION
    SELECT DISTINCT patientunitstayid, bloc FROM nurse_gcs
    UNION
    SELECT DISTINCT patientunitstayid, bloc FROM vaso
)
SELECT
    b.patientunitstayid,
    b.bloc,
    pb.gender,
    COALESCE(mv.mechvent, 0) AS mechvent,
    COALESCE(vso.max_dose_vaso, 0) AS max_dose_vaso,
    pb.re_admission,
    pb.age,
    pb.admissionweight,
    ng.gcs,
    vt.hr,
    vt.sysbp,
    vt.meanbp,
    vt.diabp,
    vt.rr,
    vt.temp_c,
    f2.fio2both,
    lb.potassium,
    lb.sodium,
    lb.chloride,
    lb.glucose,
    lb.magnesium,
    lb.calcium,
    lb.hb,
    lb.wbc_count,
    lb.platelets_count,
    lb.ptt,
    lb.pt,
    lb.arterial_ph,
    lb.pao2,
    lb.paco2,
    lb.arterial_be,
    lb.hco3,
    lb.arterial_lactate,
    ap.sofa,
    ap.sirs,
    vt.shock_index,
    CASE
        WHEN lb.pao2 IS NOT NULL AND f2.fio2both IS NOT NULL AND f2.fio2both > 0
        THEN lb.pao2 / (f2.fio2both / 100.0)
    END AS pao2_fio2,
    fw.cumulated_balance_tev,
    vt.spo2,
    lb.bun,
    lb.creatinine,
    lb.sgot,
    lb.sgpt,
    lb.total_bili,
    lb.inr,
    fw.input_total_tev,
    fw.input_4hourly_tev,
    fw.output_total,
    fw.output_4hourly,
    pb.hospmortality
FROM all_blocs b
JOIN patient_base pb ON pb.patientunitstayid = b.patientunitstayid
LEFT JOIN vitals vt ON vt.patientunitstayid = b.patientunitstayid AND vt.bloc = b.bloc
LEFT JOIN fio2 f2 ON f2.patientunitstayid = b.patientunitstayid AND f2.bloc = b.bloc
LEFT JOIN (
    SELECT patientunitstayid, bloc, 1 AS mechvent
    FROM mechvent
    GROUP BY 1, 2
) mv ON mv.patientunitstayid = b.patientunitstayid AND mv.bloc = b.bloc
LEFT JOIN vaso vso ON vso.patientunitstayid = b.patientunitstayid AND vso.bloc = b.bloc
LEFT JOIN fluids_with_cum fw ON fw.patientunitstayid = b.patientunitstayid AND fw.bloc = b.bloc
LEFT JOIN nurse_gcs ng ON ng.patientunitstayid = b.patientunitstayid AND ng.bloc = b.bloc
LEFT JOIN labs lb ON lb.patientunitstayid = b.patientunitstayid AND lb.bloc = b.bloc
LEFT JOIN apache ap ON ap.patientunitstayid = b.patientunitstayid;

CREATE INDEX ON ai_clinician.eicu_hourly_features (patientunitstayid, bloc);
