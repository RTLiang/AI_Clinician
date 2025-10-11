WITH sampled AS (
    SELECT patientunitstayid
    FROM ai_clinician.eicu_hourly_features
    GROUP BY patientunitstayid
    ORDER BY patientunitstayid
    LIMIT 5000
)
SELECT *
FROM ai_clinician.eicu_hourly_features
WHERE patientunitstayid IN (SELECT patientunitstayid FROM sampled)
  AND bloc <= 24
ORDER BY patientunitstayid, bloc;
