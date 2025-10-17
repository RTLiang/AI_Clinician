# Repository Guidelines

## Project Structure & Module Organization
- Core pipeline: `AIClinician_core_160219.m` orchestrates cohort prep, policy learning, and evaluation. Keep helper logic close to call sites.
- Cohorts: `AIClinician_sepsis3_def_160219.m` and `AIClinician_mimic3_dataset_160219.m` define features; update these before downstream scripts.
- Data extraction: `AIClinician_Data_extract_MIMIC3_140219.ipynb` (restart kernel before execution).
- Utilities: `SAH.m`, `fixgaps.m`, `fastknnsearch.m` at repo root.
- MDP code: changes live only in `MDPtoolbox/`. Ensure it is on the MATLAB path, e.g., `addpath(genpath('MDPtoolbox')); savepath`.

## Build, Test, and Development Commands
- Run full pipeline: `matlab -batch "run('AIClinician_core_160219.m')"` — end-to-end cohort→policy→evaluation.
- Regenerate cohorts: `jupyter nbconvert --to notebook --execute AIClinician_Data_extract_MIMIC3_140219.ipynb` — fresh extracts with a clean kernel.
- Refresh evaluation: `matlab -batch "run('offpolicy_eval_tdlearning.m')"` and `matlab -batch "run('offpolicy_eval_wis.m')"` — record seeds and metric deltas.
- Style check: `matlab -batch "checkcode('file.m')"` — static analysis before PRs.

## Coding Style & Naming Conventions
- MATLAB: 4-space indentation; prefer vectorized operations; preallocate with `NaN(...)`/`zeros(...)` following the core script style.
- Script names: `AIClinician_<target>_<YYMMDD>.m`.
- Tables: snake_case variable names for downstream compatibility.
- Comments: short block comments above non-obvious logic.

## Testing Guidelines
- No formal unit suite; validate behavior via data checks.
- Before/after cohort changes: diff counts for patients, ICU stays, and interventions; flag anomalies inline in the script.
- Policy edits: rerun the core on a reduced cohort and log `recqvi`, mortality deltas, and WIS/TIS.
- Notebooks: restart kernels prior to execution to avoid stale outputs.

## Commit & Pull Request Guidelines
- Commits: single-focus, imperative (e.g., `Tighten reward transition smoothing`), referencing the affected dataset or script.
- PRs: include motivation, methodology, commands executed, and behavioral shifts (tables/plots preferred). Link issues, highlight metric changes, and call out any PHI dependencies.

## Security & Data Handling
- Never commit raw MIMIC-III/eICU exports.
- Keep credentials/PHI out of VCS (use env vars or MATLAB preferences).
- Share only de-identified aggregates; ensure compliance with source data licenses.
