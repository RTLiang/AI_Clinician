# Repository Guidelines

## Project Structure & Module Organization
- Core pipeline: `AIClinician_core_160219.m` orchestrates cohort prep, policy learning, and evaluation; keep helper logic near call sites.
- Cohorts: `AIClinician_sepsis3_def_160219.m` and `AIClinician_mimic3_dataset_160219.m` define features; update these before downstream scripts.
- Data extraction: `AIClinician_Data_extract_MIMIC3_140219.ipynb` (restart kernel before executing).
- Utilities at repo root: `SAH.m`, `fixgaps.m`, `fastknnsearch.m`.
- MDP changes live only in `MDPtoolbox/` to keep merges predictable.

## Build, Test, and Development Commands
- Run full pipeline: `matlab -batch "run('AIClinician_core_160219.m')"` (ensure `MDPtoolbox/` is on the MATLAB path).
- Regenerate cohort extracts: `jupyter nbconvert --to notebook --execute AIClinician_Data_extract_MIMIC3_140219.ipynb`.
- Refresh evaluation: `matlab -batch "run('offpolicy_eval_tdlearning.m')"` and `matlab -batch "run('offpolicy_eval_wis.m')"`; capture seeds and metric deltas in logs.
- Style checks: `matlab -batch "checkcode('file.m')"` before opening a PR.

## Coding Style & Naming Conventions
- MATLAB: 4-space indentation; favor vectorized operations; preallocate with `NaN(...)` or `zeros(...)` in the core script style.
- Filenames: `AIClinician_<target>_<YYMMDD>.m` for scripts.
- Tables: snake_case variable names for downstream compatibility.
- Comments: brief block comments above non-obvious logic.

## Testing Guidelines
- Before/after cohort changes, diff counts for patients, ICU stays, and interventions; flag anomalies inline in the corresponding script.
- When adjusting policies, rerun the core on a reduced cohort and record `recqvi`, mortality deltas, and WIS/TIS.
- Restart notebooks prior to execution so stored outputs reflect current code.

## Commit & Pull Request Guidelines
- Commits: single-focus, imperative (e.g., `Tighten reward transition smoothing`), and reference the affected dataset or script.
- PRs: include motivation, methodology, executed commands, and behavioral shifts (tables/plots preferred). Link issues, highlight metric changes, and call out any PHI dependencies.

## Security & Data Handling
- Do not commit raw MIMIC-III or eICU exports.
- Keep credentials and PHI out of version control (use env vars or MATLAB preferences).
- Share only de-identified aggregates and ensure exported artifacts comply with source data licenses.

