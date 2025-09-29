# Repository Guidelines

## Project Structure & Module Organization
- `AIClinician_core_160219.m` orchestrates cohort prep, policy learning, and evaluation; keep helper functions scoped near their call sites.
- Cohort definitions live in `AIClinician_sepsis3_def_160219.m` and `AIClinician_mimic3_dataset_160219.m`; adjust feature tables here before touching training scripts.
- `MDPtoolbox/` contains Markov Decision Process utilities; confine custom edits to this folder so upstream merges stay clean.
- Data extraction runs through `AIClinician_Data_extract_MIMIC3_140219.ipynb`; standalone helpers such as `SAH.m`, `fixgaps.m`, and `fastknnsearch.m` sit at the repository root.

## Build, Test, and Development Commands
- `matlab -batch "run('AIClinician_core_160219.m')"` executes the end-to-end training pipeline; ensure `MDPtoolbox/` is on the MATLAB path.
- `jupyter nbconvert --to notebook --execute AIClinician_Data_extract_MIMIC3_140219.ipynb` regenerates cohort extracts without manual notebook steps.
- `matlab -batch "run('offpolicy_eval_tdlearning.m')"` and `matlab -batch "run('offpolicy_eval_wis.m')"` rerun evaluation suites; capture metric deltas and seeds in logs.
- Use `matlab -batch "checkcode('file.m')"` before submitting to surface style or performance warnings.

## Coding Style & Naming Conventions
- Use 4-space indentation, favor vectorised MATLAB operations, and preallocate arrays with `NaN(...)` or `zeros(...)` as in `AIClinician_core_160219.m`.
- Script names follow `AIClinician_<target>_<YYMMDD>.m`; table variables stay snake_case for downstream compatibility.
- Keep local tweaks inside existing modules; avoid introducing new top-level folders unless architecture changes demand it.

## Testing Guidelines
- For cohort logic changes, diff patient, ICU stay, and intervention counts before/after updates and print anomalies.
- When adjusting policies, rerun the core script on a reduced cohort and record `recqvi`, mortality deltas, and WIS/TIS metrics.
- Restart notebooks and execute end-to-end so stored outputs reflect the current code state.

## Commit & Pull Request Guidelines
- Write imperative, single-focus commits (e.g., `Tighten reward transition smoothing`) and reference affected datasets or scripts.
- Pull requests should explain motivation, methodology, rerun commands, linked issues, and include tables or plots for behavioural shifts.
- Highlight any reliance on protected health information so reviewers can coordinate secure handling.

## Security & Data Handling
- Never commit raw MIMIC-III or eICU exports; store credentials and PHI outside version control via environment variables or MATLAB preferences.
- Share only de-identified aggregates, and validate that exported artefacts comply with source data licenses before distribution.
