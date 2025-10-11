# Repository Guidelines

## Project Structure & Module Organization
`AIClinician_core_160219.m` orchestrates cohort preparation, policy learning, and evaluation; keep any helper logic clustered near its call sites. Cohort definitions live in `AIClinician_sepsis3_def_160219.m` and `AIClinician_mimic3_dataset_160219.m`; update feature tables here before modifying downstream scripts. Data extraction flows through `AIClinician_Data_extract_MIMIC3_140219.ipynb`, while standalone utilities (`SAH.m`, `fixgaps.m`, `fastknnsearch.m`) remain at the repo root. Confine custom Markov Decision Process changes to `MDPtoolbox/` so upstream merges stay predictable.

## Build, Test, and Development Commands
- `matlab -batch "run('AIClinician_core_160219.m')"` — execute the full training pipeline; confirm `MDPtoolbox/` is on the MATLAB path.
- `jupyter nbconvert --to notebook --execute AIClinician_Data_extract_MIMIC3_140219.ipynb` — regenerate cohort extracts from a clean kernel.
- `matlab -batch "run('offpolicy_eval_tdlearning.m')"` / `matlab -batch "run('offpolicy_eval_wis.m')"` — refresh evaluation metrics; capture seeds and deltas in logs.
- `matlab -batch "checkcode('file.m')"` — surface MATLAB style and performance warnings before opening a PR.

## Coding Style & Naming Conventions
Indent MATLAB code with four spaces, favor vectorised operations, and preallocate arrays using `NaN(...)` or `zeros(...)` as in the core script. Script files follow `AIClinician_<target>_<YYMMDD>.m`. Table variables stay snake_case for downstream compatibility. When documenting helper behaviour, prefer succinct comments above non-obvious blocks.

## Testing Guidelines
Before and after cohort logic changes, diff patient, ICU stay, and intervention counts; flag anomalies inline with the corresponding script. When adjusting policies, rerun the core pipeline on a reduced cohort and record `recqvi`, mortality deltas, and WIS/TIS metrics. Restart notebooks prior to execution so stored outputs reflect current code.

## Commit & Pull Request Guidelines
Write single-focus, imperative commits (e.g., `Tighten reward transition smoothing`), and reference the affected dataset or script. PR descriptions should cover motivation, methodology, executed commands, and behavioural shifts (tables or plots preferred). Link relevant issues, highlight metric changes, and call out any dependencies on protected health information so reviewers can coordinate secure handling.

## Security & Data Handling
Never commit raw MIMIC-III or eICU exports. Keep credentials and PHI outside version control using environment variables or MATLAB preferences. Share only de-identified aggregates and confirm exported artifacts comply with source data licenses before distribution.
