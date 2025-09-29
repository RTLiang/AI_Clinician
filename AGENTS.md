# Repository Guidelines

## Project Structure & Module Organization
Core training logic lives in `AIClinician_core_160219.m`, which wires cohort prep, policy learning, and evaluation. Input cohorts and feature matrices are defined in `AIClinician_sepsis3_def_160219.m` and `AIClinician_mimic3_dataset_160219.m`. Keep Markov Decision Process utilities inside `MDPtoolbox/`; local tweaks should stay scoped there so upstream updates remain easy. Data extraction workflows are captured in `AIClinician_Data_extract_MIMIC3_140219.ipynb`, while standalone helpers such as `SAH.m`, `fixgaps.m`, and `fastknnsearch.m` sit at the repository root for reuse.

## Build, Test, and Development Commands
- `jupyter nbconvert --to notebook --execute AIClinician_Data_extract_MIMIC3_140219.ipynb`: regenerate cohort extracts headlessly.
- `matlab -batch "run('AIClinician_core_160219.m')"`: execute the end-to-end training pipeline; ensure `MDPtoolbox/` is on the MATLAB path.
- `matlab -batch "run('offpolicy_eval_tdlearning.m')"` and `matlab -batch "run('offpolicy_eval_wis.m')"`: run evaluation suites; log metric deltas alongside dataset snapshot and random seeds.

## Coding Style & Naming Conventions
Use 4-space indentation and avoid tabs. Favour vectorised MATLAB operations and preallocate arrays with `NaN(...)` or zeros to match `AIClinician_core_160219.m`. Script filenames follow `AIClinician_<target>_<YYMMDD>.m`; table variables stay snake_case. Run `matlab -batch "checkcode('file.m')"` before submitting to catch style or performance warnings.

## Testing Guidelines
When adjusting cohort logic, diff aggregate counts (patients, ICU stays, interventions) before and after changes and print anomalies. For policy updates, rerun the core script on a reduced cohort and record `recqvi`, mortality deltas, and any WIS/TIS metrics. Restart and run notebooks end to end so outputs capture the executed state in Git.

## Commit & Pull Request Guidelines
Write imperative, single-focus commits (e.g., `Tighten reward transition smoothing`). Reference affected datasets or scripts and summarize observed metric shifts. Pull requests should state motivation, methodology, rerun commands, linked issues, and include tables or plots for behavioural changes. Highlight any reliance on protected health information so reviewers can coordinate secure handling.

## Security & Data Handling
Never commit raw MIMIC-III or eICU exports. Keep credentials and PHI outside version control using environment variables or MATLAB preferences. Share only de-identified aggregates and respect source data license constraints when distributing artefacts.
