# Repository Guidelines

## Project Structure & Module Organization
The repository centers on MATLAB scripts for end-to-end policy training. `AIClinician_core_160219.m` orchestrates model building and evaluation, while `AIClinician_sepsis3_def_160219.m` and `AIClinician_mimic3_dataset_160219.m` prepare cohorts and features. The `MDPtoolbox/` folder contains a lightly modified copy of the Markov Decision Process toolbox—keep local changes isolated there. Data extraction resides in `AIClinician_Data_extract_MIMIC3_140219.ipynb`, and supporting utilities (e.g., `SAH.m`, `fixgaps.m`) live at the repository root.

## Build, Test, and Development Commands
Run data extraction in a Python environment with `jupyter notebook AIClinician_Data_extract_MIMIC3_140219.ipynb` or the headless alternative `jupyter nbconvert --to notebook --execute`. Use MATLAB batch mode for repeatable runs: `matlab -batch "run('AIClinician_core_160219.m')"`. Evaluation helpers such as `offpolicy_eval_tdlearning.m` and `offpolicy_eval_wis.m` can be exercised with `matlab -batch "run('offpolicy_eval_tdlearning.m')"`. Set MATLAB’s `path` to include `MDPtoolbox/` before executing.

## Coding Style & Naming Conventions
Follow MATLAB’s 4-space indentation and avoid tabs. Favour vectorised operations and preallocation (`NaN(...)`) as done in `AIClinician_core_160219.m`. Name scripts with the `AIClinician_<target>_<YYMMDD>.m` pattern and table columns in snake_case to match existing variable names. Keep helper functions near the call sites unless they are shared widely—in that case place them beside peers at the repository root or under `MDPtoolbox/`. Run `matlab -batch "checkcode('file.m')"` before submission when possible.

## Testing Guidelines
Validate dataset preparation by diffing summary counts before and after changes and logging anomalies to MATLAB’s console. For policy changes, re-run `AIClinician_core_160219.m` on a small cohort subset and compare key metrics (`recqvi`, mortality deltas). Document the dataset snapshot and random seeds used. Notebook modifications must export executed output (Kernel → Restart & Run All) so reviewers can confirm results.

## Commit & Pull Request Guidelines
Commits should be small, scoped to a single concern, and written in the imperative mood (e.g., `Refine reward shaping parameters`). Reference datasets or scripts touched, and attach brief result notes when policies shift. Pull requests must outline motivation, methodology, rerun commands, and link to any tracked issues; include plots or tables when behaviour changes. Flag any dependency on protected health information so maintainers can coordinate secure handling.

## Data & Access Notes
This project depends on licensed clinical datasets; never commit raw MIMIC-III or eICU extracts. Store credentials and PHI outside the repository (use environment variables or MATLAB preferences). When sharing artefacts, strip patient identifiers and ensure aggregated counts respect the source data agreements.
