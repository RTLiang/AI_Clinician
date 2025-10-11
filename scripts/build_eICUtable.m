%% build_eICUtable.m
% Convert the feature matrices exported by eicu_feature_pipeline.py into a MATLAB table.
%
% Loads exportdir/eicu/eicu_features.mat (which contains numeric matrix eICU_data
% and cell array of column names eICU_columns), builds a table named eICUtable,
% and saves it to exportdir/eicu/eICUtable_ready.mat for AIClinician_core_160219.m.

scriptDir = fileparts(mfilename('fullpath'));
repoRoot = fileparts(scriptDir);

infile = fullfile(repoRoot, 'exportdir', 'eicu', 'eicu_features.mat');
outfile = fullfile(repoRoot, 'exportdir', 'eicu', 'eICUtable_ready.mat');

S = load(infile, 'eICU_data', 'eICU_columns');

column_names = cellstr(S.eICU_columns(:));
eICUtable = array2table(S.eICU_data, 'VariableNames', column_names);

save(outfile, 'eICUtable', '-v7.3');
fprintf('Saved table with %d rows x %d columns -> %s\n', ...
    height(eICUtable), width(eICUtable), outfile);
