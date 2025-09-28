# Run SOP

## 目标
提供一套在 Windows + MATLAB 客户端环境下可重复执行的流程，明确每一步需要运行的文件顺序，确保数据准备、特征构建、策略训练与评估可靠完成，并满足数据安全要求。

## 先决条件
- 已获 MIMIC-III v1.3+ 和（可选）eICU-RI 数据的合法访问许可。
- 已安装 MATLAB R2018b 及以上版本（含 Statistics and Machine Learning Toolbox），建议使用桌面客户端。
- 安装 Python 3.8+，并准备 Jupyter Notebook：`pip install jupyter nbconvert pandas numpy`。
- 预留 80–100 GB 磁盘空间，用于原始数据副本、中间产物和多轮模型输出；若仅分析子集，可适当减少但需保证运行余量。
- 在具备 PHI 保护措施的环境内工作，任何原始数据禁止写入 Git 仓库。

## 运行流程总览
1. 准备 Python 与 MATLAB 环境（一次性）。
2. 使用 `AIClinician_Data_extract_MIMIC3_140219.ipynb` 抽取基础数据。
3. 在 MATLAB 中运行 `AIClinician_sepsis3_def_160219.m` 生成 sepsis-3 队列。
4. 运行 `AIClinician_mimic3_dataset_160219.m` 构建特征矩阵。
5. 运行 `AIClinician_core_160219.m` 完成 500 轮策略训练与评估。
6. 如需额外评估，再运行 `offpolicy_eval_tdlearning.m`、`offpolicy_eval_wis.m` 或其它评估脚本。
7. 执行验证、记录、归档与安全清理。

## 环境准备（一次性）
1. 克隆或更新仓库，例如 `D:\Projects\AI_Clinician`。
2. 打开 PowerShell，创建并激活虚拟环境：
   ```powershell
   cd D:\Projects\AI_Clinician
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install jupyter nbconvert pandas numpy
   ```
   CMD 激活命令：`\.venv\Scripts\activate.bat`。
3. 启动 MATLAB，执行：
   ```matlab
   addpath(pwd);
   addpath(fullfile(pwd, 'MDPtoolbox'));
   savepath; % 如需下次自动加载，可保留
   ```
4. 在 MATLAB 中运行 `ver` 和 `which mdp_value_iteration`，确认工具箱与路径配置成功。

## 第一步：Notebook 数据抽取
1. 确认原始数据位于安全目录（如 `D:\Data\MIMIC\extracts\`）。建议将路径写入环境变量：
   ```powershell
   setx MIMIC_DATA_ROOT "D:\Data\MIMIC\extracts"
   setx EICU_DATA_ROOT "D:\Data\eICU\extracts"
   ```
2. 在 PowerShell 中启动 Notebook：
   ```powershell
   jupyter notebook AIClinician_Data_extract_MIMIC3_140219.ipynb
   ```
3. 按顺序执行 Notebook：
   - 修改顶部参数单元，指向 Windows 路径及过滤条件。
   - 逐单元运行，遇到缺失列或 SQL 报错须及时修改。
   - 完成后选择 “Kernel → Restart & Run All”，生成稳定输出（CSV/MAT）。
4. 无人值守运行：
   ```powershell
   jupyter nbconvert --to notebook --execute AIClinician_Data_extract_MIMIC3_140219.ipynb \
     --output AIClinician_Data_extract_MIMIC3_executed.ipynb
   ```
   将执行版 Notebook 与生成的数据备份至 `results\YYYY-MM-DD\`。

## 第二步：构建 sepsis 队列（MATLAB）
1. 在 MATLAB 客户端中，确保 Notebook 输出已在 MATLAB 当前路径可访问。
2. 运行：
   ```matlab
   run('AIClinician_sepsis3_def_160219.m');
   ```
3. 检查 Command Window 中的入组统计与过滤规则，若与预期偏差>5%，需复核数据或参数。
4. 将生成的队列表格保存到 `results\YYYY-MM-DD\cohort\`。

## 第三步：构建特征矩阵（MATLAB）
1. 确保第二步输出在 MATLAB 工作空间中或以 `.mat` 形式载入。
2. 运行：
   ```matlab
   run('AIClinician_mimic3_dataset_160219.m');
   ```
3. 确认变量 `MIMICtable`、`MIMICraw`、`MIMICzs` 维度与列名正确。
4. 将特征矩阵导出至 `results\YYYY-MM-DD\features\`，并记录生成时间。

## 第四步：策略训练核心脚本
1. 在 MATLAB 客户端执行：
   ```matlab
   run('AIClinician_core_160219.m');
   ```
2. 该脚本依赖前述特征输入，会迭代构建 500 个 MDP 模型，时间较长。运行期间：
   - 保持 MATLAB 与系统处于唤醒状态，禁用睡眠。
   - 监控 Command Window 输出，记录异常消息。
3. 如需命令行批处理，可在 PowerShell 中执行：
   ```powershell
   matlab -batch "run('AIClinician_core_160219.m')"
   ```
4. 训练完成后，收集 `recqvi`、`OA`、`allpols` 等结果，保存到 `results\YYYY-MM-DD\policies\`。

## 第五步：策略评估脚本（可选）
- 针对特定评估需求，在 MATLAB 中顺序运行：
  ```matlab
  run('offpolicy_eval_tdlearning.m');
  run('offpolicy_eval_wis.m');
  run('offpolicy_eval_tdlearning_with_morta.m');
  ```
- 每个脚本运行后导出指标表格或图表，归档到 `results\YYYY-MM-DD\evaluation\`。

## 验证与质量检查
- 将队列人数与关键指标（年龄、SOFA、死亡率等）同历史基线比较；偏差>5% 时需重新检查数据流程。
- 检查 `recqvi` 是否存在 NaN 或极端奖励值；如有异常，复核输入和模型参数。
- 绘制策略效果图（死亡率变化、液体/升压剂剂量），与上一版本结果对比。
- 记录 MATLAB/ Python 版本、抽取数据的时间戳、随机种子等信息，存入 `results\YYYY-MM-DD\run_report.md`。

## 报告与交付
- 在结果目录撰写 README 或报告，总结本次运行的目的、修改内容、关键结论。
- 提交 PR 或内部分享时，附上执行版 Notebook、MATLAB 命令日志、关键图表与数值表。
- 若执行过程中偏离本 SOP（如减少模型数量、替换数据源），需在报告中说明原因及风险控制措施。

## 数据安全与清理
- 原始数据与包含 PHI 的中间文件必须存放在受控目录，可使用 BitLocker 或企业加密方案保护。
- 删除不再需要的中间 `.mat`/`.csv` 文件：
  ```powershell
  Remove-Item D:\Data\AI_Clinician\temp\*.mat -Force
  ```
  如需安全擦除，使用组织认可的工具。
- 在提交代码前运行 `git status`，确保未添加数据制品或执行后的 Notebook。

## 常见故障排查
- **MATLAB 内存不足**：在 `AIClinician_core_160219.m` 中将 `nr_reps` 从 500 调低以做烟囱测试，或在 Windows 中扩展虚拟内存。
- **找不到 MDPtoolbox 函数**：确认 `MDPtoolbox` 文件夹完整并已加入 `path`，必要时重新克隆。
- **Notebook 卡顿或连接数据库失败**：检查路径变量是否正确、网络/磁盘是否可用，任务管理器观察资源占用。
- **结果与上次差异过大**：核对随机种子、数据快照、Notebook 参数单元；如需追溯差异，可将关键输入/输出保存为版本化文件。

## 后续建议
- 每次完成运行后，将 `results\YYYY-MM-DD` 打包备份。
- 对核心脚本的修改须更新本 SOP，并同步给团队成员。
- 如需加入新的评估流程或模型类型，可在“策略评估脚本”章节添加对应运行顺序。
