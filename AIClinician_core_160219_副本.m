%% AI Clinician core code

% (c) Matthieu Komorowski, Imperial College London 2015-2019
% as seen in publication: https://www.nature.com/articles/s41591-018-0213-5

% version 16 Feb 19
% Builds 500 models using MIMIC-III training data
% Records best candidate models along the way from off-policy policy evaluation on MIMIC-III validation data
% Tests the best model on eRI 
% 使用 MIMIC-III 训练数据构建 500 个模型
% 记录在 MIMIC-III 验证数据上进行离线策略评估过程中的最佳候选模型
% 在 eRI 数据上测试最佳模型

% TAKES:
        % MIMICtable = m*59 table with raw values from MIMIC
        % eICUtable = n*56 table with raw values from eICU
        

% GENERATES:
        % MIMICraw 原始矩阵 = MIMIC RAW DATA m*47 array with columns in right order
        % MIMICzs Z-score标准化后矩阵 = MIMIC ZSCORED m*47 array with columns in right order, matching MIMICraw
        % eICUraw 原始矩阵= eICU RAW DATA n*47 array with columns in right order, matching MIMICraw
        % eICUzs 标准化后矩阵 = eICU ZSCORED n*47 array with columns in right order, matching MIMICraw
        % recqvi 模型摘要指标 = summary statistics of all 500 models
        % idxs 500个模型的状态簇索引 = state membership of MIMIC test records, for all 500 models
     	% OA “最优策略”动作映射 = optimal policy, for all 500 models
        % allpols 详细信息 = detailed data about the best candidate models

% This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
% without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE

% Note: The size of the cohort will depend on which version of MIMIC-III is used.
% The original cohort from the 2018 Nature Medicine publication was built using MIMIC-III v1.3.

% ############################  MODEL PARAMETERS   #####################################
%  环境参数初始化：检查工具箱，载入数据表格，设定核心参数（状态数、动作数、折扣因子、交叉验证折数）

v = ver;  % 获取当前安装的工具箱列表
installed = {v.Name};
requiredToolboxes = {'Parallel Computing Toolbox', 'Curve Fitting Toolbox'};

for i = 1:numel(requiredToolboxes)
    if ~ismember(requiredToolboxes{i}, installed)
        error(['❌ Missing required toolbox: ', requiredToolboxes{i}, ...
               '. Please install it via Add-On Explorer before running this script.']);
    end
end

disp('✅ All required toolboxes are installed.');

diary off
% delete('AIClinician_core_log.txt');
diary('AIClinician_core_log.txt');  %saves command window output to text file

% ensure MDP toolbox helper functions (e.g., mdp_verbose) are on the path
core_dir = fileparts(mfilename('fullpath'));
mdp_toolbox_dir = fullfile(core_dir, 'MDPtoolbox');
if ~isfolder(mdp_toolbox_dir)
    error('AIClinician:MissingMDPToolbox', ...
        'Expected MDP toolbox directory at %s. Update RUN_SOP.md if relocated.', mdp_toolbox_dir);
end
addpath(mdp_toolbox_dir);

load("./BACKUP/MIMICtable.mat")
load("./exportdir/eicu/eICUtable_ready.mat")
disp('####  INITIALISATION  ####') 


nr_reps=2;               % 迭代次数  nr of repetitions (total nr models)
nclustering=32;            % how many times we do clustering (best solution will be chosen)
% 表示用不同随机初始质心把 k-means 重复跑 32 次，每次都会收敛到某个（可能不同的）局部最优。最终返回“簇内平方和最小”的那一次作为结果。
prop=0.25;                 % 抽样多少做 k-means 聚类 proportion of the data we sample for clustering 
gamma=0.99;                % MDP 折扣因子 discount factor gamma
transthres=5;              % threshold for pruning the transition matrix 丢弃出现次数少于 5 次的状态-动作转移
polkeep=1;                 % count of saved policies 每保存一个合格模型就递增，确保后续恢复时索引正确
ncl=750;                   % nr of states
nra=5;                     % VP 和 IV 分别离散化等级，总共 nra^2 种联合动作空间 nr of actions (2 to 10)
ncv=5;                     % nr of crossvalidation runs (each is 80% training / 20% test) 控制每轮随机交叉验证拆分的折数，这里固定为 5 折（每次 80% 训练、20% 测试）。
OA=NaN(752,nr_reps);       % 记录每次模型中各状态的最优动作 record of optimal actions
recqvi=NaN(nr_reps*2,30);  % saves data about each model (1 row per model) 按模型积累训练/验证/eICU 的 WIS、QL 等关键指标，后续用来筛选表现最佳的候选模型
allpols=cell(nr_reps,15);  % saving best candidate models 缓存通过阈值的模型及其关键对象（Q 表、转移矩阵、聚类中心等），供最终挑选和复现最佳策略时加载





% #################   Convert training data and compute conversion factors    ######################

%################## 保留原始数值并将数据归一化 ##################%

% all 47 columns of interest
colbin = {'gender','mechvent','max_dose_vaso','re_admission'};% 二值表示
colnorm={'age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',...
    'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',...
    'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',...
    'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index','PaO2_FiO2','cumulated_balance'}; % 连续变量
collog={'SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR','input_total','input_4hourly','output_total','output_4hourly'}; % 偏态分布，先做 log(0.1+x) 再 z-score
% Q：为什么这些变量要做这些变换？做log和直接变换的区别是什么？为什么要log？

colbin=find(ismember(MIMICtable.Properties.VariableNames,colbin)); % 将列名转换为索引下标
colnorm=find(ismember(MIMICtable.Properties.VariableNames,colnorm)); % 将列名转换为索引下标
collog=find(ismember(MIMICtable.Properties.VariableNames,collog)); % 将列名转换为索引下标

% find patients who died in ICU during data collection period
% 可选项：剔除在 ICU 期间死亡且记录不足 24 小时的病人，让 reformat5 保留或过滤这些行。

% ii=MIMICtable.bloc==1&MIMICtable.died_within_48h_of_out_time==1& MIMICtable.delay_end_of_record_and_discharge_or_death<24;
% icustayidlist=MIMICtable.icustayid;
% ikeep=~ismember(icustayidlist,MIMICtable.icustayid(ii));
reformat5=table2array(MIMICtable);
% reformat5=reformat5(ikeep,:);
icustayidlist=MIMICtable.icustayid;
icuuniqueids=unique(icustayidlist); %list of unique icustayids from MIMIC 799557
idxs=NaN(size(icustayidlist,1),nr_reps); %record state membership test cohort 每次模型迭代后记录 MIMIC 测试集中每条记录被分到的聚类状态，共 size(icustayidlist) 行，nr_reps列

MIMICraw=MIMICtable(:, [colbin colnorm collog]);
MIMICraw=table2array(MIMICraw);  % RAW values
MIMICzs=[reformat5(:, colbin)-0.5 zscore(reformat5(:,colnorm)) zscore(log(0.1+reformat5(:, collog)))];  
%创建归一化后的训练特征矩阵：
% reformat5(:, colbin)-0.5：二值特征从 {0,1} 平移到 {−0.5,+0.5}。
% zscore(reformat5(:, colnorm))：连续变量做标准化。
% zscore(log(0.1 + reformat5(:, collog)))：对偏态变量先做 log(0.1 + x) 再标准化。
MIMICzs(:,[4])=log(MIMICzs(:,[ 4])+.6);   % 第 4 列（max_dose_vaso）再取一次对数，强调高剂量差异。 MAX DOSE NORAD 
MIMICzs(:,45)=2.*MIMICzs(:,45);   % 第 45 列的权重放大一倍（input_4hourly），在后续聚类与状态划分中提高其影响力。 increase weight of this variable

% name of cols in eICU test set 
% eICU 做相同的变换
coltbin={'gender', 'mechvent','max_dose_vaso','re_admission'}; 
coltnorm={'age','admissionweight','gcs','hr','sysbp','meanbp','diabp','rr','temp_c','fio2both',...
    'potassium','sodium','chloride','glucose','magnesium','calcium',...
    'hb','wbc_count','platelets_count','ptt','pt','arterial_ph','pao2','paco2',...
    'arterial_be','hco3','arterial_lactate','sofa','sirs','shock_index','pao2_fio2','cumulated_balance_tev'};
coltlog={'spo2','bun','creatinine','sgot','sgpt','total_bili','inr','input_total_tev','input_4hourly_tev','output_total','output_4hourly'};

coltbin=find(ismember(eICUtable.Properties.VariableNames,coltbin)); 
coltnorm=find(ismember(eICUtable.Properties.VariableNames,coltnorm)); 
coltlog=find(ismember(eICUtable.Properties.VariableNames,coltlog));

Xtest=eICUtable(:, [coltbin coltnorm coltlog]);

% shuffle columns so test match train
% 手动对齐 MIMIC 的变量顺序
Xtest = Xtest(:,[1:43 45:47 44]);Xtest = Xtest(:,[1:43 45 44 46:end]);Xtest = Xtest(:,[1:39 41:42 40 43:end]);Xtest = Xtest(:,[1:32 34 33 35:end]);
Xtest = Xtest(:,[1:13 15:32 14 33:end]);Xtest = Xtest(:,[1:13 31 14:30 32:end]);Xtest = Xtest(:,[1:14 16 15 17:end]);Xtest = Xtest(:,[1:10 12 11 13:end]);
Xtest = Xtest(:,[1:8 10:12 9 13:end]);Xtest = Xtest(:,[1 4 2:3 5:end]);Xtest = Xtest(:,[1:29 31 30 32:end]);Xtest = Xtest(:,[1:32 34 33 35:end]);
Xtest = Xtest(:,[1:44 46 45 end]);Xtest = Xtest(:,[1:43 45:46 44 end]);

eICUraw=table2array(Xtest);
eICUraw(isnan(eICUraw(:,45)),45)=0;  % 把第 45 列的 NAN 换成 0 replace NAN fluid with 0
    
% compute conversion factors using MIMIC data
% 计算“特征变换的参数”，后面拿 eICU 数据时直接套用这些参数做同样的变换，实现训练/测试的尺度对齐。
% a/b/c/d 本身之后没有再用
a=MIMICraw(:, 1:3)-0.5; 
[b]= log(MIMICraw(:, 4)+0.1);
[c,cmu,csigma]=zscore(MIMICraw(:,5:36));
[d,dmu,dsigma]=zscore(log(0.1+MIMICraw(:,37:47)));

% ZSCORE full at once XTEST using the factors from training data
% 使用前面计算 *mu/*sigma 参数对 eICU 数据集进行归一化
eICUzs=eICUraw;
eICUzs(:,1:3)=eICUzs(:,1:3)-0.5;
eICUzs(:,4)=log(eICUzs(:,4)+0.1);
eICUzs(:,5:36)=(eICUzs(:,5:36)-cmu)./csigma;
eICUzs(:,37:47)=(log(0.1+eICUzs(:,37:47))-dmu)./dsigma;

%##################归一化结束，构建策略迭代所需变量##################%

% Guard against missing vasopressor or fluid doses before continuing
% 如果 eICU 的关键剂量列（血管加压素 eICUraw(:,4)、4 小时液体量 `eICUraw(:,45)）仍有 NaN，就打印提示并中止脚本，避免后续聚类/剂量离散化依赖的核心字段缺失导致错误。
hasMissingVaso = any(isnan(eICUraw(:,4)));
hasMissingFluids = any(isnan(eICUraw(:,45)));

if hasMissingVaso || hasMissingFluids
    disp('NaNs in Xtest / drug doses');
    disp('EXECUTION STOPPED');
    return;
end

% Initialise or reuse the parallel pool and enable verbose MDP logging
% 初始化或重用并行池并启用详细 MDP 日志记录
p = gcp('nocreate');
if isempty(p)
    pool = parpool;
else
    pool = p;
end
mdp_verbose

% Configure reproducible parallel randomness and silence non-critical warnings
stream = RandStream('mlfg6331_64'); % 生成并行子流
options = statset('UseParallel', 1, 'UseSubstreams', 1, 'Streams', stream); % 开启并行、启用子流（每个 worker 从同一个主流派生独立子流）
warning('off', 'all')


for modl=1:nr_reps  % MAIN LOOP OVER ALL MODELS 
    % modl 主循环迭代计数，范围 1:nr_reps（定义的迭代次数，默认为500）
   
  N=numel(icuuniqueids); % total number of rows to choose from
  grp=floor(ncv*rand(N,1)+1);  % 交叉验证折号，随机抽取 80%/20% 作为训练/测试集 list of 1 to 5 (20% of the data in each grp) -- this means that train/test MIMIC split are DIFFERENT in all the 500 models 
  crossval=1; % 当前作为测试折的编号（固定为 1），其余折都归入训练集。
  trainidx=icuuniqueids(crossval~=grp); % 被划分为训练集的 ICU 住院号集合，不属于 crossval 的 grp 集合
  testidx=icuuniqueids(crossval==grp); % 被划分为测试集的 ICU 住院号集合，属于 crossval 的 grp 集合
  train=ismember(icustayidlist,trainidx); % 逻辑索引，标记 icustayidlist 中哪些记录属于训练住院号。
  test=ismember(icustayidlist,testidx); % 逻辑索引，标记哪些记录属于测试住院号。
  X=MIMICzs(train,:); % 训练集的特征矩阵
  Xtestmimic=MIMICzs(~train,:); % 测试集的特征矩阵
  blocs=reformat5(train,1); % 训练记录的时间步序号，帮助重建患者轨迹
  bloctestmimic=reformat5(~train,1); % 测试记录的时间步序号
  ptid=reformat5(train,2); % 训练记录对应的患者 ID，用于按患者聚合
  ptidtestmimic=reformat5(~train,2); % 测试记录对应的患者 ID
  outcome=10; %   HOSP_MORTALITY = 8 / 90d MORTA = 10   目标列的索引，这里固定 10（90 天死亡率或住院死亡，脚本通过这个索引读取标签）。
  Y90=reformat5(train,outcome);  % 训练集对应的标签列，在 reward 构造与评估时指示生存/死亡。

fprintf('########################   MODEL NUMBER : ');       fprintf('%d \n',modl);         disp( datestr(now))
          
 
% #######   find best clustering solution (lowest intracluster variability)  ####################
disp('####  CLUSTERING  ####') % BY SAMPLING
N=size(X,1); %total number of rows to choose from
sampl=X(find(floor(rand(N,1)+prop)),:); %按 prop=0.25 的概率对当前训练样本（共 N 条）做子采样，仅在 25% 的样本上运行 kmeans 以加速计算。用 floor(rand+prop) 是一个快速 Bernoulli 采样技巧。

[~,C] = kmeans(sampl,ncl,'Options',options,'MaxIter',10000,...
    'Start','plus','Display','final','Replicates',nclustering);
    % 在这个子集上用 k-means 聚 ncl=750 个簇，重复 nclustering=32 次，并采用 k-means++ 初始化（'Start','plus'），最后返回簇内方差最小的一组质心 C。
[idx]=knnsearch(C,X);  %N-D nearest point search: look for points closest to each centroid 把所有训练样本（未采样的整集 X）分配到最近的质心，得到每条记录的状态编号 idx，供后续构建 MDP 转移和奖励使用。


%  ############################# CREATE ACTIONS  ########################

% 将静脉液体 (input_4hourly) 与升压剂 (max_dose_vaso) 的连续剂量离散成 5×5 个联合动作：
%   1. 找到两种剂量列的索引，并提取全量 MIMIC 样本的历史给药记录；
%   2. 对非零剂量做分位数排名后映射到 4 个非零等级，并把 0 剂量保留为等级 1；
%   3. 组合得到 25 个 (io, vc) 等级对，为训练集中每条时间步打上动作编号 actionbloctrain；
%   4. 统计每个等级对应的中位给药量 uniqueValuesdose，后续用于 eICU 推荐剂量、可视化与解释。
disp('####  CREATE ACTIONS  ####') 
nact=nra^2; % 状态空间 nra^2 = 25
 
iol=find(ismember(MIMICtable.Properties.VariableNames,{'input_4hourly'}));
vcl=find(ismember(MIMICtable.Properties.VariableNames,{'max_dose_vaso'}));
 
a= reformat5(:,iol);                   % 提取所有 4 小时静脉补液量的连续数值
a= tiedrank(a(a>0)) / length(a(a>0));   % 仅对非零补液做秩次排名，转成 (0,1] 分位数，准备划入 4 档非零剂量

        iof=floor((a+0.2499999999)*4);  % 将分位数平移 0.25 后放大 4 倍并取整 → 0~3，代表四档非零补液
        a= reformat5(:,iol); a=find(a>0);  % 找到补液量>0 的行索引
        io=ones(size(reformat5,1),1);  % 默认所有行都归入等级 1（即 0 剂量）
        io(a)=iof+1;   % 对非零行加 1，将 0~3 映射到 2~5，形成 “0 剂量 + 4 档非零” 共 5 档补液动作
        vc=reformat5(:,vcl);  vcr= tiedrank(vc(vc~=0)) / numel(vc(vc~=0)); vcr=floor((vcr+0.249999999999)*4);  % 对升压剂的非零历史剂量重复同样步骤，得到四档非零剂量
        vcr(vcr==0)=1; vc(vc~=0)=vcr+1; vc(vc==0)=1;   % 将 0~3 映射成 1~4，再与零剂量合并，得到 5 档升压剂动作
        ma1=[ median(reformat5(io==1,iol))  median(reformat5(io==2,iol))  median(reformat5(io==3,iol))  median(reformat5(io==4,iol))  median(reformat5(io==5,iol))];  % 计算每档补液动作对应的历史中位剂量
        ma2=[ median(reformat5(vc==1,vcl))  median(reformat5(vc==2,vcl))  median(reformat5(vc==3,vcl))  median(reformat5(vc==4,vcl))  median(reformat5(vc==5,vcl))] ;  % 计算每档升压剂动作对应的历史中位剂量

med=[io vc];  % 拼接成 5x5 的联合动作网格（行：补液档位，列：升压剂档位）
[uniqueValues,~,actionbloc] = unique(array2table(med),'rows');  % 唯一化得到 25 个联合动作，actionbloc 为每条记录对应的动作索引
actionbloctrain=actionbloc(train);  % 取出训练集的动作编号，供后续策略学习
uniqueValuesdose=[ ma2(uniqueValues.med2)' ma1(uniqueValues.med1)'];  % median dose of each bin for all 25 actions
 
 
% ####################### 整理为MDP需要的表格形式，创建QLDATA3 ###############################################################################

% 构造 TD/MDP 输入矩阵 qldata3：
%   - 初始 qldata 包含原始 bloc（时间步）、聚类状态 idx、离散动作 actionbloctrain、
%     以及 0/1 生存标签 Y90（随后映射成 ±100 奖励）；
%   - 逐行复制到 qldata3，遇到 bloc==1（轨迹结束）时插入一行吸收态
%     [下一 bloc, 终结状态, 动作0, 奖励] 并写入终结奖励；
%   - qldata3 的四列依次对应 [时间步, 状态编号, 动作编号, 结局/奖励]；
%   - 结果是一个按患者轨迹拼接的矩阵，用于后续统计转移概率和奖励。

disp('####  CREATE QLDATA3  ####')
r=[100 -100]; % r = [100 -100]; 设定正负 100 的基础奖励模板（前者对应存活，后者对应死亡）。
r2=r.*(2*(1-Y90)-1);  %将 Y90（1=死亡，0=生存）映射到 ±100：当 Y90=0 时产生 +100，Y90=1 时得到 -100。
qldata=[blocs idx actionbloctrain Y90 r2];  % contains bloc / state / action / outcome&reward     %1 = died
% 将每个时间步的时间标签 bloc、状态编号 idx、离散动作、死亡标签和奖励拼成原始轨迹表。
qldata3=zeros(floor(size(qldata,1)*1.2),4); % 为最终轨迹矩阵预分配空间（4 列：时间、状态、动作、奖励），多留 20% 以便插入终结行。
c=0;
abss=[ncl+2 ncl+1]; %absorbing states numbers 定义两个吸收态 id：ncl+2 表示死亡终结态，ncl+1 表示生存终结态。
 
        for i=1:size(qldata,1)-1 %遍历原始 qldata 的每一行（倒数第二行之前，便于查看下一行）。
            c=c+1;  qldata3(c,:)=qldata(i,1:4); % 将当前行的时间步、状态、动作、死亡标签写入 qldata3。
            if qldata(i+1,1)==1 %end of trace for this patient 检测下一行是否是下一位患者的第一个时间步（说明本轨迹结束）。
                c=c+1;     qldata3(c,:)=[qldata(i,1)+1 abss(1+qldata(i,4)) 0 qldata(i,5)]; 
                % 插入终结行：时间步加一、状态切换到 “存活/死亡” 吸收态（qldata(i,4) 是 0 或 1）、动作置零并写入终结奖励。
            end
        end
        qldata3(c+1:end,:)=[]; % 删除预分配中未写入的多余行，留下实际轨迹矩阵。

 
% ############################# 构建状态转移函数 #########################################################################################

% 构建转移概率函数 T(S' | S, A)（以列存放 S'，便于按 (S,A) 归一化）
% - transitionr(S1,S0,A) 先累计 (S0→S1 在动作 A 下) 的发生次数
% - sums0a0(S0,A) 记录每个 (S0,A) 的总次数
% - 之后对每个 (S0,A) 的列归一化，得到条件概率 T(S' | S0,A)
disp('####  CREATE TRANSITION MATRIX T(S'',S,A) ####')
% transitionr用列表示 (S, A)，便于用 sums0a0 推导临床行为策略 π_phys(A|S)；
transitionr=zeros(ncl+2,ncl+2,nact);  % T(S',S,A) 维度：[S' x S x A]
sums0a0=zeros(ncl+2,nact);            % (S,A) 计数表，用于归一化和估计行为策略
 
     for i=1:size(qldata3,1)-1
 
         % 若下一行 bloc != 1，说明轨迹未结束，存在从 S0 到 S1 的一次转移
         % bloc 为该患者当前 ICU 住院轨迹中的时间步/序号
         if (qldata3(i+1,1))~=1
         S0=qldata3(i,2); S1=qldata3(i+1,2);  acid= qldata3(i,3);
         transitionr(S1,S0,acid)=transitionr(S1,S0,acid)+1;  sums0a0(S0,acid)=sums0a0(S0,acid)+1;
         end
     end

      % 稀疏裁剪：删除出现次数较少（≤ transthres）的 (S,A)，减少噪声导致的不稳定
      sums0a0(sums0a0<=transthres)=0;

     for i=1:ncl+2
         for j=1:nact
             if sums0a0(i,j)==0
                transitionr(:,i,j)=0; 
             else
                % 对每个 (S=i, A=j) 的列进行归一化，得到 T(S' | S,A)
                transitionr(:,i,j)=transitionr(:,i,j)/sums0a0(i,j);
             end
         end
     end
 
   
% 数值清理，防止 NaN/Inf 进入后续计算
transitionr(isnan(transitionr))=0;
transitionr(isinf(transitionr))=0;
 
% 临床行为策略 π_phys(A|S)：按行归一化 (S,A) 计数，得到各状态下医生动作分布
physpol=sums0a0./sum(sums0a0')';
 
% 备份构建另一种存储布局的转移张量 T(S' | S, A)：以行存放 S'
% - transitionr2(S0,S1,A) 计数 (S0→S1 | A)
% - 之后对每个 (S0,A) 的行归一化，得到 T(S' | S0,A)
disp('####  CREATE TRANSITION MATRIX T(S,S'',A)  ####')
% transitionr2把 (S, A) 放在行上，供MDP
transitionr2=zeros(ncl+2,ncl+2,nact);  % T(S,S',A) 维度：[S x S' x A]
sums0a0=zeros(ncl+2,nact);             % 重新统计 (S,A) 计数
 
     for i=1:size(qldata3,1)-1
 
         % 同上：若轨迹未结束，记录 (S0→S1 | A) 的计数
         if (qldata3(i+1,1))~=1
         S0=qldata3(i,2); S1=qldata3(i+1,2);  acid= qldata3(i,3);
         transitionr2(S0,S1,acid)=transitionr2(S0,S1,acid)+1;  sums0a0(S0,acid)=sums0a0(S0,acid)+1;
         end
     end
 
      % 稀疏裁剪（同上）
      sums0a0(sums0a0<=transthres)=0;  % IQR 范围注释仅作数据分布提示
     
     for i=1:ncl+2
         for j=1:nact
             if sums0a0(i,j)==0
                transitionr2(i,:,j)=0; 
             else
                % 对每个 (S=i, A=j) 的行进行归一化，得到 T(S' | S,A)
                transitionr2(i,:,j)=transitionr2(i,:,j)/sums0a0(i,j);
             end
         end
     end
 
% 数值清理
transitionr2(isnan(transitionr2))=0;
transitionr2(isinf(transitionr2))=0;
 
% ################################# 奖励矩阵构建 #############################################################################
disp('####  CREATE REWARD MATRIX  R(S,A) ####')
% CF sutton& barto bottom 1998 page 106. i compute R(S,A) from R(S'SA) and T(S'SA)
r3=zeros(ncl+2,ncl+2,nact); r3(ncl+1,:,:)=-100; r3(ncl+2,:,:)=100;
R=sum(transitionr.*r3);
R=squeeze(R);   %remove 1 unused dimension

% ################################## 策略迭代 #####################################################################################
disp('####   POLICY ITERATION   ####')

 [~,~,~,~,Qon] = mdp_policy_iteration_with_Q(transitionr2, R, gamma, ones(ncl+2,1));
 [~,OptimalAction]=max(Qon,[],2);  %deterministic 
 OA(:,modl)=OptimalAction; %save optimal actions
 

% ################################## OPE for 训练集 #####################################################################################
% 这一节在 MIMIC 训练集上做离线策略评估（Off-Policy Evaluation, OPE）：
%   - 行为策略：医生历史选择 `physpol`；目标策略：策略迭代得到的 `OptimalAction`；
%   - 依据这两类策略，重新整理轨迹矩阵（含 bloc、状态、动作、行为策略概率、目标策略概率、回报、患者 ID）；
%   - 配置软化参数 p，将行为/目标策略的零概率救济为极小正概率，确保重要性采样时分母不为零。
disp('#### OFF-POLICY EVALUATION - MIMIC TRAIN SET ####')
 
% 重新构造 qldata / qldata3，在原始四列基础上扩展为 8 列，用于 OPE：
%   1) bloc；2) state；3) action；4) 奖励占位（非终止步为 0，终止步写入 ±100 奖励）；
%   5) 行为策略概率 π_beh(s,a)；6) 目标策略概率 π_target(s,a)；
%   7) 目标策略推荐动作（稍后写入 OptimalAction）；8) 患者 ID（保持轨迹连续性）。
r=[100 -100];
r2=r.*(2*(1-Y90)-1); 
qldata=[blocs idx actionbloctrain Y90 zeros(numel(idx),1) r2(:,1) ptid];  % 新 qldata 扩展列：bloc/state/action/Y90/占位/奖励/患者 ID
qldata3=zeros(floor(size(qldata,1)*1.2),8);  % 预留八列，含轨迹终结行（吸收态+奖励）

% 软化策略：如果临床行为策略在某个状态对某个动作的概率是 0，而目标策略给了正概率，就会出现分母为 0 或权重无穷大的情况，评估结果失真。
% softpi：把医生策略中原本为 0 的动作均匀分配上一点概率，非零动作相应地减去同样的总量；
% softb：目标策略也是相同处理，让最优动作权重 1-p，其余动作均摊剩下的 p。
c=0;
abss=[ncl+2 ncl+1]; %absorbing states numbers
 
        for i=1:size(qldata,1)-1
            c=c+1;
              % qldata 中第 1~3 列分别是 bloc/state/action；第 5 列是占位奖励（此处为 0）；
              % 第 7 列是 ±100 奖励；第 8 列是患者 ID（用于在评估函数中重建轨迹）。
              % 这里把这些字段拷贝到 qldata3 的对应位置，便于后续统一处理。
              qldata3(c,:)=qldata(i,[1:3 5 7 7 7 7]);
            if qldata(i+1,1)==1 %end of trace for this patient
                c=c+1;
                % 轨迹结束：插入吸收态行。bloc+1（表示终止时刻），状态为生存(ncl+1)/死亡(ncl+2)；
                % 动作为 0（不再执行决策），第 4 列写入终止奖励，第 5~7 列留空，最后一列保留患者 ID。
                qldata3(c,:)=[qldata(i,1)+1 abss(1+qldata(i,4)) 0 qldata(i,6) 0 0 0 qldata(i,7)]; 
            end
        end
        qldata3(c+1:end,:)=[];  % 删除预分配的多余行，得到真实轨迹矩阵
 
% 为每个状态填充行为策略概率 π_beh 和目标策略概率 π_target
p=0.01; %softening policies  
softpi=physpol; % behavior policy = clinicians' 
 
for i=1:750
    % softpi 是医生行为策略的软化版本：
    %   - 找出原始概率为 0 的动作（ii==true），把极小量 z 均匀分配给这些动作；
    %   - 对原本非零的动作扣除相同总量 nz，保持概率和为 1；
    %   - 这样做避免重要性采样时出现 π_beh=0 导致的除零或无限权重。
    ii=softpi(i,:)==0;
    z=p/sum(ii);          % 均匀分给零概率动作的补偿量
    nz=p/sum(~ii);        % 从非零动作中扣除的量，确保总和仍为 1
    softpi(i,ii)=z;
    softpi(i,~ii)=softpi(i,~ii)-nz;
end
softb=abs(zeros(752,25)-p/24); %"optimal" policy = target policy = evaluation policy  % 目标策略保底给余下 24 个非最优动作均匀的小概率

for i=1:750
     softb(i,OptimalAction(i))=1-p;  % 将最优动作的概率设置为 1-p，保证目标策略也合法归一
end

for i=1:size(qldata3,1)  %adding the probas of policies to qldata3
    % 跳过吸收态（state>750 的行），只对真实状态填充策略概率与推荐动作
    if qldata3(i,2)<=750
        % 第 5 列：行为策略在 state=qldata3(i,2) 下选择 action=qldata3(i,3) 的概率
        qldata3(i,5)=softpi(qldata3(i,2),qldata3(i,3));
        % 第 6 列：目标策略在同一状态下选择该动作的概率
        qldata3(i,6)=softb(qldata3(i,2),qldata3(i,3));
        % 第 7 列：记录目标策略在该状态下的推荐动作 ID，供后续对比分析
        qldata3(i,7)=OptimalAction(qldata3(i,2));   %optimal action
    end
end

qldata3train=qldata3; %qldata3 保存到 qldata3train

% 调用 offpolicy_multiple_eval_010518 做离策略评估
tic
 [ bootql,bootwis ] = offpolicy_multiple_eval_010518( qldata3,physpol, 0.99,1,6,750);
toc

%模型（第 modl 次迭代）的离策略评估结果存进 recqvi
recqvi(modl,1)=modl;
recqvi(modl,4)=nanmean(bootql);
recqvi(modl,5)=quantile(bootql,0.99);
recqvi(modl,6)=nanmean(bootwis);  %we want this as high as possible
recqvi(modl,7)=quantile(bootwis,0.05);  %we want this as high as possible


 % ################################## OPE for 内部测试集 #####################################################################################

% 使用 MIMIC 测试集做一次离策略评估（OFF-POLICY EVALUATION - MIMIC TEST SET）：
%   Step1：把测试样本投影到训练阶段的聚类质心，得到状态序列；
%   Step2：用测试集的联合动作编号与 90 天结局，重建包含 8 列信息的 qldata3；
%   Step3：对医生策略/目标策略做软化并写入 qldata3，再交给 offpolicy_multiple_eval_010518 计算回报分布。
disp('#### OFF-POLICY EVALUATION - MIMIC TEST SET ####')
    
% ---------- Step1: 计算测试集的状态映射 ----------
idxtest=knnsearch(C,Xtestmimic);                 % 基于训练得到的质心 C，为每条测试记录找到最近的聚类状态
idxs(test,modl)=idxtest;                         % 记录当前模型下测试集的状态编号，用于后续分析

% ---------- Step2: 组装测试集轨迹表 ----------
actionbloctest=actionbloc(~train);               % 把 actionbloc 中对应测试集的动作编号取出来
Y90test=reformat5(~train,outcome);               % 测试集 90 天结局标签（1=死亡,0=存活）
r=[100 -100];                                    % 奖励模板：存活 +100，死亡 -100
r2=r.*(2*(1-Y90test)-1);                         % 将 0/1 结局映射为 ±100 奖励向量
% qldata 列定义（与训练段一致）：
%   1:bloc   2:state   3:action   4:Y90   5:奖励占位(0)   6:±100 奖励   7:患者 ID
qldata=[bloctestmimic idxtest actionbloctest Y90test zeros(numel(idxtest),1) r2(:,1) ptidtestmimic];
qldata3=zeros(floor(size(qldata,1)*1.2),8);       % 预分配 8 列轨迹矩阵，预留吸收态行

% ---------- Step3: 逐条复制轨迹，并在每位患者结束时添加吸收态 ----------
c=0;
abss=[ncl+2 ncl+1]; %absorbing states numbers
 
        for i=1:size(qldata,1)-1
            c=c+1;
            % 拷贝当前时间步数据：
            %   列1-3 → bloc/state/action；
            %   列4 → 奖励占位（初始化为 0）；
            %   列5-7 → 暂时写 0，后面填入行为概率、目标概率、目标策略动作；
            %   列8 → 患者 ID。
            qldata3(c,:)=qldata(i,[1:3 5 7 7 7 7]);
            if qldata(i+1,1)==1 %end of trace for this patient
                c=c+1;
                % 若下一行 bloc 重新回到 1，说明当前患者轨迹结束：
                %   插入吸收态行：bloc+1、状态改成生存/死亡吸收态、动作=0、奖励写入第 4 列；
                %   第 5~7 列保留 0（吸收态不需要策略概率），第 8 列延续患者 ID。
                qldata3(c,:)=[qldata(i,1)+1 abss(1+qldata(i,4)) 0 qldata(i,6) 0 0 0 qldata(i,7)]; 
            end
        end
        qldata3(c+1:end,:)=[];                     % 清理预分配的空行，只保留实际轨迹

% ---------- Step4: 写入软化后的行为/目标策略 ----------

p=0.01; %small correction factor // softening policies
softpi=physpol; % behavior policy = clinicians'
for i=1:750
    % 与训练段相同：对行为策略进行软化，避免 π_beh=0
    ii=softpi(i,:)==0;             % 当前状态下，原始概率为 0 的动作掩码
    z=p/sum(ii);                   % 均匀分配的补偿概率
    nz=p/sum(~ii);                 % 对非零动作需要扣除的概率
    softpi(i,ii)=z;
    softpi(i,~ii)=softpi(i,~ii)-nz;
end
softb=abs(zeros(752,25)-p/24);     % 目标策略的基准（非最优动作均分 p/24）
for i=1:750
    softb(i,OptimalAction(i))=1-p; % 把最优动作的概率设为 1-p，保持归一
end

for i=1:size(qldata3,1)  %adding the probas of policies to qldata
    if qldata3(i,2)<=750
qldata3(i,5)=softpi(qldata3(i,2),qldata3(i,3));
qldata3(i,6)=softb(qldata3(i,2),qldata3(i,3));
qldata3(i,7)=OptimalAction(qldata3(i,2));   %optimal action
    end
end

qldata3test=qldata3;                                    % 保存测试集轨迹矩阵，供日志或后续分析使用

% ---------- Step5: 调用 OPE 评估函数 ----------

tic
[ bootmimictestql,bootmimictestwis ] = offpolicy_multiple_eval_010518( qldata3,physpol, 0.99,1,6,2000);
toc

% 记录测试集上的 Q-learning (bootmimictestql) 和 WIS (bootmimictestwis) 分布统计，
% 以评估策略的平均回报与保守下界性能。
recqvi(modl,19)=quantile(bootmimictestql,0.95);   %PHYSICIANS' 95% UB
recqvi(modl,20)=nanmean(bootmimictestql);
recqvi(modl,21)=quantile(bootmimictestql,0.99);
recqvi(modl,22)=nanmean(bootmimictestwis);    
recqvi(modl,23)=quantile(bootmimictestwis,0.01);  
recqvi(modl,24)=quantile(bootmimictestwis,0.05);  %AI 95% LB, we want this as high as possible


if recqvi(modl,24) > 40 %saves time if policy is not good on MIMIC test: skips to next model 
% WIS 评估的 5% 分位数（quantile(bootmimictestwis, 0.05)），如果它大于 40，就说明保守置信下界不错，于是继续执行 eICU 评估；反之则跳过进入下一次模型训练。

% ################################## OPE for 外部测试集 #####################################################################################

 disp('########################## eICU TEST SET #############################')

 % 把eICU数据映射到聚类状态
  idxtest2=cell(size(eICUzs,1),1);
        ii=isnan(eICUzs);
        disp('####   IDENTIFY STATE MEMBERSHIP OF eICU TEST RECORDS   ####')
    tic
      parfor i=1:size(eICUzs,1)
        idxtest2(i)={knnsearch(C(:,~ii(i,:)),eICUzs(i,~ii(i,:)))};  %which ones are the k closest records in Xtrain? - only match on available data (ii columns)!
      end
    toc
    
  idxtest2=cell2mat(idxtest2);


% 将eICU的动作映射到我们在MIMIC中构建的动作上
iol=find(ismember(MIMICtable.Properties.VariableNames,{'input_4hourly'}));
vcl=find(ismember(MIMICtable.Properties.VariableNames,{'max_dose_vaso'}));
 
 a= reformat5(:,iol);                    %IV fluid
 a= tiedrank(a(a>0)) / length(a(a>0));   % excludes zero fluid (will be action 1)
 
        iof=floor((a+0.2499999999)*4);       %converts iv volume in 4 actions
        a= reformat5(:,iol); a=find(a>0);    %location of non-zero fluid in big matrix
        io=ones(size(reformat5,1),1);        %array of ones, by default     
        io(a)=iof+1;                         %where more than zero fluid given: save actual action
        vc=reformat5(:,vcl);  vcr= tiedrank(vc(vc~=0)) / numel(vc(vc~=0)); vcr=floor((vcr+0.249999999999)*4);  %converts to 4 bins
        vcr(vcr==0)=1; vc(vc~=0)=vcr+1; vc(vc==0)=1;
        ma1=[ median(reformat5(io==1,iol))  median(reformat5(io==2,iol))  median(reformat5(io==3,iol))  median(reformat5(io==4,iol))  median(reformat5(io==5,iol))];  %median dose of drug in all bins
        ma2=[ median(reformat5(vc==1,vcl))  median(reformat5(vc==2,vcl))  median(reformat5(vc==3,vcl))  median(reformat5(vc==4,vcl))  median(reformat5(vc==5,vcl))] ;
  
med=[io vc];
[uniqueValues,~,actionbloc] = unique(array2table(med),'rows');
actionbloctrain=actionbloc(train);
uniqueValuesdose=[ ma2(uniqueValues.med2)' ma1(uniqueValues.med1)'];  % median dose of each bin for all 25 actions
 
iol=find(ismember(MIMICtable.Properties.VariableNames,{'input_4hourly'}));
vcl=find(ismember(MIMICtable.Properties.VariableNames,{'max_dose_vaso'}));
ma1=[ max(reformat5(io==1,iol))  max(reformat5(io==2,iol))  max(reformat5(io==3,iol))  max(reformat5(io==4,iol))  max(reformat5(io==5,iol))];  %upper dose of drug in all bins
ma2=[ max(reformat5(vc==1,vcl))  max(reformat5(vc==2,vcl))  max(reformat5(vc==3,vcl))  max(reformat5(vc==4,vcl))  max(reformat5(vc==5,vcl))] ;

% define actionbloctest = which actions are taken in the test set ????
vct=eICUraw(:,4); vct(vct>ma2(nra-1))=nra; vct(vct==0)=1; for z=2:nra-1; vct(vct>ma2(z-1) & vct<=ma2(z))=z;end
iot=eICUraw(:,45); for z=2:nra-1; iot(iot>ma1(z-1) & iot<=ma1(z))=z; end;iot(iot>ma1(nra-1))=nra;iot(iot==0)=1;
 
med=[iot vct];
[~,~,actionbloctest] = unique(array2table(med),'rows');   %actions taken in my test samples
 
iol=eICUraw(:,45);      % DOSES IN TEST SET
vcl=eICUraw(:,4);

% CREATE QLDATA2 FOR EICU TEST SET

ptid=eICUtable.patientunitstayid;
blocstest=eICUtable.bloc;
Y90test=eICUtable.hospmortality;
r=[100 -100];
r2=r.*(2*(1-Y90test)-1); 
models=OptimalAction(idxtest2);                  %optimal action for each record
%把动作编号映射回 MIMIC 训练时计算的代表剂量，得到模型对这条 eICU 记录给出的实际剂量建议。
modeldosevaso = uniqueValuesdose(models,1);      %dose reco in this model
modeldosefluid = uniqueValuesdose(models,2);     %dose reco in this model


% 构建 eICU 版本的轨迹表 qldata2，格式比前面多了一些列，用于同时记录真实给药和模型推荐剂量
qldata=[blocstest idxtest2 actionbloctest Y90test zeros(numel(idxtest2),1) r2(:,1) ptid  iol vcl modeldosefluid modeldosevaso  Y90test ];  % contains bloc / state / action / outcome&reward     %1 = died
qldata2=zeros(floor(size(qldata,1)*1.2),13); 
c=0;
abss=[ncl+2 ncl+1]; %absorbing states numbers
 
        for i=1:size(qldata,1)-1
            c=c+1;
              qldata2(c,:)=qldata(i,[1:3 5 7 7 7 7 8:12]);
            if qldata(i+1,1)==1 %end of trace for this patient
                c=c+1;
                qldata2(c,:)=[qldata(i,1)+1 abss(1+qldata(i,4)) 0 qldata(i,6) 0 0 0 qldata(i,7) qldata(i,8:12)]; 
            end
        end
qldata2(c+1:end,:)=[];


% 软化 add pi(s,a) and b(s,a)
p=0.01; % softening policies

softpi=physpol;%physpoleicu;   
for i=1:750
    ii=softpi(i,:)==0; z=p/sum(ii); nz=p/sum(~ii); softpi(i,ii)=z; softpi(i,~ii)=softpi(i,~ii)-nz;
end

softb=abs(zeros(752,25)-p/24); %optimal policy
for i=1:750
softb(i,OptimalAction(i))=1-p;
end

for i=1:size(qldata2,1)  %adding the probas of policies to qldata
    if qldata2(i,2)<=750
qldata2(i,5)=softpi(qldata2(i,2),qldata2(i,3));
qldata2(i,6)=softb(qldata2(i,2),qldata2(i,3));
qldata2(i,7)=OptimalAction(qldata2(i,2)); 
    end
end

tic  %multiple evaluation
 [ booteicuql,booteicuwis ] = offpolicy_multiple_eval_010518( qldata2,physpol, 0.99,1,20,500);
toc

recqvi(modl,10)=nanmean(booteicuql);
recqvi(modl,11)=quantile(booteicuql,0.99);
recqvi(modl,12)=nanmean(booteicuwis); 
recqvi(modl,13)=quantile(booteicuwis,0.01);  
recqvi(modl,14)=quantile(booteicuwis,0.05); 


end

% ################################## 挑选最优模型 #####################################################################################
% ########################## 同时跑MIMIC和外部测试eICU，共同挑选最优 #####################################################################################

%MIMIC 测试集上的 WIS 5% 分位（95% 置信下界）为正且eICU 评估的对应下界也为正的模型保留
if recqvi(modl,24)>0 & recqvi(modl,14)>0   % if 95% LB is >0 : save the model (otherwise it's pointless)
    
    disp('####   GOOD MODEL FOUND - SAVING IT   ####' ) 
    allpols(polkeep,1)={modl};
    allpols(polkeep,3)={Qon};
    allpols(polkeep,4)={physpol};
    allpols(polkeep,6)={transitionr};
    allpols(polkeep,7)={transitionr2};
    allpols(polkeep,8)={R};
    allpols(polkeep,9)={C};
    allpols(polkeep,10)={train};
    allpols(polkeep,11)={qldata3train};
    allpols(polkeep,12)={qldata3test};
    allpols(polkeep,13)={qldata2};
    polkeep=polkeep+1;
    
end

 
end

recqvi(modl:end,:)=[];

tic
     save('./BACKUP/Data_160219.mat', '-v7.3');
toc


%% IDENTIFIES BEST MODEL HERE
%% eICU 的 WIS 下界 < 0 的模型过滤掉， MIMIC 测试集上 AI 策略 WIS 的 95% 下界；取这列最大的那一行

recqvi(:,31:end)=[];

r=recqvi;
r(:,30:end)=[];
r(r(:,14)<0,:)=[];  %delete models with poor value in MIMIC test set

% SORT RECQVI BY COL 24 / DESC
bestpol=r(max(r(:,24))==r(:,24),1);   % model maximising 95% LB of value of AI policy in MIMIC test set


%% RECOVER BEST MODEL and TEST IT
%% 挑选出最优模型
disp('####   RECOVER BEST MODEL   ####')
a=cell2mat(allpols(:,1));
outcome =10; %   HOSPITAL MORTALITY = 8 / 90d MORTA = 10
ii=find(a==bestpol); %position of best model in the array allpols

% RECOVER MODEL DATA
% 提取最优模型的所有数据，以便后续报告
Qoff=cell2mat(allpols(ii,2));
Qon=cell2mat(allpols(ii,3));
physpol=cell2mat(allpols(ii,4));
softpol=cell2mat(allpols(ii,5));
transitionr=cell2mat(allpols(ii,6));
transitionr2= cell2mat(allpols(ii,7));
R = cell2mat(allpols(ii,8));
C = cell2mat(allpols(ii,9));
train =  cell2mat(allpols(ii,10));
test=~train;
qldata3train=  cell2mat( allpols(ii,11));
qldata3test= cell2mat( allpols(ii,12));
qldata2 = cell2mat(allpols(ii,13));

idx=knnsearch(C,MIMICzs(train,:));  %N-D nearest point search: look for points closest to each centroid
[~,OptimalAction]=max(Qon,[],2);  %deterministic 
idxtest=idxs(test,a(ii));          %state of records from training set
actionbloctrain=actionbloc(train);
actionbloctest=actionbloc(test);        %actionbloc is constant across clustering solutions
Y90=reformat5(train,outcome);
Y90test= reformat5(test,outcome);
blocs=reformat5(train,1);
bloctestmimic=reformat5(test,1);
vcl=reformat5(test,52);
iol=reformat5(test,56);
ptid=reformat5(train,2);
ptidtestmimic=reformat5(test,2);


%recover state membership of eicu samples
% 最佳模型的聚类中心 C 来给 eICU 每条记录分配状态
disp('####   IDENTIFY STATE MEMBERSHIP OF eICU TEST RECORDS   ####')
  idxtest2=cell(size(eICUzs,1),1);
        ii=isnan(eICUzs);
    tic
      parfor i=1:size(eICUzs,1)
        idxtest2(i)={knnsearch(C(:,~ii(i,:)),eICUzs(i,~ii(i,:)))};  %which ones are the k closest records in Xtrain? - only match on available data (ii columns)!
      end
    toc
    
  idxtest2=cell2mat(idxtest2);



% ################################## 绘图与分析 #####################################################################################


%% FIB 2A plot safety of algos: 95th UB of physicians policy value vs 95th LB of AI policy
% during bulding of 500 different models
% show that the value of AI policy is always guaranteed to be better than doctors' according to the model
% 对 recqvi 中的置信界做“历史最大值”累积，逐模型展示医生与 AI 策略的安全边界走向：
%   - 列19：医生策略在 MIMIC 测试集的 95% 上界；
%   - 列24：AI 策略在 MIMIC 测试集的 95% 下界；
%   - 列14：AI 策略在 eICU 测试集的 95% 下界。

clear h
r=recqvi;   %MAKE SURE RECQVI IS SORTED BY MODEL NUMBER!!!

m=zeros(size(r,1),1);
% 对医生策略 95% 上界做逐步累积最大值，得到“目前为止最安全的医生策略”曲线
for i=1:size(r,1)
if r(i,19)>max(m)  %physicians    // OR 19 = 95th percentile!!!!!!!!!!!!
m(i)=r(i,19);
else
m(i)=max(m);
end
end
figure
h(1)=semilogx(m,'linewidth',2);
hold on

m=zeros(size(r,1),1);
% 同理，记录 AI 策略（MIMIC 测试集）95% 下界的历史最大值，越大表示越保守的保证
for i=1:size(r,1)
if r(i,24)>max(m)  %learnt policy
m(i)=r(i,24);
else
m(i)=max(m);
end
end
h(2)=semilogx(m,'linewidth',2);


m=zeros(size(r,1),1);
% eICU 测试集同理，观察 AI 策略在外部数据上的安全下界
for i=1:size(r,1)
if r(i,14)>max(m)  %learnt policy
m(i)=r(i,14);
else
m(i)=max(m);
end
end
h(3)=semilogx(m,'linewidth',2);

axis([0 500 0 100])
xlabel('Number of models built')
ylabel('Estimated policy value')
legend([h(2) h(3) h(1)],{'95% LB for best AI policy (MIMIC test set)','95% LB for best AI policy (eICU test set)','95% UB for highest valued clinician policy'},'location','se')
set(gca,'FontSize',12)
axis square
hold off


%% FIG 2B BOXPLOT OF POLICY VALUE OVER 500 MODELS -  MIMIC TEST SET ONLY
% 对比 500 个模型中不同策略的估计回报分布：
%   recqvi(:,20) = 医生策略；(:,22) = AI 策略；(:,25) = 零药物策略；(:,26) = 随机策略。
% 绿色水平线标出 AI 策略在所有模型中的最高估计值，突出最后选用的策略。

figure
clear h
boxplot(recqvi(:,[20 22 25 26]),{'Clinicians','AI','Zero drug','Random'}); % some evaluations not done here
h=line([1.5 2.5],[max(recqvi(:,22))  max(recqvi(:,22))] ,'LineWidth',2,'color','g');
axis square
axis([0.5 4.5 -100 100])
legend(h,'Chosen policy','location','sw')
ylabel('Estimated policy value')
set(gca,'FontSize',12)


%% FIG 2C = MODEL CALIBRATION

% TD learning of physicians / bootstrapped, in MIMIC train set.
% This version also records action return and mortality, for the plot (nb: no parfor here)

disp('####   MODEL CALIBRATION - CLINICIANS POLICY EVALUATION WITH TD LEARNING   ####')
tic
[bootql,prog]=offpolicy_eval_tdlearning_with_morta( qldata3train, physpol, ptid,  idx, actionbloctrain, Y90, 0.99, 100 ); %100 reps
toc

nbins=100;
a=prog(:,1);  %Q values of actual actions
qv=floor((a+100)/(200/nbins))+1;  % converts Q values to integers btw 0 and nbins
m=prog(:,2);  %outcome
h=zeros(nbins,5);  %avg mortality and other results, per bin
 
% 将 Q 值按 nbins 分箱，计算每个箱对应的真实死亡率与标准误，检查模型校准情况
for i=1:nbins
    
    ii=qv==i;
    h(i,1)=nanmean(m(ii));  %mean mortality in this bin
    if numel(m(ii))>0
     h(i,5)=nanmean(a(ii));  %record the mean of Q values in the bin (to make sure it matches what I expect)
    end
    h(i,2)=std(m(ii))/sqrt(numel(m(ii)));  %SEM of mortality in this bin
    h(i,3)=numel(m(ii));  %nb of data points in this bin
end
 
h(:,4)=h(:,1).*h(:,3)./numel(qv);%weighted average!!
[nansum(h(:,4)) mean(prog(:,2))] %check that both are close!
 
yy1=smooth(1:nbins,h(:,1),0.1,'rloess');  % 用 rloess 对死亡率曲线做平滑，提升图形可读性
figure
hold on
line([0 nbins], [0.5 0.5], 'LineStyle',':','color','k');            % 参考线：50% 死亡率
line([nbins/2 nbins/2], [0 1], 'LineStyle',':','color','k');        % 参考线：Q 值为 0 的分界
 
H=plot(h(:,1),'b','linewidth',1);                                   % 蓝线：分箱平均死亡率
plot(h(:,1)+h(:,2),'b','linewidth',0.5);                            % 虚线：+1 SEM
plot(h(:,1)-h(:,2),'b','linewidth',0.5);                            % 虚线：-1 SEM
 
ylabel('Mortality risk');
xlabel('Return of actions')
axis([0 nbins 0 1]); ax=gca;
ax.XTick=0:nbins/10:nbins; ax.XTickLabel =num2cell(-100:20:100);    % 将横坐标标签换成实际 Q 值（回报）
bw=0.5*200/nbins;  %#ok<NASGU> % 留下 bw 以兼容后续脚本（部分版本可能使用）
H=plot(yy1,'r','linewidth',1);                                      % 红线：平滑后的死亡率曲线
axis square
set(gca,'FontSize',12)
hold off


%% FIG 2D = Computes avg Q value per patient / MIMIC TRAIN SET
% 目的：统计医生策略下，每位患者的平均 Q 值分布，并按存活/死亡分两组比较。
 
r=array2table(prog);
r.Properties.VariableNames = {'Qoff','morta','id','rep'};   % Q 值、结局、患者 ID、bootstrap 编号
d=grpstats(r,{'rep','id'},{'mean','median','sum'});         % 按患者+bootstrap 分组，求均值/中位数/和
edges=-100:5:100;                                           % 直方图的边界

figure
h(1)=histogram(d.mean_Qoff(d.mean_morta==0),edges,'facecolor','b','normalization','probability'); % 蓝色：存活患者
hold on
h(2)=histogram(d.mean_Qoff(d.mean_morta==1),edges,'facecolor','r','normalization','probability'); % 红色：死亡患者
hold off
legend([h(1) h(2)],{'Survivors','Non-survivors'},'location','nw')
axis square
xlabel('Average return per patient')
ylabel('Probability')
set(gca,'FontSize',12)


%% evaluation of chosen model on eICU
% 通过大样本 bootstrap（8000 次采样）评估最优策略在 eICU 数据集上的 TD/Q 和 WIS 表现，输出四分位数供报告使用。

disp('####   TESTING CHOSEN MODEL ON eICU    ####')


tic 
 [ booteicuql,booteicuwis] = offpolicy_multiple_eval_010518( qldata2,physpol, 0.99,1,500,8000);
toc
 
booteicuql=repmat(booteicuql,floor(size(booteicuwis,1)/size(booteicuql,1)),1);  % 若 QL 样本数少于 WIS，复制扩展以对齐维度（方差影响可忽略）

[quantile(booteicuql(:,1),0.25)  quantile(booteicuql(:,1),0.5)   quantile(booteicuql(:,1),0.75)]
[quantile(booteicuwis(:,1),0.25)  quantile(booteicuwis(:,1),0.5)   quantile(booteicuwis(:,1),0.75)]


%% FIG 3A - Heatmap of Q values
% 将 eICU 上 bootstrap 得到的医生策略价值与 AI 策略价值做二维直方图（对数刻度），观察二者的联合分布。
 
a=[booteicuql booteicuwis];                                         % 列 1：医生策略，列 2：AI 策略
[counts] = hist3(a,'Edges',{-105:2.5:100 -105:2.5:100}');           % 以 2.5 为步长划 bin，统计落入每个方格的次数
 
counts = rot90(counts);                                             % 旋转矩阵匹配视觉坐标系
figure
imagesc(log10(counts))                                              % 使用 log10 展示罕见/常见区域
colormap jet
c=colorbar;
c.Label.String = 'Booststrap estimates (log10 scale)';
axis square
hold on
axis([1 83 1 83])
line([1 83],[83 1],'LineWidth',2,'color','w');                      % 白色对角线：AI=Clinician 的等值线
ax = gca;
ax.YTick=1:10:100;
ax.YTickLabel = {'100', '75','50','25','0','-25','-50','-75','-100'}; % 将像素索引转换为实际策略价值
ax.XTick=2:10:100;
ax.XTickLabel = {'-100','-75','-50','-25','0','25','50','75','100'};
xlabel('Clinicans'' policy value')
ylabel('AI policy value')
set(gca,'FontSize',12)
hold off


%%  FIGS 3B3C : 5x5 3D histogram for distrib of action from eICU
% 对 eICU 的实际临床动作与 AI 推荐动作分别绘制 5×5 条形图，
% 每个格子对应 (IV fluids, vasopressor) 档位组合，展示动作分布差异。

nra=5;                                                                % 每种治疗离散成 5 档（含 0 剂量）
iol=find(ismember(MIMICtable.Properties.VariableNames,{'input_4hourly'}));
vcl=find(ismember(MIMICtable.Properties.VariableNames,{'max_dose_vaso'}));
 
a= reformat5(:,iol);                                                 % 使用训练集统计离散化边界：IV 档位
a= tiedrank(a(a>0)) / length(a(a>0));   % excludes zero fluid (will be action 1)
 
        iof=floor((a+0.2499999999)*4);  %converts iv volume in 4 actions
        a= reformat5(:,iol); a=find(a>0);  %location of non-zero fluid in big matrix
        io=ones(size(reformat5,1),1);  %array of ones, by default     
        io(a)=iof+1;   %where more than zero fluid given: save actual action
        vc=reformat5(:,vcl);  vcr= tiedrank(vc(vc~=0)) / numel(vc(vc~=0)); vcr=floor((vcr+0.249999999999)*4);  %converts to 4 bins
        vcr(vcr==0)=1; vc(vc~=0)=vcr+1; vc(vc==0)=1;
        ma1=[ median(reformat5(io==1,iol))  median(reformat5(io==2,iol))  median(reformat5(io==3,iol))  median(reformat5(io==4,iol))  median(reformat5(io==5,iol))];  %median dose of drug in all bins
        ma2=[ median(reformat5(vc==1,vcl))  median(reformat5(vc==2,vcl))  median(reformat5(vc==3,vcl))  median(reformat5(vc==4,vcl))  median(reformat5(vc==5,vcl))] ;
 
med=[io vc];
[uniqueValues,~,actionbloc] = unique(array2table(med),'rows');
actionbloctrain=actionbloc(train);
uniqueValuesdose=[ ma2(uniqueValues.med2)' ma1(uniqueValues.med1)'];  % median dose of each bin for all 25 actions
 
iol=find(ismember(MIMICtable.Properties.VariableNames,{'input_4hourly'}));
vcl=find(ismember(MIMICtable.Properties.VariableNames,{'max_dose_vaso'}));
 ma1=[ max(reformat5(io==1,iol))  max(reformat5(io==2,iol))  max(reformat5(io==3,iol))  max(reformat5(io==4,iol))  max(reformat5(io==5,iol))];  %upper dose of drug in all bins
 ma2=[ max(reformat5(vc==1,vcl))  max(reformat5(vc==2,vcl))  max(reformat5(vc==3,vcl))  max(reformat5(vc==4,vcl))  max(reformat5(vc==5,vcl))] ;
 

% 将 eICU 连续剂量映射到上述离散边界，得到实际动作的档位组合
vct=eICUraw(:,4); vct(vct>ma2(nra-1))=nra; vct(vct==0)=1; for z=2:nra-1; vct(vct>ma2(z-1) & vct<=ma2(z))=z;end
iot=eICUraw(:,45); for z=2:nra-1; iot(iot>ma1(z-1) & iot<=ma1(z))=z; end;iot(iot>ma1(nra-1))=nra;iot(iot==0)=1;
 
med=[iot vct];                                                       % 第一列 IV 档位，第二列 Vaso 档位

 
figure
subplot(1,2,1)   % /////////////   ACTUAL ACTIONS   ////////////////
 
[counts] = hist3(med,'Edges',{1:5 1:5})./size(med,1);                % 统计每个档位组合的频率
 counts = flipud(counts);                                           % 翻转使得高剂量在图的远端
b=bar3(counts);
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end

ax = gca;
ax.YTick=1:5;
ax.XTick=1:5;
ax.YTickLabel = {'>530', '180-530','50-180','1-50','0'};
ax.XTickLabel = {'0', '0.001-0.08','0.08-0.22','0.22-0.45','>0.45'};
view(45,35)
xlabel('Vasopressor dose')
ylabel('     IV fluids dose')
set(get(gca,'YLabel'),'Position',[6, 6, 0]);
set(get(gca,'XLabel'),'Position',[6, 6, 0]);

title('Clinicians'' policy')
c=colorbar;
c.Label.String = '%';
axis square
axis([0.5 5.5 0.5 5.5 0 0.3])
set(gca,'FontSize',12)
  

disp('##########   Clinician   ##########')
disp('  on vaso     ¦ on low fluid')
disp([sum(sum(counts(:,2:5))) sum(sum(counts(4:5,:)))])
disp('  on vaso and low fluids    ¦ on no vaso and high fluid')
disp([sum(sum(counts(3:5,2:5)))  sum(sum(counts(1:2,1)))])
disp('  on low vaso ')
disp([sum(sum(counts(:,2:4)))  ])


subplot(1,2,2)  % /////////////   OPTIMAL ACTIONS   ////////////////
OA1=OptimalAction(idxtest2);%test);              %optimal action for each record
a=[OA1 floor((OA1-0.0001)./5)+1 OA1-floor(OA1./5)*5];              % 把 1~25 的联合动作拆成 5 档补液 × 5 档升压剂
a(a(:,3)==0,3)=5;
med=a(:,[2 3]);                                                     % 列 2：补液档；列 3：升压剂档
[counts] = hist3(med,'Edges',{1:5 1:5})./size(med,1);
  counts = flipud(counts);
 
 b=bar3(counts);
for k = 1:length(b)
    zdata = b(k).ZData;
    b(k).CData = zdata;
    b(k).FaceColor = 'interp';
end

disp('##########   AI Clinician   ##########')
disp('  on vaso     ¦ on low fluid')
disp([sum(sum(counts(:,2:5))) sum(sum(counts(4:5,:)))])
disp('  on vaso and low fluids    ¦ on no vaso and high fluid')
disp([sum(sum(counts(3:5,2:5)))  sum(sum(counts(1:2,1)))])
disp('  on low vaso ')
disp([sum(sum(counts(:,2:4)))  ])


colorbar
ax = gca;
ax.YTick=1:5;
ax.XTick=1:5;
ax.YTickLabel = {'>530', '180-530','50-180','1-50','0'};
ax.XTickLabel = {'0', '0.001-0.08','0.08-0.22','0.22-0.45','>0.45'};
view(45,35)
xlabel('Vasopressor dose')
ylabel('     IV fluids dose')
set(get(gca,'YLabel'),'Position',[6, 6, 0]);
set(get(gca,'XLabel'),'Position',[6, 6, 0]);
title('AI policy')
c=colorbar;
c.Label.String = '%';
axis square
axis([0.5 5.5 0.5 5.5 0 0.3])
set(gca,'FontSize',12)


%% FIGS 3D & 3E : "Ucurves" eICU TEST SET with bootstrapped CI
% 构建 U-curve：横轴为“实际给药-模型推荐”差值，纵轴为死亡率，展示超额/不足用药对预后影响。
% 使用 bootstrap 采样（nr_reps 次）估计均值与标准误，分别对静脉液体 (IVF) 与升压剂 (vaso) 计算。

t=[-1250:100:1250]; t2=[-1.05:0.1:1.05];                           % IVF（毫升）与 Vaso（剂量倍数）的分箱边界

nr_reps=200; 
p=unique(qldata2(:,8));                                           % 患者 ID 列
prop=10000/numel(p); %10k patients of the samples are used
prop=min([prop 0.75]);  %max possible value is 0.75 (75% of the samples are used)

% ACTUAL DATA
disp('U-curves with actual doses...')
% column key:  9 given fluid    10 given vaso    11 model dose fluid     12 model dose vaso
qldata=qldata2(qldata2(:,3)~=0,:);                                 % 过滤掉吸收态行
qldata(:,14)=qldata(:,10)-qldata(:,12);                            % 列14：实际升压剂 - 推荐升压剂
qldata(:,15)=qldata(:,9)-qldata(:,11);                             % 列15：实际静脉液 - 推荐静脉液

r=array2table(qldata(:,[8 13 14 15]));  
r.Properties.VariableNames = {'id','morta','vaso','ivf'};
d=grpstats(r,'id',{'mean','median','sum'});                        % 按患者聚合，得到平均/中位超额剂量与死亡率
d3=([d.mean_morta d.mean_vaso d.mean_ivf d.median_vaso d.median_ivf d.sum_ivf d.GroupCount]);
r1=zeros(numel(t)-1,nr_reps,2);                                    % IVF：存均值/样本数/SEM
r2=zeros(numel(t2)-1,nr_reps,2);                                   % Vaso：存均值/样本数/SEM

for rep=1:nr_reps
    
disp(rep);
ii=floor(rand(size(p,1),1)+prop);     % Bernoulli 采样：约 prop 比例患者进入本次 bootstrap
d4=d3(ii==1,:);

a=[];     % IVF
b=[];     % vasopressors


for i=1:numel(t)-1
    ii=d4(:,5)>=t(i) & d4(:,5)<=t(i+1);  %median
    a=[a ; [t(i) t(i+1) sum(ii) nanmean(d4(ii,1)) nanstd(d4(ii,1))]];
end
r1(:,rep,1)=a(:,4);
r1(:,rep,2)=a(:,3);
r1(:,rep,3)=a(:,5)./sqrt(a(:,3));  % SEM !!

for i=1:numel(t2)-1
    ii=d4(:,4)>=t2(i) & d4(:,4)<=t2(i+1);   %median
    b=[b ; [t2(i) t2(i+1) sum(ii) nanmean(d4(ii,1)) nanstd(d4(ii,1))]];
end
r2(:,rep,1)=b(:,4);
r2(:,rep,2)=b(:,3);
r2(:,rep,3)=b(:,5)./sqrt(b(:,3));  % SEM !!

end

a1=nanmean(r1(:,:,1),2);                                           % IVF：各 bin 的平均死亡率
a2=nanmean(r2(:,:,1),2);                                           % Vaso：各 bin 的平均死亡率


% computing SEM in each bin
s1=nan(numel(t)-1,1);
for i=1:numel(t)-1
s1(i)=nanstd([ones(nansum(r1(i,:,1).*r1(i,:,2) ),1); zeros(nansum((1-r1(i,:,1)).*r1(i,:,2)),1)])/sqrt(nansum(r1(i,:,2)));
end
s2=nan(numel(t2)-1,1);
for i=1:numel(t2)-1
s2(i)=nanstd([ones(nansum(r2(i,:,1).*r2(i,:,2) ),1); zeros(nansum((1-r2(i,:,1)).*r2(i,:,2)),1)])/sqrt(nansum(r2(i,:,2)));
end





%% FIG 3D & 3E - "U-CURVE"  PLOT  ONLY OPTIMAL POLICY
% 使用上面得到的均值与 SEM 绘制 U 曲线，可选平滑显示最优策略推荐剂量时的死亡率走势。
t=[-1250:100:1250]; t2=[-1.05:0.1:1.05];

s=0;  %  !!!!  SMOOTHING FACTOR  !!!! use 0 for no smooth curves
f=10;   %inflation factor for SEM (for visualisation purposes)
figure
if s>0
    
yy1=smooth(1:numel(a1),a1,s,'loess');
yy2=smooth(1:numel(ar1),ar1,s,'loess');
end
subplot(1,2,1)
hold on
h=plot(a1,'b','linewidth',1);                     % 蓝线：IVF 超额剂量 vs 死亡率
plot(a1+f*s1,'b:','linewidth',1)                 % 以 f 倍 SEM 作上下界，强调不确定性
plot(a1-f*s1,'b:','linewidth',1)

plot([numel(a1)/2+.5 numel(a1)/2+.5],[0 1],'k:'); % 竖线：剂量差为 0 的位置
xlabel('Average dose excess per patient')
ylabel('Mortality')
axis([1 numel(a1) 0 1]); ax=gca;
t=t-(t(end)-t(end-1))/2;
t=round(t,2);
t=t(2:2:end);
ax.XTick=1:2:2*numel(t);
ax.XTickLabel =num2cell(t); 
rotateXLabels( gca, 90)
if s>0
plot(yy1,'b','linewidth',2);
plot(yy2,'r','linewidth',2);
end
axis square
title('Intravenous fluids')
set(gca,'FontSize',12)

hold off

subplot(1,2,2)
if s>0
yy1=smooth(1:numel(a2),a2,s,'loess');
yy2=smooth(1:numel(ar2),ar2,s,'loess');
end

hold on
h=plot(a2,'b','linewidth',1);                     % 升压剂 U 曲线
plot(a2+f*s2,'b:','linewidth',1)
plot(a2-f*s2,'b:','linewidth',1)
plot([numel(a2)/2+.5 numel(a2)/2+.5],[0 1],'k:');

xlabel('Average dose excess per patient')
ylabel('Mortality')
axis([1 numel(a2) 0 1]); ax=gca;
t2=t2-(t2(end)-t2(end-1))/2;
t2=round(t2,2);
t2=t2(2:2:end);
ax.XTick=1:2:2*numel(t2);
ax.XTickLabel =num2cell(t2); 
rotateXLabels( gca, 90)
if s>0
plot(yy1,'b','linewidth',2);
plot(yy2,'r','linewidth',2);
end
axis square
title('Vasopressors')
set(gca,'FontSize',12)
hold off




%% FIG SA - FEATURE IMPORTANCE for VASOPRESSORS, with bootstraping
% 通过随机森林 (TreeBagger) + 自助采样估计特征重要性，分别针对医生策略与 AI 推荐的升压剂用药。

nn=100;  %nr bootstraps
fi=zeros(46,nn);
fi2=zeros(46,nn);
colbin = {'gender','mechvent','max_dose_vaso','re_admission'};  %will simply substract 0.5 to center around 0
colnorm={'age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1', 'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium','Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2', 'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index','PaO2_FiO2','cumulated_balance'};
collog={'SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR','input_total','input_4hourly','output_total','output_4hourly'};
v=MIMICtable(1,[colbin colnorm collog]).Properties.VariableNames; %get he right column names!

% REMOVE COL 4 = VASOPRESSORS
v2=v([1:3 5:47]);  %this is the list of (correct) feature names
v2=regexprep(v2,'_',' ');v2=regexprep(v2,' tev','');v2=regexprep(v2,'bp',' BP');

for i=1:nn
    i                                                           % 输出当前 bootstrap 轮次
    grp=floor(100*rand(size(eICUraw,1)-1,1)+1)<=5;              % 随机挑选约 5% 样本作为训练子集

    tic
    %actual policy：医生是否给升压剂 (>0)
    br=TreeBagger(15,eICUraw(grp,[1:3 5:47]),qldata(grp,10)>0,'method','c','maxnumsplits',30,'MinLeafSize',500,'OOBVarImp','on','OOBPred','Off','MinLeaf',150,'PredictorNames',v2);
    fi(:,i)=br.OOBPermutedPredictorDeltaError;
    %optimal policy：AI 推荐是否给升压剂 (>0)
    br2=TreeBagger(15,eICUraw(grp,[1:3 5:47]),qldata(grp,12)>0,'method','c','maxnumsplits',30,'MinLeafSize',500,'OOBVarImp','on','OOBPred','Off','MinLeaf',150,'PredictorNames',v2);
    toc

    fi2(:,i)=br2.OOBPermutedPredictorDeltaError;

end


fi=mean(fi,2);  %average over all models
fi2=mean(fi2,2);


figure
subplot(1,2,1)
[~,i]=sort(fi,'asc');                           % 先按重要性排序，便于观察
barh(fi(i))
ylabel 'Feature'
xlabel 'Out-of-Bag Feature Importance'
ax=gca;
ax.YTick=1:46;
ax.YTickLabel =v2(i);
title('Clinicians'' policy')
set(gca,'FontSize',12)
subplot(1,2,2)
[~,i]=sort(fi2,'asc');
barh(fi2(i))
ylabel 'Feature'
xlabel 'Out-of-Bag Feature Importance'
ax=gca;
ax.YTick=1:46;
ax.YTickLabel =v2(i);
title('AI policy')
set(gca,'FontSize',12)

%% predict IV fluid O/N
% 同样的特征重要性分析，针对静脉补液（医生给药 vs AI 推荐）。

fi=zeros(46,nn);
fi2=zeros(46,nn);
% REMOVE COL 45 = IV Fluids
v2=v([1:44 46:47]);  %this is the list of (correct) feature names
v2=regexprep(v2,'_',' ');v2=regexprep(v2,' tev','');v2=regexprep(v2,'bp',' BP');

for i=1:nn
    grp=floor(100*rand(size(eICUraw,1)-1,1)+1)<5;  %selects a random x% of data for training
    i
    tic  %actual policy
    br=TreeBagger(10,eICUraw(grp,[1:44 46:47]),qldata(grp,9)>0,'method','c','maxnumsplits',30,'MinLeafSize',500,'OOBVarImp','on','OOBPred','Off','MinLeaf',150,'PredictorNames',v2);%100+floor(200*rand()) );
    fi(:,i)=br.OOBPermutedPredictorDeltaError;
    %optimal policy
    br2=TreeBagger(10,eICUraw(grp,[1:44 46:47]),qldata(grp,11)>0,'method','c','maxnumsplits',30,'MinLeafSize',500,'OOBVarImp','on','OOBPred','Off','MinLeaf',150,'PredictorNames',v2);%100+floor(200*rand()) );
    toc

    fi2(:,i)=br2.OOBPermutedPredictorDeltaError;
end

fi=mean(fi,2);
fi2=mean(fi2,2);
figure
subplot(1,2,1)
[~,i]=sort(fi,'asc');
barh(fi(i))
ylabel 'Feature'
xlabel 'Out-of-Bag Feature Importance'
v2=regexprep(v2,'_',' ');
ax=gca;
ax.YTick=1:46;
ax.YTickLabel =v2(i);
title('Clinicians'' policy')
set(gca,'FontSize',12)
subplot(1,2,2)
[~,i]=sort(fi2,'asc');
barh(fi2(i))
ylabel 'Feature'
xlabel 'Out-of-Bag Feature Importance'
v2=regexprep(v2,'_',' ');
ax=gca;
ax.YTick=1:46;
ax.YTickLabel =v2(i);
title('AI policy')
set(gca,'FontSize',12)


%%   #################    NUMERICAL  RESULTS    ##########################

% STATS btw given and reco doses 22/05/17 in EICU

% category 1 = dose excess negative = given less than reco
% category 3 = dose excess positive = given more than reco

% similar dose norad if given is withing +/- 10% of reco or 0.02 mkm or 10 ml/h

% KEY:
%    rectest(:,9)= rectest(:,4)- rectest(:,5);  %given  - reco  VASOPRESSORS
%    rectest(:,10)= rectest(:,6)- rectest(:,7);  %given - reco  FLUIDS

qldata=qldata2(qldata2(:,3)~=0,:);
qldata(:,14)=qldata(:,10)-qldata(:,12);
qldata(:,15)=qldata(:,9)-qldata(:,11);

% VASOPRESSORS
j=abs((qldata(:,10)-qldata(:,12))./(qldata(:,10))).*100;% PCT difference btw given and reco  VASOPRESSORS
qldata(:,17)=abs(qldata(:,14))<=0.02| j<=10;   %close dose
ii=qldata(:,17)==1;   
% sum(ii)/numel(ii)  % how many received close to optimal dose?
qldata(ii,17)=qldata(ii,17)+1;% category 2 = dose similar
ii=qldata(:,17)==0 & qldata(:,14)<0;%less than reco
qldata(ii,17)=1;% category 1
ii=qldata(:,17)==0 & qldata(:,14)>0; %more than reco
qldata(ii,17)=3;% category 3

% stats for all 3 categories
a=[];
for i=1:3
    ii=qldata(:,17)==i;   %rows in qldata who corresp to this category
    j=qldata(ii,14);   %dose

a=[a;    [sum(ii)/numel(ii) mean(qldata(ii,13)) std(qldata(ii,13))./sqrt(sum(ii)) quantile(j,0.25) median(j) quantile(j,0.75)]];
end


% FLUIDS
j= abs((qldata(:,9)-qldata(:,11))./(qldata(:,9))).*100;% PCT difference btw given and reco FLUIDS
qldata(:,18)=j<=10| abs(qldata(:,15))<=40;   %close dose (40 ml/4h = 10 ml / h)
ii=qldata(:,18)==1;   
% sum(ii)/numel(ii) % how many received close to optimal dose?
qldata(ii,18)=qldata(ii,18)+1;   % category 2 = dose similar
ii=qldata(:,18)==0 & qldata(:,15)<0;%less than reco
qldata(ii,18)=1;  %cat 1
ii=qldata(:,18)==0 & qldata(:,15)>0; %more than reco
qldata(ii,18)=3;   % cat 3


% stats for all 3 categories
% a=[];
for i=1:3
    ii=qldata(:,18)==i;   %rows in qldata who corresp to this category
    j=qldata(ii,15)./4;   %dose in ml/h

a=[a;    [sum(ii)/numel(ii) mean(qldata(ii,13)) std(qldata(ii,13))./sqrt(sum(ii)) quantile(j,0.25) median(j) quantile(j,0.75)]];
end

i=a(1,1)/(a(1,1)+a(3,1)); ii=a(6,1)/(a(4,1)+a(6,1));
a=array2table(a);
a=[ array2table({'Vaso: Less than reco', 'Vaso: Similar dose', 'Vaso: More than reco', 'Fluids: Less than reco' ,'Fluids: Similar dose' ,'Fluids: More than reco'}') a];
a.Properties.VariableNames={'category','fraction','avg_mortality','SEM','Q1_dose','Q2_dose','Q3_dose'};

disp(a)
fprintf(' Among patients who did not receive the recommended dose of vasopressors, fraction of patients who received less than recommended : ');       fprintf('%f \n',i); 
fprintf(' Among patients who did not receive the recommended dose of IV fluids, fraction of patients who received more than recommended : ');       fprintf('%f \n',ii);


%% #########    dose of drugs in the 5 bins      ###############

iol=find(ismember(MIMICtable.Properties.VariableNames,{'input_4hourly'}));
vcl=find(ismember(MIMICtable.Properties.VariableNames,{'max_dose_vaso'}));

%boundaries + median doses for each action
disp('VASOPRESSORS')
for i=1:5
[min(reformat5(vc==i,vcl)) median(reformat5(vc==i,vcl)) max(reformat5(vc==i,vcl))]
end

disp('IV FLUIDS')
for i=1:5
[min(reformat5(io==i,iol)) median(reformat5(io==i,iol)) max(reformat5(io==i,iol))]
end
