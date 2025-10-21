%% AI Clinician core code

% (c) Matthieu Komorowski, Imperial College London 2015-2019
% as seen in publication: https://www.nature.com/articles/s41591-018-0213-5

% version 16 Feb 19
% Builds 500 models using MIMIC-III training data
% Records best candidate models along the way from off-policy policy evaluation on MIMIC-III validation data
% Tests the best model on eRI data


% TAKES:
        % MIMICtable = m*59 table with raw values from MIMIC
        % eICUtable = n*56 table with raw values from eICU
        

% GENERATES:
        % MIMICraw = MIMIC RAW DATA m*47 array with columns in right order
        % MIMICzs = MIMIC ZSCORED m*47 array with columns in right order, matching MIMICraw
        % eICUraw = eICU RAW DATA n*47 array with columns in right order, matching MIMICraw
        % eICUzs = eICU ZSCORED n*47 array with columns in right order, matching MIMICraw
        % recqvi = summary statistics of all 500 models
        % idxs = state membership of MIMIC test records, for all 500 models
     	% OA = optimal policy, for all 500 models
        % allpols = detailed data about the best candidate models

% This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
% without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE

% Note: The size of the cohort will depend on which version of MIMIC-III is used.
% The original cohort from the 2018 Nature Medicine publication was built using MIMIC-III v1.3.

% ############################  MODEL PARAMETERS   #####################################


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
    % modl 主循环迭代计数，范围 1:nr_reps
   
  N=numel(icuuniqueids); % total number of rows to choose from
  grp=floor(ncv*rand(N,1)+1);  %list of 1 to 5 (20% of the data in each grp) -- this means that train/test MIMIC split are DIFFERENT in all the 500 models 交叉验证折号，随机抽取 80%/20% 作为训练/测试集
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
 
 a= reformat5(:,iol);                   %IV fluid
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
 
 
% ###################################################################################################################################
% 构造 TD/MDP 输入矩阵 qldata3：
%   - 初始 qldata 包含原始 bloc（时间步），聚类状态 idx，离散动作 actionbloctrain，
%     以及 0/1 死亡标签 Y90 转换出的 ±100 奖励；
%   - 逐行复制到 qldata3，遇到 bloc==1（轨迹结束）时插入一行吸收态
%     [下一 bloc, 终结状态, 动作0, 奖励]，并将奖励写进终结行；
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

 
% ###################################################################################################################################
% 构建转移概率张量 T(S' | S, A)（以列存放 S'，便于按 (S,A) 归一化）
% - transitionr(S1,S0,A) 先累计 (S0→S1 在动作 A 下) 的发生次数
% - sums0a0(S0,A) 记录每个 (S0,A) 的总次数
% - 之后对每个 (S0,A) 的列归一化，得到条件概率 T(S' | S0,A)
disp('####  CREATE TRANSITION MATRIX T(S'',S,A) ####')
 
transitionr=zeros(ncl+2,ncl+2,nact);  % T(S',S,A) 维度：[S' x S x A]
sums0a0=zeros(ncl+2,nact);            % (S,A) 计数表，用于归一化和估计行为策略
 
     for i=1:size(qldata3,1)-1
 
         % 若下一行 bloc != 1，说明轨迹未结束，存在从 S0 到 S1 的一次转移
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
 
% #################################################################################################################################
disp('####  CREATE REWARD MATRIX  R(S,A) ####')
% CF sutton& barto bottom 1998 page 106. i compute R(S,A) from R(S'SA) and T(S'SA)
r3=zeros(ncl+2,ncl+2,nact); r3(ncl+1,:,:)=-100; r3(ncl+2,:,:)=100;
R=sum(transitionr.*r3);
R=squeeze(R);   %remove 1 unused dimension

% ###################################################################################################################################
disp('####   POLICY ITERATION   ####')

 [~,~,~,~,Qon] = mdp_policy_iteration_with_Q(transitionr2, R, gamma, ones(ncl+2,1));
 [~,OptimalAction]=max(Qon,[],2);  %deterministic 
 OA(:,modl)=OptimalAction; %save optimal actions
 
disp('#### OFF-POLICY EVALUATION - MIMIC TRAIN SET ####')
 
% create new version of QLDATA3
r=[100 -100];
r2=r.*(2*(1-Y90)-1); 
qldata=[blocs idx actionbloctrain Y90 zeros(numel(idx),1) r2(:,1) ptid];  % contains bloc / state / action / outcome&reward     %1 = died
qldata3=zeros(floor(size(qldata,1)*1.2),8); 

c=0;
abss=[ncl+2 ncl+1]; %absorbing states numbers
 
        for i=1:size(qldata,1)-1
            c=c+1;
              qldata3(c,:)=qldata(i,[1:3 5 7 7 7 7]);
            if qldata(i+1,1)==1 %end of trace for this patient
                c=c+1;
                qldata3(c,:)=[qldata(i,1)+1 abss(1+qldata(i,4)) 0 qldata(i,6) 0 0 0 qldata(i,7)]; 
            end
        end
        qldata3(c+1:end,:)=[];

% add pi(s,a) and b(s,a)
p=0.01; %softening policies  
softpi=physpol; % behavior policy = clinicians' 

for i=1:750
    ii=softpi(i,:)==0;    z=p/sum(ii);    nz=p/sum(~ii);    softpi(i,ii)=z;   softpi(i,~ii)=softpi(i,~ii)-nz;
end
softb=abs(zeros(752,25)-p/24); %"optimal" policy = target policy = evaluation policy 

for i=1:750
     softb(i,OptimalAction(i))=1-p;
end

for i=1:size(qldata3,1)  %adding the probas of policies to qldata3
    if qldata3(i,2)<=750
qldata3(i,5)=softpi(qldata3(i,2),qldata3(i,3));
qldata3(i,6)=softb(qldata3(i,2),qldata3(i,3));
qldata3(i,7)=OptimalAction(qldata3(i,2));   %optimal action
    end
end

qldata3train=qldata3;

tic
 [ bootql,bootwis ] = offpolicy_multiple_eval_010518( qldata3,physpol, 0.99,1,6,750);
toc

recqvi(modl,1)=modl;
recqvi(modl,4)=nanmean(bootql);
recqvi(modl,5)=quantile(bootql,0.99);
recqvi(modl,6)=nanmean(bootwis);  %we want this as high as possible
recqvi(modl,7)=quantile(bootwis,0.05);  %we want this as high as possible


% testing on MIMIC-test
disp('#### OFF-POLICY EVALUATION - MIMIC TEST SET ####')
    
% create new version of QLDATA3 with MIMIC TEST samples
idxtest=knnsearch(C,Xtestmimic);
idxs(test,modl)=idxtest;  %important: record state membership of test cohort

actionbloctest=actionbloc(~train);
Y90test=reformat5(~train,outcome);
r=[100 -100];
r2=r.*(2*(1-Y90test)-1); 
qldata=[bloctestmimic idxtest actionbloctest Y90test zeros(numel(idxtest),1) r2(:,1) ptidtestmimic];  % contains bloc / state / action / outcome&reward     %1 = died
qldata3=zeros(floor(size(qldata,1)*1.2),8); 

c=0;
abss=[ncl+2 ncl+1]; %absorbing states numbers
 
        for i=1:size(qldata,1)-1
            c=c+1; qldata3(c,:)=qldata(i,[1:3 5 7 7 7 7]);
            if qldata(i+1,1)==1 %end of trace for this patient
                c=c+1; qldata3(c,:)=[qldata(i,1)+1 abss(1+qldata(i,4)) 0 qldata(i,6) 0 0 0 qldata(i,7)]; 
            end
        end
        qldata3(c+1:end,:)=[];

% add pi(s,a) and b(s,a)
p=0.01; %small correction factor // softening policies
softpi=physpol; % behavior policy = clinicians'
for i=1:750;  ii=softpi(i,:)==0;    z=p/sum(ii);    nz=p/sum(~ii);    softpi(i,ii)=z;   softpi(i,~ii)=softpi(i,~ii)-nz; end
softb=abs(zeros(752,25)-p/24); %"optimal" policy = target policy = evaluation policy
for i=1:750;softb(i,OptimalAction(i))=1-p;end

for i=1:size(qldata3,1)  %adding the probas of policies to qldata
    if qldata3(i,2)<=750
qldata3(i,5)=softpi(qldata3(i,2),qldata3(i,3));
qldata3(i,6)=softb(qldata3(i,2),qldata3(i,3));
qldata3(i,7)=OptimalAction(qldata3(i,2));   %optimal action
    end
end

qldata3test=qldata3;

tic
[ bootmimictestql,bootmimictestwis ] = offpolicy_multiple_eval_010518( qldata3,physpol, 0.99,1,6,2000);
toc

recqvi(modl,19)=quantile(bootmimictestql,0.95);   %PHYSICIANS' 95% UB
recqvi(modl,20)=nanmean(bootmimictestql);
recqvi(modl,21)=quantile(bootmimictestql,0.99);
recqvi(modl,22)=nanmean(bootmimictestwis);    
recqvi(modl,23)=quantile(bootmimictestwis,0.01);  
recqvi(modl,24)=quantile(bootmimictestwis,0.05);  %AI 95% LB, we want this as high as possible


if recqvi(modl,24) > 40 %saves time if policy is not good on MIMIC test: skips to next model

disp('########################## eICU TEST SET #############################')

  idxtest2=cell(size(eICUzs,1),1);
        ii=isnan(eICUzs);
        disp('####   IDENTIFY STATE MEMBERSHIP OF eICU TEST RECORDS   ####')
    tic
      parfor i=1:size(eICUzs,1)
        idxtest2(i)={knnsearch(C(:,~ii(i,:)),eICUzs(i,~ii(i,:)))};  %which ones are the k closest records in Xtrain? - only match on available data (ii columns)!
      end
    toc
    
  idxtest2=cell2mat(idxtest2);

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
modeldosevaso = uniqueValuesdose(models,1);      %dose reco in this model
modeldosefluid = uniqueValuesdose(models,2);     %dose reco in this model



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


% add pi(s,a) and b(s,a)
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

recqvi(:,31:end)=[];

r=recqvi;
r(:,30:end)=[];
r(r(:,14)<0,:)=[];  %delete models with poor value in MIMIC test set

% SORT RECQVI BY COL 24 / DESC
bestpol=r(max(r(:,24))==r(:,24),1);   % model maximising 95% LB of value of AI policy in MIMIC test set


%% RECOVER BEST MODEL and TEST IT
disp('####   RECOVER BEST MODEL   ####')
a=cell2mat(allpols(:,1));
outcome =10; %   HOSPITAL MORTALITY = 8 / 90d MORTA = 10
ii=find(a==bestpol); %position of best model in the array allpols

% RECOVER MODEL DATA
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
disp('####   IDENTIFY STATE MEMBERSHIP OF eICU TEST RECORDS   ####')
  idxtest2=cell(size(eICUzs,1),1);
        ii=isnan(eICUzs);
    tic
      parfor i=1:size(eICUzs,1)
        idxtest2(i)={knnsearch(C(:,~ii(i,:)),eICUzs(i,~ii(i,:)))};  %which ones are the k closest records in Xtrain? - only match on available data (ii columns)!
      end
    toc
    
  idxtest2=cell2mat(idxtest2);



%% FIB 2A plot safety of algos: 95th UB of physicians policy value vs 95th LB of AI policy
% during bulding of 500 different models
% show that the value of AI policy is always guaranteed to be better than doctors' according to the model

clear h
r=recqvi;   %MAKE SURE RECQVI IS SORTED BY MODEL NUMBER!!!

m=zeros(size(r,1),1);
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
for i=1:size(r,1)
if r(i,24)>max(m)  %learnt policy
m(i)=r(i,24);
else
m(i)=max(m);
end
end
h(2)=semilogx(m,'linewidth',2);


m=zeros(size(r,1),1);
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
 
yy1=smooth(1:nbins,h(:,1),0.1,'rloess');
figure
hold on
line([0 nbins], [0.5 0.5], 'LineStyle',':','color','k');
line([nbins/2 nbins/2], [0 1], 'LineStyle',':','color','k');
 
H=plot(h(:,1),'b','linewidth',1);
plot(h(:,1)+h(:,2),'b','linewidth',0.5);
plot(h(:,1)-h(:,2),'b','linewidth',0.5);
 
ylabel('Mortality risk');
xlabel('Return of actions')
axis([0 nbins 0 1]); ax=gca;
ax.XTick=0:nbins/10:nbins; ax.XTickLabel =num2cell(-100:20:100);
bw=0.5*200/nbins;
H=plot(yy1,'r','linewidth',1);
axis square
set(gca,'FontSize',12)
hold off


%% FIG 2D = Computes avg Q value per patient / MIMIC TRAIN SET
  
r=array2table(prog);
r.Properties.VariableNames = {'Qoff','morta','id','rep'};
d=grpstats(r,{'rep','id'},{'mean','median','sum'});
edges=-100:5:100;

figure
h(1)=histogram(d.mean_Qoff(d.mean_morta==0),edges,'facecolor','b','normalization','probability');
hold on
h(2)=histogram(d.mean_Qoff(d.mean_morta==1),edges,'facecolor','r','normalization','probability');
hold off
legend([h(1) h(2)],{'Survivors','Non-survivors'},'location','nw')
axis square
xlabel('Average return per patient')
ylabel('Probability')
set(gca,'FontSize',12)


%% evaluation of chosen model on eICU

disp('####   TESTING CHOSEN MODEL ON eICU    ####')


tic 
 [ booteicuql,booteicuwis] = offpolicy_multiple_eval_010518( qldata2,physpol, 0.99,1,500,8000);
toc
  
booteicuql=repmat(booteicuql,floor(size(booteicuwis,1)/size(booteicuql,1)),1);  % copy-paste the array, variance is low anyway

[quantile(booteicuql(:,1),0.25)  quantile(booteicuql(:,1),0.5)   quantile(booteicuql(:,1),0.75)]
[quantile(booteicuwis(:,1),0.25)  quantile(booteicuwis(:,1),0.5)   quantile(booteicuwis(:,1),0.75)]


%% FIG 3A - Heatmap of Q values
 
a=[booteicuql booteicuwis];
[counts] = hist3(a,'Edges',{-105:2.5:100 -105:2.5:100}');
 
counts = rot90(counts);
figure
imagesc(log10(counts))
colormap jet
c=colorbar;
c.Label.String = 'Booststrap estimates (log10 scale)';
axis square
hold on
axis([1 83 1 83])
line([1 83],[83 1],'LineWidth',2,'color','w');
ax = gca;
ax.YTick=1:10:100;
ax.YTickLabel = {'100', '75','50','25','0','-25','-50','-75','-100'};
ax.XTick=2:10:100;
ax.XTickLabel = {'-100','-75','-50','-25','0','25','50','75','100'};
xlabel('Clinicans'' policy value')
ylabel('AI policy value')
set(gca,'FontSize',12)
hold off


%%  FIGS 3B3C : 5x5 3D histogram for distrib of action from eICU   

nra=5;
iol=find(ismember(MIMICtable.Properties.VariableNames,{'input_4hourly'}));
vcl=find(ismember(MIMICtable.Properties.VariableNames,{'max_dose_vaso'}));
 
 a= reformat5(:,iol);                   %IV fluid
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
 

% define actionbloctest = which actions are taken in the test set ????
vct=eICUraw(:,4); vct(vct>ma2(nra-1))=nra; vct(vct==0)=1; for z=2:nra-1; vct(vct>ma2(z-1) & vct<=ma2(z))=z;end
iot=eICUraw(:,45); for z=2:nra-1; iot(iot>ma1(z-1) & iot<=ma1(z))=z; end;iot(iot>ma1(nra-1))=nra;iot(iot==0)=1;
 
med=[iot vct];

 
figure
subplot(1,2,1)   % /////////////   ACTUAL ACTIONS   ////////////////
 
[counts] = hist3(med,'Edges',{1:5 1:5})./size(med,1);
 counts = flipud(counts);
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
a=[OA1 floor((OA1-0.0001)./5)+1 OA1-floor(OA1./5)*5];
a(a(:,3)==0,3)=5;
med=a(:,[2 3]);
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

t=[-1250:100:1250]; t2=[-1.05:0.1:1.05];

nr_reps=200; 
p=unique(qldata2(:,8));
prop=10000/numel(p); %10k patients of the samples are used
prop=min([prop 0.75]);  %max possible value is 0.75 (75% of the samples are used)

% ACTUAL DATA
disp('U-curves with actual doses...')
% column key:  9 given fluid    10 given vaso    11 model dose fluid     12 model dose vaso
qldata=qldata2(qldata2(:,3)~=0,:);
qldata(:,14)=qldata(:,10)-qldata(:,12);
qldata(:,15)=qldata(:,9)-qldata(:,11);

r=array2table(qldata(:,[8 13 14 15]));  
r.Properties.VariableNames = {'id','morta','vaso','ivf'};
d=grpstats(r,'id',{'mean','median','sum'});
d3=([d.mean_morta d.mean_vaso d.mean_ivf d.median_vaso d.median_ivf d.sum_ivf d.GroupCount]);
r1=zeros(numel(t)-1,nr_reps,2);
r2=zeros(numel(t2)-1,nr_reps,2);

for rep=1:nr_reps
    
disp(rep);
ii=floor(rand(size(p,1),1)+prop);     % select a random sample of trajectories
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

a1=nanmean(r1(:,:,1),2);
a2=nanmean(r2(:,:,1),2);


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
h=plot(a1,'b','linewidth',1);
plot(a1+f*s1,'b:','linewidth',1)
plot(a1-f*s1,'b:','linewidth',1)

plot([numel(a1)/2+.5 numel(a1)/2+.5],[0 1],'k:');
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
h=plot(a2,'b','linewidth',1);
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
    i
grp=floor(100*rand(size(eICUraw,1)-1,1)+1)<=5;  %selects a random x% of data for training

tic
%actual policy
br=TreeBagger(15,eICUraw(grp,[1:3 5:47]),qldata(grp,10)>0,'method','c','maxnumsplits',30,'MinLeafSize',500,'OOBVarImp','on','OOBPred','Off','MinLeaf',150,'PredictorNames',v2);
fi(:,i)=br.OOBPermutedPredictorDeltaError;
%optimal policy
br2=TreeBagger(15,eICUraw(grp,[1:3 5:47]),qldata(grp,12)>0,'method','c','maxnumsplits',30,'MinLeafSize',500,'OOBVarImp','on','OOBPred','Off','MinLeaf',150,'PredictorNames',v2);
toc

fi2(:,i)=br2.OOBPermutedPredictorDeltaError;

end


fi=mean(fi,2);  %average over all models
fi2=mean(fi2,2);


figure
subplot(1,2,1)
[~,i]=sort(fi,'asc');
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
