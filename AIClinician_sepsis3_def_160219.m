%% AI Clinician Identifiying MIMIC-III sepsis cohort
% 结构
%%Lines 1-45 (AIClinician_sepsis3_def_160219.m:1): 
% 文件头注释、加载 reference_matrices.mat 中的映射表，并检查所需 MATLAB 工具箱是否安装。
%%Lines 47-109 (AIClinician_sepsis3_def_160219.m:47): 
% 逐个读取 exportdir/ 下的所有 CSV（培养、微生物、抗生素、人口学、各段 ce*.csv、化验、出入量、升压药、机械通气），为后续处理载入内存。
%%Lines 110-245 (AIClinician_sepsis3_def_160219.m:110): 
% 清洗微生物/培养表，填充缺失 icustay_id；修正人口学表中的 NaN；补充输液速率；将抗生素记录与具体 ICU stay 对齐。
%%Lines 246-308 (AIClinician_sepsis3_def_160219.m:246): 
% 按 Sepsis‑3 “疑似感染”规则（培养与抗生素的时间窗）为每个 ICU stay 计算感染起点 onset。
%%Lines 309-361 (AIClinician_sepsis3_def_160219.m:309): 
% 把生命体征与化验的 itemid 映射成 Refvitals/Reflabs 的列索引，便于后续直接落位到特征矩阵。
%%Lines 362-508 (AIClinician_sepsis3_def_160219.m:362): 
% 在感染起点前后 48h/24h 窗口内抽取生命体征、化验和机械通气时间点，合并成宽表 reformat，记录每个 ICU stay 的时间范围 qstime。
%%Lines 509-748 (AIClinician_sepsis3_def_160219.m:509): 
% 异常值剔除与互推逻辑（血压、FiO2、体温、血常规等），以及依据 O2 设备/流量推断缺失 FiO2、互补血压/温度/血红蛋白等派生值。
%%Lines 750-757 (AIClinician_sepsis3_def_160219.m:750): 
% 对原始宽表应用 Sample-and-Hold，将短期缺失用最近观测前向填充。
%%Lines 759-967 (AIClinician_sepsis3_def_160219.m:759): 
% 以 4 小时为步长聚合到 reformat2，同时计算每槽的液体入量、尿量、升压药统计及累计量。
%%Lines 968-986 (AIClinician_sepsis3_def_160219.m:968): 
% 将 reformat2 转成 table，丢弃缺失率 ≥70% 的变量，形成 reformat3t。
%%Lines 987-1028 (AIClinician_sepsis3_def_160219.m:987): 
% 对缺失率 <5% 的变量做线性插值，再分块执行 KNN 插补，生成 reformat4t。
%%Lines 1029-1133 (AIClinician_sepsis3_def_160219.m:1029): 
% 派生与规范化变量（性别、年龄、机械通气、升压药、P/F、Shock Index），并逐槽计算 SOFA 与 SIRS 评分。
%%Lines 1134-1197 (AIClinician_sepsis3_def_160219.m:1134): 
% 按启发式规则剔除异常住院（极端出入量、胆红素、疑似撤除治疗、窗口内死亡等）以清洗队列。
%%Lines 1198-1249 (AIClinician_sepsis3_def_160219.m:1198): 
% 汇总每个 ICU stay 的最大 SOFA/SIRS 与结局信息，筛选 max SOFA ≥2 的住院，输出 sepsis_mimiciii.csv 并保存备份 MAT 文件。
% (c) Matthieu Komorowski, Imperial College London 2015-2019
% as seen in publication: https://www.nature.com/articles/s41591-018-0213-5

% version 16 Feb 19
% IDENTIFIES THE COHORT OF PATIENTS WITH SEPSIS in MIMIC-III

% PURPOSE:
% ------------------------------
% This creates a list of icustayIDs of patients who develop sepsis at some point 
% in the ICU. records charttime for onset of sepsis. Uses sepsis3 criteria

% STEPS:
% -------------------------------
% IMPORT DATA FROM CSV FILES 从 CSV 文件导入数据
% FLAG PRESUMED INFECTION 标记疑似感染
% PREPROCESSING 预处理
% REFORMAT in 4h time slots 在 4 小时时间段内重新格式化
% COMPUTE SOFA at each time step 在每个时间步计算 SOFA
% FLAG SEPSIS 标记败血症

% note: the process generates the same features as the final MDP dataset, most of which are not used to compute SOFA
% External files required: Reflabs, Refvitals, sample_and_hold (all saved in reference_matrices.mat file)

% This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
% without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE
% reference_matrices.mat 里面包含了 Reflabs(35x11), Refvitals(28x10), sample_and_hold(2x65) 三个表
load('reference_matrices.mat', 'Reflabs', 'Refvitals', 'sample_and_hold');

v = ver;                          % 获取当前已安装的工具箱信息
installed = {v.Name};             % 提取名称列表
requiredToolboxes = { ...
    'Statistics and Machine Learning Toolbox', ...
    'Bioinformatics Toolbox' ...
};

for i = 1:numel(requiredToolboxes)
    if ~ismember(requiredToolboxes{i}, installed)
        error(['❌ Missing required toolbox: "', requiredToolboxes{i}, ...
               '". Please install it via Add-On Explorer before running this script.']);
    end
end

disp('✅ All required toolboxes are installed. Proceeding...');

%% IMPORT ALL DATA 逐个读取csv中的文件，载入内存
disp('IMPORT ALL DATA START')
% 计时器启动
tic

% 'culture' items
culture=table2array(readtable('./exportdir/culture.csv'));
%  Microbiologyevents
microbio=table2array(readtable('./exportdir/microbio.csv'));

% 调试需要，读取一个 microbo2
microbio2=table2array(readtable('./exportdir/microbio.csv'));



% Antibiotics administration
abx=table2array(readtable('./exportdir/abx.csv'));
% Demographics
demog=(readtable('./exportdir/demog.csv'));

% Vitals from Chartevents
% Divided into 10 chunks for speed. Each chunk is around 170 MB.
ce010=table2array(readtable('./exportdir/ce010000.csv'));
ce1020=table2array(readtable('./exportdir/ce1000020000.csv'));
ce2030=table2array(readtable('./exportdir/ce2000030000.csv'));
ce3040=table2array(readtable('./exportdir/ce3000040000.csv'));
ce4050=table2array(readtable('./exportdir/ce4000050000.csv'));
ce5060=table2array(readtable('./exportdir/ce5000060000.csv'));
ce6070=table2array(readtable('./exportdir/ce6000070000.csv'));
ce7080=table2array(readtable('./exportdir/ce7000080000.csv'));
ce8090=table2array(readtable('./exportdir/ce8000090000.csv'));
ce90100=table2array(readtable('./exportdir/ce90000100000.csv'));

% Labs from Chartevents and Labs from Labevents
labU=[ table2array(readtable('./exportdir/labs_ce.csv')) ; table2array(readtable('./exportdir/labs_le.csv'))  ];

% Real-time UO
UO=table2array(readtable('./exportdir/uo.csv'));
% Pre-admission UO
UOpreadm=table2array(readtable('./exportdir/preadm_uo.csv'));

% Real-time input from metavision
inputMV=table2array(readtable('./exportdir/fluid_mv.csv'));
% Real-time input from carevue
inputCV=table2array(readtable('./exportdir/fluid_cv.csv'));

% Pre-admission fluid intake
inputpreadm=table2array(readtable('./exportdir/preadm_fluid.csv'));

% Vasopressors from metavision
vasoMV=table2array(readtable('./exportdir/vaso_mv.csv'));
% Vasopressors from carevue
vasoCV=table2array(readtable('./exportdir/vaso_cv.csv'));

% Mechanical ventilation
MV=table2array(readtable('./exportdir/mechvent.csv'));



% 计时器结束
toc
disp('IMPORT ALL DATA END')

%%INITIAL DATA MANIPULATIONS 简单清洗

disp('INITIAL DATA MANIPULATIONS START')
% 将 microbio 表中缺失的 charttime 用 chartdate 填补后，删除原 chartdate 列，
% 并插入占位列以匹配后续格式

% 调试需要，把microbo2赋值给microbo
microbio = microbio2;


ii=isnan(microbio(:,3));  %if charttime is empty but chartdate isn't
microbio(ii,3)=microbio(ii,4);   %copy time
microbio( :,4)=[];    %delete chardate
% Add empty col in microbio (# 3 and #5)
microbio(:,4)=microbio(:,3);
microbio(:,[3 5])=0;
% 将 microbio 与培养结果 culture 纵向拼接成统一的 bacterio 事件表，便于同时遍历微生物与培养记录
bacterio = [microbio ; culture];

% 示例
%{
原始 microbio:
subject_id|hadm_id|charttime|chartdate
（使用的数值矩阵，没有表头这个概念，此处便于理解添加了表头） 
96|170324|5878534680.0|5878483200.0
96|170324|5879135400.0|5879088000.0
96|170324|5879145600.0|5879088000.0
96|170324||5879088000.0
处理后 microbio:
subject_id|hadm_id|icustay_id|charttime|itemid
（使用的数值矩阵，没有表头这个概念，此处便于理解添加了表头） 
96|170324|0|5878534680.0|0|
96|170324|0|5879135400.0|0|
96|170324|0|5879145600.0|0|
96|170324|0|5879088000.0|0|
---
culture:
subject_id|hadm_id|icustay_id|charttime|itemid
（使用的数值矩阵，没有表头这个概念，此处便于理解添加了表头） 
2|163353|243653.0|5318688000.0|3333
5|178980|214757.0|4199839200.0|3333
7|118037|236754.0|4777583400.0|3333
8|159514|262299.0|4666858200.0|3333
---
microbio在上，culture在下拼接
bacterio:
subject_id|hadm_id|icustay_id|charttime|itemid
（使用的数值矩阵，没有表头这个概念，此处便于理解添加了表头） 
96|170324|0|5878534680.0|0|
96|170324|0|5879135400.0|0|
96|170324|0|5879145600.0|0|
96|170324|0|5879088000.0|0|
2|163353|243653.0|5318688000.0|3333
5|178980|214757.0|4199839200.0|3333
7|118037|236754.0|4777583400.0|3333
8|159514|262299.0|4666858200.0|3333
%}

% 给人口学表 demog 中的死亡、Elixhauser 指数等缺失值置零，避免后续运算受到 NaN 影响
% correct NaNs in DEMOG
demog.morta_90(isnan(demog.morta_90))=0;
demog.morta_hosp(isnan(demog.morta_hosp))=0;
demog.elixhauser(isnan(demog.elixhauser))=0;


% compute normalized rate of infusion
% if we give 100 ml of hypertonic fluid (600 mosm/l) at 100 ml/h (given in 1h) it is 200 ml of NS equivalent
% so the normalized rate of infusion is 200 ml/h (different volume in same duration)
% 计算标准化的液体输入速率（ml/h）
% Col5: equivalent amount (mL, actual fluid volume over the interval)  FYI: AIClinician_Data_extract_MIMIC3_140219.ipynb section " Real-time input from metavision "
% Col6: rate (mL/h, pump setting; NaN if IV push / unknown)
% Col7: TEV (mL, total NS-equivalent volume for Col5)
% Col8: NS-equivalent infusion rate (mL/h) = rate * (TEV / amount)
inputMV(:,8) = inputMV(:,6) .* (inputMV(:,7) ./ inputMV(:,5));

% 说明：
% 1) 若Col6为NaN（静推无恒定速率），Col8也将为NaN——这是期望行为。
% 2) 若仅有TEV与给药起止时间，可用“TEV / 时长(h)”计算平均NS等效速率作为备选。
% 3) 计算TEV时建议以0.9%NS的osmolarity≈308 mOsm/L为基准：
%    TEV = amount * (osm_solution / 308)，此式源自osmolarity定义（osmoles/L）。


% fill-in missing ICUSTAY IDs in bacterio
% 前面将 microbio 和 culture 组合在一起时，microbio的第三列（icustay_id）为0，此时需要填充回来
% 目的是在拿到微生物/培养记录时即便原始提取缺了 icustay_id，也能通过患者信息和时间范围推断出
% 属于哪次 ICU 住院，方便后续按 ICU stay 分析脓毒症事件。
for i=1:size(bacterio,1)
if bacterio(i,3)==0   %if missing icustayid
    % 定义表头
    o=bacterio(i,4);  %charttime
    subjectid=bacterio(i,1);
    hadmid=bacterio(i,2);
   ii=find(demog.subject_id==subjectid);%返回的是列表，包含所有符合条件的demog索引值
   jj=find(demog.subject_id==subjectid & demog.hadm_id==hadmid);
    for j=1:numel(ii)
        % sepsis3 定义：前后48小时
        if o>=demog.intime(ii(j))-48*3600 & o<=demog.outtime(ii(j))+48*3600
            bacterio(i,3)=demog.icustay_id(ii(j));
        % 兜底，目标病人只有唯一一次 ICU 住院，直接认定这条细菌培养记录对应那次住院
        elseif numel(ii)==1   %if we cant confirm from admission and discharge time but there is only 1 admission: it's the one!!
            bacterio(i,3)=demog.icustay_id(ii(j));
        end
    end
end   
end
toc

% 仅靠 subject_id + hadm_id”基础上再做一次时间校验和兜底
for i=1:size(bacterio,1)
if bacterio(i,3)==0   %if missing icustayid
    subjectid=bacterio(i,1);
    hadmid=bacterio(i,2);
    jj=find(demog.subject_id==subjectid & demog.hadm_id==hadmid);
    if numel(jj)==1
        bacterio(i,3)=demog.icustay_id(jj);
    end
end
end

% fill-in missing ICUSTAY IDs in ABx
% 和上面同理，确保抗生素记录能关联到具体 ICU stay
for i=1:size(abx,1)
if isnan(abx(i,2))
    o=abx(i,3);  %time of event
    hadmid=abx(i,1);
    ii=find(demog.hadm_id==hadmid);   %row in table demographics
    for j=1:numel(ii)
        if o>=demog.intime(ii(j))-48*3600 & o<=demog.outtime(ii(j))+48*3600
            abx(i,2)=demog.icustay_id(ii(j));
        elseif numel(ii)==1   %if we cant confirm from admission and discharge time but there is only 1 admission: it's the one!!
            abx(i,2)=demog.icustay_id(ii(j));
        end
    end
end   
end
disp('INITIAL DATA MANIPULATIONS END')
%%    find presumed onset of infection according to sepsis3 guidelines

disp('FIND PRESUMED ONSET OF INFECTION ACCORDING TO SEPSIS3 GUIDELINES START')
% METHOD:
% I loop through all the ABx (AntiBiotic eXecution) given, and as soon as there
% is a sample present within the required time criteria I pick this flag and
% break the loop.

% 初始化疑似感染开始时间 onset矩阵：预留 100000 行（每行对应一个 ICU stay），三列分别存放
% 1) subject_id，2) icustay_id，3) 推断的感染发生时间。若后续 ICU stay 数量不足 100000，多余行会保持为零。
% 详细信息：
% 第1列：subject_id
% 第2列：ICU stay 的内部索引（循环变量 icustayid，1..100000；真实 icustay_id = icustayid + 200000）
% 第3列：依据 Sepsis‑3 规则推断的“疑似感染起点时间”（秒）。没有命中则为 0。
onset = zeros(100000, 3);

for icustayid = 1:100000

    % 按照 ICU stay（原始 id +200000）取出该住院的所有抗生素给药时间戳，单位秒
    % 在 sepsis_mimiciii.csv 中，icustayid是以200000开头的，例如200003，200855，203232等
    ab = abx(abx(:,2) == icustayid + 200000, 3);

    % 为同一 ICU stay 抽取所有微生物培养采样时间戳，以及对应的 subject_id
    bact = bacterio(bacterio(:,3) == icustayid + 200000, 4);
    subj_bact = bacterio(bacterio(:,3) == icustayid + 200000, 1);

    % 只有同时存在抗生素和培养记录时才有可能识别感染起点
    if ~isempty(ab) && ~isempty(bact)

        % 计算所有抗生素与培养时间的两两距离，换算成小时，为判定时间窗做准备
        D = pdist2(ab, bact) / 3600;

        % 把抗生素给药按时间顺序逐条检查，从最早的事件往后找匹配的培养记录
        for i = 1:size(D, 1)
            [M, I] = min(D(i,:));    % 找出当前抗生素与所有培养之间的最小时间差
            ab1 = ab(i);
            bact1 = bact(I);
            % If the culture is obtained, the antibiotic is required to be administered within 72 hours, whereas if the antibiotic is first, the culture is required within 24 hours.
            % 详细信息见：https://pmc.ncbi.nlm.nih.gov/articles/PMC4968574/#S17:~:text=For%20example%2C%20if%20the%20culture%20is%20obtained%2C%20the%20antibiotic%20is%20required%20to%20be%20administered%20within%2072%20hours%2C%20whereas%20if%20the%20antibiotic%20is%20first%2C%20the%20culture%20is%20required%20within%2024%20hours.
            % 情形1：先给抗生素、后采样，间隔不超过 24 小时 -> 取抗生素时间为感染起点
            if M <= 24 && ab1 <= bact1
                onset(icustayid,1) = subj_bact(1);
                onset(icustayid,2) = icustayid;
                onset(icustayid,3) = ab1;
                break

            % 情形2：先采样、后给抗生素，间隔不超过 72 小时 -> 取采样时间为感染起点
            elseif M <= 72 && ab1 >= bact1
                onset(icustayid,1) = subj_bact(1);
                onset(icustayid,2) = icustayid;
                onset(icustayid,3) = bact1;
                break
            end
        end
    end
end
toc

%sum of records found
% 有多少次 ICU 住院被标记为感染
sum(onset(:,3)>0)
disp('FIND PRESUMED ONSET OF INFECTION ACCORDING TO SEPSIS3 GUIDELINES END')

%% Replacing item_ids with column numbers from reference tables
disp('REPLACING ITEM_IDS WITH COLUMN NUMBERS FROM REFERENCE TABLES START')
% replace itemid in labs with column number
% this will accelerate process later

% 把实验室和生命体征记录里原本的 itemid（即项目编号）替换成对应参考表中的“列号”，
% 以便后续把这些记录直接映射到特征矩阵的列索引上，提升后续处理效率。


% 把每条化验记录的原始 itemid（labs_ce.csv的第三列）替换为“规范化概念的行号”，也就是把 itemid 映射成 reference_matrices.mat 中 Reflabs 的索引号。
tic
for i=1:size(labU,1)
[~,locb]=ismember(Reflabs,labU(i,3));
labU(i,3)=find(max(locb')');
end
toc

% replace itemid in vitals with col number
% 同理，把生命体征记录的 itemid 替换为 Refvitals 的索引号
for i=1:size(ce010,1)
[~,locb]=ismember(Refvitals,ce010(i,3));ce010(i,3)=find(max(locb')');
end
for i=1:size(ce1020,1)
[~,locb]=ismember(Refvitals,ce1020(i,3));ce1020(i,3)=find(max(locb')');
end
for i=1:size(ce2030,1)
[~,locb]=ismember(Refvitals,ce2030(i,3));ce2030(i,3)=find(max(locb')');
end
for i=1:size(ce3040,1)
[~,locb]=ismember(Refvitals,ce3040(i,3));ce3040(i,3)=find(max(locb')');
end
for i=1:size(ce4050,1)
[~,locb]=ismember(Refvitals,ce4050(i,3));ce4050(i,3)=find(max(locb')');
end
for i=1:size(ce5060,1)
[~,locb]=ismember(Refvitals,ce5060(i,3));ce5060(i,3)=find(max(locb')');
end
for i=1:size(ce6070,1)
[~,locb]=ismember(Refvitals,ce6070(i,3));ce6070(i,3)=find(max(locb')');
end
for i=1:size(ce7080,1)
[~,locb]=ismember(Refvitals,ce7080(i,3));ce7080(i,3)=find(max(locb')');
end
for i=1:size(ce8090,1)
[~,locb]=ismember(Refvitals,ce8090(i,3));ce8090(i,3)=find(max(locb')');
end
for i=1:size(ce90100,1)
[~,locb]=ismember(Refvitals,ce90100(i,3));ce90100(i,3)=find(max(locb')');
end

disp('REPLACING ITEM_IDS WITH COLUMN NUMBERS FROM REFERENCE TABLES END')


%%           INITIAL REFORMAT WITH CHARTEVENTS, LABS AND MECHVENT
disp('INITIAL REFORMAT WITH CHARTEVENTS, LABS AND MECHVENT START')
% gives an array with all unique charttime (1 per row) and all items in columns.
% ################## IMPORTANT !!!!!!!!!!!!!!!!!!
% Here i use -48 -> +24 because that's for sepsis3 cohort defintion!!
% I need different time period for the MDP (-24 -> +48)

% 论文原文参考：
% In MIMIC-III, data were included from up to 24 h preceding until 48 h following the estimated onset of sepsis, in order to capture the early phase of its management, including initial resuscitation.


% 预分配矩阵空间
reformat=NaN(2000000,68);  %final table 
qstime=zeros(100000,4);

winb4=49;   %lower limit for inclusion of data (48h before time flag)
winaft=25;  % upper limit (24h after)
irow=1;  %recording row for summary table
h = waitbar(0,'Initializing waitbar...');

tic
for icustayid=1:100000
% onset 表：
% 第1列：subject_id
% 第2列：ICU stay 的内部索引
% 第3列：依据 Sepsis‑3 规则推断的“疑似感染起点时间”（秒）。没有命中则为 0。
qst=onset(icustayid,3); %flag for presumed infection

if qst>0  % if we have a flag
d1=table2array(demog(demog.icustay_id==icustayid+200000,[11 5])); %age of patient + discharge time

if d1(1)>6574  % if older than 18 years old

    waitbar(icustayid/100000,h,icustayid/1000) %moved here to save some time
% CHARTEVENTS
    if icustayid<10000
    temp=ce010(ce010(:,1)==icustayid+200000,:);
    elseif icustayid>=10000 & icustayid<20000
    temp=ce1020(ce1020(:,1)==icustayid+200000,:);
    elseif icustayid>=20000 & icustayid<30000
    temp=ce2030(ce2030(:,1)==icustayid+200000,:);
    elseif icustayid>=30000 && icustayid<40000
    temp=ce3040(ce3040(:,1)==icustayid+200000,:);
    elseif icustayid>=40000 & icustayid<50000
    temp=ce4050(ce4050(:,1)==icustayid+200000,:);
    elseif icustayid>=50000 & icustayid<60000
    temp=ce5060(ce5060(:,1)==icustayid+200000,:);
    elseif icustayid>=60000 & icustayid<70000
    temp=ce6070(ce6070(:,1)==icustayid+200000,:);
    elseif icustayid>=70000 & icustayid<80000
    temp=ce7080(ce7080(:,1)==icustayid+200000,:);
    elseif icustayid>=80000 & icustayid<90000
    temp=ce8090(ce8090(:,1)==icustayid+200000,:);
    elseif icustayid>=90000
    temp=ce90100(ce90100(:,1)==icustayid+200000,:);
    end

% 4h步长分箱，所以首尾加了4h
ii=temp(:,2)>= qst-(winb4+4)*3600 & temp(:,2)<=qst+(winaft+4)*3600; %time period of interest -4h and +4h
temp=temp(ii,:);   %only time period of interest

%LABEVENTS
ii=labU(:,1)==icustayid+200000;
temp2=labU(ii,:);
ii=temp2(:,2)>= qst-(winb4+4)*3600 & temp2(:,2)<=qst+(winaft+4)*3600; %time period of interest -4h and +4h
temp2=temp2(ii,:);   %only time period of interest

%Mech Vent + ?extubated    看不懂+后面写的什么东西
%Mechvent
ii=MV(:,1)==icustayid+200000;
temp3=MV(ii,:);
ii=temp3(:,2)>= qst-(winb4+4)*3600 & temp3(:,2)<=qst+(winaft+4)*3600; %time period of interest -4h and +4h
temp3=temp3(ii,:);   %only time period of interest

% 把三路数据的时间戳合并
t=unique([temp(:,2);temp2(:,2); temp3(:,2)]);   %list of unique timestamps from all 3 sources / sorted in ascending order

if t
for i=1:numel(t)
    % 维度总览：reformat = NaN(2000000, 68)，前3列元数据，4..31为生命体征，32..66为化验，67..68为机械通气

    % 没看懂后面的算法，但总之是把三路数据按时间戳合并到 reformat 里，表头信息在 reference_matrices.mat 的 sample_and_hold 表格里面。

    % ''Height_cm''	''Weight_kg''	''GCS''	''RASS''	''HR''	''SysBP''	''MeanBP''	''DiaBP''	''RR''	''SpO2''	''Temp_C''	''Temp_F''	''CVP''	''PAPsys''	''PAPmean''	''PAPdia''	''CI''	''SVR''	''Interface''	''FiO2_100''	''FiO2_1''	''O2flow''	''PEEP''	''TidalVolume''	''MinuteVentil''	''PAWmean''	''PAWpeak''	''PAWplateau''	''Potassium''	''Sodium''	''Chloride''	''Glucose''	''BUN''	''Creatinine''	''Magnesium''	''Calcium''	''Ionised_Ca''	''CO2_mEqL''	''SGOT''	''SGPT''	''Total_bili''	''Direct_bili''	''Total_protein''	''Albumin''	''Troponin''	''CRP''	''Hb''	''Ht''	''RBC_count''	''WBC_count''	''Platelets_count''	''PTT''	''PT''	''ACT''	''INR''	''Arterial_pH''	''paO2''	''paCO2''	''Arterial_BE''	''Arterial_lactate''	''HCO3''	''ETCO2''	''SvO2''	''mechvent''	''extubated''

    %CHARTEVENTS
    % ce*.csv
    ii=temp(:,2)==t(i);
    col=temp(ii,3);
    value=temp(ii,4);  
    reformat(irow,1)=i; %timestep  
    reformat(irow,2)=icustayid;
    reformat(irow,3)=t(i); %charttime
    reformat(irow,3+col)=value;%(locb(:,1)); %store available values
    % reformat 构成（从temp提取）
    % 1)第一列：标号？
    % 2)第二列：icustayid（-200000）
    % 3)第三列：时间戳
      
    %LAB VALUES
    %labs_ce.csv and labs_le.csv
    ii=temp2(:,2)==t(i);
    col=temp2(ii,3);
    value=temp2(ii,4);
    reformat(irow,31+col)=value; %store available values
    


    %MV
    %machvent.csv
    ii=temp3(:,2)==t(i);
    if nansum(ii)>0
    value=temp3(ii,3:4);
      reformat(irow,67:68)=value; %store available values
    else
      reformat(irow,67:68)=NaN;
    end
    
    irow=irow+1;
     
end


% qstime 是每个 ICU stay 的时间边界记录表（100000×4 的数组），用来保存窗口与出院等关键时间，便于后续派生时长指标与校验覆盖范围。
% 行号即内部索引：第 r 行对应 icustayid = r（循环里的 1..100000）。
% 列1：疑似感染起点时间 qst（秒）
% 列2：该 stay 在选定窗口内的首个观测时间 t(1)
% 列3：该 stay 在选定窗口内的最后观测时间 t(end)
% 列4：出院时间（dischargetime）

qstime(icustayid,1)=qst; %flag for presumed infection / this is time of sepsis if SOFA >=2 for this patient
%HERE I SAVE FIRST and LAST TIMESTAMPS, in QSTIME, for each ICUSTAYID
qstime(icustayid,2)=t(1);  %first timestamp
qstime(icustayid,3)=t(end);  %last timestamp
qstime(icustayid,4)=d1(2); %dischargetime

end
end
end
end
toc

close(h);
reformat(irow:end,:)=[];  %delete extra unused rows
disp('INITIAL REFORMAT WITH CHARTEVENTS, LABS AND MECHVENT END')


%% ########################################################################
%                                   OUTLIERS 
% ########################################################################
disp('OUTLIERS START')
% 异常值与规则性修正（按变量的物理/生理合理范围处理）。
% 列名参考：reformat表头信息.csv
% 说明：deloutabove/Deloutbelow 用于把超阈值的观测置为 NaN，避免极端值影响后续聚合与SOFA计算。

% Weight_kg (col 5)：>300 kg 基本不可能，视为测量/录入错误
reformat=deloutabove(reformat,5,300);  % delete outlier above 300 kg

% HR (col 8)：心率 >250 bpm 罕见，通常为错误或脉搏测不准
reformat=deloutabove(reformat,8,250);

% 血压：SysBP/MeanBP/DiaBP（col 9/10/11）
% - 收缩压 >300 mmHg、舒张/平均压 <0 或 >200 mmHg 视为不合理
reformat=deloutabove(reformat,9,300);   % SysBP
reformat=deloutbelow(reformat,10,0);    % MeanBP 下界
reformat=deloutabove(reformat,10,200);  % MeanBP 上界
reformat=deloutbelow(reformat,11,0);    % DiaBP 下界
reformat=deloutabove(reformat,11,200);  % DiaBP 上界

% RR (col 12)：呼吸频率 >80 次/分多为录入问题
reformat=deloutabove(reformat,12,80);

% SpO2 (col 13)：脉氧 >150% 不合理；>100% 的值截断为 100%
reformat=deloutabove(reformat,13,150);
ii=reformat(:,13)>100; reformat(ii,13)=100;

% 体温：Temp_C (col 14), Temp_F (col 15)
% 有些记录温度单位列被颠倒：
% - 若 Temp_F 处出现 25~45（更像摄氏度），则移到 Temp_C
% - 若 Temp_C >70（更像华氏度），则移到 Temp_F
% 然后对 Temp_C >90°C 的极端值做截断
ii=reformat(:,14)>90 & isnan(reformat(:,15)); reformat(ii,15)=reformat(ii,14);
reformat=deloutabove(reformat,14,90);

% Interface (col 22)：氧疗装置类型，后续用于估算缺失的 FiO2

% FiO2：百分比 col 23（0-100），小数 col 24（0-1）
% - 百分数 >100% 或 <20%（低于空气21%）视为不合理
% - 若 col23<1，推断其实际是小数形式，转为百分比
% - 小数形式 >1.5 明显不合理
reformat=deloutabove(reformat,23,100);
ii=reformat(:,23)<1; reformat(ii,23)=reformat(ii,23)*100;
reformat=deloutbelow(reformat,23,20);
reformat=deloutabove(reformat,24,1.5);

% O2flow (col 25, L/min)：>70 L/min 视为不合理
reformat=deloutabove(reformat,25,70);

% PEEP (col 26, cmH2O)：<0 或 >40 多为异常
reformat=deloutbelow(reformat,26,0);
reformat=deloutabove(reformat,26,40);

% TidalVolume (col 27, mL)：>1800 mL 远超常规设置
reformat=deloutabove(reformat,27,1800);

% MinuteVentil (col 28, L/min)：>50 L/min 极端
reformat=deloutabove(reformat,28,50);

% Potassium (col 32, mEq/L)：[1, 15] 之外视为不合理
reformat=deloutbelow(reformat,32,1);
reformat=deloutabove(reformat,32,15);

% Sodium (col 33, mEq/L)：<95 或 >178 罕见
reformat=deloutbelow(reformat,33,95);
reformat=deloutabove(reformat,33,178);

% Chloride (col 34, mEq/L)：<70 或 >150 异常
reformat=deloutbelow(reformat,34,70);
reformat=deloutabove(reformat,34,150);

% Glucose (col 35, mg/dL)：<1 或 >1000 mg/dL 视为不合理
reformat=deloutbelow(reformat,35,1);
reformat=deloutabove(reformat,35,1000);

% Creatinine (col 37, mg/dL)：>150 极端，按出错处理
reformat=deloutabove(reformat,37,150);

% Magnesium (col 38, mg/dL)：>10 异常
reformat=deloutabove(reformat,38,10);

% Calcium (col 39, mg/dL)：>20 异常
reformat=deloutabove(reformat,39,20);

% Ionised_Ca (col 40, mmol/L 或 mg/dL 对应统一单位)：>5 异常
reformat=deloutabove(reformat,40,5);

% CO2_mEqL (col 41, 总二氧化碳/碳酸氢盐等效)：>120 异常
reformat=deloutabove(reformat,41,120);

% 肝酶：SGOT (col 42)/SGPT (col 43)：>10000 明显异常
reformat=deloutabove(reformat,42,10000);
reformat=deloutabove(reformat,43,10000);

% Hb/Ht：Hemoglobin (col 50, g/dL) >20 或 Hematocrit (col 51, %) >65 为异常
reformat=deloutabove(reformat,50,20);
reformat=deloutabove(reformat,51,65);

% WBC_count (col 53, 10^3/µL)：>500 极端
reformat=deloutabove(reformat,53,500);

% Platelets_count (col 54, 10^3/µL)：>2000 极端
reformat=deloutabove(reformat,54,2000);

% INR (col 58)：>20 视为异常
reformat=deloutabove(reformat,58,20);

% 动脉血 pH (col 59)：限制在 [6.7, 8]
reformat=deloutbelow(reformat,59,6.7);
reformat=deloutabove(reformat,59,8);

% paO2 (col 60, mmHg)：>700 极端
reformat=deloutabove(reformat,60,700);

% paCO2 (col 61, mmHg)：>200 极端
reformat=deloutabove(reformat,61,200);

% Base Excess (col 62, mEq/L)：< -50 视为异常
reformat=deloutbelow(reformat,62,-50);

% Arterial_lactate (col 63, mmol/L)：>30 视为异常
reformat=deloutabove(reformat,63,30);

% ####################################################################
% 进一步的规则性修正 / 利用已知变量间关系做补全

% 用 RASS 估算缺失的 GCS（col 7 -> col 6），参考 Wesley JAMA 2003 的近似对应关系
ii=isnan(reformat(:,6))&reformat(:,7)>=0;
reformat(ii,6)=15;
ii=isnan(reformat(:,6))&reformat(:,7)==-1;
reformat(ii,6)=14;
ii=isnan(reformat(:,6))&reformat(:,7)==-2;
reformat(ii,6)=12;
ii=isnan(reformat(:,6))&reformat(:,7)==-3;
reformat(ii,6)=11;
ii=isnan(reformat(:,6))&reformat(:,7)==-4;
reformat(ii,6)=6;
ii=isnan(reformat(:,6))&reformat(:,7)==-5;
reformat(ii,6)=3;


% FiO2 百分比/小数互补：若一项缺失，用另一项换算
ii=~isnan(reformat(:,23)) & isnan(reformat(:,24));
reformat(ii,24)=reformat(ii,23)./100;
ii=~isnan(reformat(:,24)) & isnan(reformat(:,23));
reformat(ii,23)=reformat(ii,24).*100;


% 估算缺失的 FiO2（基于装置类型 Interface(col 22) 与氧流量 O2flow(col 25) 的启发式规则）

reformatsah=SAH(reformat,sample_and_hold);  % do SAH first to handle this task

% 情形A：FiO2 缺失，O2flow 有值，且无装置或鼻导管（Interface=0/2）
% 依据常见流量-吸氧浓度经验公式给出区间映射（单位：%）
ii=find(isnan(reformatsah(:,23))&~isnan(reformatsah(:,25))&(reformatsah(:,22)==0|reformatsah(:,22)==2)); 
reformat(ii(reformatsah(ii,25)<=15),23)=70;
reformat(ii(reformatsah(ii,25)<=12),23)=62;
reformat(ii(reformatsah(ii,25)<=10),23)=55;
reformat(ii(reformatsah(ii,25)<=8),23)=50;
reformat(ii(reformatsah(ii,25)<=6),23)=44;
reformat(ii(reformatsah(ii,25)<=5),23)=40;
reformat(ii(reformatsah(ii,25)<=4),23)=36;
reformat(ii(reformatsah(ii,25)<=3),23)=32;
reformat(ii(reformatsah(ii,25)<=2),23)=28;
reformat(ii(reformatsah(ii,25)<=1),23)=24;

% 情形B：FiO2 与 O2flow 均缺失，且无装置/鼻导管（Interface=0/2）→ 视为室内空气 21%
ii=find(isnan(reformatsah(:,23))&isnan(reformatsah(:,25))&(reformatsah(:,22)==0|reformatsah(:,22)==2));
reformat(ii,23)=21;

% 情形C：FiO2 缺失，O2flow 有值，面罩/其他装置/呼吸机（假定按面罩估算）
% 说明：代码里包含 "==NaN" 判断在 MATLAB 中恒为 false，保留原始写法但不影响其余条件
ii=find(isnan(reformatsah(:,23))&~isnan(reformatsah(:,25))&(reformatsah(:,22)==NaN|reformatsah(:,22)==1|reformatsah(:,22)==3|reformatsah(:,22)==4|reformatsah(:,22)==5|reformatsah(:,22)==6|reformatsah(:,22)==9|reformatsah(:,22)==10)); 
reformat(ii(reformatsah(ii,25)<=15),23)=75;
reformat(ii(reformatsah(ii,25)<=12),23)=69;
reformat(ii(reformatsah(ii,25)<=10),23)=66;
reformat(ii(reformatsah(ii,25)<=8),23)=58;
reformat(ii(reformatsah(ii,25)<=6),23)=40;
reformat(ii(reformatsah(ii,25)<=4),23)=36;

% 情形D：FiO2、O2flow 缺失，且为面罩/呼吸机等（无法推断）→ 仍置为 NaN
ii=find(isnan(reformatsah(:,23))&isnan(reformatsah(:,25))&(reformatsah(:,22)==NaN|reformatsah(:,22)==1|reformatsah(:,22)==3|reformatsah(:,22)==4|reformatsah(:,22)==5|reformatsah(:,22)==6|reformatsah(:,22)==9|reformatsah(:,22)==10));
reformat(ii,23)=NaN;

% 情形E：非再吸入面罩（NRM，Interface=7），按流量给出更高的 FiO2 估计
ii=find(isnan(reformatsah(:,23))&~isnan(reformatsah(:,25))&reformatsah(:,22)==7); 
reformat(ii(reformatsah(ii,25)>=10),23)=90;
reformat(ii(reformatsah(ii,25)>=15),23)=100;
reformat(ii(reformatsah(ii,25)<10),23)=80;
reformat(ii(reformatsah(ii,25)<=8),23)=70;
reformat(ii(reformatsah(ii,25)<=6),23)=60;

%NO FiO2, NO O2 flow, NRM
ii=find(isnan(reformatsah(:,23))&isnan(reformatsah(:,25))&reformatsah(:,22)==7);  %no fio2 given and o2flow given, no interface OR cannula
reformat(ii,23)=NaN;

% 同步 FiO2 百分比/小数两列，保持一致性
ii=~isnan(reformat(:,23)) & isnan(reformat(:,24));
reformat(ii,24)=reformat(ii,23)./100;
ii=~isnan(reformat(:,24)) & isnan(reformat(:,23));
reformat(ii,23)=reformat(ii,24).*100;

% 血压三者互推：
% MAP ≈ (SBP + 2*DBP)/3；其余按等式重排补齐缺失一项
ii=~isnan(reformat(:,9))&~isnan(reformat(:,10)) & isnan(reformat(:,11));
reformat(ii,11)=(3*reformat(ii,10)-reformat(ii,9))./2;
ii=~isnan(reformat(:,09))&~isnan(reformat(:,11)) & isnan(reformat(:,10));
reformat(ii,10)=(reformat(ii,9)+2*reformat(ii,11))./3;
ii=~isnan(reformat(:,10))&~isnan(reformat(:,11)) & isnan(reformat(:,9));
reformat(ii,9)=3*reformat(ii,10)-2*reformat(ii,11);

% 体温互换与单位修正：
% - Temp_F 落在 25~45（更像摄氏）→ 移到 Temp_C
% - Temp_C >70（更像华氏）→ 移到 Temp_F
% - 若只存在一列，则按 C↔F 关系互补：F = C*1.8+32；C = (F-32)/1.8
ii=reformat(:,15)>25&reformat(:,15)<45; % Temp_F 里出现疑似摄氏
reformat(ii,14)=reformat(ii,15);
reformat(ii,15)=NaN;
ii=reformat(:,14)>70;  % Temp_C >70°C，疑似华氏
reformat(ii,15)=reformat(ii,14);
reformat(ii,14)=NaN;
ii=~isnan(reformat(:,14)) & isnan(reformat(:,15));
reformat(ii,15)=reformat(ii,14)*1.8+32;
ii=~isnan(reformat(:,15)) & isnan(reformat(:,14));
reformat(ii,14)=(reformat(ii,15)-32)./1.8;

% Hb/Ht 互推：经验线性关系，缺失一项时用另一项估算
ii=~isnan(reformat(:,50)) & isnan(reformat(:,51));
reformat(ii,51)=(reformat(ii,50)*2.862)+1.216;
ii=~isnan(reformat(:,51)) & isnan(reformat(:,50));
reformat(ii,50)=(reformat(ii,51)-1.216)./2.862;

% 胆红素互推：Total_bili (col 44) 与 Direct_bili (col 45) 的经验线性换算
ii=~isnan(reformat(:,44)) & isnan(reformat(:,45));
reformat(ii,45)=(reformat(ii,44)*0.6934)-0.1752;
ii=~isnan(reformat(:,45)) & isnan(reformat(:,44));
reformat(ii,44)=(reformat(ii,45)+0.1752)./0.6934;
disp('OUTLIERS END')

%% ########################################################################
%                      SAMPLE AND HOLD on RAW DATA
% ########################################################################
disp('SAMPLE AND HOLD ON RAW DATA START')
% 把最近一次观测在同一 ICU stay 内向前填充，补齐短期缺失
reformat=SAH(reformat(:,1:68),sample_and_hold);
disp('SAMPLE AND HOLD ON RAW DATA END')

% reformat：导入数据。后面 65 列分别放生命体征、化验、通气标志。之后对这一矩阵做异常值截断、互补换算等清洗。
% reformatsah：传给 SAH.m，按变量对应的“保持时间”做 sample-and-hold 向前填充的结果
% reformat2：重新按 4 小时时间槽聚合，生成的 84 列矩阵中，列 1-3 是时间槽元数据，列 4-84 是人口学和聚合后的特征。
% reformat2t → reformat3t：先把 reformat2 转成表对象并加上列名，然后根据缺失率剔除缺失超过 70% 的特征列，得到 reformat3t，
%                          仍保留前 11 列元数据和后面的关键派生列。
% reformat4t / reformat4：reformat3t 在进行 kNN/线性插补、单位标准化、派生 P/F 比、Shock Index、SOFA/SIRS 等指标后，
%                          拷贝为 reformat4t（表格）和 reformat4（数值矩阵）。这一步还包含对极端入量/尿量的住院级剔除及若干筛查，最终的 reformat4t 就是写入 .mat 的 Sepsis 队列表。



%% ########################################################################
%                             DATA COMBINATION
% ########################################################################
disp('DATA COMBINATION START')
% WARNING: the time window of interest has been defined above (here -48 -> +24)! 

% ============================ 重采样与合并阶段初始化 ============================
% 将按 4 小时为步长（time slots）把前面按时间点堆叠的宽表 reformat 聚合成 reformat2。
% 目标：每个 ICU stay × 每个 4h 槽 -> 1 行，便于后续计算 SOFA/SIRS 与下游建模。

timestep = 4;                      % 每个时间槽的宽度（小时）
irow = 1;                          % reformat2 的写入行指针
icustayidlist = unique(reformat(:,2));  % 需要处理的 ICU stay 内部索引列表

% 预分配输出矩阵（84 列），以减少动态扩容的开销：
% 列布局（见后续写入处）：
%  1  = bloc（该住院内的第几个 4h 槽，1..）
%  2  = icustayid（内部索引，1..100000）
%  3  = t0（本 4h 槽左边界时间戳，秒）
%  4:11   = 人口学与结局（gender, age, elixhauser, re_admission, died_in_hosp,
%           died_within_48h_of_out_time, mortality_90d, delay_end_of_record...）
%  12:78  = 该 4h 槽内的生命体征+化验（来自 reformat 第 4..end 列聚合后的 67 列）
%  79     = 升压药剂量中位数（median_dose_vaso）
%  80     = 升压药剂量最大值（max_dose_vaso）
%  81     = 累计入量（total fluid given, 截至当前槽）
%  82     = 本槽入量（4 小时内给入量）
%  83     = 累计尿量（total UO, 截至当前槽）
%  84     = 本槽尿量（4 小时尿量）
% 
% bloc | icustayid | charttime           | HR | MeanBP | FiO2_1 | PaO2 | input_4hourly | output_4hourly | ...
-----+-----------+---------------------+----+--------+--------+------+---------------+----------------+------
1    |     34567 | 2121-05-10 08:00:00 | 95 |     72 |  0.30  |  85  |      550      |       80       | ...
2    |     34567 | 2121-05-10 12:00:00 | 88 |     70 |  0.40  |  92  |      300      |      120       | ...
3    |     34567 | 2121-05-10 16:00:00 | 92 |     68 |  0.45  |  78  |      260      |      100       | ...

reformat2 = nan(size(reformat,1), 84);  % output array

h = waitbar(0,'Initializing waitbar...');  % 进度条
npt = numel(icustayidlist);                % 住院数，用于循环与进度
% Adding 2 empty cols for future shock index=HR/SBP and P/F
reformat(:,69:70)=NaN(size(reformat,1),2);

tic
% 外层住院循环：逐个 icustay 汇总 4 小时槽内的各来源观测，并写入 reformat2
for i = 1:npt
    % 取当前住院（内部索引）的数据，准备做 4h 槽聚合
    icustayid = icustayidlist(i);  % 1..100000（真实 icustay_id = +200000）
    
    % CHARTEVENTS & LABS 子表：同一住院的时间点堆叠宽表，以及首条时间戳（槽起点参考）
    temp = reformat(reformat(:,2)==icustayid,:);   % 本住院的子表
    beg  = temp(1,3);                              % 本次住院首条记录的时间戳（秒）
    
    % IV 液体记录（输入量）：分别来自 MetaVision 连续记录 inputMV 与 CareVue 事件 inputCV
    % 下面取出与本住院相关的行，并拆出列：开始时间、结束时间、速率
    % 可能找不到与本住院相关的行，即 input 为空，得到的iv、input、startt、endt、rate均为空数组
    iv     = find(inputMV(:,1)==icustayid+200000);   % MV infusion rows of interest
    input  = inputMV(iv,:);                          % MV subset
    iv     = find(inputCV(:,1)==icustayid+200000);   % CV bolus rows of interest
    input2 = inputCV(iv,:);                          % CV subset
    startt = input(:,2);   % 输注/推注开始时间（秒）
    endt   = input(:,3);   % 输注/推注结束时间（秒）
    rate   = input(:,8);   % 输注速率（mL/h 等；推注处常为 NaN）
    
    % 住院记录开始之前的“入量”基线（preadmission volume）：若有则纳入累计入量起点
    pread = inputpreadm(inputpreadm(:,1)==icustayid+200000, 2);
    if ~isempty(pread) % pread不为空，即有入科前入量记录
        totvol = nansum(pread);          % 初始累计入量（mL）
        waitbar(i/npt, h, i/npt*100)     % 更新进度条
    else
        totvol = 0;                      % 无记录按 0 处理
    end
    
    % 计算“记录开始之前”的输入量（以 beg 为界）：
    % - 连续输注 infu：考虑与 [t0,t1] 的四种相交情形求和
    % - 推注 bolus：取时间落在 [t0,t1] 内的事件量并求和
    t0 = 0;              % 窗口左界（住院时间轴起点）
    t1 = beg;            % 窗口右界（记录开始时刻）
    % 连续输注 4 种覆盖情形合计（单位小时换算）
    infu = nansum( rate.*(endt-startt).*(endt<=t1 & startt>=t0)/3600 ...
                 + rate.*(endt-t0).*(startt<=t0 & endt<=t1 & endt>=t0)/3600 ...
                 + rate.*(t1-startt).*(startt>=t0 & endt>=t1 & startt<=t1)/3600 ...
                 + rate.*(t1-t0).*(endt>=t1 & startt<=t0) /3600 );
    % 推注：MV（用 rate NaN 标识）与 CV（直接剂量列）
    bolus  = nansum(input( isnan(input(:,6)) & input(:,2)>=t0 & input(:,2)<=t1, 7)) ...
           + nansum(input2(                        input2(:,2)>=t0 & input2(:,2)<=t1, 5));
    totvol = nansum([totvol, infu, bolus]);   % 累计入量起点（mL）
            
    % 升压药（vasopressors）两路来源：
    % - MV 连续记录 vasoMV：用开始/结束时间区间与剂量速率列
    % - CV 事件 vasoCV：用事件时间与当时剂量
    iv    = find(vasoMV(:,1)==icustayid+200000);
    vaso1 = vasoMV(iv,:);      % MV subset
    iv    = find(vasoCV(:,1)==icustayid+200000);
    vaso2 = vasoCV(iv,:);      % CV subset
    startv= vaso1(:,3);        % 开始时间（秒）
    endv  = vaso1(:,4);        % 结束时间（秒）
    ratev = vaso1(:,5);        % 连续输注速率（单位依剂量标准）
            

        % 人口学与结局向量 dem（将写入 reformat2 列 4..11）
        % 顺序对应：gender, age, elixhauser, re_admission, died_in_hosp,
        %           died_within_48h_of_out_time, mortality_90d,
        %           delay_end_of_record_and_discharge_or_death (小时)
        demogi=find(demog.icustay_id==icustayid+200000);
        dem=[  demog.gender(demogi) ; ...
               demog.age(demogi) ; ...
               demog.elixhauser(demogi) ; ...
               demog.adm_order(demogi)>1 ; ...
               demog.morta_hosp(demogi); ...
               abs(demog.dod(demogi)-demog.outtime(demogi))<(24*3600*2); ...
               demog.morta_90(demogi) ; ...
               (qstime(icustayid,4)-qstime(icustayid,3))/3600];
        
        
        % 尿量（URINE OUTPUT）：同一住院的尿量事件与入科前尿量基线
        iu     = find(UO(:,1)==icustayid+200000);
        output = UO(iu,:);                                  % 尿量事件子集（mL）
        pread  = UOpreadm(UOpreadm(:,1)==icustayid,4);      % 入科前尿量（mL）
        if ~isempty(pread)
            UOtot = nansum(pread);                          % 初始累计尿量
        else
            UOtot = 0;
        end
        % 记录开始之前的尿量：落在 [t0,t1] 的事件累加
        UOnow = nansum(output(output(:,2)>=t0 & output(:,2)<=t1, 4));
        UOtot = nansum([UOtot, UOnow]);
    
    
    % 4 小时时间槽循环：将本住院的时间点聚合为 4h 槽，并写入 reformat2 对应列
    for j=0:timestep:79 % -52 until +28 = 80 hours in total
        % 槽边界（秒）与本槽内观测筛选
        t0=3600*j+ beg;   %left limit of time window
        t1=3600*(j+timestep)+beg;   %right limit of time window
        ii=temp(:,3)>=t0 & temp(:,3)<=t1;  %index of items in this time period
        if sum(ii)>0
            
            
        % 基本信息与人口学/结局（reformat2 列1..11）
        reformat2(irow,1)=(j/timestep)+1;   % 列1 bloc：该住院内第几个 4h 槽 (1,2,3...)
        reformat2(irow,2)=icustayid;        % 列2 icustayid：内部索引
        reformat2(irow,3)=3600*j+ beg;      % 列3 t0：本槽左边界时间戳
        reformat2(irow,4:11)=dem;           % 列4..11：gender/age/elixhauser/再入院/死亡标记/90天死亡/延迟
            
        
        % 槽内生命体征 + 化验聚合（reformat2 列12..78，对应 reformat 列4..end）
        value=temp(ii,:);%records all values in this timestep
        % 若仅 1 条观测，直接写入；多条观测取列均值
        if sum(ii)==1   %if only 1 row of values at this timestep
          reformat2(irow,12:78)=value(:,4:end);
        else
          reformat2(irow,12:78)=nanmean(value(:,4:end)); %mean of all available values
        end
        
        
        % 升压药 VASOPRESSORS（reformat2 列79..80）：中位/最大剂量
            % for CV: dose at timestamps.
            % for MV: 4 possibles cases, each one needing a different way to compute the dose of VP actually administered:
            %----t0---start----end-----t1----
            %----start---t0----end----t1----
            %-----t0---start---t1---end
            %----start---t0----t1---end----

        
        % MV 连续输注：判断与 [t0,t1] 是否相交
        % 说明：每条 MV 输注记录是一个时间区间 [startv(k), endv(k)]，在该区间内用恒定速率 ratev(k) 给药。
        % 与 4 小时槽 [t0, t1] 存在重叠当且仅当 max(startv, t0) < min(endv, t1)。
        % 代码将这一“区间相交”条件展开为 4 类常见形态的并集，便于直观理解：
        %   1) 结束点在槽内：         ---- t0 --- start ---- end --- t1 ----
        %      条件：endv ∈ [t0, t1]                           → (endv>=t0 & endv<=t1)
        %   2) 整段落在槽内：         ---- t0 --- start --- end --- t1 ----
        %      条件：startv,endv ∈ [t0, t1]                    → (startv>=t0 & endv<=t1)
        %   3) 起点在槽内：           ---- t0 --- start ---- t1 ---- end ----
        %      条件：startv ∈ [t0, t1]                          → (startv>=t0 & startv<=t1)
        %   4) 覆盖整个槽：           ---- start --- t0 ==== t1 --- end ----
        %      条件：startv ≤ t0 且 endv ≥ t1                   → (startv<=t0 & endv>=t1)
        % 处理：将满足任一形态的 MV 速率 ratev(v) 与本槽内 CV 的离散剂量一起汇总，
        %       用中位数作为“代表剂量”（抗极值）并用最大值反映“峰值剂量”。请确保 MV/CV 单位一致。
        v=(endv>=t0&endv<=t1)|(startv>=t0&endv<=t1)|(startv>=t0&startv<=t1)|(startv<=t0&endv>=t1);
        % CV 离散事件：直接取本槽内剂量列
        v2=vaso2(vaso2(:,3)>=t0&vaso2(:,3)<=t1,4);
        v1=nanmedian([ratev(v); v2]);
        v2=nanmax([ratev(v); v2]);
        if ~isempty(v1)&~isnan(v1)&~isempty(v2)&~isnan(v2)
        reformat2(irow,79)=v1;    % 列79：median_dose_vaso
        reformat2(irow,80)=v2;    % 列80：max_dose_vaso
        end
        
        % 输入液体 INPUT FLUID（reformat2 列81..82）
        % input from MV (4 ways to compute)
        infu=  nansum(rate.*(endt-startt).*(endt<=t1&startt>=t0)/3600   +    rate.*(endt-t0).*(startt<=t0&endt<=t1&endt>=t0)/3600 +     rate.*(t1-startt).*(startt>=t0&endt>=t1&startt<=t1)/3600 +      rate.*(t1-t0).*(endt>=t1&startt<=t0)   /3600);
        % all boluses received during this timestep, from inputMV (need to check rate is NaN) and inputCV (simpler):
        bolus=nansum(input(isnan(input(:,6))& input(:,2)>=t0&input(:,2)<=t1,7)) + nansum(input2(input2(:,2)>=t0&input2(:,2)<=t1,5));  
        % sum fluid given：累计入量与本槽入量
        totvol=nansum([totvol,infu,bolus]);
        reformat2(irow,81)=totvol;    % 列81：累计入量 total fluid given
        reformat2(irow,82)=nansum([infu,bolus]);   % 列82：本槽入量 input_4hourly
        
        % 尿量 UO（reformat2 列83..84）
        UOnow=nansum(output(output(:,2)>=t0&output(:,2)<=t1,4));  
        UOtot=nansum([UOtot UOnow]);
        reformat2(irow,83)=UOtot;    % 列83：累计尿量 output_total
        reformat2(irow,84)=nansum(UOnow);   % 列84：本槽尿量 output_4hourly

        %CUMULATED BALANCE
        reformat2(irow,85)=totvol-UOtot;    %cumulated balance

        irow=irow+1;
        end
    end
end
toc

reformat2(irow:end,:)=[];
close(h);

disp('DATA COMBINATION END')
%% ########################################################################
%    CONVERT TO TABLE AND DELETE VARIABLES WITH EXCESSIVE MISSINGNESS
% ########################################################################
disp('CONVERT TO TABLE AND DELETE VARIABLES WITH EXCESSIVE MISSINGNESS START')

% missing value 过多的变量剔除（>70% 缺失）
dataheaders=[sample_and_hold(1,:) {'Shock_Index' 'PaO2_FiO2'}]; 
dataheaders=regexprep(dataheaders,'['']','');
dataheaders = ['bloc','icustayid','charttime','gender','age','elixhauser','re_admission', 'died_in_hosp', 'died_within_48h_of_out_time','mortality_90d','delay_end_of_record_and_discharge_or_death',...
    dataheaders,  'median_dose_vaso','max_dose_vaso','input_total','input_4hourly','output_total','output_4hourly','cumulated_balance'];

reformat2t=array2table(reformat2);
reformat2t.Properties.VariableNames=dataheaders;
miss=sum(isnan(reformat2))./size(reformat2,1);

% if values have less than 70% missing values (over 30% of values present): I keep them
% 生成 reformat3t 逻辑：保留前 11 列元数据/结局 + 仅保留第 12–74 列里缺失率 <70% 的变量 + 末尾 11 列固定保留
reformat3t=reformat2t(:,[true(1,11) miss(12:74)<0.70 true(1,11)]) ; 
disp('CONVERT TO TABLE AND DELETE VARIABLES WITH EXCESSIVE MISSINGNESS END')
%% ########################################################################
%             HANDLING OF MISSING VALUES  &  CREATE REFORMAT4T
% ########################################################################
disp('HANDLING OF MISSING VALUES AND CREATE REFORMAT4T START')
% Do linear interpol where missingness is low (kNN imputation doesnt work if all rows have missing values)

% 少于 5% 的缺失值时，做线性插值（针对时间序列的同一变量）
reformat3=table2array(reformat3t);
miss=sum(isnan((reformat3)))./size(reformat3,1);
ii=miss>0&miss<0.05;  %less than 5% missingness 
mechventcol=find(ismember(reformat3t.Properties.VariableNames,{'mechvent'})); %第 11 列到 mechventcol-1 列是需要处理的变量，由于前面已经删除了一些变量，不能用下标定位，直接用字段头名定位

for i=11:mechventcol-1 % correct col by col, otherwise it does it wrongly
  if ii(i)==1
    reformat3(:,i)=fixgaps(reformat3(:,i));
  end
end

reformat3t(:,11:mechventcol-1)=array2table(reformat3(:,11:mechventcol-1));

% KNN IMPUTATION -   每1万行切分成块来挨个处理，剩余仍然缺失的变量使用KNN

mechventcol=find(ismember(reformat3t.Properties.VariableNames,{'mechvent'}));
ref=reformat3(:,11:mechventcol-1);  %columns of interest

tic
for i=1:10000:size(reformat3,1)-9999   %dataset divided in 5K rows chunks (otherwise too large)
    i
    ref(i:i+9999,:)=knnimpute(ref(i:i+9999,:)',1, 'distance','seuclidean')'; %在向量空间里面按照相似度插值
end

ref(end-9999:end,:)=knnimpute(ref(end-9999:end,:)',1, 'distance','seuclidean')';  %the last bit is imputed from the last 10K rows

toc

% I paste the data interpolated, but not the demographics and the treatments
reformat3t(:,11:mechventcol-1)=array2table(ref);  

reformat4t=reformat3t;
reformat4=table2array(reformat4t);

disp('HANDLING OF MISSING VALUES AND CREATE REFORMAT4T END')
%% ########################################################################
%        COMPUTE SOME DERIVED VARIABLES: P/F, Shock Index, SOFA, SIRS...
% ########################################################################
disp('COMPUTE SOME DERIVED VARIABLES: P/F, Shock Index, SOFA, SIRS... START')
% 本节：在插补后的表 reformat4t 上，进一步规范字段并派生二次指标：
% 1) 纠正性别/年龄边界；2) 标准化 mechvent；3) 填补 Elixhauser；
% 4) 规范升压药缺失；5) 重算 P/F 与 Shock Index；6) 逐时刻计算 SOFA 与 SIRS。

% 1) 性别编码规范化（原始 1/2 → 0/1），便于后续建模
reformat4t.gender=reformat4t.gender-1; 

% 2) 年龄上限处理：极端年龄值统一设为 91.4 岁（按 MIMIC 去标识规则）
ii=reformat4t.age>150*365.25;              % 以天为单位，150y 以上视为异常
reformat4t.age(ii)=91.4*365.25;            % 91.4y 的匿名上限

% 3) 机械通气列：NaN→0，>0→1（二值指示），是否进行了机械通气
reformat4t.mechvent(isnan(reformat4t.mechvent))=0;
reformat4t.mechvent(reformat4t.mechvent>0)=1;

% 4) Elixhauser 合并症指数：少量缺失以全体中位数补齐
reformat4t.elixhauser(isnan(reformat4t.elixhauser))=nanmedian(reformat4t.elixhauser);

% 5) 升压药剂量：median/max 缺失置 0，避免 NaN 影响 SOFA 心血管评分
a=find(ismember(reformat4t.Properties.VariableNames,{'median_dose_vaso'}));
ii=isnan(reformat4(:,a));
reformat4t(ii,a)=array2table(zeros(sum(ii),1));
a=find(ismember(reformat4t.Properties.VariableNames,{'max_dose_vaso'}));
ii=isnan(reformat4(:,a));
reformat4t(ii,a)=array2table(zeros(sum(ii),1));

% 6) P/F 比值（PaO2 / FiO2）重算：强制使用小数形式 FiO2_1，避免混用百分比
p=find(ismember(reformat4t.Properties.VariableNames,{'paO2'}));
f=find(ismember(reformat4t.Properties.VariableNames,{'FiO2_1'}));
a=find(ismember(reformat4t.Properties.VariableNames,{'PaO2_FiO2'}));
reformat4t(:,a)=array2table(reformat4(:,p)./reformat4(:,f));  

% 7) Shock Index（HR/SBP）重算：消除 Inf，然后以全体均值填补 NaN（~0.8）
p=find(ismember(reformat4t.Properties.VariableNames,{'HR'}));
f=find(ismember(reformat4t.Properties.VariableNames,{'SysBP'}));
a=find(ismember(reformat4t.Properties.VariableNames,{'Shock_Index'}));
reformat4(:,a)=reformat4(:,p)./reformat4(:,f);  
reformat4(isinf(reformat4(:,a)),a)=NaN;
d=nanmean(reformat4(:,a));
reformat4(isnan(reformat4(:,a)),a)=d;  % 用均值替换 NaN
reformat4t(:,a)=array2table(reformat4(:,a));

% 8) SOFA 逐时刻计算：基于 PaO2/FiO2、PLT、总胆红素、MAP/升压药、GCS、肾功能/尿量
%    注意：此处 s6（肾功能）将 Cr 与 UO 组合（UO<某阈值提高评分）。
a=zeros(8,1); % indices of vars used in SOFA
a(1)=find(ismember(reformat4t.Properties.VariableNames,{'PaO2_FiO2'}));
a(2)=find(ismember(reformat4t.Properties.VariableNames,{'Platelets_count'}));
a(3)=find(ismember(reformat4t.Properties.VariableNames,{'Total_bili'}));
a(4)=find(ismember(reformat4t.Properties.VariableNames,{'MeanBP'}));
a(5)=find(ismember(reformat4t.Properties.VariableNames,{'max_dose_vaso'}));
a(6)=find(ismember(reformat4t.Properties.VariableNames,{'GCS'}));
a(7)=find(ismember(reformat4t.Properties.VariableNames,{'Creatinine'}));
a(8)=find(ismember(reformat4t.Properties.VariableNames,{'output_4hourly'}));
s=table2array(reformat4t(:,a));  

p=[0 1 2 3 4];

% 各系统评分的阈值划分（按 SOFA 标准）：
s1=[s(:,1)>400 s(:,1)>=300 &s(:,1)<400 s(:,1)>=200 &s(:,1)<300 s(:,1)>=100 &s(:,1)<200 s(:,1)<100 ];   % 呼吸：P/F
s2=[s(:,2)>150 s(:,2)>=100 &s(:,2)<150 s(:,2)>=50 &s(:,2)<100 s(:,2)>=20 &s(:,2)<50 s(:,2)<20 ];        % 凝血：PLT
s3=[s(:,3)<1.2 s(:,3)>=1.2 &s(:,3)<2 s(:,3)>=2 &s(:,3)<6 s(:,3)>=6 &s(:,3)<12 s(:,3)>12 ];              % 肝脏：TBil
s4=[s(:,4)>=70 s(:,4)<70&s(:,4)>=65 s(:,4)<65 s(:,5)>0 &s(:,5)<=0.1 s(:,5)>0.1 ];                       % 心血管：MAP/升压药
s5=[s(:,6)>14 s(:,6)>12 &s(:,6)<=14 s(:,6)>9 &s(:,6)<=12 s(:,6)>5 &s(:,6)<=9 s(:,6)<=5 ];               % 神经：GCS
s6=[s(:,7)<1.2 s(:,7)>=1.2 &s(:,7)<2 s(:,7)>=2 &s(:,7)<3.5 (s(:,7)>=3.5 &s(:,7)<5)|(s(:,8)<84) (s(:,7)>5)|(s(:,8)<34) ]; % 肾：Cr / UO

nrcol=size(reformat4,2);   % 当前列数
reformat4(1,nrcol+1:nrcol+7)=0;  % 预留 7 列：各系统分项 + 总 SOFA
for i=1:size(reformat4,1)
    % 取每个系统满足阈值的最高分，并求和得到总 SOFA
    t=max(p(s1(i,:)))+max(p(s2(i,:)))+max(p(s3(i,:)))+max(p(s4(i,:)))+max(p(s5(i,:)))+max(p(s6(i,:)));
    if t
        reformat4(i,nrcol+1:nrcol+7)=[max(p(s1(i,:))) max(p(s2(i,:))) max(p(s3(i,:))) max(p(s4(i,:))) max(p(s5(i,:))) max(p(s6(i,:))) t];
    end
end

% 9) SIRS 逐时刻计算（需要 Temp, HR, RR, PaCO2, WBC）：满足条件计 1 分，累加 0..4
a=zeros(5,1);
a(1)=find(ismember(reformat4t.Properties.VariableNames,{'Temp_C'}));
a(2)=find(ismember(reformat4t.Properties.VariableNames,{'HR'}));
a(3)=find(ismember(reformat4t.Properties.VariableNames,{'RR'}));
a(4)=find(ismember(reformat4t.Properties.VariableNames,{'paCO2'}));
a(5)=find(ismember(reformat4t.Properties.VariableNames,{'WBC_count'}));
s=table2array(reformat4t(:,a));  

s1=[s(:,1)>=38| s(:,1)<=36];   % 体温
s2=[s(:,2)>90 ];               % 心率
s3=[s(:,3)>=20|s(:,4)<=32];    % 呼吸/二氧化碳分压
s4=[s(:,5)>=12| s(:,5)<4];     % 白细胞
reformat4(:,nrcol+8)=s1+s2+s3+s4;  % SIRS 总分

% adds 2 cols for SOFA and SIRS, if necessary
if sum(ismember(reformat4t.Properties.VariableNames,{'SIRS'}))== 0
reformat4t(:,end+1:end+2)=array2table(0);
reformat4t.Properties.VariableNames(end-1:end)= {'SOFA','SIRS'};  
end

% records values
reformat4t(:,end-1)=array2table(reformat4(:,end-1));
reformat4t(:,end)=array2table(reformat4(:,end));

disp('COMPUTE SOME DERIVED VARIABLES: P/F, Shock Index, SOFA, SIRS... END')
%% ########################################################################
%                            EXCLUSION OF SOME PATIENTS 
% ########################################################################
disp('EXCLUSION OF SOME PATIENTS START')
% 说明：以下按启发式规则剔除疑似录入异常/非生理值及撤除治疗导致的早期死亡等样本，
% 以提高后续队列质量。先打印剔除前的唯一 icustayid 数。
numel(unique(reformat4t.icustayid))  % count before

% 规则1：极端尿量（output_4hourly）整住院剔除
% 判定：>12000 mL / 4h（≈12 L/4h）。原注释写 40 L/4h，此处以阈值为准。
a=find(reformat4t.output_4hourly>12000);
i=unique(reformat4t.icustayid(a));
i=find(ismember(reformat4t.icustayid,i));
reformat4t(i,:)=[];

% 规则2：总胆红素 Total_bili 出现占位非物理值（如 999999）整住院剔除
a=find(reformat4t.Total_bili>10000); 
i=unique(reformat4t.icustayid(a));
i=find(ismember(reformat4t.icustayid,i));
reformat4t(i,:)=[];

% 规则3：极端入量（input_4hourly）整住院剔除
% 判定：>10000 mL / 4h（≈10 L/4h）
a=find(reformat4t.input_4hourly>10000);
i=unique(reformat4t.icustayid(a));  % 28 ids
i=find(ismember(reformat4t.icustayid,i));
reformat4t(i,:)=[];


% 规则4：可能“撤除治疗”的早期死亡剔除（withdrawal bias）
% 思路：若 90 天死亡=1，且该住院的最后一个 4h 槽：
%  - max_dose_vaso（升压药）降为 0，但历史峰值 > 0.3（提示曾用过较大剂量）；
%  - 此时 SOFA 仍较高（>= 本次住院 max SOFA 的一半）；
%  - 观测步数较少（GroupCount < 20，即 <80 小时）。
% 满足时整住院剔除。
% 先按住院聚合出每位病人的极值统计：
q=reformat4t.bloc==1;
% fence_posts=find(q(:,1)==1);
num_of_trials=numel(unique(reformat4t.icustayid));%size(fence_posts,1);
a=array2table([reformat4t.icustayid reformat4t.mortality_90d reformat4t.max_dose_vaso reformat4t.SOFA]);
a.Properties.VariableNames={'id','mortality_90d','vaso','sofa'};
d=grpstats(a,'id','max');

% 基于上面的统计，筛出满足撤除条件的住院 id 并删除
e=zeros(num_of_trials,1);
for i=1:num_of_trials
    if d.max_mortality_90d(i) ==1
    ii=reformat4t.icustayid==d.id(i) & reformat4t.bloc==d.GroupCount(i);  %last row for this patient
    e(i)=sum((reformat4t.max_dose_vaso(ii)==0 & d.max_vaso(i)>0.3 & reformat4t.SOFA(ii)>=d.max_sofa(i)/2))>0;
    end
end
r=d.id(e==1 & d.GroupCount<20); % ids to be removed
ii=ismember(reformat4t.icustayid,r);
reformat4t(ii,:)=[];

% 规则5：在数据采集窗口内于 ICU 去世者剔除（窗口可能被截断，数据不完整）
% 条件：第一槽（bloc==1）且 died_within_48h_of_out_time==1，且记录结束到出院/死亡的延迟 <24 h
ii=reformat4t.bloc==1 & reformat4t.died_within_48h_of_out_time==1 & reformat4t.delay_end_of_record_and_discharge_or_death<24;
ii=ismember(icustayidlist,reformat4t.icustayid(ii));
reformat4t(ii,:)=[];

% 打印剔除后的唯一 icustayid 数
numel(unique(reformat4t.icustayid))   % count after
disp('EXCLUSION OF SOME PATIENTS END')
%% #######################################################################
%                       CREATE SEPSIS COHORT
% ########################################################################
% 先进行了初筛以减少数量，后所有特征填补完毕精确筛选脓毒症队列
disp('CREATE SEPSIS COHORT START')
% 本节：基于已计算的逐时刻 SOFA/SIRS，从每次 ICU 住院提取一行摘要，
% 并以 “监测窗内 max SOFA >= 2” 作为 Sepsis-3 的筛选条件构建脓毒症队列。
% 假设基线 SOFA=0（与文献一致）。

% 预分配 cohort 摘要矩阵（每次住院一行，共5列）：
% 1 icustayid（真实编号）
% 2 morta_90d（90天死亡）
% 3 max_sofa（窗口内最大 SOFA）
% 4 max_sirs（窗口内最大 SIRS）
% 5 sepsis_time（疑似感染起点 qst）
sepsis=zeros(30000,5);
irow=1;

tic
for icustayid=1:100000
    % 收集该住院的所有 4h 槽行，若存在则记录汇总指标
    ii=find(ismember(reformat4t.icustayid,icustayid));
    if mod(icustayid,10000)==0;disp([num2str(icustayid/1000), ' %']);end
    if ii
    
         sofa=reformat4t.SOFA(ii);
         sirs=reformat4t.SIRS(ii);
         sepsis(irow,1)=icustayid+200000;                 % 真实 icustay_id（+200000）
         sepsis(irow,2)=reformat4t.mortality_90d(ii(1));  % 90-day mortality（住院级常量）
         sepsis(irow,3)=max(sofa);                        % 住院内最大 SOFA
         sepsis(irow,4)=max(sirs);                        % 住院内最大 SIRS
         sepsis(irow,5)=qstime(icustayid);                % 疑似感染起点（秒）
         irow=irow+1;
    end
end
toc
sepsis(irow:end,:)=[];   % 裁掉未使用的预分配行

% 转为 table 并命名列，便于下游保存/分析
sepsis=array2table(sepsis);
sepsis.Properties.VariableNames={'icustayid','morta_90d','max_sofa','max_sirs','sepsis_time'};

% 筛选：仅保留 max SOFA >= 2 的住院（Sepsis-3 条件之一）
sepsis(sepsis.max_sofa<2,:)=[];

% 队列最终规模
size(sepsis,1)  

% 保存队列至 CSV（已去标识，只含汇总指标与时间旗标）
writetable(sepsis,'sepsis_mimiciii.csv','Delimiter',',');
disp('CREATE SEPSIS COHORT END')


save('./BACKUP/AIClinician_sepsis3_def_160219.mat')