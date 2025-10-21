%% AI Clinician Building MIMIC-III dataset 

%% 重新筛选出我们关注的-24h到48h患者状态，def则为筛选sepsis患者而选择-48-24h
%% sepsis_def因为只需要筛选出sepsis定义人群，所使用的特征并非最终所有状态特征
%结构
%AIClinician_mimic3_dataset_160219.m:21）：
% 加载 Sepsis-3 队列与原始事件表，筛出疑似感染前 24h 至后 48h（含 ±4h 缓冲）的 CHARTEVENTS/LAB/MV 记录，写入预分配矩阵 reformat 并记录各 stay 的关键时间戳。
%异常值清理（AIClinician_mimic3_dataset_160219.m:154）：
% 针对体重、血压、电解质等列按生理范围做上下限截断或单位修正，确保后续统计稳定。
%变量互补/纠偏（AIClinician_mimic3_dataset_160219.m:260）：
% 利用变量关系填补缺失（如 RASS→GCS、FiO₂ 百分比↔小数、温度单位、Hb↔Ht、总胆红素↔直接胆红素），并再次对 FiO₂ 做设备逻辑估算。
%样本保持（SAH）（AIClinician_mimic3_dataset_160219.m:390）：
% 对 68 列原始时间序列执行 sample-and-hold 插补，使观测在时间轴上连续。
%数据合并成 4 小时格（AIClinician_mimic3_dataset_160219.m:402）：
% 按 ICU stay 聚合到 4h 时间步，累积/分段计算液体输入、尿量、升压药剂量，并附加人口学与结局信息，形成 reformat2。
%表裁剪（AIClinician_mimic3_dataset_160219.m:578）：
% 给聚合结果命名列、挑出策略学习需要的核心变量，得到 reformat3t。
%插补前调整（AIClinician_mimic3_dataset_160219.m:608）：
% 规范性别/年龄、填补 Elixhauser 与升压药缺失，评估缺失率，并为 Shock Index、P/F 置占位。
%缺失值处理（AIClinician_mimic3_dataset_160219.m:649）：
% 对缺失率 <5% 的列做线性插值，剩余部分按 10k 行块使用 kNN（seuclidean）补齐。
%派生指标（AIClinician_mimic3_dataset_160219.m:693）：
% 重算 P/F、Shock Index（含极值处理），基于阈值规则生成逐时间步的 SOFA 与 SIRS，并修正液体相关变量。
%最终表保存（AIClinician_mimic3_dataset_160219.m:784）：
% 把 reformat4t 定名为 MIMICtable 并存入 ./BACKUP/MIMICtable.mat，供后续策略学习与评估调用。

% 本脚本注释说明
% -------------------------------------------------------------------------
% 目的：在已识别的 Sepsis-3 队列基础上，围绕疑似感染时间重建 4 小时步长的特征时间序列，
%       清洗/插补/派生变量，最终生成供 MDP 学习与评估的 MIMICtable。
% 关系与差异：
% - 上游（Sepsis 脚本）用于筛选队列（-48 → +24），本脚本用于构建策略学习数据（-24 → +48）。
% - 两者有重叠的清洗/派生步骤，但服务的目标与时间窗不同。

% (c) Matthieu Komorowski, Imperial College London 2015-2019
% as seen in publication: https://www.nature.com/articles/s41591-018-0213-5

% version 16 Feb 19
% uses the sepsis-3 cohort previously defined
% builds the MIMIC-III dataset

% This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
% without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE

%% ########################################################################
%           INITIAL REFORMAT WITH CHARTEVENTS, LABS AND MECHVENT
%           使用 CHARTEVENTS、LABS 和机械通气数据进行初始重整
% ########################################################################

load('./BACKUP/AIClinician_sepsis3_def_160219.mat')

disp('INITIAL REFORMAT START');

% gives an array with all unique charttime (1 per row) and all items in columns.
% ################## IMPORTANT !!!!!!!!!!!!!!!!!!
% Here i use -24 -> +48 because that's for the MDP
% MDP 的时间范围是 -24 -> +48，与 sepsis3 定义的 -48 -> +24 不同
% 注：为避免 4h 分箱边界截断，原始时间戳筛选时额外加入 ±4h 缓冲，
% 例如：例如正好在感染前 24 小时这个点，直接截断可能导致它被排除或分到下一格。
%     后续再在固定 80 小时时间带内（约 -28 → +52）做聚合。

% 以下代码和 AIClinician_sepsis3_def_160219.m 高度相似，详细注释请参见该脚本。

reformat=NaN(2000000,68);  % 原始“逐时间点”宽表（行：时间点；列：变量）
qstime=zeros(100000,4);    % 每个 stay 的关键时间：[疑似感染, 首条时间, 末条时间, 出院时间]
winb4=25;                  % 前窗（小时）：24h + 1h 缓冲（配合下方 ±4h）
winaft=49;                 % 后窗（小时）：48h + 1h 缓冲
irow=1;                    % reformat 写入行指针
h = waitbar(0,'Initializing waitbar...');  % 进度条

tic

% sepsis表头：
% 1 icustayid（真实编号）
% 2 morta_90d（90天死亡）
% 3 max_sofa（窗口内最大 SOFA）
% 4 max_sirs（窗口内最大 SIRS）
% 5 sepsis_time（疑似感染起点 qst）

for icustayidrow=1:size(sepsis,1)  % 逐条遍历已筛出的 sepsis 队列（每行一个 ICU stay）
    
qst=sepsis.sepsis_time(icustayidrow);%,3); %flag for presumed infection
icustayid=sepsis.icustayid(icustayidrow)-200000;
waitbar(icustayidrow/size(sepsis,1),h,icustayidrow/size(sepsis,1)*100) %moved here to save some time


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

% 4h 分箱缓冲：在理论窗口两端各延伸 4h，避免分箱边界问题
ii=temp(:,2)>= qst-(winb4+4)*3600 & temp(:,2)<=qst+(winaft+4)*3600; %time period of interest -4h and +4h
temp=temp(ii,:);   %only time period of interest

%LABEVENTS
ii=labU(:,1)==icustayid+200000;
temp2=labU(ii,:);
ii=temp2(:,2)>= qst-(winb4+4)*3600 & temp2(:,2)<=qst+(winaft+4)*3600; %time period of interest -4h and +4h
temp2=temp2(ii,:);   %only time period of interest

%Mech Vent + ?extubated
ii=MV(:,1)==icustayid+200000;
temp3=MV(ii,:);
ii=temp3(:,2)>= qst-(winb4+4)*3600 & temp3(:,2)<=qst+(winaft+4)*3600; %time period of interest -4h and +4h
temp3=temp3(ii,:);   %only time period of interest

t=unique([temp(:,2);temp2(:,2); temp3(:,2)]);   % 三路时间戳合并去重，升序排列

if t
for i=1:numel(t)
    
    %CHARTEVENTS（第1列=序号；第2列=内部icustayid；第3列=charttime）：
    ii=temp(:,2)==t(i);
    col=temp(ii,3);
    value=temp(ii,4);  
    reformat(irow,1)=i; %timestep  
    reformat(irow,2)=icustayid;
    reformat(irow,3)=t(i); %charttime
    reformat(irow,3+col)=value; %store available values
      
    %LAB VALUES：按映射后的列号直接落位到宽表
    ii=temp2(:,2)==t(i);
    col=temp2(ii,3);
    value=temp2(ii,4);
    reformat(irow,31+col)=value; %store available values
      
    %MV  机械通气/拔管标志（若此时刻存在则写入，否则置 NaN）
    ii=temp3(:,2)==t(i);
    if nansum(ii)>0
    value=temp3(ii,3:4);
      reformat(irow,67:68)=value; %store available values
    else
      reformat(irow,67:68)=NaN;
    end
    
    irow=irow+1;
     
end

% 记录本 stay 的关键时间，用于后续派生/校验
qstime(icustayid,1)=qst; %time of sepsis
qstime(icustayid,2)=t(1);  %first timestamp
qstime(icustayid,3)=t(end);  %last timestamp
qstime(icustayid,4)=table2array(demog(demog.icustay_id==icustayid+200000,5)); % discharge time

end

% end
end
toc

close(h);
reformat(irow:end,:)=[];  %delete extra unused rows（裁掉未使用的预分配行）

disp('INITIAL REFORMAT END');


%% ########################################################################
%                                   OUTLIERS 
% ########################################################################

disp('OUTLIER CLEANING START');
% 说明：按变量物理/生理范围做异常值截断与单位纠偏；列索引含义与上游脚本一致。
% 以下代码和 AIClinician_sepsis3_def_160219.m 高度相似，详细注释请参见该脚本。

%weight
reformat=deloutabove(reformat,5,300);  %delete outlier above a threshold (300 kg), for variable # 5

%HR
reformat=deloutabove(reformat,8,250);

%BP
reformat=deloutabove(reformat,9,300);
reformat=deloutbelow(reformat,10,0);
reformat=deloutabove(reformat,10,200);
reformat=deloutbelow(reformat,11,0);
reformat=deloutabove(reformat,11,200);

%RR
reformat=deloutabove(reformat,12,80);

%SpO2
reformat=deloutabove(reformat,13,150);
ii=reformat(:,13)>100;reformat(ii,13)=100;
reformat=deloutbelow(reformat,13,50);

%temp
ii=reformat(:,14)>90 & isnan(reformat(:,15));reformat(ii,15)=reformat(ii,14);
reformat=deloutabove(reformat,14,90);
reformat=deloutbelow(reformat,14,25);

%interface / is in col 22

% FiO2
reformat=deloutabove(reformat,23,100);
ii=reformat(:,23)<1;reformat(ii,23)=reformat(ii,23)*100;
reformat=deloutbelow(reformat,23,20);
reformat=deloutabove(reformat,24,1.5);

% O2 FLOW
reformat=deloutabove(reformat,25,70);

%PEEP
reformat=deloutbelow(reformat,26,0);
reformat=deloutabove(reformat,26,40);

%TV
reformat=deloutabove(reformat,27,1800);

%MV
reformat=deloutabove(reformat,28,50);

%K+
reformat=deloutbelow(reformat,32,1);
reformat=deloutabove(reformat,32,15);

%Na
reformat=deloutbelow(reformat,33,95);
reformat=deloutabove(reformat,33,178);

%Cl
reformat=deloutbelow(reformat,34,70);
reformat=deloutabove(reformat,34,150);

%Glc
reformat=deloutbelow(reformat,35,1);
reformat=deloutabove(reformat,35,1000);

%Creat
reformat=deloutabove(reformat,37,150);

%Mg
reformat=deloutabove(reformat,38,10);

%Ca
reformat=deloutabove(reformat,39,20);

%ionized Ca
reformat=deloutabove(reformat,40,5);

%CO2
reformat=deloutabove(reformat,41,120);

%SGPT/SGOT
reformat=deloutabove(reformat,42,10000);
reformat=deloutabove(reformat,43,10000);

%Hb/Ht
reformat=deloutabove(reformat,50,20);
reformat=deloutabove(reformat,51,65);

%WBC
reformat=deloutabove(reformat,53,500);

%plt
reformat=deloutabove(reformat,54,2000);

%INR
reformat=deloutabove(reformat,58,20);

%pH
reformat=deloutbelow(reformat,59,6.7);
reformat=deloutabove(reformat,59,8);

%po2
reformat=deloutabove(reformat,60,700);

%pco2
reformat=deloutabove(reformat,61,200);

%BE
reformat=deloutbelow(reformat,62,-50);

%lactate
reformat=deloutabove(reformat,63,30);

disp('OUTLIER CLEANING END');

% ####################################################################
% some more data manip / imputation from existing values

disp('DATA MANIPULATION START');
% 说明：基于变量间已知关系做进一步补全/纠偏（如 RASS→GCS、FiO2 百分比/小数互转、温度单位互换、
%       Hb/Ht 互推、Total/Direct_bili 线性关系等）。
% 以下代码和 AIClinician_sepsis3_def_160219.m 高度相似，详细注释请参见该脚本。

% estimate GCS from RASS - data from Wesley JAMA 2003
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


% FiO2
ii=~isnan(reformat(:,23)) & isnan(reformat(:,24));
reformat(ii,24)=reformat(ii,23)./100;
ii=~isnan(reformat(:,24)) & isnan(reformat(:,23));
reformat(ii,23)=reformat(ii,24).*100;


%ESTIMATE FiO2 /// with use of interface / device (cannula, mask, ventilator....)

reformatsah=SAH(reformat,sample_and_hold);  % do SAH first to handle this task

%NO FiO2, YES O2 flow, no interface OR cannula
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

%NO FiO2, NO O2 flow, no interface OR cannula
ii=find(isnan(reformatsah(:,23))&isnan(reformatsah(:,25))&(reformatsah(:,22)==0|reformatsah(:,22)==2));  %no fio2 given and o2flow given, no interface OR cannula
reformat(ii,23)=21;

%NO FiO2, YES O2 flow, face mask OR.... OR ventilator (assume it's face mask)
ii=find(isnan(reformatsah(:,23))&~isnan(reformatsah(:,25))&(reformatsah(:,22)==NaN|reformatsah(:,22)==1|reformatsah(:,22)==3|reformatsah(:,22)==4|reformatsah(:,22)==5|reformatsah(:,22)==6|reformatsah(:,22)==9|reformatsah(:,22)==10)); 
reformat(ii(reformatsah(ii,25)<=15),23)=75;
reformat(ii(reformatsah(ii,25)<=12),23)=69;
reformat(ii(reformatsah(ii,25)<=10),23)=66;
reformat(ii(reformatsah(ii,25)<=8),23)=58;
reformat(ii(reformatsah(ii,25)<=6),23)=40;
reformat(ii(reformatsah(ii,25)<=4),23)=36;

%NO FiO2, NO O2 flow, face mask OR ....OR ventilator
ii=find(isnan(reformatsah(:,23))&isnan(reformatsah(:,25))&(reformatsah(:,22)==NaN|reformatsah(:,22)==1|reformatsah(:,22)==3|reformatsah(:,22)==4|reformatsah(:,22)==5|reformatsah(:,22)==6|reformatsah(:,22)==9|reformatsah(:,22)==10));  %no fio2 given and o2flow given, no interface OR cannula
reformat(ii,23)=NaN;

%NO FiO2, YES O2 flow, Non rebreather mask
ii=find(isnan(reformatsah(:,23))&~isnan(reformatsah(:,25))&reformatsah(:,22)==7); 
reformat(ii(reformatsah(ii,25)>=10),23)=90;
reformat(ii(reformatsah(ii,25)>=15),23)=100;
reformat(ii(reformatsah(ii,25)<10),23)=80;
reformat(ii(reformatsah(ii,25)<=8),23)=70;
reformat(ii(reformatsah(ii,25)<=6),23)=60;

%NO FiO2, NO O2 flow, NRM
ii=find(isnan(reformatsah(:,23))&isnan(reformatsah(:,25))&reformatsah(:,22)==7);  %no fio2 given and o2flow given, no interface OR cannula
reformat(ii,23)=NaN;

% update again FiO2 columns
ii=~isnan(reformat(:,23)) & isnan(reformat(:,24));
reformat(ii,24)=reformat(ii,23)./100;
ii=~isnan(reformat(:,24)) & isnan(reformat(:,23));
reformat(ii,23)=reformat(ii,24).*100;

%BP
ii=~isnan(reformat(:,9))&~isnan(reformat(:,10)) & isnan(reformat(:,11));
reformat(ii,11)=(3*reformat(ii,10)-reformat(ii,9))./2;
ii=~isnan(reformat(:,09))&~isnan(reformat(:,11)) & isnan(reformat(:,10));
reformat(ii,10)=(reformat(ii,9)+2*reformat(ii,11))./3;
ii=~isnan(reformat(:,10))&~isnan(reformat(:,11)) & isnan(reformat(:,9));
reformat(ii,9)=3*reformat(ii,10)-2*reformat(ii,11);

%TEMP
%some values recorded in the wrong column
ii=reformat(:,15)>25&reformat(:,15)<45; %tempF close to 37deg??!
reformat(ii,14)=reformat(ii,15);
reformat(ii,15)=NaN;
ii=reformat(:,14)>70;  %tempC > 70?!!! probably degF
reformat(ii,15)=reformat(ii,14);
reformat(ii,14)=NaN;
ii=~isnan(reformat(:,14)) & isnan(reformat(:,15));
reformat(ii,15)=reformat(ii,14)*1.8+32;
ii=~isnan(reformat(:,15)) & isnan(reformat(:,14));
reformat(ii,14)=(reformat(ii,15)-32)./1.8;

% Hb/Ht
ii=~isnan(reformat(:,50)) & isnan(reformat(:,51));
reformat(ii,51)=(reformat(ii,50)*2.862)+1.216;
ii=~isnan(reformat(:,51)) & isnan(reformat(:,50));
reformat(ii,50)=(reformat(ii,51)-1.216)./2.862;

%BILI
ii=~isnan(reformat(:,44)) & isnan(reformat(:,45));
reformat(ii,45)=(reformat(ii,44)*0.6934)-0.1752;
ii=~isnan(reformat(:,45)) & isnan(reformat(:,44));
reformat(ii,44)=(reformat(ii,45)+0.1752)./0.6934;

disp('DATA MANIPULATION END');


%% ########################################################################
%                      SAMPLE AND HOLD on RAW DATA
% ########################################################################
% 以下代码和 AIClinician_sepsis3_def_160219.m 高度相似，详细注释请参见该脚本。

disp('SAMPLE AND HOLD START');

reformat=SAH(reformat(:,1:68),sample_and_hold);

disp('SAMPLE AND HOLD END');


%% ########################################################################
%                             DATA COMBINATION
% ########################################################################

disp('DATA COMBINATION START');
% 以下代码和 AIClinician_sepsis3_def_160219.m 高度相似，详细注释请参见该脚本。

tic
     save('./BACKUP/dataset_before_combination.mat', '-v7.3');
toc


% WARNING: the time window of interest has been defined above (here -24 -> +48)!

timestep=4;  % 分箱步长（小时）
irow=1;      % reformat2 写指针
icustayidlist=unique(reformat(:,2));  % 需要聚合的 ICU stay 内部索引列表
reformat2=nan(size(reformat,1),84);   % 输出宽表（4h 槽 × 特征），84 列布局如下：
%  1=bloc（timestep 序号）  2=icustayid（内部索引）  3=t0（槽左边界时间戳）
%  4:11=人口学与结局（gender, age, elixhauser, re_admission, died_in_hosp,
%       died_within_48h_of_out_time, mortality_90d, delay_end_of_record...）
%  12:78=生命体征+化验（来自 reformat 第 4..end 的聚合均值）
%  79:80=升压药剂量：median_dose_vaso / max_dose_vaso（4h 槽内）
%  81:84=液体与尿量：input_total / input_4hourly / output_total / output_4hourly
h = waitbar(0,'Initializing waitbar...');
npt=numel(icustayidlist);  % 住院数
% 为 Shock_Index / PaO2_FiO2 预留空列（稍后派生再赋值）
reformat(:,69:70)=NaN(size(reformat,1),2);

tic
for i=1:npt
    
    icustayid=icustayidlist(i);  %1 to 100000, NOT 200 to 300K!
     
        %CHARTEVENTS AND LAB VALUES
        temp=reformat(reformat(:,2)==icustayid,:);   %subtable of interest
        beg=temp(1,3);   %timestamp of first record
    
        % IV 液体（输入量）：MetaVision 连续记录 + CareVue 事件
        iv=find(inputMV(:,1)==icustayid+200000);   %rows of interest in inputMV
        input=inputMV(iv,:);    %subset of interest
        iv=find(inputCV(:,1)==icustayid+200000);   %rows of interest in inputCV
        input2=inputCV(iv,:);    %subset of interest
        startt=input(:,2); % 连续/推注开始时间（秒）
        endt=input(:,3);   % 连续/推注结束时间（秒）
        rate=input(:,8);   % 连续输注速率（推注处为 NaN）
        
        % 入科前输入量作为累计入量起点
        pread=inputpreadm(inputpreadm(:,1)==icustayid+200000,2) ;
        if ~isempty(pread)
            totvol=nansum(pread);  % 初始累计入量（mL）
            waitbar(i/npt,h,i/npt*100)
        else
            totvol=0;
        end
       
        % 记录开始前的输入量（以 beg 为界）：连续输注四种区间相交情形 + 推注事件量
        t0=0;
        t1=beg;
        %input from MV (4 ways to compute)
        infu=  nansum(rate.*(endt-startt).*(endt<=t1&startt>=t0)/3600   +    rate.*(endt-t0).*(startt<=t0&endt<=t1&endt>=t0)/3600 +     rate.*(t1-startt).*(startt>=t0&endt>=t1&startt<=t1)/3600 +      rate.*(t1-t0).*(endt>=t1&startt<=t0)   /3600);
        %all boluses received during this timestep, from inputMV (need to check rate is NaN) and inputCV (simpler):
        bolus=nansum(input(isnan(input(:,6))& input(:,2)>=t0&input(:,2)<=t1,7)) + nansum(input2(input2(:,2)>=t0&input2(:,2)<=t1,5));  
        totvol=nansum([totvol,infu,bolus]); 
            
        % 升压药（VASOPRESSORS）：MetaVision 连续 + CareVue 事件
        iv=find(vasoMV(:,1)==icustayid+200000);   %rows of interest in vasoMV
        vaso1=vasoMV(iv,:);    %subset of interest
        iv=find(vasoCV(:,1)==icustayid+200000);   %rows of interest in vasoCV
        vaso2=vasoCV(iv,:);    %subset of interest
        startv=vaso1(:,3); % 连续输注开始
        endv=vaso1(:,4);   % 连续输注结束
        ratev=vaso1(:,5);  % 连续输注速率
            

        % 人口学与结局（8 列）：gender / age / elixhauser / re-admission /
        % died_in_hosp / died_within_48h_of_out_time / morta_90d /
        % delay_end_of_record_and_discharge_or_death（小时）
        demogi=find(demog.icustay_id==icustayid+200000);        
        dem=[  demog.gender(demogi) ; demog.age(demogi) ;demog.elixhauser(demogi) ; demog.adm_order(demogi)>1 ;  demog.morta_hosp(demogi); abs(demog.dod(demogi)-demog.outtime(demogi))<(24*3600*2); demog.morta_90(demogi) ; (qstime(icustayid,4)-qstime(icustayid,3))/3600];     
        
        
        % 尿量（URINE OUTPUT）：入科前尿量作为累计起点 + 记录开始前尿量
        iu=find(UO(:,1)==icustayid+200000);   %rows of interest in inputMV
        output=UO(iu,:);    %subset of interest
        pread=UOpreadm(UOpreadm(:,1)==icustayid,4) ;%preadmission UO
            if ~isempty(pread)     %store the value, if available
                UOtot=nansum(pread);
            else
                UOtot=0;
            end
        % adding the volume of urine produced before start of recording!    
        UOnow=nansum(output(output(:,2)>=t0&output(:,2)<=t1,4));  %t0 and t1 defined above
        UOtot=nansum([UOtot UOnow]);
    
    
    for j=0:timestep:79 % 固定 80 小时（-28 → +52），与理论窗口（-24 → +48）边界对齐
        t0=3600*j+ beg;                 % 槽左边界（秒）
        t1=3600*(j+timestep)+beg;       % 槽右边界（秒）
        ii=temp(:,3)>=t0 & temp(:,3)<=t1;  % 本槽内的观测行
        if sum(ii)>0
            
            
        %ICUSTAY_ID, OUTCOMES, DEMOGRAPHICS
        reformat2(irow,1)=(j/timestep)+1;   % bloc：时间槽序号（1..）
        reformat2(irow,2)=icustayid;        % icustayid：内部索引
        reformat2(irow,3)=3600*j+ beg;      % t0：槽左边界时间戳
        reformat2(irow,4:11)=dem;           % 人口学与结局（8 列）
            
        
        %CHARTEVENTS and LAB VALUES：本 4h 槽内的多个原始观测取均值（若仅 1 条则直接写入）
        value=temp(ii,:);
                
        if sum(ii)==1   %if only 1 row of values at this timestep
          reformat2(irow,12:78)=value(:,4:end);
        else
          reformat2(irow,12:78)=nanmean(value(:,4:end)); %mean of all available values
          
        end
        
        
        %VASOPRESSORS
            % for CV: dose at timestamps.
            % for MV: 4 possibles cases, each one needing a different way to compute the dose of VP actually administered:
            %----t0---start----end-----t1----
            %----start---t0----end----t1----
            %-----t0---start---t1---end
            %----start---t0----t1---end----

        
        %MV：连续/事件与 t0/t1 的区间相交（4 情形），提取本槽剂量
        v=(endv>=t0&endv<=t1)|(startv>=t0&endv<=t1)|(startv>=t0&startv<=t1)|(startv<=t0&endv>=t1);
        %CV
        v2=vaso2(vaso2(:,3)>=t0&vaso2(:,3)<=t1,4);
        v1=nanmedian([ratev(v); v2]);
        v2=nanmax([ratev(v); v2]);
        if ~isempty(v1)&~isnan(v1)&~isempty(v2)&~isnan(v2)
        reformat2(irow,79)=v1;    % 升压药中位剂量（本槽）
        reformat2(irow,80)=v2;    % 升压药最大剂量（本槽）
        end
        
        %INPUT FLUID：MV 连续（4 种覆盖情形）+ MV/CV 推注 → 槽内量与累计量
        infu=  nansum(rate.*(endt-startt).*(endt<=t1&startt>=t0)/3600   +    rate.*(endt-t0).*(startt<=t0&endt<=t1&endt>=t0)/3600 +     rate.*(t1-startt).*(startt>=t0&endt>=t1&startt<=t1)/3600 +      rate.*(t1-t0).*(endt>=t1&startt<=t0)   /3600);
        %all boluses received during this timestep, from inputMV (need to check rate is NaN) and inputCV (simpler):
        bolus=nansum(input(isnan(input(:,6))& input(:,2)>=t0&input(:,2)<=t1,7)) + nansum(input2(input2(:,2)>=t0&input2(:,2)<=t1,5));  
 
        %sum fluid given
        totvol=nansum([totvol,infu,bolus]);
        reformat2(irow,81)=totvol;           % input_total（累计）
        reformat2(irow,82)=nansum([infu,bolus]);   % input_4hourly（本槽）
        
        %UO：4h 槽内尿量与累计尿量
        UOnow=nansum(output(output(:,2)>=t0&output(:,2)<=t1,4));  
        UOtot=nansum([UOtot UOnow]);
        reformat2(irow,83)=UOtot;           % output_total（累计）
        reformat2(irow,84)=nansum(UOnow);   % output_4hourly（本槽）

        %CUMULATED BALANCE：累计入量 - 累计尿量
        reformat2(irow,85)=totvol-UOtot;    % cumulated balance

        irow=irow+1;
        end
    end
end
toc

reformat2(irow:end,:)=[];  % 裁掉未使用的预分配行
close(h);


tic
     save('./BACKUP/dataset_after_combination.mat', '-v7.3');
toc

disp('DATA COMBINATION END');

%% ########################################################################
%             CONVERT TO TABLE AND KEEP ONLY WANTED VARIABLE
% ########################################################################

disp('TABLE REDUCTION START');
% 说明：拼装列名，构建宽表为 table，并裁剪为策略学习所需的变量子集（见 dataheaders5）。


dataheaders=[sample_and_hold(1,:) {'Shock_Index' 'PaO2_FiO2'}]; 
dataheaders=regexprep(dataheaders,'['']','');
dataheaders = ['bloc','icustayid','charttime','gender','age','elixhauser','re_admission', 'died_in_hosp', 'died_within_48h_of_out_time','mortality_90d','delay_end_of_record_and_discharge_or_death',...
    dataheaders,  'median_dose_vaso','max_dose_vaso','input_total','input_4hourly','output_total','output_4hourly','cumulated_balance'];

reformat2t=array2table(reformat2);
reformat2t.Properties.VariableNames=dataheaders;

% headers I want to keep
% 保留列
dataheaders5 = {'bloc','icustayid','charttime','gender','age','elixhauser','re_admission', 'died_in_hosp', 'died_within_48h_of_out_time','mortality_90d','delay_end_of_record_and_discharge_or_death','SOFA','SIRS',...
    'Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','SpO2','Temp_C','FiO2_1','Potassium','Sodium','Chloride','Glucose',...
    'BUN','Creatinine','Magnesium','Calcium','Ionised_Ca','CO2_mEqL','SGOT','SGPT','Total_bili','Albumin','Hb','WBC_count','Platelets_count','PTT','PT','INR',...
    'Arterial_pH','paO2','paCO2','Arterial_BE','HCO3','Arterial_lactate','mechvent','Shock_Index','PaO2_FiO2',...
    'median_dose_vaso','max_dose_vaso','input_total','input_4hourly','output_total','output_4hourly','cumulated_balance'};

ii=find(ismember(reformat2t.Properties.VariableNames,dataheaders5));
reformat3t=reformat2t(:,ii); 

disp('TABLE REDUCTION END');


%% SOME DATA MANIP BEFORE IMPUTATION

disp('PRE-IMPUTATION ADJUSTMENTS START');
% 说明：规范布尔/人口学变量；将升压药剂量缺失置零；评估缺失率；
%       为 kNN 插补预留 Shock_Index / PaO2_FiO2 占位。
% 以下代码和 AIClinician_sepsis3_def_160219.m 高度相似，详细注释请参见该脚本。

% CORRECT GENDER
reformat3t.gender=reformat3t.gender-1; 

%CORRECT AGE > 200 yo
ii=reformat3t.age>150*365.25;
reformat3t.age(ii)=91.4*365.25;

% FIX MECHVENT
reformat3t.mechvent(isnan(reformat3t.mechvent))=0;
reformat3t.mechvent(reformat3t.mechvent>0)=1;

% FIX Elixhauser missing values
reformat3t.elixhauser(isnan(reformat3t.elixhauser))=nanmedian(reformat3t.elixhauser);  %use the median value / only a few missing data points 

%vasopressors / no NAN
a=find(ismember(reformat3t.Properties.VariableNames,{'median_dose_vaso'}));
ii=isnan(table2array(reformat3t(:,a)));
reformat3t(ii,a)=array2table(zeros(sum(ii),1));
a=find(ismember(reformat3t.Properties.VariableNames,{'max_dose_vaso'}));
ii=isnan(table2array(reformat3t(:,a)));
reformat3t(ii,a)=array2table(zeros(sum(ii),1));

% check prop of missingness here
miss=array2table(sum(isnan(table2array(reformat3t)))./size(reformat3t,1));
miss.Properties.VariableNames=reformat3t.Properties.VariableNames;
figure; bar(sum(isnan(table2array(reformat3t)))./size(reformat3t,1))

% I fill the values temporarily with zeros, otherwise kNN imp doesnt work
reformat3t.Shock_Index=zeros(size(reformat3t,1),1);
reformat3t.PaO2_FiO2=zeros(size(reformat3t,1),1);

disp('PRE-IMPUTATION ADJUSTMENTS END');


%% ########################################################################
%        HANDLING OF MISSING VALUES & CREATE REFORMAT4T
% ########################################################################

disp('MISSING VALUE IMPUTATION START');
% 说明：先对缺失率 <5% 的列做线性插值（fixgaps），其余按 10K 行块做 kNN（seuclidean 距离）。

% Do linear interpol where missingness is low (kNN imputation doesnt work if all rows have missing values)
reformat3=table2array(reformat3t);
miss=sum(isnan((reformat3)))./size(reformat3,1);
ii=miss>0&miss<0.05;  %less than 5% missingness
mechventcol=find(ismember(reformat3t.Properties.VariableNames,{'mechvent'}));

for i=11:mechventcol-1 % correct col by col, otherwise it does it wrongly
  if ii(i)==1
    reformat3(:,i)=fixgaps(reformat3(:,i));
  end
end

reformat3t(:,11:mechventcol-1)=array2table(reformat3(:,11:mechventcol-1));

% KNN IMPUTATION -  Done on chunks of 10K records.

mechventcol=find(ismember(reformat3t.Properties.VariableNames,{'mechvent'}));
ref=reformat3(:,11:mechventcol-1);  %columns of interest

tic
for i=1:10000:size(reformat3,1)-9999   %dataset divided in 5K rows chunks (otherwise too large)
    i
    ref(i:i+9999,:)=knnimpute(ref(i:i+9999,:)',1, 'distance','seuclidean')';
end

ref(end-9999:end,:)=knnimpute(ref(end-9999:end,:)',1, 'distance','seuclidean')';  %the last bit is imputed from the last 10K rows

toc

% I paste the data interpolated, but not the demographics and the treatments
reformat3t(:,11:mechventcol-1)=array2table(ref);  

reformat4t=reformat3t;
reformat4=table2array(reformat4t);

disp('MISSING VALUE IMPUTATION END');

%% ########################################################################
%        COMPUTE SOME DERIVED VARIABLES: P/F, Shock Index, SOFA, SIRS...
% ########################################################################
disp('DERIVED VARIABLE COMPUTATION START');
% 说明：计算 P/F、Shock Index（含极端值处理），并基于阈值规则计算 SOFA/SIRS。


% re-compute P/F with no missing values...
p=find(ismember(reformat4t.Properties.VariableNames,{'paO2'}));
f=find(ismember(reformat4t.Properties.VariableNames,{'FiO2_1'}));
a=find(ismember(reformat4t.Properties.VariableNames,{'PaO2_FiO2'}));
reformat4t(:,a)=array2table(reformat4(:,p)./reformat4(:,f));  

%recompute SHOCK INDEX without NAN and INF
p=find(ismember(reformat4t.Properties.VariableNames,{'HR'}));
f=find(ismember(reformat4t.Properties.VariableNames,{'SysBP'}));
a=find(ismember(reformat4t.Properties.VariableNames,{'Shock_Index'}));
reformat4(:,a)=reformat4(:,p)./reformat4(:,f);  
reformat4(isinf(reformat4(:,a)),a)=NaN;
d=nanmean(reformat4(:,a));
reformat4(isnan(reformat4(:,a)),a)=d;  %replace NaN with average value ~ 0.8
reformat4t(:,a)=array2table(reformat4(:,a));
ii=reformat4t.Shock_Index>=quantile(reformat4t.Shock_Index,0.999); %replace outliers with 99.9th percentile
reformat4t.Shock_Index(ii)=quantile(reformat4t.Shock_Index,0.999);

% SOFA - at each timepoint
% need (in this order):  P/F  MV  PLT  TOT_BILI  MAP  NORAD(max)  GCS  CR  UO
a=zeros(8,1);                              % indices of vars used in SOFA
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

s1=[s(:,1)>400 s(:,1)>=300 &s(:,1)<400 s(:,1)>=200 &s(:,1)<300 s(:,1)>=100 &s(:,1)<200 s(:,1)<100 ];   %count of points for all 6 criteria of sofa
s2=[s(:,2)>150 s(:,2)>=100 &s(:,2)<150 s(:,2)>=50 &s(:,2)<100 s(:,2)>=20 &s(:,2)<50 s(:,2)<20 ];
s3=[s(:,3)<1.2 s(:,3)>=1.2 &s(:,3)<2 s(:,3)>=2 &s(:,3)<6 s(:,3)>=6 &s(:,3)<12 s(:,3)>12 ];
s4=[s(:,4)>=70 s(:,4)<70&s(:,4)>=65 s(:,4)<65 s(:,5)>0 &s(:,5)<=0.1 s(:,5)>0.1 ];
s5=[s(:,6)>14 s(:,6)>12 &s(:,6)<=14 s(:,6)>9 &s(:,6)<=12 s(:,6)>5 &s(:,6)<=9 s(:,6)<=5 ];
s6=[s(:,7)<1.2 s(:,7)>=1.2 &s(:,7)<2 s(:,7)>=2 &s(:,7)<3.5 (s(:,7)>=3.5 &s(:,7)<5)|(s(:,8)<84) (s(:,7)>5)|(s(:,8)<34) ];

nrcol=size(reformat4,2);   %nr of variables in data
reformat4(1,nrcol+1:nrcol+7)=0;  
for i=1:size(reformat4,1)  
    t=max(p(s1(i,:)))+max(p(s2(i,:)))+max(p(s3(i,:)))+max(p(s4(i,:)))+max(p(s5(i,:)))+max(p(s6(i,:)));  %SUM OF ALL 6 CRITERIA
    
    if t
    reformat4(i,nrcol+1:nrcol+7)=    [max(p(s1(i,:))) max(p(s2(i,:))) max(p(s3(i,:))) max(p(s4(i,:))) max(p(s5(i,:))) max(p(s6(i,:))) t];
    end
end

% SIRS - at each timepoint |  need: temp HR RR PaCO2 WBC 
a=zeros(5,1); % indices of vars used in SOFA
a(1)=find(ismember(reformat4t.Properties.VariableNames,{'Temp_C'}));
a(2)=find(ismember(reformat4t.Properties.VariableNames,{'HR'}));
a(3)=find(ismember(reformat4t.Properties.VariableNames,{'RR'}));
a(4)=find(ismember(reformat4t.Properties.VariableNames,{'paCO2'}));
a(5)=find(ismember(reformat4t.Properties.VariableNames,{'WBC_count'}));
s=table2array(reformat4t(:,a));  

s1=[s(:,1)>=38| s(:,1)<=36];   %count of points for all criteria of SIRS
s2=[s(:,2)>90 ];
s3=[s(:,3)>=20|s(:,4)<=32];
s4=[s(:,5)>=12| s(:,5)<4];
reformat4(:,nrcol+8)=s1+s2+s3+s4;

% adds 2 cols for SOFA and SIRS, if necessary
if sum(ismember(reformat4t.Properties.VariableNames,{'SIRS'}))== 0
reformat4t(:,end+1:end+2)=array2table(0);
reformat4t.Properties.VariableNames(end-1:end)= {'SOFA','SIRS'};  
end

% more IO corrections
 ii=reformat4t.input_total<0;
 reformat4t.input_total(ii)=0;
 ii=reformat4t.input_4hourly<0;
 reformat4t.input_4hourly(ii)=0;

% records values
reformat4t(:,end-1)=array2table(reformat4(:,end-1));
reformat4t(:,end)=array2table(reformat4(:,end));


disp('DERIVED VARIABLE COMPUTATION END');

%% ########################################################################
%                     CREATE FINAL MIMIC_TABLE
% ########################################################################

disp('FINAL TABLE CREATION START');
% 说明：生成最终的 MIMICtable，并持久化为 .mat（-v7.3 以支持大体量）。
MIMICtable = reformat4t;
% 保存MIMICtable，下次使用就可以直接load
save('./BACKUP/MIMICtable.mat', 'MIMICtable', '-v7.3');
disp('FINAL TABLE CREATION END');
