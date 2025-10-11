function [ bootql,prog ] = offpolicy_eval_tdlearning_with_morta( qldata3, physpol, ptid, idx, actionbloctrain, Y90, gamma, num_iter )

bootql=cell(num_iter,1);
p=unique(qldata3(:,8));
prop=5000/numel(p); %5000 patients of the samples are used
prop=min([prop 0.75]);  %max possible value is 0.75 (75% of the samples are used)

jprog=1;
prog=NaN(floor(size(ptid,1)*1.01*prop*num_iter),4);

ii=qldata3(:,1)==1;
a=qldata3(ii,2);
ncl = size(physpol,1)-2;
d=zeros(ncl,1);

for i=1:ncl; d(i)=sum(a==i); end  % state distribution

num_states = size(physpol,1);
num_actions = size(physpol,2);
 
fprintf('Progress:\n');
fprintf(['\n' repmat('.',1,num_iter) '\n\n']);


for i=1:num_iter
fprintf('\b|\n');

ii=floor(rand(size(p,1),1)+prop);     % select a random sample of trajectories
j=ismember(qldata3(:,8),p(ii==1));
q=qldata3(j==1,1:4);

if isempty(q)
    bootql(i) = {NaN};
    continue
end

[Qoff, ~]=OffpolicyQlearning150816( q , gamma, 0.1, 300000, num_states, num_actions);


if size(Qoff,1) < num_states
    Qoff(num_states, size(Qoff,2)) = 0;
elseif size(Qoff,1) > num_states
    Qoff = Qoff(1:num_states,:);
end

if size(Qoff,2) < num_actions
    Qoff(:, num_actions) = 0;
elseif size(Qoff,2) > num_actions
    Qoff = Qoff(:,1:num_actions);
end

V = physpol .* Qoff;
Vs = sum(V,2);
bootql(i)={nansum(Vs(1:ncl).*d)/sum(d)};
jj=find(ismember(ptid,p(ii==1)));

     for ii=1:numel(jj)  % record offline Q value in training set & outcome - for plot
     state_idx = round(idx(jj(ii)));
     action_idx = round(actionbloctrain(jj(ii)));
     if state_idx>=1 && state_idx<=num_states && action_idx>=1 && action_idx<=num_actions 
         prog(jprog,1)=Qoff(state_idx,action_idx);
     else
         prog(jprog,1)=NaN;
     end
     prog(jprog,2)=Y90(jj(ii));
     prog(jprog,3)=ptid(jj(ii));   %HERE EACH ITERATION GIVES A DIFFERENT PT_ID  //// if I just do rep*ptid it bugs and mixes up ids, for ex with id3 x rep 10 = 30 (which already exists)
     prog(jprog,4)=i;
     jprog=jprog+1;
     end


end

bootql=cell2mat(bootql);
prog(jprog:end,:)=[];


end

