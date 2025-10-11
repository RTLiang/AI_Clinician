function [ bootql ] = offpolicy_eval_tdlearning( qldata3, physpol, gamma, num_iter )
% V value averaged over state population
% hence the difference with mean(V) stored in recqvi(:,3)

ncl=size(physpol,1)-2;
bootql=cell(num_iter,1);
p=unique(qldata3(:,8));
prop=5000/numel(p); %5000 patients of the samples are used
prop=min([prop 0.75]);  %max possible value is 0.75 (75% of the samples are used)

ii=qldata3(:,1)==1;
a=qldata3(ii,2);
d=zeros(ncl,1);
 for i=1:ncl
  d(i)=sum(a==i);    % intitial state disctribution
 end
 
fprintf('Progress of Q-Learning:\n');
fprintf(['\n' repmat('.',1,num_iter) '\n\n']);

num_states = size(physpol,1);
num_actions = size(physpol,2);

parfor i=1:num_iter
fprintf('\b|\n');

ii=floor(rand(size(p,1),1)+prop);     % select a random sample of trajectories
mask=ismember(qldata3(:,8),p(ii==1));
q=qldata3(mask,1:4);

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

% Vs=nansum((physpol.*Qoff)')';
% bootql(i)={sum(Vs(1:ncl).*d)/sum(d)};
end

bootql=cell2mat(bootql);

end

