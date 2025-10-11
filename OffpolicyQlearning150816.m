function [ Q, sumQ] = OffpolicyQlearning150816( qldata3 , gamma, alpha, numtraces, ncl_total, nact_total)
% OFF POLICY Q LEARNING

%initialisation of variables
sumQ=zeros(numtraces,1);  %record sum of Q after each iteration

if nargin < 5 || isempty(ncl_total)
    ncl_total = 0;
end
if nargin < 6 || isempty(nact_total)
    nact_total = 0;
end

valid_actions = qldata3(:,3);
valid_states = qldata3(:,2);
valid_actions = valid_actions(isfinite(valid_actions));
valid_states = valid_states(isfinite(valid_states));

if isempty(valid_actions)
    max_action = 1;
else
    max_action = max(valid_actions);
end

if isempty(valid_states)
    max_state = 1;
else
    max_state = max(valid_states);
end

if max_action < 1
    max_action = 1;
end

if max_state < 1
    max_state = 1;
end

nact=max([nact_total,max_action]);   % ensure action dimension large enough
ncl=max([ncl_total,max_state]);    % ensure state dimension large enough
Q=zeros (ncl, nact);  
maxavgQ=1;
modu=100;
listi=find(qldata3(:,1)==1);   %position of 1st step of each episodes in dataset
nrepi=numel(listi);  %nr of episodes in the dataset
jj=1;

if nrepi < 2
    return;
end

 for j=1:numtraces
    
    
    i=listi(floor(rand()*(nrepi-2))+1);  %pick one episode randomly (not the last one!)
    trace = [];
    
    while qldata3(i+1,1)~=1 
    S1=qldata3(i+1,2);
    a1=qldata3(i+1,3);
    r1=qldata3(i+1,4);
     step = [ r1, S1, a1 ];
     trace = [trace ; step];
    i=i+1;
    end

    tracelength = size(trace,1);
    if tracelength == 0
        continue
    end
    return_t = trace(tracelength,1); % get last reward as return for penultimate state and action.
    
   for t=tracelength-1:-1:1       %Step through time-steps in reverse order
        s = round(trace(t,2)); % get state index from trace at time t
        a = round(trace(t,3)); % get action index
        if s < 1 || s > ncl || a < 1 || a > nact || ~isfinite(s) || ~isfinite(a)
            continue
        end
        Q(s,a) = (1-alpha)*Q(s,a) + alpha*return_t; % update Q.
        return_t = return_t*gamma + trace(t,1); % return for time t-1 in terms of return and reward at t
    end
    
     sumQ(jj,1)=sum(sum(Q));
     jj=jj+1;
     
 if mod(j,500*modu)==0  %check if can stop iterating (when no more improvement is seen)
%      sumQ(jj,1)=sum(sum(Q));
%      jj=jj+1;
     s=mean(sumQ(j-49999:j));
     d=(s-maxavgQ)/maxavgQ;
     if abs(d)<0.001
         break   %exit routine
     end
     maxavgQ=s;
 end
 

 end

sumQ(jj:end)=[];


if nargin >=5 && ncl_total>0
    if size(Q,1) < ncl_total
        Q = [Q; zeros(ncl_total-size(Q,1), size(Q,2))];
    elseif size(Q,1) > ncl_total
        Q = Q(1:ncl_total,:);
    end
end

if nargin >=6 && nact_total>0
    if size(Q,2) < nact_total
        Q = [Q zeros(size(Q,1), nact_total-size(Q,2))];
    elseif size(Q,2) > nact_total
        Q = Q(:,1:nact_total);
    end
end

end
