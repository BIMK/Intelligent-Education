function [Fitness,Popobj,Popcon] = CalFitness2(Population,state,knowledge_log,num)
% Calculate the fitness of each solution

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

N=size(Population,1);
Popobj=zeros(N,3);
Popcon=zeros(N,2);


for i=1:N
    Q=Population(i,:);
    Q(Q>=1)=1;
%     disp(size(Population(i,:)));
    Popobj(i,1) = corr(state',Population(i,:)','type','Pearson');
    Popobj(i,2) = -(sum(Q) - sum(Q.*knowledge_log))/(123-sum(knowledge_log));
    Popobj(i,3) = (123 - sum(Q))/123;
    Popcon(i,1) = num - sum(Population(i,:));
    Popcon(i,2) = sum(Population(i,:))- num-round(num/2); 
    
end

   
    
Fitness = CalFitness1(Popobj,Popcon);

end

