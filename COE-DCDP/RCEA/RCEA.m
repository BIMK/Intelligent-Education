 function RCEA(Global)
% <algorithm> <R>
% Evolutionary algorithm for sparse multi-objective optimization problems
% stuId ---  1 --- Id of student
% num   --- 100 --- number of recommended exercises
%------------------------------- Reference --------------------------------
% Y. Tian, X. Zhang, C. Wang, and Y. Jin, An evolutionary algorithm for
% large-scale sparse multi-objective optimization problems, IEEE
% Transactions on Evolutionary Computation, 2019.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

    %% Population initialization
    [stuId,num] = Global.ParameterSet(1,100);
    str    = {'student_emb','k_difficulty','Q_matrix','exercise_log','knowledge_log','predition'};
    CallStack = dbstack('-completenames');
    load(fullfile(fileparts(CallStack(1).file),'dataSet2.mat'),'dataSet2');
    load(fullfile(fileparts(CallStack(1).file),'student_emb2.mat'),'student_emb2');
    real_emb=student_emb2(235,:);
    stu_state = dataSet2.(str{1})(stuId,:);
    exer_difficulty = dataSet2.(str{2});
    Q_matrix = dataSet2.(str{3});
    exer_log = dataSet2.(str{4})(stuId,:);
    knowledge_log = dataSet2.(str{5})(stuId,:);
    predition = dataSet2.(str{6})(stuId,:);
    
    indexc1 = find(exer_log==1);
    indexc2 = find(predition<0.2 | predition>0.9);  
    decision=[setdiff(indexc1,intersect(indexc1,indexc2)) indexc2];
    Decision=(1:Global.D);
    Decision=setdiff(Decision,decision);
    dec=zeros(Global.N,length(Decision));
    tec=zeros(Global.N,length(decision));
    Dec=zeros(Global.N,Global.D);
     
    for i=1:length(predition)
        if predition(1,i)>=0.65
              predition(1,i)=1;
        else
              predition(1,i)=0;
        end      
    end
    
    
    predition=predition(1,Decision);
    group=zeros(1,length(Decision));
    for i=1:length(Decision)
        index = find(Q_matrix(Decision(i),:)==1);
        [~,z]=min(stu_state(index));
        group(i)= index(z);
    end

    Q = Q_matrix(Decision,:); 
 
     
    fitness=zeros(1,length(Decision));
    for i = 1 : Global.N
        dec(i,TournamentSelection(2,round(rand*length(Decision)),fitness)) = 1;
    end

   
         
    
    Dec(:,Decision)=dec;
    Dec(:,decision)=tec;
    Population = INDIVIDUAL(Dec);
    [Population,dec,tec,Fitness] = EnvironmentalSelection1(Population,dec,tec,Global.N);



     
     upper=zeros(1,size(Q_matrix,2));
     for i=1:size(Q_matrix,2)
         upper(i)=min(num,sum(Q_matrix(:,i)));

     end
     
     lower=zeros(1,size(Q_matrix,2));
     
     assistPop=zeros(Global.N,size(Q_matrix,2));
     U=rand(1,Global.N);
     for i=1:Global.N
         count=round(num+rand*num/2);
         weight=0.5+U(i)*0.5;
         number1=ceil(weight*count/round(size(Q_matrix,2)/2));
         number2=ceil((count-weight*count)/(size(Q_matrix,2)-round(size(Q_matrix,2)/2)));
         [~,index]=sort(stu_state);
         assistPop(i,index(1:round(size(Q_matrix,2)/2)))=number1;
         assistPop(i,index(round(size(Q_matrix,2)/2)+1:end))=number2;
         
     end
     
%      assistPop(Global.N/2+1:end,:)= round(rand(Global.N/2,size(Q_matrix,2)).*upper);
     [assistFitness,~,~]=CalFitness2(assistPop,stu_state,knowledge_log,num);
     assist=[];
     P=1;
     g=6;

    %% Optimization
    while Global.NotTermination(Population)

         MatingPool1       = TournamentSelection(2,2*Global.N,assistFitness);
         assistOffspring        = myGA(assistPop(MatingPool1,:),upper,lower); 
         [assistPop,assistFitness,~] = EnvironmentalSelection2([assistPop;assistOffspring],Global.N,stu_state,knowledge_log,num); 
         assist=[assist;assistPop];
         [Fit,~,~]=CalFitness2(assist,stu_state,knowledge_log,num);
         x=Fit<1;
         assist=assist(Fit<1,:);
         if(size(assist,1)>100)
             Fit=Fit(x);
             [~,rank]=sort(Fit);
             assist=assist(rank<=100,:);
         end

         MatingPool2       = TournamentSelection(2,2*Global.N,Fitness); 
         [Offdec,Offtec] = Operator(dec(MatingPool2,:),tec(MatingPool2,:),fitness,assist,group,P,Q);
         %% local search
         for i=1:size(Offdec,1)
             if sum(Offdec(i,:))>num
                 index=find(Offdec(i,:)==1);
                 randSample=randperm(length(index));
                 Offdec(i,index(randSample(1:sum(Offdec(i,:))-num)))=0;   
             elseif sum(Offdec(i,:))<num
                 index=find(Offdec(i,:)~=1);
                 randSample=randperm(length(index));
                 Offdec(i,index(randSample(1:num-sum(Offdec(i,:)))))=1;
             end
             rightNum=sum(Offdec(i,:).*predition);
             if rightNum<0.6*num
                 upp=0.8*num-rightNum;
                 low=0.6*num-rightNum;
                 percent=randi([low,upp]);
                 index1=find(Offdec(i,:)==0&predition==1);
                 index2=find(Offdec(i,:)==1&predition==0);
                 percent=min(percent,min(length(index1),length(index2)));
                 randSample1=randperm(length(index1)); 
                 randSample2=randperm(length(index2));
                 Offdec(i,index1(randSample1(1:percent)))=1;
                 Offdec(i,index2(randSample2(1:percent)))=0;  
             elseif rightNum>0.8*num 
                 upp=rightNum-0.6*num;
                 low=rightNum-0.8*num;                
                 percent=randi([low,upp]);
                 index1=find(Offdec(i,:)==1&predition==0);
                 index2=find(Offdec(i,:)==0&predition==1);
                 percent=min(percent,min(length(index1),length(index2)));
                 randSample1=randperm(length(index1)); 
                 randSample2=randperm(length(index2));                
                 Offdec(i,index1(randSample1(1:percent)))=0;
                 Offdec(i,index2(randSample2(1:percent)))=1;                    
             end  
         end
         
         Dec=zeros(Global.N,Global.D);
         Dec(:,Decision)=Offdec;
         Dec(:,decision)=Offtec;
         Offspring = INDIVIDUAL(Dec); 
         [Population,dec,tec,Fitness] = EnvironmentalSelection1([Population,Offspring],[dec;Offdec],[tec;Offtec],Global.N); 
        
%          P=max(length(find(SS==1))/total,0.1);
%          disp(P);
         P=0.5*(1-cos((1-Global.gen/Global.maxgen)*pi));

%          if g==6
%             [hv,~]=HV(Population.objs,[1 1 1]);
%              g=1;
%              disp(hv);
%          end
         g=g+1;
         if Global.gen==Global.maxgen
             [hv,~]=HV(Population.objs,[1 1 1]);
             disp(hv);
             
         end

         
         
         



%            if Global.gen==Global.maxgen
%                 number=zeros(size(Q_matrix,2),1);
%                 a=Population.objs;
%                 a=a(:,1);
%                 a=find(a==min(a));
%                 res=Population(a).decs;
%                 disp(Population(a).objs);
%                 disp('1111111111111111111111111111111111');
%                 [~,index]=sort(real_emb);
%                 for i=1:size(Q_matrix,2)
%                     number(i,1)=res*Q_matrix(:,index(i));
%                 end
%                 disp(number);
% 
%                 Q = res*Q_matrix;
%                 A = corr(real_emb',Q','type', 'Pearson');
%                 B = corr(real_emb',Q','type', 'Spearman');
%                 C = corr(real_emb',Q','type', 'Kendall');
%                 disp(A);
%                 disp(B);
%                 disp(C);   
%                 
%            end

%            if Global.gen==Global.maxgen
%                 a=Population.objs;
%                 a=a(:,1);
%                 a=find(a==min(a));
%                 [~,index]=sort(stu_state);
%                 for i=1:size(Q_matrix,2)
%                     number(i,1)=res*Q_matrix(:,index(i));
%                 end
%               disp(Population(a).objs);
%            end

%          if Global.gen==Global.maxgen
%            [hv,~]=HV(Population.objs,[1 1 1]);
%            disp(hv);
%             disp("11111111111111111"); 
%             x=Population.objs;
%             y=Population.cons;
%            Accuracy=0.5*(max(-1*x(:,1),0))+0.5*(30-max(y(:,6),0))/30;
%             disp(mean(Accuracy));
%            disp("11111111111111111");
%            Diversity=0.5*(abs(x(:,2)))+0.5*(1-x(:,3));
%            disp(mean(Diversity));
%            disp("11111111111111111");
%          end
          


              
              
              
              
              
              
          

        
         
            %³åÍ»ÐÔ·ÖÎö
%              x=Population.objs;
%              a=-x(:,1);
%              b=x(:,3);
%              c=[mean(a) mean(b)];
%              disp(c);
       
    end
end

