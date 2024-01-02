classdef Exercise_RC < PROBLEM
% <problem> <Exercise Recommendation>
% The exercise recommendation problem
% stuId ---  1 --- Id of student
% num   --- 100 --- number of recommended exercises
%------------------------------- Reference --------------------------------
% Y. Tian, C. Lu, X. Zhang, K. C. Tan, and Y. Jin, Solving large-scale
% multi-objective optimization problems with sparse optimal solutions via
% unsupervised neural networks, IEEE Transactions on Cybernetics, 2020.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2016-2017 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB Platform
% for Evolutionary Multi-Objective Optimization [Educational Forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------

% The datasets are taken from the Alex Arenas Website in
% https://sites.google.com/site/assistmentsdata/home/assistment-2009-2010-data/skill-builder-data-2009-2010
% No.   Students	Exercises   Knowledge Concepts
% 1      4163	      17746             123
      properties(Access = private)
        stu_state;
        exer_difficulty;
        Q_matrix;
        exer_log;
        knowledge_log;
        predition;
        num;
        pred;
      end
      methods
        %% Initialization
         function obj = Exercise_RC()
             % Load data
             [stuId,obj.num] = obj.Global.ParameterSet(1,100);
             str    = {'student_emb','k_difficulty','Q_matrix','exercise_log','knowledge_log','predition'};
             CallStack = dbstack('-completenames');
             load(fullfile(fileparts(CallStack(1).file),'dataSet2.mat'),'dataSet2');
             obj.stu_state = dataSet2.(str{1})(stuId,:);
             obj.exer_difficulty = dataSet2.(str{2});
             obj.Q_matrix = dataSet2.(str{3});
             obj.exer_log = dataSet2.(str{4})(stuId,:);
             obj.knowledge_log = dataSet2.(str{5})(stuId,:);
             obj.predition = dataSet2.(str{6})(stuId,:);
             obj.pred = obj.predition;
             for i=1:length(obj.pred)
                 if obj.pred(1,i)>=0.65
                     obj.pred(1,i)=1;
                 else
                     obj.pred(1,i)=0;
                 end      
             end
             
             % Parameter setting
             obj.Global.M        = 3;
             obj.Global.D        = length(obj.exer_difficulty);
             obj.Global.lower    = zeros(1,obj.Global.D);
             obj.Global.upper    = ones(1,obj.Global.D);
             obj.Global.encoding = 'binary';
         end
         function PopObj = CalObj(obj,PopDec)
            PopDec = logical(PopDec);
            PopObj = zeros(size(PopDec,1),obj.Global.M);
            for i = 1 : size(PopDec,1)
                Q = PopDec(i,:)*obj.Q_matrix;
                PopObj(i,1) = corr(obj.stu_state',Q','type','Pearson');
                Q(Q>=1) = 1;
                PopObj(i,2) = -(sum(Q) - sum(Q.*obj.knowledge_log))/(size(obj.stu_state,2)-sum(obj.knowledge_log));
                PopObj(i,3) = (size(obj.stu_state,2) - sum(Q))/size(obj.stu_state,2);
            end
         end
         
         
         function PopCon = CalCon(obj,PopDec)
             PopDec = logical(PopDec);
             PopCon = zeros(size(PopDec,1),8);
             for i=1 : size(PopDec,1)
                 A = PopDec(i,:).*obj.predition;
                 B = PopDec(i,:).* obj.pred;
                 PopCon(i,1) = 0.6*obj.num - sum(B);
                 PopCon(i,2) = sum(B) - 0.8*obj.num;
                 PopCon(i,3) = obj.num -sum(PopDec(i,:));
                 PopCon(i,4) = sum(PopDec(i,:))-obj.num;
                 PopCon(i,5) = length(find(A>=0.2 & A<=0.9))-sum(PopDec(i,:));
                 PopCon(i,6) = sum(PopDec(i,:)) - length(find(A>=0.2 & A<=0.9));
                 PopCon(i,7) = sum(obj.exer_log.*PopDec(i,:))-0;
                 PopCon(i,8) = 0 - sum(obj.exer_log.*PopDec(i,:));             
             end
             
         end
                  
      end
    
            
end