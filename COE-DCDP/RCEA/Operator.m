function [Offdec,Offtec] = Operator(dec,tec,fitness,assist,group,P,Q)
% The operator of RCEA

%------------------------------- Copyright --------------------------------
% Copyright (c) 2018-2019 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
%     Pcro=0.5;
    [N,~]       = size(dec);
    Parent1dec = dec(1:N/2,:);
    Parent2dec = dec(N/2+1:end,:);
    
    Offdec = Parent1dec;
    Offtec = tec(1:N/2,:);
    Offtec(:,:) = 0;
    order=zeros(1,N/2);
    
    
    for i=1:N/2
        if rand<P
            randSample=randperm(size(assist,1));
            order(i)=randSample(1);
            guide=assist(order(i),:);
            for j=1:size(assist,2)
                  Index=find(Q(:,j)==1)';
                if sum(Offdec(i,Index))>guide(j)
%                     index=find(Parent1dec(i,Index)&~Parent2dec(i,Index));
%                     croNum=min(length(index),sum(Offdec(i,Index))-guide(j));
%                     Offdec(i,Index(index(randperm(croNum))))=0; 
                      index=find(Parent1dec(i,Index)~=Parent2dec(i,Index));
                      index0=find(~Parent1dec(i,Index)&Parent2dec(i,Index));
                      index1=find(Parent1dec(i,Index)&~Parent2dec(i,Index));
                      p0=max(length(index)-(sum(Offdec(i,Index))-guide(j)),0)/(2*length(index0));
                      p1=(length(index)+sum(Offdec(i,Index))-guide(j))/(2*length(index1));
                      for z=1:length(index0)
                          if rand<p0
                              Offdec(i,Index(index0(z)))=1;
                          end
                      end
                      for z=1:length(index1)
                          if rand<p1
                              Offdec(i,Index(index1(z)))=0;
                          end
                      end      
                     
                elseif sum(Offdec(i,Index))<guide(j)                
%                     index=find(~Parent1dec(i,Index)&Parent2dec(i,Index));
%                     croNum=min(length(index),guide(j)-sum(Offdec(i,Index)));
%                     Offdec(i,Index(index(randperm(croNum))))=1;
                      index=find(Parent1dec(i,Index)~=Parent2dec(i,Index));
                      index0=find(~Parent1dec(i,Index)&Parent2dec(i,Index));
                      index1=find(Parent1dec(i,Index)&~Parent2dec(i,Index));
                      p0=(length(index)+guide(j)-sum(Offdec(i,Index)))/(2*length(index0));
                      p1=max(0,length(index)-(guide(j)-sum(Offdec(i,Index))))/(2*length(index1));
                      for z=1:length(index0)
                          if rand<p0
                              Offdec(i,Index(index0(z)))=1;
                          end
                      end
                      for z=1:length(index1)
                          if rand<p1
                              Offdec(i,Index(index1(z)))=0;
                          end
                      end                      
                end 
            end        

            for j=1:size(assist,2)
                Index=find(Q(:,j)==1)';                
                if sum(Offdec(i,Index))>guide(j)
                    index=find(Offdec(i,Index)==1);
                      Offdec(i,Index(index(randperm(min(sum(Offdec(i,Index))-guide(j),length(index))))))=0;
%                        index=find(Offdec(i,Index)==1);                        
%                        p1=(sum(Offdec(i,Index))-guide(j))/length(index);
%                        for z=1:length(index)
%                           if rand<p1
%                               Offdec(i,Index(index(z)))=0;
%                           end
%                        end 

                elseif sum(Offdec(i,Index))<guide(j) 
                    index=find(Offdec(i,Index)==1);
                    Offdec(i,Index(index(randperm(min(guide(j)-sum(Offdec(i,Index)),length(index))))))=1;
%                        index=find(Offdec(i,Index)==0);                        
%                        p0=(guide(j)-sum(Offdec(i,Index)))/length(index);
%                        for z=1:length(index)
%                           if rand<p0
%                               Offdec(i,Index(index(z)))=1;
%                           end
%                        end                  
                    
                    
                    
                end
            end
            
        else
            if rand < 0.5
                index = find(Parent1dec(i,:)&~Parent2dec(i,:));
                index = index(TS(-fitness(index)));
                Offdec(i,index) = 0;
            else
                index = find(~Parent1dec(i,:)&Parent2dec(i,:));
                index = index(TS(fitness(index)));
                Offdec(i,index) = Parent2dec(i,index);
            end
                
            if rand < 0.5
                index = find(Offdec(i,:));
                index = index(TS(-fitness(index)));
                Offdec(i,index) = 0;
            else
                index = find(~Offdec(i,:));
                index = index(TS(fitness(index)));
                Offdec(i,index) = 1;
            end
%               Offdec(i,:)=GAhalf([Parent1dec(i,:);Parent2dec(i,:)]);
           
%            croPos=round(rand*size(Parent1dec(i,:),2));
%            if croPos~=1||croPos~=size(Parent1dec(i,:),2)
%                Offdec(i,croPos:end)=Parent2dec(i,croPos:end);
%            end
%            mutPos=ceil(rand*size(Parent1dec(i,:),2));
%            if  Offdec(i,mutPos)==1
%                Offdec(i,mutPos)=0;
%            else
%                Offdec(i,mutPos)=1;
%            end

            
        end
    end
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
%           for i=1:N/2
%                 Index=[find(Parent1dec(i,:)&~Parent2dec(i,:)) find(~Parent1dec(i,:)&Parent2dec(i,:))];
%                 if sum(Parent1dec(i,:))>num
%                     index1=find(Parent1dec(i,:)&~Parent2dec(i,:));
%                     index0=find(~Parent1dec(i,:)&Parent2dec(i,:));               
%                     n0=ceil((Pcro*length(Index)-sum(Parent1dec(i,:))+ num)/2);
%                     n1=floor((Pcro*length(Index)+sum(Parent1dec(i,:))- num)/2);
% 
%                     x=randperm(length(index1));
%                     y=randperm(length(index0));
%                     if length(index1)>0
%                        Offdec(i,index1(x(1:min(n1,length(index1))))) = 0;                        
%                     end
%                     if length(index0)>0
%                         Offdec(i,index0(y(1:min(n0,length(index0))))) = 1;   
%                     end
%                 elseif sum(Parent1dec(i,:))<num   
%                     index1=find(Parent1dec(i,:)&~Parent2dec(i,:));
%                     index0=find(~Parent1dec(i,:)&Parent2dec(i,:));                   
%                     n0=ceil((Pcro*length(Index)-sum(Parent1dec(i,:))+ num)/2);
%                     n1=floor((Pcro*length(Index)+sum(Parent1dec(i,:))- num)/2);
% 
%                     x=randperm(length(index1));
%                     y=randperm(length(index0));
%                     if length(index1)>0
%                        Offdec(i,index1(x(1:min(n1,length(index1))))) = 0;
%                     end
%                     if length(index0)>0
%                         Offdec(i,index0(y(1:min(n0,length(index0))))) = 1;
%                     end
%                 else
%                     index1=find(Parent1dec(i,:)&~Parent2dec(i,:));
%                     index0=find(~Parent1dec(i,:)&Parent2dec(i,:));            
%                     
%                     n0=ceil(Pcro*length(Index)/2);
%                     n1=floor(Pcro*length(Index)/2);
% 
%                     x=randperm(length(index1));
%                     y=randperm(length(index0));
%                     if length(index1)>0
%                        Offdec(i,index1(x(1:n1))) = 0;
%                     end
%                     if length(index0)>0
%                         Offdec(i,index0(y(1:n0))) = 1;
%                     end
%                 end 
%           end

        
    
    

    %% Mutation for mask
       
    
    
    
    
    
    
    
    
    
    
    
    
    
    
%      for i=1:N/2
%          x=Offdec(i,:).*predition;
%          y=sum(Offdec(i,:));
%               if sum(x)<round(0.6*y)
%                  high=round(0.8*y)-sum(x);
%                  low=round(0.6*y)-sum(x);
%                  b=randi([low,high]);
%                  index1=find(Offdec(i,:)==0&predition==1);
%                  c=randperm(length(index1));
%                  Offdec(i,index1(c(1:b)))=1;
%                  index2=find(Offdec(i,:)==1&predition==0);
%                  d=randperm(length(index2));
%                  Offdec(i,index2(d(1:b)))=0;
%               elseif sum(x)>round(0.8*y)
%                  high=sum(x)-round(0.6*y);
%                  low=sum(x)-round(0.8*y);
%                  b=randi([low,high]);
%                  index1=find(Offdec(i,:)==1&predition==0);
%                  if b>length(index1)
%                      b=length(index1);
%                  end
%                  c=randperm(length(index1));
%                  Offdec(i,index1(c(1:b)))=0;
%                  index2=find(Offdec(i,:)==0&predition==1);
%                  d=randperm(length(index2));
%                  Offdec(i,index2(d(1:b)))=1; 
%               else
%                  if rand<0.5
%                       Index=find(predition==1);
%                       Index = Index(find(Offdec(i,Index)));
%                       Index = Index(TS(-fitness(Index)));
%                       Offdec(i,Index) = 0;
%                       Index = find(~Offdec(i,Index));
%                       Index = Index(TS(fitness(Index)));
%                       Offdec(i,Index) = 1;
%                  else
%                       Index=find(predition==0);
%                       Index = Index(find(Offdec(i,Index)));
%                       Index = Index(TS(-fitness(Index)));
%                       Offdec(i,Index) = 0;
%                       Index = find(~Offdec(i,Index));
%                       Index = Index(TS(fitness(Index)));
%                       Offdec(i,Index) = 1;
%                  end
%                   
%                   
%               end
%          
%      end

       
 



end


function index = TS(Fitness)
% Binary tournament selection

    if isempty(Fitness)
        index = [];
    else
       index = TournamentSelection(2,1,Fitness);
        
    end
end
