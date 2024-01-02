classdef RBF<handle
    properties(SetAccess = private)
       NumSample=100;      %
       Sample=[];
       HidNum   = 10;     % 
       InDim=10;
       OutNum=1;
       Output=[];
       centers=[];
       gap=10;
       maxIteration=50;
       
       Delta=[];
       lamda=1;
       Green=[];
       method='Gaussian';
       ID=[];
       Weight=[];
       
       
        
    end
    
    methods
        %% ���캯��
        function obj=RBF(Sample,Output,NumSample,HidNum,InDim,OutNum,gap,method)
            obj.NumSample=NumSample;      %
            obj.Sample=Sample;
            obj.Output=Output;
            obj.HidNum   = HidNum;     %
            obj.InDim=InDim;
            obj.OutNum=OutNum; 
            obj.gap=gap;            %�ж�k-prototypeֹͣ
            obj.centers=cell(1,obj.HidNum);
            obj.method=method;
            obj.Green=zeros(obj.NumSample,obj.HidNum );
        end
        
        %% k-prototype
        function k_prototype(obj)
            
            AA=randperm(obj.NumSample);
            Index_center=AA(1:obj.HidNum);
            for i=1:obj.HidNum
                obj.centers{i}=obj.Sample(Index_center(i),:);
            end
            iDist=obj.gap+1;
            j=1;
            while(iDist>obj.gap && j<obj.maxIteration)
                % �������
                [index,~]=obj.Hami_Distance();
                %�����µ�����
                for i=1:obj.HidNum
                    center{i}=mode(obj.Sample(find(index==i),:),1);%����ȡ����
                    Dist(i,1)= sum(obj.centers{i}~=center{i});
                    obj.centers{i}=center{i};
%                     index=obj.Hami_Distance();
                end
                iDist=sum(Dist);
                j=j+1;
            end
                      
            
        end
       
        %% ��������
        function [index,minDist,ID]=Hami_Distance(obj)
            for i=1:obj.HidNum
                ID(:,i)=sum(obj.Sample~=repmat(obj.centers{i},obj.NumSample,1),2);
            end
            [minDist,index]=min(ID,[],2);
        end
        
        
        %% ����Delta
        function CalculatedDelta(obj)
            for i=1:obj.HidNum
                center(i,:)=obj.centers{i};
            end
          
                for i=1:obj.HidNum
                    
                ID1(:,i)=sum(center()~=repmat(obj.centers{i},obj.HidNum,1),2);
                ID1(i,i)=inf;
                end
            
            d=min(ID1,[],2);
            obj.Delta=obj.lamda*d;
            
        end
        %% ����Green����
        
        function CalculateGreen(obj)
            
            [~,~,ID1]=obj.Hami_Distance();
            obj.ID=ID1;
            
            switch obj.method
                case 'Gaussian' 
                    for i=1:length(obj.Delta)
                        
                    obj.Green(:,i)=exp(-1/(2*(obj.Delta(i).^2)).*(obj.ID(:,i).^2));
                    end
            end
                
        end
        %% ����ȨֵW
        function  CalculateWeight(obj)
            obj.Weight=(obj.Green'*obj.Green)^(-1)*obj.Green'*obj.Output;            
        end
        
        function predict=Predict(obj,Test,TestNum)
            
            
             for i=1:obj.HidNum
                ID1(:,i)=sum(Test~=repmat(obj.centers{i},TestNum,1),2);
            end
            
            
            switch obj.method
                case 'Gaussian' 
                    for i=1:length(obj.Delta)
                        
                    Green1(:,i)=exp(-1/(2*(obj.Delta(i).^2)).*(ID1(:,i).^2));
                    end
            end
            
            predict=Green1*obj.Weight;
             
        end
        
        function run(obj)
           
            obj.k_prototype();
            
            obj.CalculatedDelta();
            obj.CalculateGreen();
            obj.CalculateWeight();
        end
        
        
    end
    
end