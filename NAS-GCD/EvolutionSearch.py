
import torch.backends.cudnn as cudnn
import torch, random,time,os,logging
import numpy as np
import matplotlib.pyplot as plt

from utils.config import get_common_search_config
from utils.utils import get_dataset,write_txt
from utils.Evaluation_Model import solution_evaluation
from Tree.Genetic_Operator import  Generate,Generate_crossover_mutation, Generate_from_existing
from genotypes import Genotype_mapping
from EMO_public import F_distance,NDsort,F_mating,F_EnvironmentSelect
from Tree.Node import Tree
from Tree.Node import Node as TreeNode
import sys,gc,pickle
import math 

from copy import deepcopy

from threading import Thread
class MyThread(Thread):
    def __init__(self, func, args):
        super(MyThread, self).__init__()
        self.func = func
        self.args = args

    def run(self):
        self.result = self.func(*self.args)

    def get_result(self):
        try:
            return self.result
        except Exception:
            return None

class Individual():
    def __init__(self, Dec=None, num_Nodes=5,mapping=None,config=None, gen=0,id=0):
        self.config = config
        self.id = id
        self.gen = gen
        self.pre_inputs=['Stu','Exer','Conc']

        self.mapping =mapping
        self.Reverse_mapping = dict([val, key] for key, val in self.mapping.items())


        if Dec is None:
            self.Dec = Dec
            self.numNodes = num_Nodes
            self.RandomBuildTree()
        else:
            self.numNodes = len(Dec)//3
            self.build_treeFromDec(Dec)
            self.UpdateShape(self.tree.root)
            self.RepairConstraint(self.tree.root)
            self.getNumNode()
        self.Get_DecArrary()#给节点编号
        # a=1
    def Deletion(self,index):  # the index is not same as No. (index is based on level travel, No. based on Post-Order travel)

        subtree = self.get_subTree(index)
        if subtree.left is not None and subtree.left.item not in self.pre_inputs: # select the left or right
            if np.random.rand()<0.5:
                used_tree = subtree.right
            else:
                used_tree = subtree.left
        else:
            used_tree = subtree.right
        self.set_subTree(subtree.No,used_tree)
        self.After_Genetic()

    def Insertion(self,index): # Insert a node as the parent pf the Node(index)
        subtree = self.get_subTree(index)
        #------- -------------generate the Insert_node --------------
        if index ==1 and np.random.rand()<0.7:  # big probability for binary operator？原0.7
                randi = np.random.randint(0,2)
                if randi ==0:
                    op ='add'
                elif randi==1:
                    op = 'mul'
                else:
                    op = 'concat'
        else:
                op = np.random.randint(0,len(self.mapping))
                op = self.mapping[op]

        Insert_node = TreeNode(item=op)

        #-----------------Set the Insert_node ---------------------
        # here we donot consider the feasiability  in terms of "shape", will be done by Repairing  -------

        if Insert_node.item in ['add','mul','concat']:
            Insert_node.right = subtree
            #--------- set the right
            candidate = np.random.randint(0,len(self.pre_inputs))
            Insert_node.left = TreeNode(item=self.pre_inputs[candidate],shape='same')
        else:
            Insert_node.right = subtree

        #-----------
        self.set_subTree(subtree.No,Insert_node)
        self.After_Genetic()

    def Replacement(self,index):
        subtree = self.get_subTree(index)

        #-------------generate the Replace_node --------------
        op = subtree.item
        while op == subtree.item:
            op = np.random.randint(0,len(self.mapping))
            op = self.mapping[op]
        Replace_node = TreeNode(item=op)
        #---------------Set the Replace_node, here we donot consider the feasiability  in terms of "shape", will be done by Repairing  --------------
        if subtree.item in ['add','mul','concat'] and Replace_node.item in ['add','mul','concat']:
            Replace_node.left = subtree.left
            Replace_node.right = subtree.right
        elif subtree.item in ['add','mul','concat'] and Replace_node.item not in ['add','mul','concat']:
            if np.random.rand()<0.5:
                Replace_node.right = subtree.right
            else:
                Replace_node.right = subtree.left
        elif subtree.item not in ['add','mul','concat'] and Replace_node.item in ['add','mul','concat']:
            Replace_node.right = subtree.right
            # randomly adding a input as child node
            candidate = np.random.randint(0,len(self.pre_inputs))
            Replace_node.left = TreeNode(item=self.pre_inputs[candidate],shape='same')
        else:
            Replace_node.right = subtree.right
        #---------------------
        self.set_subTree(subtree.No,Replace_node)
        self.After_Genetic()

    def After_Genetic(self):
        self.BTS()
        self.UpdateShape(self.tree.root)
        self.RepairConstraint(self.tree.root)
        self.getNumNode()
        self.Get_DecArrary()
    
    def BTS(self):
        #后序遍历，重新计算每个节点的nodesum
        s1 = []
        s2 = []
        s1.append(self.tree.root)  # post order travel by two stacks
        while len(s1)>0:
            cur = s1.pop()
            s2.append(cur)
            if cur.left is not None and cur.left.item not in self.pre_inputs:
                s1.append(cur.left)
            if cur.right is not None and cur.right.item not in self.pre_inputs:
                s1.append(cur.right)
        for idx,node_i in enumerate(s2[::-1]): #先调整编号方便交换子树
            node_i.No = idx+3
        for node_i in s2[::-1]:
            if node_i.left is not None and node_i.right is not None:
                if  node_i.left.nodesum>node_i.right.nodesum:
                    t=node_i.left
                    node_i.left=node_i.right
                    node_i.right=t
                node_i.nodesum = node_i.left.nodesum+node_i.right.nodesum+self.Reverse_mapping[node_i.item]

            if node_i.left is not None and node_i.right is None:
                node_i.nodesum = node_i.left.nodesum+self.Reverse_mapping[node_i.item]
                node_i.right=node_i.left
                node_i.left=None
            if node_i.left is  None and node_i.right is  not None:
                 node_i.nodesum = node_i.right.nodesum+self.Reverse_mapping[node_i.item]
            else:
               continue
            s2[-1-idx]=node_i
        self.tree.root=s2[0]
         
        

    def Get_DecArrary(self): # return int arrary for building NASCDNet, Post-Order travel
        # post order travel by two stacks
        s2 = []
        s1=[self.tree.root]
        res = [] 
        Dec_postorder=[]
        while s1:
            cur = s1.pop()
            if cur:
                s1.append(cur.left)
                s1.append(cur.right)
                if cur.item not in self.pre_inputs:
                    s2.append(cur)
                res.append(cur)
        for node_i in res:
            Dec_postorder.append(node_i.item)
            
        self.Dec_postorder=Dec_postorder.reverse()
 
        Dec = []
        candidate_inputs = deepcopy(self.pre_inputs)    
        for idx,node_i in enumerate(s2[::-1]):
            node_i.No = idx+3  
            if node_i.right.item == 'Stu':
                x1=0
            elif node_i.right.item == 'Exer':
                x1=1
            elif node_i.right.item == 'Conc':
                x1=2
            else:
                x1 = node_i.right.No
            if node_i.item in ['add','mul','concat']:
                if node_i.left.item == 'Stu':
                    x2=0
                    x1,x2 = x2,x1  # 0, [0,1,2]
                elif node_i.left.item == 'Exer':   # exchange for unique encoding
                    x2=1
                    if x1>x2:
                        x1,x2 = x2,x1  # 1,2
                                       # else:[0,1],1
                elif node_i.left.item == 'Conc':
                    x2=2   # [0,1,2],2
                else:
                    x2 = node_i.left.No 
            else:
                x2 = 0
            candidate_inputs.append(node_i.No)
            op_num = self.Reverse_mapping[node_i.item]
            Dec.extend([x1,x2,op_num])
        self.Dec = Dec
        # inorder traversal
                
        s3=[]
        s4=[]
        cur=self.tree.root
        # s3.append(self.tree.root)
        while cur or s3:
            while cur:
                s3.append(cur)
                cur=cur.left
            cur=s3.pop()
            s4.append(cur)
            cur=cur.right    
        Dec_inorder = []
        for node_i in s4:
            Dec_inorder.append(node_i.item)
            
        self.Dec_inorder=Dec_inorder

    def set_subTree(self,Tree_No,another_subTree): # set tree according to the No., which is based on Post-Order

        if self.tree.root.No ==Tree_No:
            self.tree.root = another_subTree
            return

        Queue = [self.tree.root]
        while len(Queue)>0:
            cur = Queue.pop(0)
            if cur.right!=None:
                if cur.right.No ==Tree_No:
                    cur.right= another_subTree
                    return
                else:
                    Queue.append(cur.right)
            if cur.left!=None:
                if cur.left.No ==Tree_No:
                    cur.left = another_subTree
                    return
                else:
                    Queue.append(cur.left)


    def get_subTree(self, index): # counting from root node to maxi: level travel
        subtree = []
        Queue = [self.tree.root]
        while index>0:
            cur = Queue.pop(0)
            if cur.item in self.mapping.values():
                index -=1
                subtree = cur
            if cur.right!=None:
                Queue.append(cur.right)
            if cur.left!=None:
                Queue.append(cur.left)

        return subtree


    def getNumNode(self):
        num = 0
        Queue = [self.tree.root]
        while len(Queue)!=0:
            cur = Queue.pop(0)

            if cur.item in self.mapping.values():
                num +=1
            if cur.right!=None:
                Queue.append(cur.right)
            if cur.left!=None:
                Queue.append(cur.left)
        self.numNodes = num#计算节点数
        return self.numNodes
    def getLeafNum(self):
        num = 0
        Queue = [self.tree.root]
        while len(Queue)!=0:
            cur = Queue.pop(0)
            if cur.item in self.pre_inputs:
                num +=1
            if cur.right!=None:
                Queue.append(cur.right)
            if cur.left!=None:
                Queue.append(cur.left)
            
        self.leafNum = num
        return self.leafNum



    def tree_deep(self,node): # include root node and leaf node递归计算树的深度
        if node is None:
            return 0
        left, right = 0,0
        if node.right is not None:
            right = self.tree_deep(node.right)
        if node.left is not None:
            left = self.tree_deep(node.left)
       
        return max(left,right)+1

    def RandomBuildTree(self):
        tree = Tree()


        # 随机采样
        tree.sample(self.mapping,self.numNodes)


        self.tree = tree
        self.AddLeafNode(self.tree.root)

        # basic steps after a solution is generated
        self.UpdateShape(self.tree.root)
        self.RepairConstraint(self.tree.root)
        self.getNumNode()
        #------------------
        # abc = self.get_subTree(2)
        a = 1


    def RepairConstraint(self,node):
        if node is None:
            return
        elif node.item in self.pre_inputs:
            return

        self.RepairConstraint(node.left)
        self.RepairConstraint(node.right)

        if node.right.shape =='single' and node.item in ['mean','sum','ffn','concat','ffn_d']:   # 修复  mean 后续不能再直接 follow mean等操作
            op = node.item
            while op  in ['mean','sum','ffn','concat','ffn_d']:
                op = np.random.randint(0,len(self.mapping))
                op = self.mapping[op]
            if node.item =='concat' and op in  ['add','mul']: # for concat, directly used binary operator for replacement
                node.item = op
            elif op in ['add','mul']: #for ['mean','sum','ffn'],  binary operator
                node.item = op
                # adding right child
                candidate = np.random.randint(0,len(self.pre_inputs))
                node.left = TreeNode(item=self.pre_inputs[candidate],shape='same')
                node.shape = 'same'
            else: # unary operator
                node.item = op
                node.left = None # used for 'concat'
        #------------------------- 修复concat-----------
        if node.item=='concat' and node.left.shape!=node.right.shape:
            if np.random.rand()<0.5:
                node.item = 'add'
            else:
                node.item = 'mul'


        #----------------------------------------------
        if node.item not in ['add','mul','concat']: # 修复  连续相同的 （unary）操作
            if node.item == node.right.item:
                node.right = node.right.right

        #------------------ update shape information ----------------------------------
        if node.item in ['add','mul','concat']:
            if node.left.shape==node.right.shape:
                node.shape = node.right.shape
            else:
                node.shape ='same'

        elif node.item in ['sum','mean','ffn','ffn_d']:
            node.shape = 'single'
        else:
            node.shape = node.right.shape


    def AddLeafNode(self,node):
        if node is None:
            return
        elif node.item in self.pre_inputs:
            return

        if node.left is None and node.right is not None and node.item in ['add','mul','concat']:
            candidate = np.random.randint(0,len(self.pre_inputs))
            node.left = TreeNode(item=self.pre_inputs[candidate],shape='same')

        if node.right ==None and node.left==None:
            if node.item not in ['add','mul','concat']:
                candidate = np.random.randint(0,len(self.pre_inputs)-1)    # only select from stu and Exer to avoid mistakes
                   # only select from stu and Exer to avoid mistakes
                node.right = TreeNode(item=self.pre_inputs[candidate],shape='same')
            else:
                # candidate = np.random.randint(0,len(self.pre_inputs)-1,2)  # only select from stu and Exer to avoid mistakes
                candidate = np.random.choice(range(len(self.pre_inputs)),2,replace=False) # avoid same inputs
                node.right = TreeNode(item=self.pre_inputs[candidate[0]],shape='same')
                node.left = TreeNode(item=self.pre_inputs[candidate[1]],shape='same')
        self.AddLeafNode(node.right)
        self.AddLeafNode(node.left)

    def UpdateShape(self,node):
        if node is None:
            return
        elif node.item in self.pre_inputs:
            return

        self.UpdateShape(node.left)
        self.UpdateShape(node.right)

        if node.item in ['add','mul','concat']:
            if node.left.shape==node.right.shape:
                node.shape = node.right.shape
            else:
                node.shape ='same'

        elif node.item in ['sum','mean','ffn','ffn_d']:
            node.shape = 'single'
        else:
            node.shape = node.right.shape

    def build_treeFromDec(self,Dec):
        self.Dec = Dec#如果两个节点编号相同，节点深度相同一定是同一节点
        nodes = [TreeNode('Stu',shape='same',nodesum=0.001),TreeNode('Exer',shape='same',nodesum=0.01),TreeNode('Conc',shape='same',nodesum=0.1)]
        self.numNodes = len(Dec)//3
        for i in range(self.numNodes):
            temp = Dec[3*i:3*(i+1)]#一个子树最多3个节点
            x1,x2,op = temp[0],temp[1],temp[2]#得到三个节点应该放的操作
            if self.mapping[op] in ['add','mul','concat']:
                if nodes[x2].nodesum<nodes[x1].nodesum:
                    node_i=TreeNode(item=self.mapping[op], left=nodes[x2], right=nodes[x1],nodesum=nodes[x1].nodesum+nodes[x2].nodesum+op+x1+x2)          
                elif nodes[x2].nodesum>=nodes[x1].nodesum:
                    node_i=TreeNode(item=self.mapping[op], left=nodes[x1], right=nodes[x2],nodesum=nodes[x1].nodesum+x1+x2+nodes[x2].nodesum+op)    
            else:#单元操作只计算x1
                node_i = TreeNode(item=self.mapping[op], left=None, right=nodes[x1], nodesum=nodes[x1].nodesum+op)
            nodes.append(node_i)

        self.tree = Tree(root=nodes[-1])
    def visualization(self,path=None):
        self.tree.visualization(path)



    def mkdir(self):

        self.save_dir = "{}/Gen_{}/[{}]/".format(self.config.exp_name,self.gen,self.id)
        self.training_log = self.save_dir+'training_log.txt'

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def save(self):
        # saving basic information
        self.dec_dir = self.save_dir+'dec.txt'
        self.fitness_dit = self.save_dir+'fitness.txt'
        self.user_dict=self.save_dir+'user_feature.txt'
        # information = 'Deep num:{}, LeafNode num:{}, Node num:{}'.format(self.deep_number,self.leafNum,self.numNodes)
        self.info_dir = self.save_dir+'infomation.txt'
        #存信息及可视化
        # write_txt(self.info_dir,information)
        write_txt(self.dec_dir,self.Dec)
        self.visualization(self.save_dir)


    def evaluation(self,device,user_feature):
        self.fitness=[]
        self.user_feature=[]
        self.mkdir()
        f = open(self.training_log, "w+")
        print('Evaluating {}-th solution'.format(self.id), file=f,flush=True)
        print('Evaluating {}-th solution'.format(self.id), file=sys.stdout)
        logging.info('Evaluating {}-th solution'.format(self.id))
        self.save()
        all_best_auc=[]
        Settings = [device,self.config,self.Dec ,self.save_dir,f]
        for i in self.config.dataset:
            best_acc,best_auc = solution_evaluation(Settings,i)
            print('{}-th solution: {}_best valid acc:{}, auc:{}  '.format(self.id,i,best_acc, best_auc),file=sys.stdout)
            logging.info('{}-th solution: {}_best valid acc:{}, auc:{}'
                     .format(self.id,i,best_acc, best_auc))
            all_best_auc.append(1-best_auc)
        mean_auc=np.mean(all_best_auc)
        std_auc=1-np.std(all_best_auc)
        self.fitness.append(mean_auc)
        self.fitness.append(std_auc)    
        logging.info('{}-th solution: Best result:{}'.format(self.id,self.fitness))



        np.savetxt( self.fitness_dit, np.array(self.fitness), delimiter=' ')
        if user_feature:
            np.savetxt( self.user_dict, np.array(self.user_feature), delimiter=' ')
        gc.collect()
        f.close()



class EvolutionAlgorithm():
    def __init__(self,config):
        self.config = config


        if config.dataset=='assistment2009':
            self.threshold = 0.75
        elif config.dataset=='slp':
            self.threshold = 0.82
        elif config.dataset=='junyi':
            self.threshold = 0.8
        else:
            self.threshold = 0.78

        self.Maxi_Gen = 100#100,40
        self.gen =0
        self.Popsize = 100#100,50
        #--------Population and offspring information-------------
        self.Population = []
        self.Pop_fitness = []

        self.offspring = []
        self.off_fitness=[]
        #-------other information--------------------
        self.tour_index = []
        self.FrontValue = []
        self.CrowdDistance =[]
        self.select_index = []
        self.Archive = []

        # self.LoadDataset()
        self.get_Boundary_Mapping()

    def LoadDataset(self):
        print('Loading Dataset....')
        self.config.student_n,self.config.exer_n,self.config.knowledge_n,\
        self.train_loader, self.val_loader = get_dataset(self.config)
        print('Loading Finish！')
    def get_Boundary_Mapping(self):
        self.mapping = Genotype_mapping
        logging.info('Genotype_mapping: '+str(self.mapping))
        print('Genotype_mapping: '+str(self.mapping))

    def Initialization(self):
        if config.Continue_path is None:
            self.set_dir(path='initial')
            self.Population=[]
            for idx in range(0,self.Popsize):
                num_nodes = np.random.randint(config.Num_Nodes[0],config.Num_Nodes[1]+1) # +1
                self.Population.append(Individual(num_Nodes=num_nodes,mapping=self.mapping,config=self.config,gen='initial',id=idx))
            self.Pop_fitness = self.Evaluation(self.Population)
            self.set_dir(path='initial')
            self.Save()
        else:
            pathdir = os.path.expandvars(config.Continue_path)[-4]
            curdir = os.path.expandvars(config.Continue_path)[-3]

            self.gen = int(curdir[-1])
            self.Population = pickle.load(open(config.Continue_path+'/'+pathdir+'/Population.pkl'),'rb')
            self.Pop_fitness = np.loadtxt(config.Continue_path+'/'+pathdir+'/fitness.txt')

        for x_individual in self.Population:
            self.Archive.append(x_individual.Dec)
    
    def Initialization_from_existing(self):

        self.set_dir(path='initial')
        self.Population = []
        self.Population.append(Individual(Dec=[1,0,9, 3,0,0, 0,0,9,  4,5,10, 1,0,9, 6,7,11, 8,0,6 ],
                                          mapping=self.mapping,config=self.config,gen='initial',id=0) ) # IRT
        self.Population.append(Individual(Dec=[0,1,12, 3,0,9, 4,0,6],mapping=self.mapping,
                                          config=self.config,gen='initial',id=2) ) # MCD
        self.Population.append(Individual(Dec=[1,0,6,3,0,0,0,0,6,4,5,10,2,6,11,1,0,9,8,0,6,7,9,11],
                                      mapping=self.mapping,config=self.config,gen='initial',id=3) ) # NCD
        self.Population.append(Individual(Dec=[0,2,12,   3,0,6,   1,2,12,   5,0,6, 6,0,0,  4,7,10,  8,0,14,   9,0,6,  10,0,13],
                                        mapping=self.mapping,config=self.config,gen='initial',id=4))#RCD
        self.Decinorder=[]
        self.Decpostorder=[]
        
        for i in self.Population:
            self.Decinorder.append(i.Dec_inorder)
            self.Decpostorder.append(i.Dec_postorder)
        self.repeat=0
        while len(self.Population)< self.Popsize:
            for idx in range(4, self.Popsize,4):#已知结构增删换后产生子代。
                offspring_temp = Generate_from_existing(self.Population[:4],gen='initial')#不断用已知结构做父代产生子代
                if len(self.Population)>self.Popsize:
                            break
                #需要判断产生的子代是否已经产生过
                for i in  offspring_temp[::-1]:
                    # inpostorder=[]
                    # inpostorder=(i.Dec)
                    if i.Dec_inorder in  self.Decinorder and i.Dec_postorder in self.Decpostorder:
                        offspring_temp.remove(i)
                        idx=idx-1
                        self.repeat+=1
                    else:
                        self.Decinorder.append(i.Dec_inorder)
                        self.Decpostorder.append(i.Dec_postorder)
                for id_indi, individual_i in enumerate(offspring_temp):
                    individual_i.id = idx+id_indi
                    self.Population.append(individual_i)
                #if len(offspring_temp)!=5:
                #   count =count+(5-len(offspring_temp))
            #ncount=len(self.Population)-5
            #count=0      
            if len(self.Population)>self.Popsize:
                self.Population=self.Population[:self.Popsize]
                self.Decinorder=self.Decinorder[:self.Popsize]
                self.Decpostorder=self.Decpostorder[:self.Popsize]
                break
        for idx,individual in enumerate(self.Population):
                individual.id=idx
        self.Pop_fitness,self.user= self.Evaluation(self.Population,user_feature=True)
        self.set_dir(path='initial')
        self.Save(path='initial')

        for x_individual in self.Population:
            self.Archive.append(x_individual.Dec)



    def Evaluation(self,Population,user_feature):
        if self.config.parallel_evaluation and self.config.n_gpu>1:
            fitness =[]
            user=[]
            for i in range(0,len(Population),self.config.n_gpu):
                # one GPU for one solution executed in one thread
                logging.info('solution:{0:>2d} --- {1:>2d}(Parallel evaluation)'.format(i,i+self.config.n_gpu-1))

                solution_set = Population[i:i+self.config.n_gpu]
                self.Para_Evaluation(solution_set)

            fitness = [x.fitness for x in Population]
            fitness = np.array(fitness)
        else:
            # evaluation in Serial model
            fitness = np.zeros((len(Population),2))#100,2目标
            user=np.zeros((len(Population),2))
            for i,solution in enumerate(Population):
                # solution = Population[66]
                solution.evaluation(self.config.device_ids,user_feature)#评价指标
                fitness[i] = solution.fitness
                if user_feature:
                   user[i] = solution.user_feature
            if user_feature:
                return fitness,user
        return fitness

    def Para_Evaluation(self,solution_set):
        thread = [MyThread(solution.evaluation, args=(id,)) for id, solution in enumerate(solution_set)]
        #---------------------------------------
        # (1):execute each thread, but some error(block) may appear due to same dataloader sub-thread are called
        # A = [x.start() for x in thread]
        #---------------------------------
        # (2):wait several seconds after starting each thread
        # to avoid same dataloader sub-thread are used
        for x in thread:
            x.start()
            time.sleep(3)
        # ---------------------------------------
        # synchronize all threads for (returning outputs)/get final outputs
        A = [print(x.is_alive()) for x in thread]
        B = [x.join() for x in thread]
        # C = [x._stop() for x in thread]
        # del A,B,C,thread
        del  A,B,thread
        gc.collect()


    def MatingPoolSelection(self):
        self.MatingPool, self.tour_index = F_mating.F_mating(self.Population.copy(), self.FrontValue,
                                                             self.CrowdDistance)
    def Genetic_operation(self,user_feature=False):
        # if user_feature==False:
            self.offspring = []
            self.user=[]
            i=0
            while len(self.offspring)<self.Popsize:
                # offspring_temp = Generate(self.MatingPool,self.gen)
                offspring_temp = Generate_crossover_mutation(self.MatingPool,self.gen)
                for offspring_i in offspring_temp:#去重
                    if offspring_i.Dec in self.Archive or (offspring_i.Dec_inorder in self.Dec_inorder and offspring_i.Dec_postorder in self.Dec_postorder):
                        self.repeat+=1
                    offspring_i.id = i
                    self.offspring.append(offspring_i)
                    self.Archive.append(offspring_i.Dec)
                    self.Dec_inorder.append(offspring_i.Dec_inorder)
                    self.Dec_postorder.append(offspring_i.Dec_postorder)
                    i+=1
                    if i>=self.Popsize:
                        break

                self.MatingPoolSelection()#上述去重操作后子代产生数量不足种群数量需要补齐self.MatingPool在变
            if user_feature:
                self.off_fitness,self.user = self.Evaluation(self.offspring,user_feature)
            else:
                self.off_fitness = self.Evaluation(self.offspring,user_feature)
      
    def First_Selection(self,Population,Fitness):
        pass



    def EvironmentSelection(self):
        Population = []
        Population.extend(self.Population)
        Population.extend(self.offspring)
        FunctionValue = np.vstack((self.Pop_fitness, self.off_fitness))

        Population, FunctionValue, FrontValue, CrowdDistance, select_index = F_EnvironmentSelect. \
            F_EnvironmentSelect(Population, FunctionValue, self.Popsize)

        self.Population = Population
        self.Pop_fitness = FunctionValue
        self.FrontValue = FrontValue
        self.CrowdDistance = CrowdDistance
        self.select_index = select_index


    def print_logs(self,since_time=None,initial=False):
        if initial:

            logging.info('********************************************************************Initializing**********************************************')
            print('********************************************************************Initializing**********************************************')
        else:
            used_time = (time.time()-since_time)/60

            logging.info('*******************************************************{0:>2d}/{1:>2d} processing, time spent so far:{2:.2f} min******'
                         '*****************************************'.format(self.gen+1,self.Maxi_Gen,used_time))

            print('*******************************************************{0:>2d}/{1:>2d} processing, time spent so far:{2:.2f} min******'
                  '*****************************************'.format(self.gen+1,self.Maxi_Gen,used_time))

    def set_dir(self,path=None):
        if path is None:
            path = self.gen
        self.whole_path = "{}/Gen_{}/".format(self.config.exp_name, path)

        if not os.path.exists(self.whole_path):
            os.makedirs(self.whole_path)

    def Save(self,path=None):
        # return
        fitness_file = self.whole_path + 'fitness.txt'
        np.savetxt(fitness_file, self.Pop_fitness, delimiter=' ')
        if path is None:
            user_file = self.whole_path + 'user_feature.txt'
            np.savetxt(user_file, self.user, delimiter=' ')
        Pop_file = self.whole_path +'Population.txt'
        with open(Pop_file, "w") as file:
            for j,solution in enumerate(self.Population):
                file.write('solution {}: {} \n'.format(j, solution.Dec))

        for i,solution in enumerate(self.Population):
            solution.visualization(self.whole_path+str(i)+'_')

        #------------save as pkl for re-loading------------
        name =  self.whole_path +'Population.pkl'
        f = open(name,'wb')
        pickle.dump(self.Population,f)
        f.close()



    def Plot(self):
        if self.config.parallel_evaluation:
            return
        plt.clf()
        plt.plot(1-self.Pop_fitness[:,0],1-self.Pop_fitness[:,1],'o')
        plt.xlabel('ACC')
        # plt.xlabel('Complexity')
        plt.ylabel('AUC')
        plt.title('Generation {0}/{1} \n best AUC1: {2:.4f}, best AUC2: {3:.4f}'.format(self.gen+1,self.Maxi_Gen,max(1-self.Pop_fitness[:,0]), max(1-self.Pop_fitness[:,1])) )
        # plt.show()
        plt.pause(0.2)
        plt.savefig(self.whole_path+'figure.jpg')

    def Main_Loop(self):


        # plt.ion()
        since_time = time.time()
        self.print_logs(initial=True)
        self.Dec_postorder=[]
        self.Dec_inorder=[]
        self.Initialization_from_existing()
        self.Plot()
        logging.info("inital repeat number:{}".format(self.repeat))
        self.FrontValue = NDsort.NDSort(self.Pop_fitness, self.Popsize)[0]#指标非支配排序
        self.CrowdDistance = F_distance.F_distance(self.Pop_fitness, self.FrontValue)

        while self.gen<self.Maxi_Gen:
            self.set_dir()
            self.print_logs(since_time=since_time)
            self.repeat=0
            self.MatingPoolSelection()#交配池
            if self.gen==self.Maxi_Gen-1:
                self.Genetic_operation(user_feature=True)
            else:
                self.Genetic_operation(user_feature=True)
            logging.info("{}gen repeat num:{}".format({self.gen},{self.repeat}))
            self.EvironmentSelection()
            if self.gen==self.Maxi_Gen-1:
                self.Save()
            else:
                self.Save(path='notnone')
            self.Plot()
            self.gen += 1
        

        # plt.ioff()


if __name__ == '__main__':

    config = get_common_search_config()
       
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    cudnn.enabled = True
    cudnn.benchmark = True
    cudnn.deterministic = True
    EA = EvolutionAlgorithm(config)
    EA.Main_Loop()


