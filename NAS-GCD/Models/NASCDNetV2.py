import torch
import torch.nn as nn
from Models.Operations import *

from  copy import deepcopy
import torch.nn.functional as F
from genotypes import Genotype_mapping as mapping # for NAS search: exp, log are not allowed
# from genotypes import Genotype_mapping_1 as mapping  # for building NCD
# from genotypes import Genotype_mapping__ as mapping   # used for main function's other method


class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class NASCDNet(nn.Module):
    def __init__(self,args,NAS_dec=None):
        super(NASCDNet, self).__init__()

        #-----------Settings------------
        self.NAS_dec = NAS_dec
        self.student = nn.Embedding(args['n_student'],args['dim'])
        self.exercise = nn.Embedding(args['n_exercise'],args['dim'])
        # self.concept = nn.Embedding(args['n_concept'],args['dim'])
        self.concept = nn.Linear(args['n_concept'],args['dim'])
        # self.concept = Identity()  # concept 直接作为输出，可能会报错 [2, 0, 8, 3, 0, 0]
        #-----------NASGraph------------
        self.Graph = Graph()
        #-----------classifier------------
        self.prednet_input_len = args['dim']
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)
        #-----------classifier------------

        self.BN = nn.BatchNorm1d(args['dim'])


    def get_embedding(self,x):
        # return self.student(x[0].long()), self.exercise(x[1].long()),self.concept(x[2].long())
        return self.student(x[0].long()), self.exercise(x[1].long()),self.concept(x[2].float())

    def forward(self,x, input_NAS=None):
        if input_NAS is None and self.NAS_dec is not  None:
            input_NAS = self.NAS_dec
        stu_embedding, exer_embedding, conc_embedding = self.get_embedding(x)

        y = self.Graph([stu_embedding, exer_embedding, conc_embedding],NAS = input_NAS)

        if y.shape[1]!=1:
            # y=self.BN(y)
            input_x = self.drop_1(torch.sigmoid(self.prednet_full1(y)))
            input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
            y = torch.sigmoid(self.prednet_full3(input_x))
        else:
            # y = torch.sigmoid(y)   # 完全不适合，效果差
            # y = (F.hardtanh(y)+1)/2 # 强制限制在  0~1之间,效果不如 clip
            y = torch.clip(y,0,1)   # 目测最接近  y=y的效果

        return y.view(-1)



class Node(nn.Module):
    def __init__(self):
        super(Node, self).__init__()
        self.Operation = nn.ModuleDict()
        for primitive in mapping.values():
            op = NAS_OPS[primitive](None)
            self.Operation[primitive] = op

        self.linear_single = nn.Sequential(nn.Linear(1,10),nn.Linear(10,1))
        self.linear_same = nn.Linear(123,123)

    def forward(self,candidate_inputs,NAS_node):
        x1,x2,Op = NAS_node[0],NAS_node[1],NAS_node[2]
        Op_name = mapping[Op]

        # if candidate_inputs[x1].shape[1]==1:
        #     x = self.linear_single(candidate_inputs[x1])
        # else:
        #     x = self.linear_same(candidate_inputs[x1])

        if Op_name not in ['add','mul','concat']:
            # y = self.Operation[Op_name](x)
            y = self.Operation[Op_name](candidate_inputs[x1])
        else:
            # if candidate_inputs[x2].shape[1]==1:
            #     x_2 = self.linear_single(candidate_inputs[x2])
            # else:
            #     x_2= self.linear_same(candidate_inputs[x2])
            # y = self.Operation[Op_name](x,x_2)
            y = self.Operation[Op_name](candidate_inputs[x1],candidate_inputs[x2])
        return y

class Graph(nn.Module):
    def __init__(self,Maxi_num_node=12):
        super(Graph, self).__init__()
        self.Nodes = nn.ModuleList([])
        for i in range(Maxi_num_node):
            self.Nodes.extend([Node()])

    def forward(self,concat_states, NAS):
        for i,node in enumerate(self.Nodes):
            if 3*(i+1)>len(NAS):
                break
            state = node(concat_states,NAS[i*3:(i+1)*3])
            concat_states.append(state)
        return state


if __name__ == '__main__':
    args = {'n_student':100, 'n_exercise': 150, 'n_concept':50, 'dim':128}

    stu=15
    exer = 20
    conc = 30
    onehot_stu = torch.Tensor([stu]).cuda().long()
    onehot_exer = torch.Tensor([exer]).cuda().long()
    onehot_conc = torch.Tensor([conc]).cuda().long()

    Net = NASCDNet(args).cuda()
    input_nas = [0,1,5,  1,1,6, 4,0,5,  4,0,0, 0,6,7, 6,7,8]
    out = Net([onehot_stu,onehot_exer,onehot_conc],input_NAS=input_nas)




    a=0