import torch
import torch.nn as nn
import torch.nn.functional as F
from model_search_paths import Model_paths as Model
from DCGCD.fusion import Fusion

class Net(nn.Module):
    def __init__(self,args,all_map,node_types,local_map):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.stu_n = args.student_n
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256
        self.map = all_map
        self.node_type=node_types
        self.directed_g = local_map['directed_g'].to(self.device)
        self.undirected_g = local_map['undirected_g'].to(self.device)
        self.e_from_k = local_map['e_from_k'].to(self.device)

        super(Net,self).__init__()

        #network structure
        self.stu_emb = nn.Embedding(self.stu_n,self.knowledge_dim).to(self.device)
        self.kn_emb = nn.Embedding(self.knowledge_dim, self.knowledge_dim)  
        self.exer_emb = nn.Embedding(self.exer_n, self.knowledge_dim)  

        self.index = torch.LongTensor(list(range(self.stu_n))).to(self.device)
        self.exer_index = torch.LongTensor(list(range(self.exer_n))).to(self.device)  
        self.k_index = torch.LongTensor(list(range( self.knowledge_dim))).to(self.device)  


        #学生的嵌入
        self.FusionLayer1 = Model(args.gpu,args.knowledge_n,args.n_hid,3,len(self.map),args.knowledge_n, [4],
                                  args.ratio,[3,4],args.k,args.lam_seq,args.lam_res)                         
        self.FusionLayer3 = Fusion(args, local_map)
        self.FusionLayer4 = Fusion(args, local_map)

        self.prednet_full3 = nn.Linear(1 * args.knowledge_n, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                print(name)
                nn.init.xavier_normal_(param)


    def forward(self, stu_id, exer_id, kn_r):
        all_stu_emb = self.stu_emb(self.index).to(self.device)
        exer_emb = self.exer_emb(self.exer_index).to(self.device)
        kn_emb = self.kn_emb(self.k_index).to(self.device)

  
        all_emb = torch.cat((all_stu_emb,exer_emb,kn_emb),0)
        all_stu_emb1 = self.FusionLayer1(all_emb, self.node_type, self.map)
        all_stu_emb1 = all_stu_emb1[self.knowledge_dim+self.exer_n:self.stu_n + self.exer_n+self.stu_n, :]
        #all_stu_emb2 = all_stu_emb1[1747:3714, :]

        exer_emb1,kn_emb1 = self.FusionLayer3(exer_emb,kn_emb)
        exer_emb2,kn_emb2 = self.FusionLayer4(exer_emb,kn_emb1)



        batch_exer_emb = exer_emb2[exer_id]  # 32 123
        batch_exer_vector = batch_exer_emb.repeat(1, batch_exer_emb.shape[1]).reshape(batch_exer_emb.shape[0],
                                                                                      batch_exer_emb.shape[1],
                                                                                      batch_exer_emb.shape[1])

        # get batch student data
        batch_stu_emb = all_stu_emb2[stu_id]  # 32 123
        batch_stu_vector = batch_stu_emb.repeat(1, batch_stu_emb.shape[1]).reshape(batch_stu_emb.shape[0],
                                                                                   batch_stu_emb.shape[1],
                                                                                   batch_stu_emb.shape[1])

        # get batch knowledge concept data
        kn_vector = kn_emb2.repeat(batch_stu_emb.shape[0], 1).reshape(batch_stu_emb.shape[0], kn_emb.shape[0],
                                                                     kn_emb.shape[1])


        # C认知诊
        alpha = batch_exer_vector + kn_vector
        betta = batch_stu_vector + kn_vector
        o = torch.sigmoid(self.prednet_full3(alpha * betta))


        sum_out = torch.sum(o * kn_r.unsqueeze(2), dim=1)             
        count_of_concept = torch.sum(kn_r, dim=1).unsqueeze(1)              
        return output

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)
class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)