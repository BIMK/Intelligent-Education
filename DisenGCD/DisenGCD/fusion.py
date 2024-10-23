import torch
import torch.nn as nn
import torch.nn.functional as F
from GraphLayer import GraphLayer

class Fusion(nn.Module):
    def __init__(self, args, local_map):
        self.device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')
        self.knowledge_dim = args.knowledge_n
        self.exer_n = args.exer_n
        self.emb_num = args.student_n
        self.stu_dim = self.knowledge_dim

        # data structure
        self.directed_g = local_map['directed_g'].to(self.device)
        self.undirected_g = local_map['undirected_g'].to(self.device)
        self.k_from_e = local_map['k_from_e'].to(self.device)

        self.e_from_k = local_map['e_from_k'].to(self.device)


        super(Fusion, self).__init__()

        self.directed_gat = GraphLayer(self.directed_g, args.knowledge_n, args.knowledge_n)
        self.undirected_gat = GraphLayer(self.undirected_g, args.knowledge_n, args.knowledge_n)
        self.e_from_k = GraphLayer(self.e_from_k, args.knowledge_n, args.knowledge_n)  # src: k

        self.k_from_e = GraphLayer(self.k_from_e, args.knowledge_n, args.knowledge_n)  # src: e

        self.k_attn_fc1 = nn.Linear(2 * args.knowledge_n, 1, bias=True)
        self.k_attn_fc2 = nn.Linear(2 * args.knowledge_n, 1, bias=True)
        self.k_attn_fc3 = nn.Linear(2 * args.knowledge_n, 1, bias=True)

        self.e_attn_fc1 = nn.Linear(2 * args.knowledge_n, 1, bias=True)

    def forward(self, exer_emb,kn_emb):
        k_directed = self.directed_gat(kn_emb)  
        k_undirected = self.undirected_gat(kn_emb)

        e_k_graph = torch.cat((exer_emb, kn_emb), dim=0)
        e_from_k_graph = self.e_from_k(e_k_graph)
        # update concepts
        A = kn_emb
        B = k_directed
        C = k_undirected
        concat_c_1 = torch.cat([A, B], dim=1)
        concat_c_2 = torch.cat([A, C], dim=1)
        score1 = self.k_attn_fc1(concat_c_1) 
        score2 = self.k_attn_fc2(concat_c_2)  
        score = F.softmax(torch.cat([score1, score2], dim=1), dim=1)
                         
        kn_emb = A + score[:, 0].unsqueeze(1) * B + score[:, 1].unsqueeze(1) * C


        # updated exercises
        A = exer_emb
        B = e_from_k_graph[0:self.exer_n]
        concat_e_1 = torch.cat([A, B], dim=1)
        score1 = self.e_attn_fc1(concat_e_1)
        exer_emb = exer_emb + score1[:, 0].unsqueeze(1) * B

        return exer_emb,kn_emb
