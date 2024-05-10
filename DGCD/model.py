import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.linear2 = nn.Linear(in_features, out_features)
        self.linear3 = nn.Linear(in_features, out_features)
        self.linear4 = nn.Linear(in_features, out_features)
        self.linear5 = nn.Linear(in_features, out_features)
        self.linear6 = nn.Linear(in_features, out_features)
 
    def forward(self, sparse_adj_t,sparse_adj_f, class_emb,stu_embs,exer_embs):
        s2c_emb = torch.mean(self.linear3(stu_embs),dim=0,keepdim=True)
        c2s_emb = self.linear3(class_emb) / stu_embs.size(0)
        s2s_emb = self.linear4(stu_embs)
        c2c_emb = self.linear5(class_emb)
        e2e_emb = self.linear6(exer_embs)
        c2s_emb_entropy = self.entropy(c2s_emb).clone().detach()
        if sparse_adj_f == None:
            # true GCN
            stu_emb_t = self.linear1(sparse_adj_t @ exer_embs)
            exer_embs_t = self.linear1(sparse_adj_t.T @ stu_embs)
            stu_emb_t_entropy = self.entropy(stu_emb_t).clone().detach()
            stu_emb_entropy = stu_emb_t_entropy + c2s_emb_entropy

            stu_embs_new = (stu_emb_t * (stu_emb_t_entropy/stu_emb_entropy) + c2s_emb*(c2s_emb_entropy/stu_emb_entropy))/2 + s2s_emb
            exer_embs_new = exer_embs_t + e2e_emb
            class_emb_new = s2c_emb + c2c_emb


        elif sparse_adj_t == None:
            # false GCN
            stu_emb_f = self.linear2(sparse_adj_f @ exer_embs)
            exer_embs_f = self.linear2(sparse_adj_f.T @ stu_embs)
            stu_emb_f_entropy = self.entropy(stu_emb_f ).clone().detach()
            stu_emb_entropy = stu_emb_f_entropy + c2s_emb_entropy

            stu_embs_new =  (stu_emb_f * (stu_emb_f_entropy/stu_emb_entropy) + c2s_emb*(c2s_emb_entropy/stu_emb_entropy))/2 + s2s_emb
            exer_embs_new = exer_embs_f + e2e_emb
            class_emb_new = s2c_emb + c2c_emb
            
        else:
            # true GCN
            stu_emb_t = self.linear1(sparse_adj_t @ exer_embs)
            exer_embs_t = self.linear1(sparse_adj_t.T @ stu_embs)
            stu_emb_t_entropy = self.entropy(stu_emb_t).clone().detach()
            exer_embs_t_entropy = self.entropy(exer_embs_t).clone().detach()
            # false GCN
            stu_emb_f = self.linear2(sparse_adj_f @ exer_embs)
            exer_embs_f = self.linear2(sparse_adj_f.T @ stu_embs)
            stu_emb_f_entropy = self.entropy(stu_emb_f).clone().detach()
            exer_embs_f_entropy = self.entropy(exer_embs_f).clone().detach()

            stu_emb_entropy = stu_emb_t_entropy + stu_emb_f_entropy + c2s_emb_entropy
            exer_emb_entropy = exer_embs_t_entropy + exer_embs_f_entropy

            stu_embs_new = (stu_emb_t * (stu_emb_t_entropy/stu_emb_entropy) + stu_emb_f * (stu_emb_f_entropy/stu_emb_entropy) \
                            + c2s_emb*(c2s_emb_entropy/stu_emb_entropy))/3 + s2s_emb
            exer_embs_new = (exer_embs_t * (exer_embs_t_entropy/exer_emb_entropy) + exer_embs_f * (exer_embs_f_entropy/exer_emb_entropy))/2 + e2e_emb
            class_emb_new = s2c_emb + c2c_emb 

        return class_emb_new,stu_embs_new,exer_embs_new

    def entropy(self,embedding):
        # 将 embedding 转化为概率分布
        prob_distribution = F.softmax(embedding, dim=1)
        # 计算信息熵
        entropy_value = - torch.sum(prob_distribution * torch.log2(prob_distribution + 1e-10),dim=1)/math.sqrt(embedding.size(1))

        return entropy_value.unsqueeze(1)

class NoneNegClipper(object):
    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)

class DGCD(nn.Module):
    def __init__(self,class_n,stu_n,exer_n,skill_n,t):
        self.class_n = class_n
        self.stu_n = stu_n
        self.exer_n = exer_n
        self.skill_n = skill_n
        self.prednet_input_len = skill_n
        self.prednet_len1,self.prednet_len2,self.prednet_len3 = 256,128,1
        self.pi = t 
        super(DGCD, self).__init__()

        self.class_emb = nn.Embedding(self.class_n, self.skill_n)
        self.stu_emb = nn.Embedding(self.stu_n, self.skill_n)
        self.exer_diff = nn.Embedding(self.exer_n, self.skill_n)
        self.exer_dis = nn.Embedding(self.exer_n, 1)
        self.mu = nn.Linear(2*self.skill_n, 1)
        self.logvar = nn.Linear(2*self.skill_n, 1)
        self.conv1 = GCNLayer(self.skill_n, self.skill_n)
        self.conv2 = GCNLayer(self.skill_n, self.skill_n)
        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)


        for name,param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
    
    def forward(self,edge_t,edge_f,class_id,stu_list,kn_emb,exer_list,exer_test):
        if exer_test != None:
            out = self.testforward(edge_t,edge_f,class_id,stu_list,exer_list,kn_emb,exer_test)
        else:
            out = self.trainforward(edge_t,edge_f,class_id,stu_list,exer_list,kn_emb)
        return out
    
    def trainforward(self,edge_t,edge_f,class_id,stu_list,exer_list,kn_emb):
        stu_embeddings = self.stu_emb(stu_list)
        exer_embeddings = self.exer_diff(exer_list)
        class_emb = self.class_emb(class_id)
        exer_dis = torch.sigmoid(self.exer_dis(exer_list))*10

        if len(edge_f[0]) ==0:        
            # edge_t vae
            combined_embeddings_t = torch.cat([stu_embeddings[edge_t[0]], exer_embeddings[edge_t[1]]], dim=1)
            edge_mu_t = self.mu(combined_embeddings_t)
            edge_logvar_t = self.logvar(combined_embeddings_t)
            edge_std_t = torch.exp(0.5 * edge_logvar_t)
            kl_loss = self.kl_loss(edge_mu_t, edge_logvar_t)
            edge_t_1 = edge_mu_t + edge_std_t * torch.randn_like(edge_std_t)
            edge_t_1 = torch.sigmoid(edge_t_1).squeeze()
            edge_t_1 = RelaxedBernoulli(self.pi,probs=edge_t_1).rsample()
            adj_t_1 = torch.sparse.FloatTensor(edge_t,edge_t_1,(len(stu_list),len(exer_list)))
            adj_t_1_norm = self.sparse_adj_norm(adj_t_1)
            class_emb_1,stu_embs_1,exer_embs_1 = self.conv1(adj_t_1_norm,None,class_emb.unsqueeze(0),stu_embeddings,exer_embeddings)
            class_emb_2,stu_embs_2,exer_embs_2 = self.conv2(adj_t_1_norm,None,class_emb_1,stu_embs_1,exer_embs_1)
        elif len(edge_t[0]) ==0:
            #edge_f vae
            combined_embeddings_f = torch.cat([stu_embeddings[edge_f[0]], exer_embeddings[edge_f[1]]], dim=1)
            edge_mu_f = self.mu(combined_embeddings_f)
            edge_logvar_f = self.logvar(combined_embeddings_f)
            edge_std_f = torch.exp(0.5 * edge_logvar_f)
            kl_loss = self.kl_loss(edge_mu_f, edge_logvar_f)
            edge_f_1 = edge_mu_f + edge_std_f * torch.randn_like(edge_std_f)
            edge_f_1 = torch.sigmoid(edge_f_1).squeeze() 
            edge_f_1 = RelaxedBernoulli(self.pi,probs=edge_f_1).rsample()
            adj_f_1 = torch.sparse.FloatTensor(edge_f,edge_f_1,(len(stu_list),len(exer_list)))
            adj_f_1_norm = self.sparse_adj_norm(adj_f_1)
            class_emb_1,stu_embs_1,exer_embs_1 = self.conv1(None,adj_f_1_norm,class_emb.unsqueeze(0),stu_embeddings,exer_embeddings)
            class_emb_2,stu_embs_2,exer_embs_2 = self.conv2(None,adj_f_1_norm,class_emb_1,stu_embs_1,exer_embs_1)
        else:
            combined_embeddings_t = torch.cat([stu_embeddings[edge_t[0]], exer_embeddings[edge_t[1]]], dim=1)
            edge_mu_t = self.mu(combined_embeddings_t)
            edge_logvar_t = self.logvar(combined_embeddings_t)
            edge_std_t = torch.exp(0.5 * edge_logvar_t)

            combined_embeddings_f = torch.cat([stu_embeddings[edge_f[0]], exer_embeddings[edge_f[1]]], dim=1)
            edge_mu_f = self.mu(combined_embeddings_f)
            edge_logvar_f = self.logvar(combined_embeddings_f)
            edge_std_f = torch.exp(0.5 * edge_logvar_f)

            kl_loss = self.kl_loss(edge_mu_t, edge_logvar_t) + self.kl_loss(edge_mu_f, edge_logvar_f)

            edge_t_1 = edge_mu_t + edge_std_t * torch.randn_like(edge_std_t)
            edge_f_1 = edge_mu_f + edge_std_f * torch.randn_like(edge_std_f)
            edge_t_1 = torch.sigmoid(edge_t_1).squeeze() 
            edge_f_1 = torch.sigmoid(edge_f_1).squeeze() 
            edge_t_1 = RelaxedBernoulli(self.pi,probs=edge_t_1).rsample()
            edge_f_1 = RelaxedBernoulli(self.pi,probs=edge_f_1).rsample()

            adj_t_1 = torch.sparse.FloatTensor(edge_t,edge_t_1,(len(stu_list),len(exer_list)))
            adj_f_1 = torch.sparse.FloatTensor(edge_f,edge_f_1,(len(stu_list),len(exer_list)))

            adj_t_1_norm = self.sparse_adj_norm(adj_t_1)
            adj_f_1_norm = self.sparse_adj_norm(adj_f_1)

            class_emb_1,stu_embs_1,exer_embs_1 = self.conv1(adj_t_1_norm,adj_f_1_norm,class_emb.unsqueeze(0),stu_embeddings,exer_embeddings)
            class_emb_2,stu_embs_2,exer_embs_2 = self.conv2(adj_t_1_norm,adj_f_1_norm,class_emb_1,stu_embs_1,exer_embs_1)
            
   
        class_k = torch.sigmoid(class_emb_2)
        exer_k = torch.sigmoid(exer_embeddings)
        input_x = exer_dis * (class_k - exer_k) * kn_emb
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output_d1 = torch.sigmoid(self.prednet_full3(input_x))

        return output,kl_loss
    
    def testforward(self,edge_t,edge_f,class_id,stu_list,exer_list,kn_emb,exer_test):
        class_emb = self.class_emb(class_id)
        stu_embeddings = self.stu_emb(stu_list)
        exer_embeddings = self.exer_diff(exer_list)
        if len(edge_f[0]) ==0 :
            # edge_t vae
            combined_embeddings_t = torch.cat([stu_embeddings[edge_t[0]], exer_embeddings[edge_t[1]]], dim=1)
            edge_mu_t = self.mu(combined_embeddings_t)
            edge_t_1 = torch.sigmoid(edge_mu_t)
            edge_t_1 = edge_t_1.squeeze() 
            edge_t_1 = RelaxedBernoulli(self.pi,probs=edge_t_1).rsample()
            adj_t_1 = torch.sparse.FloatTensor(edge_t,edge_t_1,(len(stu_list),len(exer_list)))
            adj_t_1_norm = self.sparse_adj_norm(adj_t_1)
            class_emb_1,stu_embs_1,exer_embs_1 = self.conv1(adj_t_1_norm,None,class_emb.unsqueeze(0),stu_embeddings,exer_embeddings)
            class_emb_2,stu_embs_2,exer_embs_2 = self.conv2(adj_t_1_norm,None,class_emb_1,stu_embs_1,exer_embs_1)
        elif len(edge_t[0]) ==0:
            #edge_f vae
            combined_embeddings_f = torch.cat([stu_embeddings[edge_f[0]], exer_embeddings[edge_f[1]]], dim=1)
            edge_mu_f = self.mu(combined_embeddings_f)
            edge_f_1 = torch.sigmoid(edge_mu_f)
            edge_f_1 = edge_f_1.squeeze() 
            edge_f_1 = RelaxedBernoulli(self.pi,probs=edge_f_1).rsample() 
            adj_f_1 = torch.sparse.FloatTensor(edge_f,edge_f_1,(len(stu_list),len(exer_list)))
            adj_f_1_norm = self.sparse_adj_norm(adj_f_1)
            class_emb_1,stu_embs_1,exer_embs_1 = self.conv1(None,adj_f_1_norm,class_emb.unsqueeze(0),stu_embeddings,exer_embeddings)
            class_emb_2,stu_embs_2,exer_embs_2 = self.conv2(None,adj_f_1_norm,class_emb_1,stu_embs_1,exer_embs_1)
        else:
            combined_embeddings_t = torch.cat([stu_embeddings[edge_t[0]], exer_embeddings[edge_t[1]]], dim=1)
            edge_mu_t = self.mu(combined_embeddings_t)

            combined_embeddings_f = torch.cat([stu_embeddings[edge_f[0]], exer_embeddings[edge_f[1]]], dim=1)
            edge_mu_f = self.mu(combined_embeddings_f)
            edge_t_1 = torch.sigmoid(edge_mu_t).squeeze() 
            edge_f_1 = torch.sigmoid(edge_mu_f).squeeze()
            edge_t_1 = RelaxedBernoulli(self.pi,probs=edge_t_1).rsample()
            edge_f_1 = RelaxedBernoulli(self.pi,probs=edge_f_1).rsample()

            adj_t_1 = torch.sparse.FloatTensor(edge_t,edge_t_1,(len(stu_list),len(exer_list)))
            adj_f_1 = torch.sparse.FloatTensor(edge_f,edge_f_1,(len(stu_list),len(exer_list)))

            adj_t_1_norm = self.sparse_adj_norm(adj_t_1)
            adj_f_1_norm = self.sparse_adj_norm(adj_f_1)
            class_emb_1,stu_embs_1,exer_embs_1 = self.conv1(adj_t_1_norm,adj_f_1_norm,class_emb.unsqueeze(0),stu_embeddings,exer_embeddings)
            class_emb_2,stu_embs_2,exer_embs_2 = self.conv2(adj_t_1_norm,adj_f_1_norm,class_emb_1,stu_embs_1,exer_embs_1)
        
        class_k = torch.sigmoid(class_emb_2)
        exer_dis = torch.sigmoid(self.exer_dis(exer_test))*10
        exer_k = torch.sigmoid(self.exer_diff(exer_test))
        input_x = exer_dis * (class_k - exer_k) * kn_emb
        input_x = self.drop_1(torch.tanh(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.tanh(self.prednet_full2(input_x)))
        output = torch.sigmoid(self.prednet_full3(input_x))

        return output
    
    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)
    
    def kl_loss(self,mu,logvar):
        return -0.5 * torch.mean(1 + logvar - (mu).pow(2) - logvar.exp())
    
    def sparse_adj_norm(self,adj):
        adj_dense = adj.to_dense()
        dm = torch.diag(1.0 / torch.sum(adj_dense, dim=1))
        dm[dm == float('inf')] = 0
        dmt = torch.diag(1.0 / torch.sum(adj_dense.T, dim=1))
        dmt[dmt == float('inf')] = 0
        norm_adj = torch.matmul(torch.matmul(dm, adj_dense), dmt)
        return norm_adj
