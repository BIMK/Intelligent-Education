import sys

import torch
import torch.nn as nn
from pyod.models.ecod import ECOD
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from baseCDM.NCD import NCD
from baseCDM.KANCD import KANCD
from baseCDM.DINA import DINA
from baseCDM.KSCD import KSCD
from baseCDM.IRT import IRT
from baseCDM.MIRT import MIRT
import warnings

warnings.filterwarnings("ignore")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Net(nn.Module):
    '''
    NeuralCDM
    '''

    def __init__(self, student_n, exer_n, knowledge_n, time_graph):
        super(Net, self).__init__()
        self.time_graph = time_graph
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.boundaries = torch.tensor([0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048], device=device)
        self.time_embedding = nn.Embedding(self.boundaries.size(0) + 1, 128)
        self.time_effect_embedding = nn.Embedding(self.boundaries.size(0) + 1, 128)

        self.multihead_attn_a = nn.MultiheadAttention(128, 2, batch_first=True)
        self.multihead_attn_b = nn.MultiheadAttention(128, 2, batch_first=True)
        self.self_multihead_attn = nn.MultiheadAttention(128, 1, batch_first=True)

        self.hint_embedding = nn.Embedding(self.knowledge_dim, 128)
        self.hint_s_embedding = nn.Embedding(self.student_n, 128)
        self.vae_encode = nn.Sequential(
            nn.Linear(self.knowledge_dim, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, 4),
        )
        self.vae_decode = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, self.knowledge_dim),
        )
        self.hintBN = nn.BatchNorm1d(128)
        self.hint_FC = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 128)
        )
        self.FC = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(16, 2),
            nn.Sigmoid()
        )

        # network structure
        self.baseCDM_type = sys.argv[3]
        print(f"use {self.baseCDM_type} model")
        assert self.baseCDM_type in ['NCD', "KANCD", "DINA", "KSCD", "IRT", "MIRT"]
        if self.baseCDM_type == 'NCD':
            self.baseCDM = NCD(self.student_n, self.exer_n, self.knowledge_dim)
        if self.baseCDM_type == 'KANCD':
            self.baseCDM = KANCD(self.student_n, self.exer_n, self.knowledge_dim)
        if self.baseCDM_type == 'DINA':
            self.baseCDM = DINA(self.student_n, self.exer_n, self.knowledge_dim)
        if self.baseCDM_type == 'KSCD':
            self.baseCDM = KSCD(self.student_n, self.exer_n, self.knowledge_dim)
        if self.baseCDM_type == 'IRT':
            self.baseCDM = IRT(self.student_n, self.exer_n)
        if self.baseCDM_type == 'MIRT':
            self.baseCDM = MIRT(self.student_n, self.exer_n)

        self.add_or_not = sys.argv[4] == "add"
        if self.add_or_not:
            print("add my additional framework")
        else:
            print("no additional charges")

        self.baseAD_type = sys.argv[5]
        print(f"use {self.baseAD_type} AD")
        assert self.baseAD_type in ["ECOD", "IFOREST", "KNN", "LOF", "OCSVM"]
        if self.baseAD_type == 'ECOD':
            self.baseAD = ECOD()
        if self.baseAD_type == 'IFOREST':
            self.baseAD = IForest()
        if self.baseAD_type == 'KNN':
            self.baseAD = KNN(n_neighbors=1)
        if self.baseAD_type == 'LOF':
            self.baseAD = LOF()
        if self.baseAD_type == 'OCSVM':
            self.baseAD = OCSVM()

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name and "BN" not in name:
                nn.init.xavier_normal_(param)

    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

    def forward(self, stu_id, exer_id, kn_emb, time_taken, skill_index):
        output = None
        recon_loss = torch.zeros(1, device=device)
        mu = torch.zeros(1, device=device)
        logvar = torch.zeros(1, device=device)

        if self.baseCDM_type == 'NCD':
            output = self.baseCDM(stu_id, exer_id, kn_emb)
        if self.baseCDM_type == 'KANCD':
            output = self.baseCDM(stu_id, exer_id, kn_emb)
        if self.baseCDM_type == 'DINA':
            output = self.baseCDM(stu_id, exer_id, kn_emb)
        if self.baseCDM_type == 'KSCD':
            output = self.baseCDM(stu_id, exer_id, kn_emb)
        if self.baseCDM_type == 'IRT':
            output = self.baseCDM(stu_id, exer_id)
        if self.baseCDM_type == 'MIRT':
            output = self.baseCDM(stu_id, exer_id)

        if self.add_or_not:
            attn_output_a = []

            for index, item in enumerate(stu_id):
                a, i = self.time_graph.get_all_problem_time(item, time_taken[index])
                a_embed = self.time_embedding(torch.bucketize(a.to(device), self.boundaries))
                if a.size()[0] != 1:
                    a = torch.unsqueeze(a, dim=1)
                    all_AD_result = self.baseAD.fit(a).decision_scores_
                    AD_result = all_AD_result[i]
                else:
                    all_AD_result = [1.0]
                    AD_result = [1.0]
                all_AD_result = torch.tensor(all_AD_result, device=device)
                AD_result = torch.tensor(AD_result, device=device)
                weight = -torch.pow((all_AD_result - AD_result), 2)
                weight = torch.softmax(weight, dim=0)
                weight = torch.unsqueeze(weight, dim=1)
                result = torch.mul(weight, a_embed)
                result = torch.sum(result, dim=0)
                result = torch.squeeze(result, dim=0)
                attn_output_a.append(result)
            attn_output_a = torch.stack(attn_output_a)
            attn_output_a = torch.unsqueeze(attn_output_a, dim=1).float()

            attn_output_b = []

            for index, item in enumerate(exer_id):
                a, i = self.time_graph.get_all_student_time(item, time_taken[index])
                a_embed = self.time_embedding(torch.bucketize(a.to(device), self.boundaries))
                if a.size()[0] != 1:
                    a = torch.unsqueeze(a, dim=1)
                    all_AD_result = self.baseAD.fit(a).decision_scores_
                    AD_result = all_AD_result[i]
                else:
                    all_AD_result = [1.0]
                    AD_result = [1.0]
                all_AD_result = torch.tensor(all_AD_result, device=device)
                AD_result = torch.tensor(AD_result, device=device)
                weight = -torch.pow((all_AD_result - AD_result), 2)
                weight = torch.softmax(weight, dim=0)
                weight = torch.unsqueeze(weight, dim=1)
                result = torch.mul(weight, a_embed)
                result = torch.sum(result, dim=0)
                result = torch.squeeze(result, dim=0)
                attn_output_b.append(result)
            attn_output_b = torch.stack(attn_output_b)
            attn_output_b = torch.unsqueeze(attn_output_b, dim=1).float()

            hint_embeding = self.hint_embedding(skill_index)
            hint_s_embeding = self.hint_s_embedding(stu_id)
            knowledge_low_emb = self.hint_embedding(torch.arange(self.knowledge_dim).to(hint_s_embeding.device))
            stu_all_k_emb = torch.mm(hint_s_embeding, knowledge_low_emb.T)

            stu_all_k_emb_a = self.vae_encode(stu_all_k_emb)
            mu, logvar = stu_all_k_emb_a.chunk(2, dim=1)
            z = self.reparameterise(mu, logvar)
            stu_all_k_emb_recon = self.vae_decode(z)
            recon_loss = torch.pow(stu_all_k_emb - stu_all_k_emb_recon, 2).sum(dim=1, keepdim=True) / self.knowledge_dim

            hint_embeding = torch.mul(recon_loss, hint_embeding)
            hint_embeding_new = self.hintBN(hint_embeding)
            hint_embeding_new = self.hint_FC(hint_embeding_new)
            hint_embeding_new = hint_embeding_new + hint_embeding
            hint_embeding_new = torch.unsqueeze(hint_embeding_new, dim=1)
            cat_data = torch.concat([attn_output_a, attn_output_b, hint_embeding_new], dim=1)
            self_multihead_attn_output, self_multihead_attn_output_w = self.self_multihead_attn(cat_data, cat_data,
                                                                                                cat_data)
            self_multihead_attn_output = torch.reshape(self_multihead_attn_output,
                                                       (self_multihead_attn_output.size(0), -1))
            finally_data = self.FC(self_multihead_attn_output)

            output = output * finally_data[:, 0] + (1 - output) * finally_data[:, 1]
        return output, recon_loss, mu, logvar

    def get_knowledge_status(self, stu_id):
        stat_emb = torch.sigmoid(self.student_emb(stu_id))
        return stat_emb.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10
        return k_difficulty.data, e_discrimination.data
