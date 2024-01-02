import torch
import torch.nn as nn
from utils import main, seed_torch
import math
from torch.distributions import Normal

params = main()

class Net(nn.Module):

    def __init__(self, student_n, exer_n, knowledge_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.student_n = student_n
        self.device = params.device
        self.dropout = params.dropout
        self.p = params.p

        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = params.prednet_len1, params.prednet_len2  # changeable


        super(Net, self).__init__()

        # seed_torch(0)
        # network structure
        self.student_emb_mean = nn.Embedding(self.student_n, self.knowledge_dim)
        self.student_emb_covariance = nn.Embedding(self.student_n, self.knowledge_dim)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim)
        self.e_discrimination = nn.Embedding(self.exer_n, 1)

        self.prednet_full1 = nn.Linear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=self.dropout)
        self.prednet_full2 = nn.Linear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=self.dropout)
        self.prednet_full3 = nn.Linear(self.prednet_len2, 1)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        with open("../0Reli-NCD/five_fold_params/data1/assist2009_student_mean.txt", "r") as f:
            self.data_mean = f.read()
        self.data_mean = eval(self.data_mean)
        self.data_mean = torch.tensor(self.data_mean)
        self.data_mean_mean = torch.mean(self.data_mean, dim = 0).to(self.device)


    def forward(self, stu_id, exer_id, kn_id, d_type):     
        if d_type == 'train':
            stu_emb, stu_mean, log_stu_covariance = self.get_individ_Distribution_train(stu_id)
            stu_emb = torch.sigmoid(stu_emb)

            k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
            e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10

            output_0 = e_discrimination * (stu_emb - k_difficulty) * kn_id  
            output_0 = self.drop_1(torch.sigmoid(self.prednet_full1(output_0)))
            output_0 = self.drop_2(torch.sigmoid(self.prednet_full2(output_0)))
            output = torch.sigmoid(self.prednet_full3(output_0))

            return output, stu_mean, log_stu_covariance, kn_id, self.data_mean_mean
        
        else:
            stu_mean, log_stu_covariance = self.get_individ_Distribution_val_test(stu_id)

            k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
            e_discrimination = torch.sigmoid(self.e_discrimination(exer_id)) * 10

            stu_emb = torch.sigmoid(stu_mean)

            output_0 = e_discrimination * (stu_emb - k_difficulty) * kn_id
            output_0 = self.drop_1(torch.sigmoid(self.prednet_full1(output_0)))
            output_0 = self.drop_2(torch.sigmoid(self.prednet_full2(output_0)))
            output = torch.sigmoid(self.prednet_full3(output_0))

            return output, stu_mean, log_stu_covariance

    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)

    def get_individ_Distribution_train(self, stu_id):
        stu_mean = self.student_emb_mean(stu_id)  

        log_stu_covariance = self.student_emb_covariance(stu_id)
        stu_covariance = log_stu_covariance.exp()

        stu_covariance = stu_covariance - 1.0 / (2 * math.e * math.pi)
        stu_covariance = torch.dropout(stu_covariance, p=self.p, train=True)
        stu_covariance = stu_covariance + 1.0 / (2 * math.e * math.pi)

        standard_nor_distribution = torch.randn(stu_mean.shape).to(self.device)
        stu_state = stu_mean + torch.mul(standard_nor_distribution, stu_covariance) 

        return stu_state, stu_mean, log_stu_covariance

    def get_individ_Distribution_val_test(self, stu_id):    
        log_stu_covariance = self.student_emb_covariance(stu_id)
        stu_mean = self.student_emb_mean(stu_id)   

        return stu_mean, log_stu_covariance

    def get_knowledge_status(self, stu_id):
        stu_mean = self.student_emb_mean(stu_id)  
        log_stu_covariance = self.student_emb_covariance(stu_id)
        stu_emb = torch.sigmoid(stu_mean)

        return stu_emb.data, log_stu_covariance.data

    def get_exer_params(self, exer_id):
        k_difficulty = torch.sigmoid(self.k_difficulty(exer_id))
        e_discrimination = torch.sigmoid(self.e_discrimination(exer_id))
        return k_difficulty.data, e_discrimination.data


class NoneNegClipper(object):
    # 取正

    def __init__(self):
        super(NoneNegClipper, self).__init__()

    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            a = torch.relu(torch.neg(w))
            w.add_(a)
