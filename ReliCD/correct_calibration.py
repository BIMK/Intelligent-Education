import numpy as np
import torch
import torch.nn.functional as F
from utils import main

params = main()
device = params.device


# correctness history class
class History(object):
    def __init__(self, student_n, exer_n, knowledge_n):
        self.correctness = np.zeros((student_n, knowledge_n))
        self.max_response_num = np.ones((student_n, knowledge_n))

    # correctness update
    def correctness_update(self, input_stu_ids, correct_concept):

        # first
        # non_indx = torch.nonzero(correct_concept)
        # non_indx = non_indx.to(torch.device('cpu')).tolist()
        # non_indx = np.array(non_indx)
        # input_stu_ids = input_stu_ids.to(torch.device('cpu')).tolist()
        # input_stu_ids = np.array(input_stu_ids)
        # index = np.stack((input_stu_ids[non_indx[:,0]], non_indx[:,1]), axis=1)
        # self.correctness[index[:,0],index[:,1]] += 1
        # self.correctness[input_stu_ids[non_indx[:,0]]][non_indx[:,1]] += 1

        # second
        non_indx = torch.nonzero(correct_concept)
        for i in range(len(non_indx)):
            self.correctness[input_stu_ids[non_indx[i][0]]][non_indx[i][1]] += 1
        

    # max num update
    def max_response_num_update(self, input_stu_ids, kn_id):

        non_indx = torch.nonzero(kn_id)
        # first
        # non_indx = non_indx.to(torch.device('cpu')).tolist()
        # non_indx = np.array(non_indx)
        # input_stu_ids = input_stu_ids.to(torch.device('cpu')).tolist()
        # input_stu_ids = np.array(input_stu_ids)
        # index = np.stack((input_stu_ids[non_indx[:,0]], non_indx[:,1]), axis=1)

        # second
        for i in range(len(non_indx)):
            self.max_response_num[input_stu_ids[non_indx[i][0]]][non_indx[i][1]] += 1

    # correctness normalize (0 ~ 1) range
    def correctness_normalize(self, cum_correctness, stu_idx, kno_idx):

        data_min = self.correctness.min()
        data_max = float(self.max_response_num.max())


        return (cum_correctness - data_min) / (data_max - data_min)


    def get_target_margin(self, stu_idx1, stu_idx2, kno_idx1, kno_idx2):

        stu_idx1 = stu_idx1.to(torch.device('cpu')).tolist()
        stu_idx2 = stu_idx2.to(torch.device('cpu')).tolist()
        kno_idx1 = kno_idx1.to(torch.device('cpu')).tolist()
        kno_idx2 = kno_idx2.to(torch.device('cpu')).tolist()

        cum_correctness1 = np.sum(self.correctness[stu_idx1] * kno_idx1, axis=1) / np.sum(self.max_response_num[stu_idx1] * kno_idx1, axis=1)
        cum_correctness2 = np.sum(self.correctness[stu_idx2] * kno_idx2, axis=1) / np.sum(self.max_response_num[stu_idx2] * kno_idx2, axis=1)

        # make target pair
        n_pair = len(cum_correctness1)
        target1 = (cum_correctness1[:n_pair])
        target2 = (cum_correctness2[:n_pair])
        # calc target
        greater = np.array(target1 > target2, dtype='float')
        less = np.array(target1 < target2, dtype='float') * (-1)

        target = greater + less
        target = torch.tensor(target).to(device)
        # calc margin
        margin = np.abs(target1 - target2)
        margin = torch.tensor(margin).to(device)

        return target, margin

    
    def out_put(self, ):

        return self.correctness, self.max_response_num

