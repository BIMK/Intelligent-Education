import torch
import json
class TrainDataLoader(object):

    def __init__(self,data_id='0',data_path=''):
        self.ptr = 0
        self.data1 = []

        data_file = data_path+'train.json'
        with open(data_file,encoding='utf8') as i_f:
            self.data1 = json.load(i_f)

    def next_batch(self):
        if self.is_end():
            return None,None,None,None,None,None
        log = self.data1[self.ptr]
        self.ptr = self.ptr + 1
        class_id = log['student_class_id']
        stu_list = log['stu_list']
        exer_list = log['exer_list']
        labels = torch.Tensor(log['score_list'])
        stu_exer_true = log['s_e_t']
        stu_exer_false = log['s_e_f']
        knowledge_code = log['skill_list']
        kn_emb = torch.Tensor(knowledge_code)
        return torch.LongTensor(stu_exer_true),torch.LongTensor(stu_exer_false),kn_emb,torch.tensor(stu_list),torch.tensor(class_id),torch.tensor(exer_list),labels

    def is_end(self):
        if self.ptr + 1 > len(self.data1):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0

class  ValTestDataLoader(object):
    def __init__(self,data_id='0',data_path=''):
        self.ptr = 0
        self.data1 = []
        
        data_file = data_path + 'test.json'
        with open(data_file,encoding='utf8') as i_f:
            self.data1 = json.load(i_f)

    def next_batch(self):
        if self.is_end():
            return None,None,None,None,None,None,None
        log = self.data1[self.ptr]
        self.ptr = self.ptr + 1
        class_id = log['student_class_id']
        stu_list = log['stu_list']
        exer_list = log['exer_list']
        exer_test = log['exer_test']
        labels = torch.Tensor(log['test_score'])
        stu_exer_true = log['s_e_t']
        stu_exer_false = log['s_e_f']
        knowledge_code = log['skill_list']
        kn_emb = torch.Tensor(knowledge_code)
        return torch.LongTensor(stu_exer_true),torch.LongTensor(stu_exer_false),kn_emb, torch.tensor(stu_list), torch.tensor(class_id), torch.tensor(exer_list), torch.tensor(exer_test),labels

    def is_end(self):
        if self.ptr + 1 > len(self.data1):
            return True
        else:
            return False

    def reset(self):
        self.ptr = 0