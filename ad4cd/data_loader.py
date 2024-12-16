import json
import sys
import torch
from torch.utils.data import Dataset
import pandas as pd
from util.getGraph import Graph

dataset = sys.argv[1]
k_fold = sys.argv[2]  # 0-4

with open('data/' + dataset + '/num.json') as a:
    b = json.load(a)
    exer_n = b["problem_n"]
    knowledge_n = b["skill_n"]
    student_n = b["user_n"]
    time_graph = Graph(pd.read_csv(f"data/{dataset}/{dataset}.csv"))


class MyDataSet(Dataset):
    def __init__(self, name):
        assert name in ["train", "valid", "test"]
        if name == "test":
            self.csv = pd.read_csv(f"data/{dataset}/test.csv")
        if name == "valid":
            self.csv = pd.read_csv(f"data/{dataset}/train_{k_fold}.csv")
        if name == "train":
            arr = ["0", "1", "2", "3", "4"]
            arr.remove(k_fold)
            csv_arr = []
            for i in arr:
                csv_arr.append(pd.read_csv(f"data/{dataset}/train_{i}.csv"))
            self.csv = pd.concat(csv_arr, ignore_index=True)

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        data = self.csv.iloc[[index]]
        studentId = data["user_id"][index]
        problemId = data["problem_id"][index]
        skill_index = data["skill_id"][index]
        skill_num_arr = [0] * knowledge_n
        skill_num_arr[skill_index] = 1
        skill_num = torch.LongTensor(skill_num_arr)
        labels = data["correct"][index]
        time_taken = data["timeTaken"][index]
        return studentId, problemId, skill_num, labels, time_taken, skill_index
