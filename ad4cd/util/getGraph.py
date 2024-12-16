import numpy as np
import torch
from pandas import DataFrame


class Graph:
    def __init__(self, df: DataFrame):
        df = df[["user_id", "problem_id", "timeTaken"]]
        self.graph = df

    def get_all_problem_time(self, stu_id, time):
        data = self.graph[self.graph["user_id"] == stu_id.item()]
        data = data["timeTaken"].values
        index = np.where(data == time.item())[0]
        assert len(index) != 0
        index = index[0]
        return torch.tensor(data), index

    def get_all_student_time(self, exe_id, time):
        data = self.graph[self.graph["problem_id"] == exe_id.item()]
        data = data["timeTaken"].values
        index = np.where(data == time.item())[0]
        assert len(index) != 0
        index = index[0]
        return torch.tensor(data), index
