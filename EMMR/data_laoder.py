import numpy as np
import pandas as pd
import json


class Dataset:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path, sep=',', usecols=[0, 1], names=["user", "item"])
        self.Mi = self.data["item"].value_counts()
        self.user_num = self.data['user'].max() + 1
        self.item_num = self.data['item'].max() + 1
        self.buy_record = self.data.groupby('user')['item'].apply(list)
        self.Si = None




