import sys
import json
import datetime
import logging

import torch
import numpy as np
import pandas as pd

sys.path.append('..')
import pyat

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

dataset = 'DS'
# read datasets
test_triplets = pd.read_csv(f'../datasets/PTADisc/{dataset}/test_20_sort.csv', encoding='utf-8').to_records(index=False)
concept_map = json.load(open(f'../datasets/PTADisc/{dataset}/concept_map_{dataset}.json', 'r'))
concept_map = {int(k):v for k,v in concept_map.items()}
dataInfo = json.load(open(f'../datasets/PTADisc/{dataset}/info.json', 'r'))

test_data = pyat.AdapTestDataset(test_triplets, concept_map,
                                 dataInfo['test_cnt'], dataInfo['problem_cnt'], dataInfo['concept_cnt'])
print(dataInfo['test_cnt'])
print(dataInfo['problem_cnt'])
config = {
    'learning_rate': 0.0025,
    'batch_size': 2048,
    'num_epochs': 10,
    'num_dim': 1,
    'device': 'cuda',
    'THRESHOLD': 300,
    'start': 0,
    'end': 3000
}

logging.basicConfig(
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S',
    format='[%(asctime)s %(levelname)s] %(message)s',
)

test_length = 5
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

# strategies = (pyat.RandomStrategy(), pyat.MAATStrategy(n_candidates=10), pyat.NCATs())
strategies = (pyat.NCATs(),)

for strategy in strategies:
    model = pyat.NCDModel(**config)
    model.adaptest_init(test_data,dataset)

    model.adaptest_preload(f'../models/ncd/{dataset}_80_train.pt')
    test_data.reset()
    if strategy.name == 'NCAT':
        selected_questions = strategy.adaptest_select(test_data, concept_map, config, test_length)
        for it in range(test_length):
            for student, questions in selected_questions.items():
                test_data.apply_selection(student, questions[it])
            model.adaptest_update(test_data)
            results = model.adaptest_evaluate(test_data)
            # log results
            logging.info(f'Iteration {it}')
            for name, value in results.items():
                logging.info(f'{name}:{value}')
        continue
    # pyat.AdapTestDriver.run(model, strategy, test_data, test_length, f'../results/{now}')
