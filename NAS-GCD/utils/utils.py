import os,torch,logging

from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np






def write_txt(path_file,content):
    with open(path_file, "w") as file:
        file.write('{}'.format(content))

def get_dataset(config,Search = True):
    
    student_n,exer_n,knowledge_n,train_loader,test_loader = eval('get_{name}_dataset'.format(name =config.dataset[0]))(config)
    if Search:
        return student_n,exer_n,knowledge_n,train_loader, test_loader
    else:
        return student_n,exer_n,knowledge_n,train_loader, test_loader, test_loader


from utils.utils_dataset import get_dataset as get_datasetjunyi

def get_junyi_dataset(config):

    # train_set, valid_set, test_set ,user_n,item_n,knowledge_n = get_datasetjunyi(name='junyi',batch_size=config.batch_size,num_workers=config.num_workers)
    train_set, test_set ,user_n,item_n,knowledge_n = get_datasetjunyi(name='junyi',batch_size=config.batch_size,num_workers=config.num_workers)

    return  user_n,item_n,knowledge_n,train_set, test_set
def get_slp_dataset(config):

    train_set, test_set ,user_n,item_n,knowledge_n = get_datasetjunyi(name='slp',batch_size=config.batch_size,num_workers=config.num_workers)
    
    return  user_n,item_n,knowledge_n,train_set, test_set#valid_set不用？

def get_assistment2009_dataset(config):

    train_set, valid_set, test_set ,user_n,item_n,knowledge_n = get_datasetjunyi(name='assist',batch_size=config.batch_size,num_workers=config.num_workers)

    return  user_n,item_n,knowledge_n,train_set, valid_set, test_set

def get_bbk_dataset(config):

    train_set, test_set ,user_n,item_n,knowledge_n = get_datasetjunyi(name='bbk',batch_size=config.batch_size,num_workers=config.num_workers)

    return  user_n,item_n,knowledge_n,train_set,test_set

def get_Assistment_dataset(config):
    train_data = pd.read_csv("../comparison/EduCDM-main/EduCDM-main/data/a0910/train.csv")
    valid_data = pd.read_csv("../comparison/EduCDM-main/EduCDM-main/data/a0910/valid.csv")
    test_data = pd.read_csv("../comparison/EduCDM-main/EduCDM-main/data/a0910/test.csv")
    df_item = pd.read_csv("../comparison/EduCDM-main/EduCDM-main/data/a0910/item.csv")

    item2knowledge = {}
    knowledge_set = set()
    for i, s in df_item.iterrows():
        item_id, knowledge_codes = s['item_id'], list(set(eval(s['knowledge_code'])))
        item2knowledge[item_id] = knowledge_codes
        knowledge_set.update(knowledge_codes)

    user_n = np.max(train_data['user_id'])
    item_n = np.max([np.max(train_data['item_id']), np.max(valid_data['item_id']), np.max(test_data['item_id'])])
    knowledge_n = np.max(list(knowledge_set))


    def transform(user, item, item2knowledge, score, batch_size,num_workers):
        knowledge_emb = torch.zeros((len(item), knowledge_n))
        for idx in range(len(item)):
            knowledge_emb[idx][np.array(item2knowledge[item[idx]]) - 1] = 1.0

        data_set = TensorDataset(
            torch.tensor(user, dtype=torch.int64) - 1,  # (1, user_n) to (0, user_n-1)
            torch.tensor(item, dtype=torch.int64) - 1,  # (1, item_n) to (0, item_n-1)
            knowledge_emb,
            torch.tensor(score, dtype=torch.float32)
        )
        return DataLoader(data_set, batch_size=batch_size, shuffle=True,num_workers=num_workers)


    train_set, valid_set, test_set = [
        transform(data["user_id"], data["item_id"], item2knowledge, data["score"], config.batch_size, config.num_workers)
        for data in [train_data, valid_data, test_data]
]
    return  user_n,item_n,knowledge_n,train_set, valid_set, test_set











#---------------------------------------------------------
def setup_device(n_gpu_use):
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine, training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU\'s configured to use is {}, but only {} are available on this machine.".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:1' if n_gpu_use > 0 else 'cpu')
    # list_ids = list(range(n_gpu_use))#改
    list_ids = [1]
    return device, list_ids



# TODO
def count_parameters_in_MB(model):
    return sum([m.numel() for m in model.parameters()])/1e6

# DIR and log handle
def process_config(config):
    print(' *************************************** ')
    print(' The experiment name is {} '.format(config.exp_name))
    print(' *************************************** ')

    if not os.path.exists(config.exp_name):
        print('-----------------making experiment dir: \"{}\" -----------------'.format(config.exp_name))
        os.makedirs(config.exp_name)

    message = ''
    message += '           ----------------- Config ---------------\n'
    for k, v in sorted(vars(config).items()):
        comment = ''
        message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '           ----------------- End -------------------'
    print(message)

def save_log(config):
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(filename='{}/log.log'.format(config.exp_name), level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    message = '\n           '
    message += '----------------- Config ---------------\n'
    for k, v in sorted(vars(config).items()):
        comment = ''
        message += '{:>35}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '            ----------------- End -------------------'
    logging.info(message)
