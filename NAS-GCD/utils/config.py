import json
import argparse,os,logging

from datetime import datetime

from utils.utils import process_config,save_log,setup_device




class Settings(dict):
    """Experiment configuration options.

    Wrapper around in-built dict class to access members through the dot operation.

    Experiment parameters:
        "expt_name": Name/description of experiment, used for logging.
        "gpu_id": Available GPU ID(s)

        "train_filepath": Training set path
        "val_filepath": Validation set path
        "test_filepath": Test set path

        "num_nodes": Number of nodes in TSP tours
        "num_neighbors": Number of neighbors in k-nearest neighbor input graph (-1 for fully connected)

        "node_dim": Number of dimensions for each node
        "voc_nodes_in": Input node signal vocabulary size
        "voc_nodes_out": Output node prediction vocabulary size
        "voc_edges_in": Input edge signal vocabulary size
        "voc_edges_out": Output edge prediction vocabulary size

        "beam_size": Beam size for beamsearch procedure (-1 for disabling beamsearch)

        "hidden_dim": Dimension of model's hidden state
        "num_layers": Number of GCN layers
        "mlp_layers": Number of MLP layers
        "aggregation": Node aggregation scheme in GCN (`mean` or `sum`)

        "max_epochs": Maximum training epochs
        "val_every": Interval (in epochs) at which validation is performed
        "test_every": Interval (in epochs) at which testing is performed

        "batch_size": Batch size
        "batches_per_epoch": Batches per epoch (-1 for using full training set)
        "accumulation_steps": Number of steps for gradient accumulation (DO NOT USE: BUGGY)

        "learning_rate": Initial learning rate
        "decay_rate": Learning rate decay parameter
    """

    def __init__(self, config_dict):
        super().__init__()
        for key in config_dict:
            self[key] = config_dict[key]

    def __getattr__(self, attr):
        return self[attr]

    def __setitem__(self, key, value):
        return super().__setitem__(key, value)

    def __setattr__(self, key, value):
        return self.__setitem__(key, value)

    __delattr__ = dict.__delitem__


def get_default_config():
    """Returns default settings object.
    """
    return Settings(json.load(open("./configs/assist.json")))

def get_config(filepath):
    """Returns settings from json file.
    """
    config = get_default_config()
    config.update(Settings(json.load(open(filepath))))
    return config

def get_common_search_config():
    parser = argparse.ArgumentParser("Searching Settings of Evolutionary Algorithm")
    parser.add_argument("--seed",type=int, default=111, help="random seed")
    # dataset
    parser.add_argument("--data-dir", type=str, default='./data', help='data folder')
    # parser.add_argument("--dataset", type=list, default=['slp', 'junyi', 'assistment2009', 'bbk'], help="dataset for evaluation")
    # parser.add_argument("--dataset", type=list, default=['slp', 'junyi', 'bbk'], help="dataset for evaluation")
    parser.add_argument("--dataset", type=list, default=['bbk'], help="dataset for evaluation")
    # parser.add_argument("--dataset", type=list, default=['slp', 'bbk'], help="dataset for evaluation")
    # parser.add_argument("--dataset", type=list, default=['slp'], help="dataset for evaluation")
    # base model setting
    parser.add_argument("--Continue-path",type=str,default=None, help='the path of experiemts to be continued')
    parser.add_argument("--dropout", type=float, default=0.0, help='dropout rate for FFN and self-attention')
    parser.add_argument("--Num-Nodes", type=list, default=[2,4], help='dropout rate for FFN and self-attention')#原[2,4]
    #hyperparameters for training process
    parser.add_argument("--epochs", type=int, default=30, help="number of training/fine-tunning epochs")#改为30
    parser.add_argument("--batch-size", type=int, default=128, help="batch size")
    parser.add_argument("--lr", type=int, default=0.001, help="batch size")
    # GPU settings
    parser.add_argument("--num-workers", type=int, default=0, help="number of workers")
    parser.add_argument("--parallel_evaluation", type=bool, default=False, help="train solutions under parallel mode")
    parser.add_argument("--n-gpu", type=int, default=1, help="number of gpus to use")
    # Saving DIR setiing
    parser.add_argument('--exp-name', type=str, default="./experiment/main_results", help="experiment name")
    config = parser.parse_args()
    if config.Continue_path is None:
        config.exp_name = config.exp_name +'/'+ 'Search'+('_' + datetime.now().strftime('%y-%m-%d_%H-%M-%S'))+'/'
    else:
        config.exp_name = config.Continue_path+'/Continue/'

    config = eval("get_search_config")(config)
    # device
    config.device, config.device_ids = setup_device(config.n_gpu)

    process_config(config)
    save_log(config)

    return config

def get_search_config(config):
    config.epochs = 30
    config.lr = 0.002
    config.wd = 0.0
    return config

def get_junyi_search_config(config):
    config.epochs = 30
    config.lr = 0.002
    config.wd = 0.0
    return config

def get_slp_search_config(config):
    config.epochs = 30
    config.lr = 0.002
    config.wd = 0.0
    return config

def get_bbk_search_config(config):
    config.epochs = 30
    config.lr = 0.002
    config.wd = 0.0
    return config