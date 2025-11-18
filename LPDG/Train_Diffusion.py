from torch.utils.data import Dataset, DataLoader
import os
import torch
import argparse
from tqdm import tqdm
from utils import get_dataset_information
from DiffModel import DiffCDR
from run import get_metrics
import torch.optim as optim
def set_seed(seed):
    import  numpy as np
    import  random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MyDataset(Dataset):
    def __init__(self, patterns) -> None:
        super().__init__()
        ori_problem_seqs = []
        ori_skill_seqs = []
        ori_correct_seqs = []
        mod_problem_seqs = []
        mod_skill_seqs = []
        mod_correct_seqs = []
        for ele in patterns:
            ori_problem_seqs.append(ele['ori_problem'])
            ori_skill_seqs.append(ele['ori_skill'])
            ori_correct_seqs.append(ele['ori_correct'])
            mod_problem_seqs.append(ele['model_problem'])
            mod_skill_seqs.append(ele['model_skill'])
            mod_correct_seqs.append(ele['model_correct'])
        self.ori_problem = ori_problem_seqs
        self.ori_skill = ori_skill_seqs
        self.ori_correct = ori_correct_seqs
        self.mod_problem = mod_problem_seqs
        self.mod_skill = mod_skill_seqs
        self.mod_correct = mod_correct_seqs

    def __len__(self):
        return len(self.ori_correct)

    def __getitem__(self, index):
        ele = {
            "original_logdict": {
                "problem_seq": self.ori_problem[index],
                "skill_seq": self.ori_skill[index],
                "correct_seq": self.ori_correct[index],
            },
            "model_logdict": {
                "problem_seq": self.mod_problem[index],
                "skill_seq": self.mod_skill[index],
                "correct_seq": self.mod_correct[index],
            }

        }
        return ele

set_seed(2026)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Assistment09', help='choose a dataset.')
parser.add_argument('--model', type=str, default='dk', help='choose a model.')
parser.add_argument('--device', type=str, default='cuda', help='choose a device.')
parser.add_argument('--max_length', type=int, default=50, help='choose a value for max length.')
parser.add_argument('--epoch_num', type=int, default=50, help='')
parser.add_argument('--early_stop', type=int, default=5, help='number of early stop for AUC.')
parser.add_argument('--k_fold', type=int, default=5, help='number of folds for cross_validation.')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay.')
parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden_size.')
parser.add_argument('--embed_size', type=int, default=64, help='embed_size.')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--K', type=int, default=1)
args = parser.parse_args()
root_path = './dataset/' + args.dataset
pair_path = os.path.join(root_path, 'pair_p_s_c.pth')
patterns = torch.load(pair_path)
train_ratio = 0.8
trian_patterns = patterns[:int(len(patterns) * train_ratio)]
test_patterns = patterns[int(len(patterns) * train_ratio):]
train_data = MyDataset(trian_patterns)
test_data = MyDataset(test_patterns)
model_len = 16
args.model_len = model_len

def batch_fn(many_batch_dict):
    max_length = 50
    sorted_batch = sorted(many_batch_dict, key=lambda x: len(x['original_logdict']['problem_seq']), reverse=True)
    ori_len = []
    mod_len = []
    for x in sorted_batch:
        ori_len.append(len(x['original_logdict']['problem_seq']))
        if len(x['model_logdict']['problem_seq']) > model_len:
            mod_len.append(model_len)
        else:
            mod_len.append(len(x['model_logdict']['problem_seq']))
    ori_seqs_length = torch.LongTensor(ori_len)
    mod_seqs_length = torch.LongTensor(mod_len)
    ori_problem_seqs_tensor = torch.zeros(len(sorted_batch), max_length).long()
    ori_skill_seqs_tensor = torch.zeros(len(sorted_batch), max_length).long()
    ori_correct_seqs_tensor = torch.full((len(sorted_batch), max_length), -1).long()
    mod_problem_seqs_tensor = torch.zeros(len(sorted_batch), model_len).long()
    mod_skill_seqs_tensor = torch.zeros(len(sorted_batch), model_len).long()
    mod_correct_seqs_tensor = torch.full((len(sorted_batch), model_len), -1).long()
    ori_problem_seqs = [x['original_logdict']['problem_seq'] for x in sorted_batch]
    ori_skill_seqs = [x['original_logdict']['skill_seq'] for x in sorted_batch]
    ori_correct_seqs = [x['original_logdict']['correct_seq'] for x in sorted_batch]
    mod_problem_seqs = [x['model_logdict']['problem_seq'] for x in sorted_batch]
    mod_skill_seqs = [x['model_logdict']['skill_seq'] for x in sorted_batch]
    mod_correct_seqs = [x['model_logdict']['correct_seq'] for x in sorted_batch]
    for idx, (
            ori_problem_seq, ori_skill_seq, ori_correct_seq, ori_seq_len, mod_problem_seq, mod_skill_seq,
            mod_correct_seq,
            mod_seq_len) in enumerate(
        zip(ori_problem_seqs, ori_skill_seqs, ori_correct_seqs, ori_seqs_length, mod_problem_seqs, mod_skill_seqs,
            mod_correct_seqs, mod_seqs_length)):
        ori_problem_seqs_tensor[idx, :ori_seq_len] = torch.LongTensor(ori_problem_seq)
        ori_skill_seqs_tensor[idx, :ori_seq_len] = torch.LongTensor(ori_skill_seq)
        ori_correct_seqs_tensor[idx, :ori_seq_len] = torch.LongTensor(ori_correct_seq)
        if mod_seq_len < model_len:
            mod_problem_seqs_tensor[idx, :mod_seq_len] = torch.LongTensor(mod_problem_seq)
            mod_skill_seqs_tensor[idx, :mod_seq_len] = torch.LongTensor(mod_skill_seq)
            mod_correct_seqs_tensor[idx, :mod_seq_len] = torch.LongTensor(mod_correct_seq)
        else:
            mod_problem_seqs_tensor[idx, :] = torch.LongTensor(mod_problem_seq[:model_len])
            mod_skill_seqs_tensor[idx, :] = torch.LongTensor(mod_skill_seq[:model_len])
            mod_correct_seqs_tensor[idx, :] = torch.LongTensor(mod_correct_seq[:model_len])

    return_dict = {
        "original_logdict": {
            'problem_seqs_tensor': ori_problem_seqs_tensor,
            'skill_seqs_tensor': ori_skill_seqs_tensor,
            'correct_seqs_tensor': ori_correct_seqs_tensor,
            'seqs_length': ori_seqs_length
        },
        "model_logdict": {
            'problem_seqs_tensor': mod_problem_seqs_tensor,
            'skill_seqs_tensor': mod_skill_seqs_tensor,
            'correct_seqs_tensor': mod_correct_seqs_tensor,
            'seqs_length': mod_seqs_length
        }
    }
    return return_dict

batchsize = 64
train_Dataloder = DataLoader(train_data, batch_size=batchsize, shuffle=True, drop_last=True, collate_fn=batch_fn)
test_Dataloder = DataLoader(test_data, batch_size=batchsize, shuffle=True, drop_last=True, collate_fn=batch_fn)
path = './dataset/' + args.dataset + '/settings.json'
dataset_information = get_dataset_information(dataset=args.dataset, max_length=['max_length'], path=path)
args.skill_num = int(dataset_information['skill_num'])
args.problem_num = int(dataset_information['problem_num'])
args.sequence_num = int(dataset_information['sequence_num'])
args.kernel_size = 4
args.stride = 3
diff_model = DiffCDR(args).to(args.device)
best = {
    'auc': 0.0,
    'parameters': ''
}

optimzer = optim.Adam(diff_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

for epoch in range(args.epoch_num):
    print(epoch)
    diff_model.train()
    for logdict in tqdm(train_Dataloder):
        diff_model.zero_grad()
        diffusion_loss = diff_model.diffusion_loss_fn(batchsize, logdict, True)
        diffusion_loss.backward()
        optimzer.step()
    predictions_list = list()
    labels_list = list()
    diff_model.eval()
    for logdict in tqdm(test_Dataloder):
        problem = logdict['model_logdict']['problem_seqs_tensor'].to("cuda")
        length = logdict['model_logdict']['seqs_length'].to("cuda")
        cond_emb, cond_mask = diff_model.getcond_emb(logdict['original_logdict'], problem, length, False)
        h = diff_model.p_sample_loop(cond_emb, 'cuda')
        preds = diff_model.getPreds(h, problem)[:, 1:].reshape(-1)
        label = logdict['model_logdict']['correct_seqs_tensor'][:, 1:].reshape(-1)
        mask = label > -1
        masked_labels = label[mask].float()
        masked_preds = preds[mask]
        predictions_list.append(torch.squeeze(masked_preds.detach()))
        labels_list.append(torch.squeeze(masked_labels.detach()))
    metrics = get_metrics(predictions_list, labels_list)
    print("metrics:", metrics)
    if best['auc'] < metrics['acc'] + metrics['auc']:
        best['parameters'] = diff_model.state_dict()
        best['auc'] = metrics['acc'] + metrics['auc']
torch.save(best['parameters'], './dataset/' + args.dataset + '/diffusion_model_pooling.pth')
