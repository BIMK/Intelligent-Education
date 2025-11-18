import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from DiffModel import DiffCDR
from utils import get_dataset_information
import random
from argparse import ArgumentParser
import pandas as pd
from utils import normal_initialization
from module.layers import SeqPoolingLayer
from tqdm import tqdm
import json
from torch.utils.data import Dataset
import os
import numpy as np
os.environ['cuda_LAUNCH_BLOCKING'] = "1"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def batch_fn(many_batch_dict):
    max_length = 50
    sorted_batch = sorted(many_batch_dict, key=lambda x: len(x['original_logdict']['problem_seq']), reverse=True)
    ori_len = []
    mod_len = []
    for x in sorted_batch:
        ori_len.append(len(x['original_logdict']['problem_seq']))
        mod_len.append(len(x['model_logdict']['problem_seq']))
    ori_seqs_length = torch.LongTensor(ori_len)
    mod_seqs_length = torch.LongTensor(mod_len)
    ori_problem_seqs_tensor = torch.zeros(len(sorted_batch), max_length).long()
    ori_skill_seqs_tensor = torch.zeros(len(sorted_batch), max_length).long()
    ori_correct_seqs_tensor = torch.full((len(sorted_batch), max_length), -1).long()
    mod_problem_seqs_tensor = torch.zeros(len(sorted_batch), max_length).long()
    mod_skill_seqs_tensor = torch.zeros(len(sorted_batch), max_length).long()

    ori_problem_seqs = [x['original_logdict']['problem_seq'] for x in sorted_batch]
    ori_skill_seqs = [x['original_logdict']['skill_seq'] for x in sorted_batch]
    ori_correct_seqs = [x['original_logdict']['correct_seq'] for x in sorted_batch]
    mod_problem_seqs = [x['model_logdict']['problem_seq'] for x in sorted_batch]
    mod_skill_seqs = [x['model_logdict']['skill_seq'] for x in sorted_batch]

    for idx, (ori_problem_seq, ori_skill_seq, ori_correct_seq, ori_seq_len, mod_problem_seq, mod_skill_seq,
              mod_seq_len) in enumerate(
        zip(ori_problem_seqs, ori_skill_seqs, ori_correct_seqs, ori_seqs_length, mod_problem_seqs, mod_skill_seqs,
            mod_seqs_length)):
        ori_problem_seqs_tensor[idx, :ori_seq_len] = torch.LongTensor(ori_problem_seq)
        ori_skill_seqs_tensor[idx, :ori_seq_len] = torch.LongTensor(ori_skill_seq)
        ori_correct_seqs_tensor[idx, :ori_seq_len] = torch.LongTensor(ori_correct_seq)
        mod_problem_seqs_tensor[idx, :mod_seq_len] = torch.LongTensor(mod_problem_seq)
        mod_skill_seqs_tensor[idx, :mod_seq_len] = torch.LongTensor(mod_skill_seq)

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
            'correct_seqs_tensor': [],
            'seqs_length': mod_seqs_length
        }
    }
    return return_dict

class Data_for_inference(Dataset):
    def __init__(self, a) -> None:
        super().__init__()
        self.a = a

    def __len__(self):
        return len(self.a)

    def __getitem__(self, index):
        return self.a[index]

class ConditionEncoder(nn.Module):
    def __init__(self, K) -> None:
        super().__init__()
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=2,
            dim_feedforward=256,
            dropout=0.5,
            activation='gelu',
            layer_norm_eps=1e-12,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=transformer_layer,
            num_layers=2,
        )
        self.condition_layer = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, K),
        )
        self.pooling_layer = SeqPoolingLayer('mean')
        self.tau = 1

    def forward(self, trm_input, src_mask, memory_key_padding_mask, src_seqlen):
        trm_out = self.encoder(
            src=trm_input,
            mask=src_mask,  
            src_key_padding_mask=memory_key_padding_mask,
        )
        trm_out = self.pooling_layer(trm_out, src_seqlen)  
        condition = self.condition_layer(trm_out)  
        condition = F.gumbel_softmax(condition, tau=self.tau, dim=-1)  
        self.condition4loss = condition
        self.tau = max(self.tau * 0.995, 0.1)
        return condition
class Generator(nn.Module):
    def __init__(self, dataset_name, K) -> None:
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=64,
            nhead=2,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=256,
            dropout=0.5,
            activation='gelu',
            layer_norm_eps=1e-12,
            batch_first=True,
        )
        self.K = K
        self.condition_linear = nn.Sequential(
            nn.Linear(64, 64 * K),
            nn.ReLU(),
            nn.Linear(64 * K, 64 * K)
        )
        self.dropout = nn.Dropout(0.5)
        self.position_embedding = torch.nn.Embedding(52, 64)
        self.condition_encoder = ConditionEncoder(K)
        self.device = 'cuda'
        self.apply(normal_initialization)
        self.load_pretrained(dataset_name)

    def load_pretrained(self, dataset_name):
        path = './dataset/' + dataset_name + '/ktmodel.pth'
        saved = torch.load(path, map_location='cpu')
        pretrained = saved['problem_emb.weight']
        pretrained = torch.cat([
            pretrained,
            nn.init.normal_(torch.zeros(2, 64), std=0.02)
        ])
        self.item_embedding = nn.Embedding.from_pretrained(pretrained, padding_idx=0, freeze=False)
        self.item_embedding_decoder = self.item_embedding

    def condition_mask(self, logits, src):
        mask = torch.zeros_like(logits, device=logits.device, dtype=torch.bool)
        mask = mask.scatter(-1, src.unsqueeze(-2).repeat(1, mask.shape[1], 1), 1)
        logits = torch.masked_fill(logits, ~mask, -torch.inf)
        return logits

    def forward(self, src, tgt, src_mask, tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                memory_key_padding_mask,
                src_seqlen,
                tgt_seqlen,
                ):
        position_ids = torch.arange(src.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.reshape(1, -1)
        src_position_embedding = self.position_embedding(position_ids)
        src_emb = self.dropout(self.item_embedding(src) + src_position_embedding)

        memory = self.transformer.encoder(src_emb, src_mask, src_padding_mask)
        B, L, D = memory.shape
        memory = self.condition_linear(memory).reshape(B, L, self.K, D)

        position_ids = torch.arange(tgt.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.reshape(1, -1)
        tgt_position_embedding = self.position_embedding(position_ids)
        tgt_emb = self.dropout(self.item_embedding(tgt) + tgt_position_embedding)

        condition = self.condition_encoder(tgt_emb, tgt_mask, tgt_padding_mask, tgt_seqlen)  # BK
        condition = condition.reshape(B, 1, self.K, 1)
        memory_cond = (memory * condition).sum(-2)

        outs = self.transformer.decoder(tgt_emb, memory_cond, tgt_mask, None, tgt_padding_mask, memory_key_padding_mask)

        logits = outs @ self.item_embedding_decoder.weight.T
        logits = self.condition_mask(logits, src)

        return logits

    def encode(self, src, src_mask):
        position_ids = torch.arange(src.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.reshape(1, -1)
        src_position_embedding = self.position_embedding(position_ids)

        item_embedd = self.item_embedding(src)
        src_emb = self.dropout(item_embedd + src_position_embedding)

        return self.transformer.encoder(src_emb, src_mask)

    def set_condition(self, condition):
        self.condition = condition

    def decode(self, tgt, memory, tgt_mask):
        B, L, D = memory.shape
        memory = self.condition_linear(memory).reshape(B, L, self.K, D)[:, :, self.condition]
        position_ids = torch.arange(tgt.size(1), dtype=torch.long, device=self.device)
        position_ids = position_ids.reshape(1, -1)
        tgt_position_embedding = self.position_embedding(position_ids)
        tgt_emb = self.dropout(self.item_embedding(tgt) + tgt_position_embedding)

        return self.transformer.decoder(tgt_emb, memory, tgt_mask)
    
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device='cuda')) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, -100000).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = generate_square_subsequent_mask(src_seq_len)

    src_padding_mask = (src == 0)
    tgt_padding_mask = (tgt == 0)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
def inference_mask_generative(logits, src, ys):
    mask = torch.ones_like(logits, device=logits.device, dtype=torch.bool)
    mask = mask.scatter(-1, ys, 0)
    logits = torch.masked_fill(logits, ~mask, -torch.inf)
    return logits


def inference_mask(logits, src, ys):
    mask = torch.zeros_like(logits, device=logits.device, dtype=torch.bool)
    mask = mask.scatter(-1, src, 1)
    mask = mask.scatter(-1, ys, 0)
    logits = torch.masked_fill(logits, ~mask, -torch.inf)
    return logits


def greedy_decode(model, y, src, src_mask, max_len, start_symbol):
    src = src.to('cuda')
    src_mask = src_mask.to('cuda')
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to('cuda')
    for i in range(max_len - 1):
        memory = memory.to('cuda')
        tgt_mask = (generate_square_subsequent_mask(ys.size(1))
                    .type(torch.bool)).to('cuda')
        out = model.decode(ys, memory, tgt_mask)
        prob = out[:, -1] @ model.item_embedding_decoder.weight.T
        if random.random() > y or i <= 1:
            prob = inference_mask(prob, src, ys)
        else:
            prob = inference_mask_generative(prob, src, ys)
        _, next_word = torch.max(prob, dim=-1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=1)
        if next_word == EOS:
            break
    return ys


def translate(model: torch.nn.Module, y, src):
    model.eval()
    model = model.to('cuda')
    src = src.reshape(1, -1)
    src = src.to('cuda')
    num_tokens = src.shape[1]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model, y, src, src_mask, max_len=25, start_symbol=SOS).flatten()
    return tgt_tokens

def process_sequence(seq, problem2skill):
    a = []
    problem_Seq = []
    skill_Seq = []
    correct_Seq = []
    prob_1 = 0.5
    for ele in seq:
        problem_Seq.append(ele)
        skill = problem2skill[str(int(ele))]
        skill_Seq.append(skill)
        random_number_python = random.random()
        if random_number_python < prob_1:
            correct_Seq.append(1)
        else:
            correct_Seq.append(0)
    a.append(problem_Seq)
    a.append(skill_Seq)
    a.append(correct_Seq)
    return a

def getProblem_Skill(seq, problem2skill):
    problem_Seq = []
    skill_Seq = []
    for ele in seq:
        if ele > len(problem2skill):
            continue
        problem_Seq.append(ele)
        skill = problem2skill[str(int(ele))]
        skill_Seq.append(skill)
    return problem_Seq, skill_Seq


def preprocess(seq):
    return torch.tensor([SOS] + seq + [EOS], device='cuda')


def f(logdict, correct, seqlen):
    problem = logdict['problem_seqs_tenso'].tolist()
    skill = logdict['skill_seqs_tensor'].tolist()
    d = []
    for p, s, c, l in zip(problem, skill, correct, seqlen):
        d.append([p[:l], s[:l], c])
    return d

def convert_with_dynamic_threshold(numbers, initial_threshold=0.5, decrement=0.003):
    initial_threshold=sum(numbers)/len(numbers)
    threshold=initial_threshold
    result = []
    for num in numbers:
        if num < threshold:
            result.append(0)
            threshold = max(0, threshold - decrement) 
        else:
            threshold = initial_threshold
            result.append(1)
    return result

def cont(ss):
    s=[]
    for i in ss:
        s+=i
    return s

def getdiffmodel(dataset_name):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=dataset_name, help='choose a dataset.')
    parser.add_argument('--model', type=str, default='dk', help='choose a model.')
    parser.add_argument('--device', type=str, default='cuda', help='choose a device.')
    parser.add_argument('--max_length', type=int, default=50, help='choose a value for max length.')
    parser.add_argument('--early_stop', type=int, default=5, help='number of early stop for AUC.')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size.')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden_size.')
    parser.add_argument('--embed_size', type=int, default=64, help='embed_size.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    params = parser.parse_args()
    model_len =16
    params.model_len = model_len
    path = './dataset/' + dataset_name + '/settings.json'
    dataset_information = get_dataset_information(dataset=dataset, max_length=['max_length'], path=path)
    params.skill_num = int(dataset_information['skill_num'])
    params.problem_num = int(dataset_information['problem_num'])
    params.sequence_num = int(dataset_information['sequence_num'])
    params.kernel_size = 4
    params.stride = 3
    params.dataset = dataset_name
    diff_model = DiffCDR(params)
    diff_path = './dataset/' + dataset_name + '/diffusion_model_pooling.pth'
    diff_model.load_state_dict(torch.load(diff_path))
    diff_model = diff_model.to(params.device)
    return diff_model


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root_path', type=str, default='./dataset/Assistment09',
                        help='The path to the dataset.')
    parser.add_argument('--ckpt_name', type=str, default="regenerator-", help='The name of pretrained regenerator')
    parser.add_argument('--answer_type', type=str, default="model", help='The name of pretrained regenerator')
    parser.add_argument('--model_name', type=str, default="diffusion", help='The name of pretrained regenerator')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('-K', type=int, default=5)
    args = parser.parse_args()
    decrement = 0.0001
    model_len=16
    set_seed(2026)
    t='TrainSet_Z_3in1'+str(decrement)+'_r_1.0.pth'
    out_path = os.path.join(args.root_path, t)
    max_length = 50
    import os
    train_set = set()
    dataset_name = args.root_path.split('/')[-1]
    num_item_dict = {
        'Assistment12': 50918,
        'Assistment17': 3162,
        'Assistment09': 17752,
    }
    dataset = args.root_path.split('/')[-1]
    num_item = num_item_dict[dataset]
    SOS = num_item
    EOS = num_item + 1
    model = Generator(dataset_name, 5).to('cuda')
    model.load_state_dict(torch.load(os.path.join(args.root_path, args.ckpt_name + str(args.K) + '.pth')))
    flag=t.split('_')[0]

    all_interaction_data = os.path.join(args.root_path, 'TrainSet/train_data.csv')
    
    all_interaction_data_df = pd.read_csv(all_interaction_data)
    problem_seq_list = all_interaction_data_df['problem_seq'].tolist()
    skill_seq_list = all_interaction_data_df['skill_seq']
    correct_seq_list = all_interaction_data_df['correct_seq']
    problem_seq_list = [list(map(int, seq.strip('[').strip(']').split(','))) for seq in problem_seq_list]
    skill_seq_list = [list(map(int, seq.strip('[').strip(']').split(','))) for seq in skill_seq_list]
    correct_seq_list = [list(map(eval, seq.strip('[').strip(']').split(','))) for seq in correct_seq_list]
    seqlist = [preprocess(_) for _ in problem_seq_list]
    problem2skillfile = os.path.join(args.root_path, 'problem2skill.json')
    with open(problem2skillfile, 'r') as f:
        problem2skill = json.load(f)
    filtered_sequences = []
    ori_logdict = []
    for i in zip(problem_seq_list, skill_seq_list, correct_seq_list):
        problem_seqs_tensor = torch.zeros(1, max_length).long()
        skill_seqs_tensor = torch.zeros(1, max_length).long()
        correct_seqs_tensor = torch.full((1, max_length), -1).long()
        problem_seqs_tensor[0][:len(i[0])] = torch.tensor(i[0])
        skill_seqs_tensor[0][:len(i[0])] = torch.tensor(i[1])
        correct_seqs_tensor[0][:len(i[0])] = torch.tensor(i[2])
        seqs_length = torch.zeros(1, 1).long()
        seqs_length[0] = torch.Tensor([len(i[0])])
        ori_logdict.append(
            {
                'problem_seqs_tensor': problem_seqs_tensor,
                'skill_seqs_tensor': skill_seqs_tensor,
                'correct_seqs_tensor': correct_seqs_tensor,
                'seqs_length': seqs_length
            }
        )
    diff_model=getdiffmodel(dataset_name)
    diff_model.eval()
    ori_logdict = ori_logdict
    idx = 0
    sequences=[]
    filtered_problem_set = []
    batch_problem_list = []
    batch_skill_list = []
    computation_sequence = []
    batch_porblem_tensor_set = []
    lengths = []
    train = []
    p_index=0
    for i in range(args.K):
        model.set_condition(i)
        index = 0
        for seq in tqdm(seqlist):
            rst = translate(model, 1.0, seq)
            seq = rst.tolist()[1:-1]
            if len(seq) < 5:
                continue
            else:
                seq = seq[:model_len]
                if tuple(seq) not in train_set:
                    train_set.add(tuple(seq))
                    train.append(seq)
                    problem, skill = getProblem_Skill(seq, problem2skill)
                    lengths.append(len(problem))
                    temp_problem = torch.zeros(1, model_len).long()
                    if len(problem) > model_len:
                        problem = problem[:model_len]
                        skill = skill[:model_len]
                    batch_problem_list.append(problem)
                    batch_skill_list.append(skill)
                    temp_problem[0][:len(problem)] = torch.Tensor(problem)
                    computation_sequence.append(ori_logdict[index])
                    batch_porblem_tensor_set.append(temp_problem)
                    if len(lengths) ==63:
                        problem_seqs_tensor=[]
                        skill_seqs_tensor=[]
                        correct_seqs_tensor=[]
                        for i in computation_sequence:
                            problem_seqs_tensor.append(i['problem_seqs_tensor'])
                            skill_seqs_tensor.append(i['skill_seqs_tensor'])
                            correct_seqs_tensor.append(i['correct_seqs_tensor'])
                        ori_log={
                            "problem_seqs_tensor": torch.cat(problem_seqs_tensor,dim=0),
                            'skill_seqs_tensor': torch.cat(skill_seqs_tensor,dim=0),
                            'correct_seqs_tensor': torch.cat(correct_seqs_tensor,dim=0)
                        }
                        computation_problem=torch.cat(batch_porblem_tensor_set,dim=0)
                        cond_emb, cond_mask = diff_model.getcond_emb(ori_log, computation_problem.to("cuda"),
                                                                     lengths, True)
                        h = diff_model.p_sample_loop(cond_emb, 'cuda')
                        correct_seq = diff_model.getPreds(h, computation_problem.to("cuda")).squeeze(-1)
                        correct_seq = correct_seq.float()
                        for j in range(63):
                            correct = convert_with_dynamic_threshold(correct_seq[j].tolist(), decrement=decrement)[:lengths[j]]
                            if sum(correct) == len(correct) or sum(correct) == 0:
                                continue
                            else:
                                if lengths[j]==len(correct):
                                    filtered_sequences.append([batch_problem_list[j], batch_skill_list[j], correct])
                                    idx += 1
                                else:
                                    continue
                            if idx % 3 == 0:
                                # To accelerate the computational efficiency, we concatenate the three sequences together for the calculation of the downstream model
                                pro = cont([filtered_sequences[idx - 3][0],filtered_sequences[idx - 2][0],filtered_sequences[idx - 1][0]])
                                sk = cont([filtered_sequences[idx - 3][1],filtered_sequences[idx - 2][1],filtered_sequences[idx - 1][1]])
                                co = cont([filtered_sequences[idx - 3][2],filtered_sequences[idx - 2][2],filtered_sequences[idx - 1][2]])
                                if (str(pro) not in filtered_problem_set ):
                                    sequences.append([pro, sk, co])
                                    print(pro)
                                    print(co)
                                    filtered_problem_set.append(str(pro))
                                    p_index+=1
                        batch_problem_list.clear()
                        batch_skill_list.clear()
                        batch_porblem_tensor_set.clear()
                        computation_sequence.clear()
                        lengths.clear()
            index += 1
    print(len(sequences))
    torch.save(sequences, out_path)
