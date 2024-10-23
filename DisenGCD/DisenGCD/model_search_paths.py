import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from torch import Tensor

class Op(nn.Module):
    '''
    operation for one link in the DAG search space
    '''

    def __init__(self, k):
        super(Op, self).__init__()
        self.k = k   # nc: k=1   lr: k=2


    def forward(self, x, adjs, ws):
        num_op = len(ws)
        num = int(num_op//self.k)
        idx = random.sample(range(num_op),num)
        return sum(ws[i] * torch.spmm(adjs[i], x) for i in range(num_op) if i in idx) / num #self.k  #num   #self.k


class Cell(nn.Module):
    '''
    the DAG search space
    '''
    def __init__(self, n_step, n_hid_prev, n_hid, cstr, k, use_norm = True, use_nl = True, ratio = 1):
        super(Cell, self).__init__()
        
        self.affine = nn.Linear(n_hid_prev, n_hid)
        self.n_step = n_step               #* number of intermediate states (i.e., K)
        self.norm = nn.LayerNorm(n_hid, elementwise_affine = False) if use_norm is True else lambda x : x
        self.use_nl = use_nl
        assert(isinstance(cstr, list))
        self.cstr = cstr                   #* type constraint
        self.ratio = ratio
        op = Op(k)

        self.ops_seq = nn.ModuleList()     #* state (i - 1) -> state i, 1 <= i < K,  AI,  seq: sequential
        for i in range(1, self.n_step):
            self.ops_seq.append(op)
        self.ops_res = nn.ModuleList()     #* state j -> state i, 0 <= j < i - 1, 2 <= i < K,  AIO,  res: residual
        for i in range(2, self.n_step):
            for j in range(i - 1):
                self.ops_res.append(op)

        self.last_seq = op               #* state (K - 1) -> state K,  /hat{A}
        self.last_res = nn.ModuleList()    #* state i -> state K, 0 <= i < K - 1,  /hat{A}IO
        for i in range(self.n_step - 1):
            self.last_res.append(op)


    

    def forward(self, x, adjs, ws_seq, ws_res):
        #assert(isinstance(ws_seq, list))
        #assert(len(ws_seq) == 2)

        x = self.affine(x)
        states = [x]
        offset = 0
        edge = 1
        for i in range(self.n_step - 1):
            seqi = self.ops_seq[i](states[i], adjs[:-1], ws_seq[0][i])   #! exclude zero Op
            resi = sum(self.ops_res[offset + j](h, adjs, ws_res[0][offset + j]) for j, h in enumerate(states[:i]))
            offset += i
            states.append((seqi + self.ratio * resi)/edge)
        #assert(offset == len(self.ops_res))

        adjs_cstr = [adjs[i] for i in self.cstr]
        out_seq = self.last_seq(states[-1], adjs_cstr, ws_seq[1])

        adjs_cstr.append(adjs[-1])
        out_res = sum(self.last_res[i](h, adjs_cstr, ws_res[1][i]) for i, h in enumerate(states[:-1]))
        output = self.norm((out_seq + self.ratio * out_res)/edge)
        if self.use_nl:
            output = F.gelu(output)
        return output


class Model_paths(nn.Module):

    def __init__(self,gpu, in_dim, n_hid, num_node_types, n_adjs, n_classes, n_steps, ratio, cstr, k, lambda_seq, lambda_res, attn_dim = 64, use_norm = True, out_nl = True):
        super(Model_paths, self).__init__()
        self.device = torch.device(('cuda:%d' % (gpu)) if torch.cuda.is_available() else 'cpu')
        self.num_node_types = num_node_types
        self.cstr = cstr  
        self.n_adjs = n_adjs  
        self.n_hid = n_hid   
        self.ws = nn.ModuleList()          #* node type-specific transformation
        self.lambda_seq = lambda_seq
        self.lambda_res = lambda_res
        for i in range(num_node_types): 
            self.ws.append(nn.Linear(in_dim, n_hid))  
        assert(isinstance(n_steps, list))  #* [optional] combine more than one meta data?
        self.metas = nn.ModuleList()
        for i in range(len(n_steps)):  
            self.metas.append(Cell(n_steps[i], n_hid, n_hid, cstr, k, use_norm = use_norm, use_nl = out_nl, ratio = ratio))  # self.metas contions 1 Cell

        self.as_seq = []                   #* arch parameters for ops_seq    k<K and i=k-1   AI
        self.as_last_seq = []              #* arch parameters for last_seq   k=K and i=k-1  /hat{A}
        for i in range(len(n_steps)):
            if n_steps[i] > 1:  # not for
                ai = 1e-3 * torch.randn(n_steps[i] - 1, (n_adjs - 1))   #! exclude zero Op   torch.randn(3, 5)  AI
                ai = ai.to(self.device)
                ai.requires_grad_(True)
                self.as_seq.append(ai)
            else:
                self.as_seq.append(None)
            ai_last = 1e-3 * torch.randn(len(cstr))  # torch.randn(2)  edge related to the evaluation  /hat{A}   actually /hat{A} I
            ai_last = ai_last.to(self.device)
            ai_last.requires_grad_(True)
            self.as_last_seq.append(ai_last)

        ks = [sum(1 for i in range(2, n_steps[k]) for j in range(i - 1)) for k in range(len(n_steps))]
        self.as_res = []                  #* arch parameters for ops_res    k<K and i<k-1    AIO
        self.as_last_res = []             #* arch parameters for last_res   k=K and i<k-1    /hat{A}IO
        for i in range(len(n_steps)):
            if ks[i] > 0:
                ai = 1e-3 * torch.randn(ks[i], n_adjs)  # (3,6)  AIO
                ai = ai.to(self.device)
                ai.requires_grad_(True)
                self.as_res.append(ai)
            else:
                self.as_res.append(None)
            
            if n_steps[i] > 1:
                ai_last = 1e-3 * torch.randn(n_steps[i] - 1, len(cstr) + 1) 
                ai_last = ai_last.to(self.device)
                ai_last.requires_grad_(True)
                self.as_last_res.append(ai_last)
            else:
                self.as_last_res.append(None)

        assert(ks[0] + n_steps[0] + (0 if self.as_last_res[0] is None else self.as_last_res[0].size(0)) == (1 + n_steps[0]) * n_steps[0] // 2)


        #* [optional] combine more than one meta data?
        self.attn_fc1 = nn.Linear(n_hid, attn_dim) 
        self.attn_fc2 = nn.Linear(attn_dim, 1)  

        self.classifier = nn.Linear(n_hid, n_classes)

    def forward(self, node_feats, node_types, adjs):
        hid = torch.zeros((node_types.size(0), self.n_hid)).to(self.device)
        for i in range(self.num_node_types):
            idx = (node_types == i)
            hid[idx] = self.ws[i](node_feats[idx])
        temps = []
        attns = []
        for i, meta in enumerate(self.metas):
            ws_seq = []
            ws_seq.append(None if self.as_seq[i] is None else F.softmax(self.as_seq[i], dim=-1))  # softmax here
            ws_seq.append(F.softmax(self.as_last_seq[i], dim=-1)) 
            ws_res = []
            ws_res.append(None if self.as_res[i] is None else F.softmax(self.as_res[i], dim=-1))
            ws_res.append(None if self.as_last_res[i] is None else F.softmax(self.as_last_res[i], dim=-1))
            hidi = meta(hid, adjs, ws_seq, ws_res)  # cell
            temps.append(hidi)  
            attni = self.attn_fc2(torch.tanh(self.attn_fc1(temps[-1]))) # attni.shape   
            attns.append(attni)

        hids = torch.stack(temps, dim=0).transpose(0, 1)  
        attns = F.softmax(torch.cat(attns, dim=-1), dim=-1) 
        out = (attns.unsqueeze(dim=-1) * hids).sum(dim=1)  # attns.unsqueeze(dim=-1) * hids 
        logits = self.classifier(out) 
        return logits


    def alphas(self):
        alphas = []
        for each in self.as_seq:
            if each is not None:
                alphas.append(each)
        for each in self.as_last_seq:
            alphas.append(each)
        for each in self.as_res:
            if each is not None:
                alphas.append(each)
        for each in self.as_last_res:
            if each is not None:
                alphas.append(each)

        return alphas

    def getid(self, seq_res, lam):
        seq_softmax = None if seq_res is None else F.softmax(seq_res, dim=-1)

        length = seq_res.size(-1)
        if len(seq_res.shape) == 1:
            max = torch.max(seq_softmax, dim=0).values
            min = torch.min(seq_softmax, dim=0).values
            threshold = lam * max + (1 - lam) * min
            return [k for k in range(length) if seq_softmax[k].item()>=threshold]
        max = torch.max(seq_softmax, dim=1).values
        min = torch.min(seq_softmax, dim=1).values
        threshold = lam * max + (1 - lam) * min
        res = [[k for k in range(length) if seq_softmax[j][k].item() >= threshold[j]] for j in range(len(seq_softmax))]
        return res

    def sample_final(self, eps):
        '''
        to sample one candidate edge type per link
        '''
        idxes_seq = []
        idxes_res = []
        if np.random.uniform() < eps:
            for i in range(len(self.metas)): 
                temp = []
                temp.append(None if self.as_seq[i] is None else torch.randint(low=0, high=self.as_seq[i].size(-1), size=self.as_seq[i].size()[:-1]).to(self.device))
                temp.append(torch.randint(low=0, high=self.as_last_seq[i].size(-1), size=(1,)).to(self.device))
                idxes_seq.append(temp)
            for i in range(len(self.metas)):
                temp = []
                temp.append(None if self.as_res[i] is None else torch.randint(low=0, high=self.as_res[i].size(-1), size=self.as_res[i].size()[:-1]).to(self.device))  # self.as_res[0]: shape [3,6]   high:  6   size :3
                temp.append(None if self.as_last_res[i] is None else torch.randint(low=0, high=self.as_last_res[i].size(-1), size=self.as_last_res[i].size()[:-1]).to(self.device)) # self.as_last_res[0]: shape [3,3]   high:  3   size :3
                idxes_res.append(temp)
        else:
            for i in range(len(self.metas)):
                temp = []
                seq = self.getid(self.as_seq[i], self.lambda_seq)
                last_seq = self.getid(self.as_last_seq[i], self.lambda_seq)
                temp.append(seq)
                temp.append(last_seq)
                idxes_seq.append(temp)

            for i in range(len(self.metas)):
                temp = []
                res = self.getid(self.as_res[i], self.lambda_res)
                last_res = self.getid(self.as_last_res[i], self.lambda_res)
                temp.append(res)
                temp.append(last_res)
                idxes_res.append(temp)
        return idxes_seq, idxes_res

    
    def parse(self):
        '''
        to derive a meta data indicated by arch parameters
        '''
        idxes_seq, idxes_res = self.sample_final(0.)

        msg_seq = []; msg_res = []
        for i in range(len(idxes_seq)):
            map_seq = [[self.cstr[item] for item in idxes_seq[i][1]]]
            msg_seq.append(map_seq if idxes_seq[i][0] is None else idxes_seq[i][0] + map_seq) #idxes_seq[0][0]+idxes_seq[0][1]

            assert(len(msg_seq[i]) == self.metas[i].n_step)
            temp_res = []
            if idxes_res[i][1] is not None:
                for res in idxes_res[i][1]:
                    temp = []
                    for item in res:
                        if item < len(self.cstr):
                            temp.append(self.cstr[item])
                        else:
                            assert(item == len(self.cstr))
                            temp.append(self.n_adjs - 1)
                    temp_res.append(temp)
                if idxes_res[i][0] is not None:
                    temp_res = idxes_res[i][0] + temp_res   # idxes_res[0][0]+idxes_res[0][1]
            assert(len(temp_res) == self.metas[i].n_step * (self.metas[i].n_step - 1) // 2)
            msg_res.append(temp_res)
        

        return msg_seq, msg_res

