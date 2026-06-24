import logging
from datetime import datetime

import torch
import torch.nn as nn
import math
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

import datetime
from pyat.utils.data import AdapTestDataset
import sys
import pyat
import json
import pandas as pd
import numpy as np
import pyat

noise_schedule = NoiseScheduleVP(schedule='linear')

sys.path.append('..')  

# ---------------------------------------------------------
def get_timestep_embedding(timesteps, embedding_dim: int):
    """
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    timesteps = timesteps.to(dtype=torch.float32)

    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    assert embedding_dim % 2 == 0
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    # emb = tf.range(num_embeddings, dtype=DEFAULT_DTYPE)[:, None] * emb[None, :]
    # emb = tf.cast(timesteps, dtype=torch.float32)[:, None] * emb[None, :]
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], axis=1)
    # if embedding_dim % 2 == 1:  # zero pad
    #    emb = torch.pad(emb, [0,1])
    assert emb.shape == torch.Size([timesteps.shape[0], embedding_dim])
    return emb


class DiffCDR(nn.Module):
    def __init__(self, num_steps=200, diff_dim=32, input_dim=32, c_scale=0.1, diff_sample_steps=30,
                 diff_task_lambda=0.1, diff_mask_rate=0.1):
        super(DiffCDR, self).__init__()

        # -------------------------------------------
        # define params
        self.num_steps = num_steps
        self.betas = torch.linspace(1e-4, 0.02, num_steps)

        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float(), self.alphas_prod[:-1]], 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        assert self.alphas.shape == self.alphas_prod.shape == self.alphas_prod_p.shape == \
               self.alphas_bar_sqrt.shape == self.one_minus_alphas_bar_log.shape \
               == self.one_minus_alphas_bar_sqrt.shape

        # -----------------------------------------------
        self.diff_dim = diff_dim
        self.input_dim = input_dim
        self.task_lambda = diff_task_lambda
        self.sample_steps = diff_sample_steps
        self.c_scale = c_scale
        self.mask_rate = diff_mask_rate
        # -----------------------------------------------

        self.linears = nn.ModuleList(
            [
                nn.Linear(input_dim, diff_dim),
                nn.Linear(diff_dim, diff_dim),
                nn.Linear(diff_dim, input_dim),
            ]
        )

        self.step_emb_linear = nn.ModuleList(
            [
                nn.Linear(diff_dim, input_dim),
            ]
        )

        self.cond_emb_linear = nn.ModuleList(
            [
                nn.Linear(input_dim, input_dim),
            ]
        )

        self.num_layers = 1

        # linear for stm
        self.al_linear = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 128),
            nn.Linear(128, input_dim),
        )

    def forward(self, x, t, cond_emb, cond_mask):
        for idx in range(self.num_layers):
            t_embedding = get_timestep_embedding(t, self.diff_dim)
            t_embedding = self.step_emb_linear[idx](t_embedding)
            cond_embedding = self.cond_emb_linear[idx](cond_emb)
            t_c_emb = t_embedding + cond_embedding * cond_mask.unsqueeze(-1)
            x = x + t_c_emb
            # x= torch.cat([t_embedding,cond_embedding * cond_mask.unsqueeze(-1),x],axis=1)
            x = self.linears[0](x)
            x = self.linears[1](x)
            x = self.linears[2](x)
        return x

    def get_al_emb(self, emb):
        return self.al_linear(emb)


# ---------------------------------------------------------
# loss
import torch.nn.functional as F


def q_x_fn(model, x_0, t, device):
    # eq(4)
    noise = torch.normal(0, 1, size=x_0.size(), device=device)

    alphas_t = model.alphas_bar_sqrt.to(device)[t]
    alphas_1_m_t = model.one_minus_alphas_bar_sqrt.to(device)[t]

    return (alphas_t * x_0 + alphas_1_m_t * noise), noise


def diffusion_loss_fn(model, x_0, cond_emb, y_input,
                      device, is_task, IRT_Model, data_log, rcd_data):
    num_steps = model.num_steps
    mask_rate = model.mask_rate

    if is_task == False:

        # ------------------------
        # sampling
        # ------------------------
        batch_size = x_0.shape[0]
        # sample t
        t = torch.randint(0, num_steps, size=(batch_size // 2,), device=device)
        if batch_size % 2 == 0:
            t = torch.cat([t, num_steps - 1 - t], dim=0)
        else:
            extra_t = torch.randint(0, num_steps, size=(1,), device=device)
            t = torch.cat([t, num_steps - 1 - t, extra_t], dim=0)
        t = t.unsqueeze(-1)

        x, e = q_x_fn(model, x_0, t, device)

        # random mask
        cond_mask = 1 * (torch.rand(cond_emb.shape[0], device=device) <= mask_rate)
        cond_mask = 1 - cond_mask.int()
        # pred noise
        output = model(x, t.squeeze(-1), cond_emb, cond_mask)

        return F.smooth_l1_loss(e, output)

    elif is_task:
        final_output = p_sample_loop(model, cond_emb, device)
        y_pred = final_output.tolist()

        file_path = f'../datasets/PTADisc/C/train_80_sort_Diff.csv' 
        data_res = pd.read_csv(file_path)
        data_log = pd.DataFrame(data_log.numpy(), columns=data_res.columns)
        score_pred = []
       
        item_id_column = data_log['item_id']
        user_id_colum = data_log['user_id']
        for i, pred_value in zip(item_id_column, y_pred):
           
            rcd_data._knowledge_embs = rcd_data._knowledge_embs.to(device)
            knowledge_embs = rcd_data._knowledge_embs[i]
            pred = IRT_Model.forward_diff(torch.tensor(pred_value), torch.LongTensor([i]), knowledge_embs)
            
            # alpha = torch.tensor(IRT_Model.get_alpha(i))
            # beta = torch.tensor(IRT_Model.get_beta(i))
            #
            
            # pred = (alpha * torch.tensor(pred_value)).sum() + beta
            #
           
            # pred = torch.sigmoid(pred)
            score_pred.append(pred)

       
        score_pred = torch.stack(score_pred)
        
        score = torch.tensor(data_log['score'].to_numpy())

        epsilon = 1e-7
        loss = -(score * torch.log(score_pred + epsilon) + (1 - score) * torch.log(1 - score_pred + epsilon)).mean()

        # MSE
        task_loss = (torch.tensor(y_pred) - y_input).square().mean()
        # RMSE
        # task_loss = (y_pred - y_input.squeeze().float()).square().sum().sqrt() / y_pred.shape[0]
        return F.smooth_l1_loss(x_0, final_output) + task_loss



# generation fun
def p_sample(model, cond_emb, device):
    # wrap for dpm_solver
    classifier_scale_para = model.c_scale
    dmp_sample_steps = model.sample_steps
    num_steps = model.num_steps

    model_kwargs = {'cond_emb': cond_emb,
                    'cond_mask': torch.zeros(cond_emb.size()[0], device=device),
                    }

    model_fn = model_wrapper(
        model,
        noise_schedule,
        is_cond_classifier=True,
        classifier_scale=classifier_scale_para,
        time_input_type="1",
        total_N=num_steps,
        model_kwargs=model_kwargs
    )

    dpm_solver = DPM_Solver(model_fn, noise_schedule)

    sample = dpm_solver.sample(
        cond_emb,
        steps=dmp_sample_steps,
        eps=1e-4,
        adaptive_step_size=False,
        fast_version=True,
    )
    return model.get_al_emb(sample).to(device)


def p_sample_loop(model, cond_emb, device):
    # source emb input
    # cur_x = cond_emb
    # noise input
    # cur_x = torch.normal(0,1,size = cond_emb.size() ,device=device)

    # reversing
    cur_x = p_sample(model, cond_emb, device)

    return cur_x
