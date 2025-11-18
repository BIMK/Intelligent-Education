import torch
import torch.nn as nn
import math
from model.dk import dk
from dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver

noise_schedule = NoiseScheduleVP(schedule='linear')




def get_timestep_embedding(timesteps, embedding_dim: int):
    timesteps=timesteps.squeeze(-1)
    assert len(timesteps.shape) == 2
    assert embedding_dim % 2 == 0
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps[:, :, None] * emb[None, None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    assert emb.shape == torch.Size([timesteps.shape[0], timesteps.shape[1], embedding_dim])
    return emb

from utils import normal_initialization
from module.layers import SeqPoolingLayer
from torch.nn import MultiheadAttention
K=5


import torch.nn.functional as F

class DiffCDR(nn.Module):
    def __init__(self,args,num_steps=1000, diff_dim=64,input_dim =64,c_scale=0.5,diff_sample_steps=32,diff_task_lambda=0.1,diff_mask_rate=0.5):
        super(DiffCDR,self).__init__()
        self.args=args
        self.num_steps = num_steps
        self.betas = torch.linspace(1e-4,0.02 ,num_steps)
        self.alphas = 1-self.betas
        self.alphas_prod = torch.cumprod(self.alphas,0)
        self.alphas_prod_p = torch.cat([torch.tensor([1]).float(),self.alphas_prod[:-1]],0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_log = torch.log(1 - self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
        assert self.alphas.shape==self.alphas_prod.shape==self.alphas_prod_p.shape==self.alphas_bar_sqrt.shape==self.one_minus_alphas_bar_log.shape==self.one_minus_alphas_bar_sqrt.shape
        self.diff_dim = diff_dim
        self.input_dim = input_dim
        self.task_lambda = diff_task_lambda
        self.sample_steps = diff_sample_steps
        self.c_scale = c_scale
        self.mask_rate = diff_mask_rate

        self.linears = nn.ModuleList(
            [
                nn.Linear(input_dim,diff_dim),    
                nn.Linear(diff_dim,diff_dim) ,     
                nn.Linear(diff_dim,input_dim),  
            ]
        )
        
        self.step_emb_linear = nn.ModuleList(
            [   
                nn.Linear(diff_dim,input_dim),
            ]
        )

        self.cond_emb_linear = nn.ModuleList(
            [   
                nn.Linear(input_dim,input_dim),
            ]
        )
        self.num_layers = 1
        self.al_linear = nn.Linear(input_dim,input_dim,False)
        self.device=args.device
        self.load_ktmodel()
        
        self.out= nn.Sequential(
            torch.nn.Linear(self.args.hidden_size+self.args.embed_size, self.args.embed_size),
            torch.nn.Linear(self.args.embed_size, 1),
            nn.Dropout(p=0.25)
            )
        self.al_linear = nn.Linear(input_dim,input_dim,False)
        self.cond_Linear=nn.Linear(input_dim*2,input_dim)
        self.cond_attention=MultiheadAttention(embed_dim=input_dim, num_heads=8, dropout=0.25, batch_first=True)
        #-----------------------------------------------
        self.loss_function = torch.nn.BCELoss()
        self.kernel_size=args.kernel_size
        self.stride=args.stride

    def load_ktmodel(self,):
        path = './dataset/'+self.args.dataset+'/ktmodel.pth'
        ktmodel=dk(self.args)
        ktmodel.load_state_dict(torch.load(path))
        self.ktmodel=ktmodel.to(self.device)

    def forward(self, x,t,cond_embedding,cond_mask, ):
        for idx in range( self.num_layers):
            t_embedding = get_timestep_embedding(t ,self.diff_dim)
            t_embedding = self.step_emb_linear[idx](t_embedding)
            t_c_emb = t_embedding + cond_embedding*cond_mask.unsqueeze(-1)
            x = x + t_c_emb
            x = self.linears[0](x) 
            x = self.linears[1](x) 
            x = self.linears[2](x) 
        return x
        
    def get_al_emb(self,emb):
        return self.al_linear (emb)

    def getPreds(self,h,problem):
        problem=self.ktmodel.getskill(problem)
        preds=self.out(torch.cat([h,problem],dim=-1)) 
        m = torch.nn.Sigmoid()
        preds = m(preds)
        return preds
    
    def generate_key_padding_mask(self,lengths, seq_len):

        batch_size = lengths.size(0)
        range_tensor = torch.arange(seq_len, device=lengths.device).unsqueeze(0)
        lengths = lengths.expand(batch_size, seq_len)

        key_padding_mask = range_tensor >= lengths  
        return key_padding_mask.to(self.device)
    
    def getcond_emb(self,original_logdict,skill_seq,model_length,is_task):
        ori_ks=self.ktmodel.getks(original_logdict).to(self.device)
        ori_ks = ori_ks.transpose(1, 2)
        ori_ks=F.max_pool1d(ori_ks, kernel_size=self.kernel_size, stride=self.stride)
        ori_ks = ori_ks.transpose(1, 2)
        skill_emb=self.ktmodel.getskill(skill_seq).to(self.device)
        x=torch.cat([ori_ks,skill_emb],dim=-1)
        cond_mermey=self.cond_Linear(x)
        cond_mask = 1 * (torch.rand((cond_mermey.shape[0],cond_mermey.shape[1]),device=self.device) <= self.mask_rate)
        cond_mask = 1 - cond_mask.int()  
        return cond_mermey.to(self.device),cond_mask.to(self.device)
    

    def compute_masked_loss(self,x, y, lengths, loss_fn=F.mse_loss, beta=1.0):
        batch_size, seq_len, embed_size = x.size()
        mask = torch.arange(seq_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).expand(batch_size, seq_len, embed_size)
        x_masked = x * mask
        y_masked = y * mask
        loss = loss_fn(x_masked, y_masked,reduction='sum')
        valid_elements = mask.sum()  
        loss = loss / valid_elements
        
        return loss
    
    def q_x_fn(self,x_0,t,device):
        noise = torch.normal(0,1,size = x_0.size() ,device=device)
        alphas_t = self.alphas_bar_sqrt.to(device)[t]
        alphas_1_m_t = self.one_minus_alphas_bar_sqrt.to(device)[t]
        return (alphas_t * x_0 + alphas_1_m_t * noise),noise
    
    def diffusion_loss_fn(self,batchsize,log_dict,is_task):

        original_logdict=log_dict['original_logdict']
        model_logdict=log_dict['model_logdict']
        num_steps = self.num_steps
        skill_seq=model_logdict['problem_seqs_tensor'].to(self.device)
        mod_length=model_logdict['seqs_length'].to(self.device)
        if is_task:
            #sample t
            ori_ks=self.ktmodel.getks(original_logdict).to(self.device)
            ori_ks = ori_ks.transpose(1, 2)
            ori_ks=F.avg_pool1d(ori_ks, kernel_size=self.kernel_size, stride=self.stride)
            ori_ks = ori_ks.transpose(1, 2)
            cond_emb,cond_mask=self.getcond_emb(original_logdict,skill_seq,mod_length,is_task)

            t = torch.randint(0,num_steps,size=(batchsize//2,self.args.model_len),device=self.args.device)
            if batchsize%2 ==0:
                t = torch.cat([t,num_steps-1-t],dim=0)
            else:
                extra_t = torch.randint(0,num_steps,size=(1,),device=self.args.device)
                t = torch.cat([t,num_steps-1-t,extra_t],dim=0)
            t = t.unsqueeze(-1)
            x,e = self.q_x_fn(ori_ks,t,self.args.device)
            output = self(x,t,cond_emb,cond_mask)
            h=self.p_sample_loop(cond_emb,self.args.device)
            preds=self.getPreds(h,skill_seq)[:, 1:].reshape(-1)
            label=model_logdict['correct_seqs_tensor'][:, 1:].reshape(-1).to('cuda')
            mask = label > -1
            masked_labels = label[mask].float()
            masked_preds = preds[mask]
            labels = torch.as_tensor(masked_labels, dtype=torch.float)
            loss = self.loss_function(masked_preds, labels)
            return F.mse_loss(e, output)+1.0*loss

    def p_sample(self,cond_emb,x,device):
        classifier_scale_para = self.c_scale
        dmp_sample_steps = self.sample_steps
        num_steps = self.num_steps

        model_kwargs ={'cond_emb':cond_emb,
                    'cond_mask':torch.zeros((cond_emb.size()[0],cond_emb.shape[1]) ,device=device),
                    }


        model_fn = model_wrapper(
            self,
            noise_schedule,
            is_cond_classifier=True,
            classifier_scale = classifier_scale_para, 
            time_input_type="1",
            total_N=num_steps,
            model_kwargs=model_kwargs
        )

        dpm_solver = DPM_Solver(model_fn, noise_schedule)
        sample = dpm_solver.sample(
                        x,
                        steps=dmp_sample_steps,
                        eps=1e-4,
                        adaptive_step_size=False,
                        fast_version=True,
                    )
        
        return self.al_linear(sample)
    
    def p_sample_loop(self,cond_emb,device): 

        cur_x = cond_emb
        cur_x= self.p_sample(cond_emb,cur_x,device)

        return cur_x