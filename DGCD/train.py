import os
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import random
from dataloader import TrainDataLoader
from model import DGCD
import argparse

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='data/SLPbio_', type=str, help='')
    parser.add_argument('--num_stu', default=3922, type=int, help='num_stu')
    parser.add_argument('--num_exer', default=120, type=int, help='num_exer')
    parser.add_argument('--num_class', default=145, type=int, help='num_class')
    parser.add_argument('--num_train_epochs', default=100, type=int, help='')
    parser.add_argument('--num_skill', default=21, type=int, help='')
    parser.add_argument('--lr', default=0.001, type=float, help='')
    parser.add_argument('--batch_size', default=256, type=int, help='')
    parser.add_argument('--t', default=0.77, type=float, help='')
    parser.add_argument('--reg_r', default=1e-4, type=float, help='')
    parser.add_argument('--kl_r', default=1e-6, type=float, help='')
    return parser.parse_args()


device = torch.device('cuda:0')

result_r_m = [[[1.0],[1.0]] for i in range(5)]
def train():
    data_loader = TrainDataLoader(data_path)
    net = DGCD(class_n,student_n,exer_n,skill_n,t)

    net = net.to(device)
    optimizer = optim.Adam(net.parameters(),lr=learn,weight_decay=reg_r)
    print('training model...')
    loss_function = nn.MSELoss()
    rmsem = 1.0
    maem = 1.0
    for epoch in range(epochs):
        data_loader.reset()
        running_loss = 0.0
        count = 1
        while not data_loader.is_end():
            edge_t,edge_f,kn_emb,stu_list,class_id,exer_list,labels= data_loader.next_batch()
            batch_size = bsize
            start = 0
            edge_t,edge_f,kn_emb, stu_list, class_id, exer_list,labels = edge_t.to(device),edge_f.to(device),kn_emb.to(device),stu_list.to(device),class_id.to(device),exer_list.to(device),\
                                                                            labels.to(device)
            while (start + batch_size) <= len(exer_list):
                optimizer.zero_grad()
                output0_1,kl_loss = net.forward(edge_t,edge_f,class_id, stu_list, kn_emb, exer_list,None)
                output1_1 = output0_1.reshape([-1])
                output_1 = output1_1[start:(start+batch_size)]
                loss= loss_function(output_1, labels[start:(start+batch_size)]) + args.kl_r * kl_loss
                loss.backward()
                optimizer.step()
                net.apply_clipper()
                running_loss += loss.item()
                start = start + batch_size
                count = count + 1
                if count%100 == 0 :
                    print('[%d,%5d] loss: %f'%(epoch+1,count,running_loss/batch_size))
                    running_loss = 0.0
                   
            if start < len(exer_list):
                optimizer.zero_grad()
                output0_1,kl_loss = net.forward(edge_t,edge_f,class_id, stu_list, kn_emb, exer_list,None)
                output1_1 = output0_1.reshape([-1])
                output_1 = output1_1[start:len(exer_list)]
                loss= loss_function(output_1, labels[start:len(exer_list)]) + args.kl_r * kl_loss
                loss.backward()
                optimizer.step()
                net.apply_clipper()
                running_loss += loss.item()
                count = count + 1
                if count%100 == 0 :
                    print('[%d,%5d] loss: %f'%(epoch+1,count,running_loss/batch_size))
                    running_loss = 0.0

if __name__ == '__main__':
    args = set_args()
    knowledge_n = args.num_skill
    student_n = args.num_stu
    class_n = args.num_class
    exer_n = args.num_exer
    epochs = args.num_train_epochs
    skill_n = args.num_skill
    learn = args.lr
    data_path = args.data_path
    bsize = args.batch_size
    t = args.t
    reg_r = args.reg_r
    train()
