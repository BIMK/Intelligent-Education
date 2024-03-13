from Models.NASCDNetV2 import NASCDNet
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from sklearn.metrics import roc_auc_score,accuracy_score
import gc

from  utils.utils import get_dataset
import sys

import threading
lock =threading.Lock()


# def train_epoch(epoch_i,train_data, Model, loss_function, optimizer,user_feature,device="cpu",file=None):
def train_epoch(epoch_i,train_data, Model, loss_function, optimizer,device="cpu",file=None):
    Model.train()
    epoch_losses = []
    lock.acquire()
    all_user=[]
    for batch_idx,batch_data in enumerate(train_data):
        lock.release()
        user_id, item_id, knowledge_emb, y = batch_data
        user_id: torch.Tensor = user_id.to(device)
        item_id: torch.Tensor = item_id.to(device)
        knowledge_emb: torch.Tensor = knowledge_emb.to(device)
        y: torch.Tensor = y.to(device)
        # if user_feature:
        #     pred,user_state = Model([user_id, item_id, knowledge_emb],user_feature)
        #     all_user.extend(user_state.tolist())
        #     # pred=torch.Tensor(pred)
        #     # user_state=torch.Tensor(user_state)
        # # pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
        # else:
            # pred: torch.Tensor = Model([user_id, item_id, knowledge_emb],user_feature)
        pred: torch.Tensor = Model([user_id, item_id, knowledge_emb])

        loss = loss_function(pred.float(), y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.mean().item())
        if batch_idx%50==0:
            print("Train Epoch: {:03d} Batch: {:05d}/{:05d} Loss: {:.4f}"
                  .format(epoch_i, batch_idx, len(train_data), loss.item()),file=file,flush=True)
        lock.acquire()
    lock.release()
    # if user_feature:
    #     return all_user

# def Validation(epoch_i,test_data, Model, loss_function,user_feature,device="cpu",file=None):
def Validation(epoch_i,test_data, Model, loss_function,device="cpu",file=None):
    Model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        lock.acquire()
        for batch_data in test_data:
            lock.release()
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            y: torch.Tensor = y.to(device)
            # if user_feature:
            #     pred,user_state = Model([user_id, item_id, knowledge_emb],user_feature)
            # pred=torch.Tensor(pred)
            # user_state=torch.Tensor(user_state)
        # pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
            # else:
                # pred: torch.Tensor = Model([user_id, item_id, knowledge_emb],user_feature)
            pred: torch.Tensor = Model([user_id, item_id, knowledge_emb])

            loss = loss_function(pred, y)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

            lock.acquire()
        lock.release()

    auc  = roc_auc_score(y_true, y_pred)
    acc = accuracy_score(y_true, np.array(y_pred) >= 0.5)
    print("Validation Epoch: {:03d},  Loss: {:.4f} ACC: {:.4f}, AUC: {:.4f}"
          .format(epoch_i,  loss.item(), acc, auc),file=file,flush=True)
    return acc, auc



def solution_evaluation(settings):

    # get input arguments
    device,config,Dec, save_dir, f = settings
    student_n,exer_n,knowledge_n, \
    trainloader, valloader =get_dataset(config)# get_dataset(config,name)

    # args={'n_student': student_n, 'n_exercise': exer_n, 'n_concept':knowledge_n, 'dim':knowledge_n} # dim = 123 is number of knowledge_n这个没用吧
    args={'n_student': student_n, 'n_exercise': exer_n, 'n_concept':knowledge_n, 'dim':128} # dim = 123 is number of knowledge_n

    if not isinstance(device,list):
    #Parallel model
        device_ids = [device]
    else:
    #Serial model
        device_ids = device.copy()
        device = device[0]
    Model  = NASCDNet(args,Dec)
    Model = Model.to(device)

    loss_function = nn.BCELoss()
    # loss_function=nn.BCELoss(input.float(), label.int())
    optimizer = optim.Adam(Model.parameters(), lr=config.lr)


    # start training
    print("start training",file=f,flush=True)
    best_auc = 0.0
    best_acc=0.0
    count = 0
    # config.epochs=1
    for epoch_i in range(config.epochs):
        # if user_feature:
        #     user_state=train_epoch(epoch_i,trainloader, Model, loss_function, optimizer,user_feature,device=device,file=f)
        #     user=np.mean(np.mat(np.array(user_state)))
        # else:
        # train_epoch(epoch_i,trainloader, Model, loss_function, optimizer,user_feature,device=device,file=f)
        # acc, auc = Validation(epoch_i,valloader, Model, loss_function,user_feature,device=device,file=f)
        train_epoch(epoch_i,trainloader, Model, loss_function, optimizer,device=device,file=f)
        acc, auc = Validation(epoch_i,valloader, Model, loss_function,device=device,file=f)

        best = False
        if auc > best_auc:
            best_auc = auc
            best_acc = acc
            best = True
            count = 0
        else:
            count +=1
        if count>10:
            print('Early  stopping',file=f,flush=True)
            print('Early  stopping',file=sys.stdout)
            break


    # if user_feature:
    #     print('Best valid acc:{}, auc:{},user:{}'.format(best_acc, best_auc,user),file=f,flush=True)
    # else:
    print('Best valid acc:{}, auc:{}'.format(best_acc, best_auc),file=f,flush=True)

    torch.cuda.empty_cache()
    del trainloader,valloader
    gc.collect()
    # if user_feature:
    #     return best_acc,best_auc,user
    # else:
    return best_acc,best_auc

