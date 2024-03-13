import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score,mean_squared_error

from utils.CDM import CDM
from Models.NCDMNet import Net as NCDNet
# from Models.NASCDNet import NASCDNet
from Models.NASCDNetV2 import NASCDNet

class NASCD(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n,dec=None):
        super(NASCD, self).__init__()
        # self.ncdm_net = NCDNet(knowledge_n, exer_n, student_n)
        args={'n_student': student_n, 'n_exercise': exer_n, 'n_concept':knowledge_n, 'dim':knowledge_n}
        args={'n_student': student_n, 'n_exercise': exer_n, 'n_concept':knowledge_n, 'dim':128}

        if dec is None:

            # self.ncdm_net = NASCDNet(args,  NAS_dec=[0,1,5,  1,1,6, 4,0,5,  4,0,0, 0,6,7]) # Wrong NCD but better performance
            # [0, 0, 6, 1, 0, 9, 4, 0, 6, 4, 0, 0, 3, 6, 10] # under Genotype_mapping
            # self.ncdm_net = NASCDNet(args,  NAS_dec=[0,0,5,  1,0,5, 1,0,6, 5,0,5,   4,0,0, 0,7,7,  6,8,8]) # NCD  based on MixedOp

            # self.ncdm_net = NASCDNet(args,  NAS_dec=[1,0,8,  1,0,10, 0,3,12,  5,0,9, 6,0,0, 4,7,11, 8,0,7]) # MIRT wrong version but similar peformance
            # self.ncdm_net = NASCDNet(args,  NAS_dec=[1,0,8,  1,0,10, 0,3,12,  5,0,9, 4,0,0, 6,7,11, 8,0,7]) # MIRT

            # self.ncdm_net = NASCDNet(args,  NAS_dec=[1,0,10,  1,0,10, 1,0,10,  3,0,8, 5,0,7, 4,0,0,
            #                                          0,8,11, 6,9,12, 10,0,7, 7,11,12, 11,12,11, 7,13,11]) #IRT wrong version: theta dim is 128 but should be 1

            # self.ncdm_net = NASCDNet(args,  NAS_dec=[0,1,13,  3,0,7]) # MCD net version-1 with similar performance: concatLinear [256,1]
            # self.ncdm_net = NASCDNet(args,  NAS_dec=[0,1,13, 3,0,10, 4,0,7]) # MCD net: concatLinear + FFN
            # [0,1,12, 3,0,9, 4,0,6] # under Genotype_mapping


            #------------------------------ Genotype_mapping Model--------------
            # self.ncdm_net = NASCDNet(args,NAS_dec=[1, 0, 10, 3, 2, 10, 4, 0, 5, 5, 0, 4, 6, 0, 3, 7, 0, 1, 8, 0, 7]) # 0.7940（train）,0.7913(test)
            # self.ncdm_net = NASCDNet(args,NAS_dec= [0, 0, 7, 3, 0, 11, 4, 1, 10, 5, 0, 11, 6, 1, 10, 7, 2, 10, 8, 0,
            #                                         13, 9, 0, 3, 10, 0, 1, 11, 0, 2, 12, 0, 6, 13, 0, 2, 14, 0, 3, 15,
            #                                         0, 6, 16, 0, 10, 17, 0, 0, 18, 0, 2] ) # 0.8131（train）, 0.807054(test)

                #NCD
            self.ncdm_net = NASCDNet(args,NAS_dec=[1,0,6,3,0,0,0,0,6,4,5,10,2,6,11,1,0,9,8,0,6,7,9,11])
            # IRT
            self.ncdm_net = NASCDNet(args,NAS_dec=[1,0,9, 3,0,0, 0,0,9,  4,5,10, 1,0,9, 6,7,11, 8,0,6 ])   #参考论文  sigmoid(a(theta-beta))
            # MIRT
            self.ncdm_net = NASCDNet(args,NAS_dec=[0,2,11, 3,0,8, 1,0,9, 4,5,10, 6,0,6 ])
            #MCD
            self.ncdm_net = NASCDNet(args,NAS_dec=[0,1,12, 3,0,9, 4,0,6])

        else:

            self.ncdm_net = NASCDNet(args,NAS_dec=dec)


        #---------------------------Junyi
        # self.ncdm_net = NASCDNet(args,NAS_dec=[0, 0, 7, 3, 0, 13, 4, 1, 10])

        #---------------------------SLP
        # self.ncdm_net = NASCDNet(args,NAS_dec=[0,1,12, 3,0,9, 4,0,6])
        # self.ncdm_net = NASCDNet(args,NAS_dec=[0, 1, 12, 3, 1, 10, 4, 2, 11, 5, 0, 0, 6, 0, 10, 7, 0, 9, 8, 0, 6]) # epoch 5 slp()
        # self.ncdm_net = NASCDNet(args,NAS_dec=[0, 0, 0, 3, 0, 6, 4, 2, 12, 5, 1, 12, 6, 0, 3, 7, 2, 11, 8, 0, 10, 9, 0, 7, 10, 0, 13, 11, 0, 4] )#  0.855991
        # self.ncdm_net = NASCDNet(args,NAS_dec=[0, 0, 0, 3, 0, 6, 4, 2, 12, 5, 1, 12, 6, 0, 3, 7, 0, 0, 8, 0, 10, 9, 2, 11, 10, 0, 7, 11
        #     , 0, 1, 12, 0, 13, 13, 0, 3, 14, 0, 4, 15, 0, 1, 16, 0, 11, 17, 1, 11]  )#  auc: 0.857120


    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False):


        acc_list,auc_lis,rmse_list=[],[],[]

        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()
        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        # optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr,weight_decay=3e-6)
        is_best_auc = 0
        for epoch_i in range(epoch):
            epoch_losses = []
            batch_count = 0





            self.ncdm_net.train()




            for batch_data in tqdm(train_data, "Epoch %s" % epoch_i):
            # for batch_data in train_data:
                batch_count += 1
                user_id, item_id, knowledge_emb, y = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                # pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
                pred: torch.Tensor = self.ncdm_net([user_id, item_id, knowledge_emb])
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())
                # print('\r batch {}, loss {}'.format(batch_count, loss.item()),end='')

            print("[Epoch %d] average loss: %.6f" % (epoch_i, float(np.mean(epoch_losses))))

            if test_data is not None:
                auc, accuracy,rmse = self.eval(test_data, device=device)
                print("[Epoch %d] auc: %.6f, accuracy: %.6f, best_auc: %.6f" % (epoch_i, auc, accuracy,is_best_auc))
                acc_list.append(accuracy)
                auc_lis.append(auc)
                rmse_list.append(rmse)
            if auc >is_best_auc:
                is_best_auc = auc
                count = 0
            else:
                count +=1
            if count>10:
                print('Early  stopping')
                break
                

        index = np.argmax(auc_lis)
        return [acc_list[index], auc_lis[index],rmse_list[index]],[accuracy,auc,rmse]

    def eval(self, test_data, device="cpu"):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred = [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, knowledge_emb, y = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.ncdm_net([user_id, item_id, knowledge_emb])
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())

        return roc_auc_score(y_true, y_pred), accuracy_score(y_true, np.array(y_pred) >= 0.5),mean_squared_error(y_true, y_pred)

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        logging.info("load parameters from %s" % filepath)

