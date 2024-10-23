import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
import time
import json
import sys
import pickle
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score
from DCGCD.data_loader import TrainDataLoader, ValTestDataLoader
from DCGCD.Diagnosis import Net
from DCGCD.util import CommonArgParser,construct_local_map
from DCGCD.preprocess import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor ,build_graph

def train(args,local_map):
    np.random.seed(args.seed) 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device(('cuda:%d' % (args.gpu)) if torch.cuda.is_available() else 'cpu')

    train_loader = TrainDataLoader()
    val_loader = ValTestDataLoader('predict')
    print(device)
    with open(("./data/bbk/82/edges.pkl"), "rb") as f: 
        edges = pickle.load(f) 
        f.close()
    adjs_pt = []
    for mx in edges:
        adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
            normalize_row(mx.astype(np.float32) + sp.eye(mx.shape[0], dtype=np.float32))).to(device))  
    adjs_pt.append(
        sparse_mx_to_torch_sparse_tensor(sp.eye(edges[0].shape[0], dtype=np.float32).tocoo()).to(device))  
    adjs_pt.append(torch.sparse.FloatTensor(size=edges[0].shape).to(device))

    node_types = np.load("./data/bbk/82/node_types.npy")
    node_types = torch.from_numpy(node_types).to(device)

    net = Net(args,adjs_pt,node_types,local_map)

    net = net.to(device)
    optimizer_w = torch.optim.Adam(
        net.parameters(),
        lr=args.lr,
        weight_decay = args.wd

    )

    optimizer_a1 = torch.optim.Adam(
        net.FusionLayer1.alphas(),
        lr=args.alr
    )


    loss_function = nn.NLLLoss()
    for epoch in range(args.epochs):
        train_loader.reset()
        running_loss = 0.0
        batch_count, batch_avg_loss = 0, 0.0
        while not train_loader.is_end():
            batch_count += 1
            train_stu_ids, train_exer_ids, train_knowledge_embs, train_labels = train_loader.next_batch()
            train_stu_ids, train_exer_ids, train_knowledge_embs, train_labels = train_stu_ids.to(device), train_exer_ids.to(
                device),train_knowledge_embs.to(device), train_labels.to(device)
            optimizer_w .zero_grad()
            output_1 = net.forward( train_stu_ids, train_exer_ids, train_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)
            loss_w = loss_function(torch.log(output + 1e-10), train_labels)
            loss_w.backward()
            optimizer_w.step()
            net.apply_clipper()

            running_loss += loss_w.item()
            if batch_count % 200 == 199:
                print('[%d, %5d] train_loss: %.3f' % (epoch + 1, batch_count + 1, running_loss / 200))
                running_loss = 0.0


        val_loader.reset()
        correct_count, exer_count = 0, 0
        batch_count, batch_avg_loss = 0, 0.0
        pred_all, label_all = [], []
        while not val_loader.is_end():
            batch_count += 1
            val_stu_ids, val_exer_ids, val_knowledge_embs, val_labels = val_loader.next_batch()
            val_stu_ids, val_exer_ids, val_knowledge_embs, val_labels = val_stu_ids.to(device), val_exer_ids.to(
                device), val_knowledge_embs.to(device), val_labels.to(device)

            optimizer_a1.zero_grad()
            #optimizer_a2.zero_grad()
            #optimizer_a3.zero_grad()

            output_1 = net.forward(val_stu_ids, val_exer_ids, val_knowledge_embs)
            output_0 = torch.ones(output_1.size()).to(device) - output_1
            output = torch.cat((output_0, output_1), 1)
            loss_a1 = loss_function(torch.log(output + 1e-10),val_labels)
            loss_a1.backward()

            optimizer_a1.step()
            #optimizer_a2.step()
            #optimizer_a3.step()

            output = output_1.view(-1)
            # compute accuracy
            for i in range(len(val_labels)):
                if (val_labels[i] == 1 and output[i] > 0.5) or (val_labels[i] == 0 and output[i] < 0.5):
                    correct_count += 1
            exer_count += len(val_labels)
            pred_all += output.to(torch.device('cpu')).tolist()
            label_all += val_labels.to(torch.device('cpu')).tolist()




        pred_all = np.array(pred_all)
        label_all = np.array(label_all)
        # compute accuracy
        accuracy = correct_count / exer_count
        # compute RMSE
        rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
        # compute AUC
        auc = roc_auc_score(label_all, pred_all)

        print('val err {}; Arch {}'.format(loss_a1,net.FusionLayer1.parse()))
        print(' accuracy= %f,rmse = %f, auc= %f ' % (accuracy,rmse, auc))
        #save_snapshot(net, 'model_epoch' + str(epoch + 1))


        """print(
                    "Epoch {}; Train err {};a1{};a2{};a3{} ; Arch {}".format(epoch + 1, loss_w.item(),loss_a1.item(), loss_a2.item(),loss_a3.item() ,net.FusionLayer1.parse() + net.FusionLayer2.parse()
                                                                         +net.FusionLayer3.parse()))"""




def save_snapshot(model, filename):
    f = open(filename, 'wb')
    torch.save(model.state_dict(), f)
    f.close()


if __name__ == '__main__':
    args = CommonArgParser().parse_args()
    train(args, construct_local_map(args))