
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss

def get_metrics(epoch, running_loss, batch_count, pred_all, label_all, pred_correct_incorrect, d_type):

    correct_count = 0


    if d_type == 'val':

        for i in range(len(label_all)):
            if (label_all[i] == 1 and pred_all[i] > 0.5) or (label_all[i] == 0 and pred_all[i] < 0.5):
                correct_count += 1

        NLL = (running_loss / batch_count)
        ECE, MCE = calc_ece(pred_correct_incorrect, label_all, bins = 10)
        pred_all = np.array(pred_all)
        label_all = np.array(label_all)
        # compute accuracy
        accuracy = correct_count / len(label_all)
        # compute RMSE
        rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
        # compute AUC
        auc = roc_auc_score(label_all, pred_all)
        brier_score = brier_score_loss(label_all, pred_all)
        print('epoch= %d, accuracy= %f, rmse= %f, auc= %f, brier_score= %f, NLL= %f, ECE= %f, MCE= %f' % (epoch+1, accuracy, rmse, auc, brier_score, NLL, ECE, MCE))
        with open('result/model_val.txt', 'a', encoding='utf8') as f:
            f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f, brier_score= %f, NLL= %f, ECE= %f, MCE= %f \n' % (epoch+1, accuracy, rmse, auc, brier_score, NLL, ECE, MCE))


    else:

        for i in range(len(label_all)):
            if (label_all[i] == 1 and pred_all[i] > 0.5) or (label_all[i] == 0 and pred_all[i] < 0.5):
                correct_count += 1
        
        NLL = (running_loss / batch_count)
        ECE, MCE = calc_ece(pred_correct_incorrect, label_all, bins = 10)
        pred_all = np.array(pred_all)
        label_all = np.array(label_all)
        # compute accuracy
        accuracy = correct_count / len(label_all)
        # compute RMSE
        rmse = np.sqrt(np.mean((label_all - pred_all) ** 2))
        # compute AUC
        auc = roc_auc_score(label_all, pred_all)
        brier_score = brier_score_loss(label_all, pred_all)
        print('epoch= %d, accuracy= %f, rmse= %f, auc= %f, brier_score= %f, NLL= %f, ECE= %f, MCE= %f' % (epoch+1, accuracy, rmse, auc, brier_score, NLL, ECE, MCE))
        with open('result/model_test.txt', 'a', encoding='utf8') as f:
            f.write('epoch= %d, accuracy= %f, rmse= %f, auc= %f, brier_score= %f, NLL= %f, ECE= %f, MCE= %f \n' % (epoch+1, accuracy, rmse, auc, brier_score, NLL, ECE, MCE))



    return rmse, auc, ECE


def coverage_risk(confidence, correctness):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list


def aurc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area


    return aurc, eaurc


def calc_ece(softmax, label, bins=10):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmax = torch.tensor(softmax)
    labels = torch.tensor(label)

    softmax_max, predictions = torch.max(softmax, 1)
    correctness = predictions.eq(labels)

    ece = torch.zeros(1)
    mce = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = softmax_max.gt(bin_lower.item()) * softmax_max.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()   #落入某个bin中的样本比率

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = softmax_max[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            if torch.abs(avg_confidence_in_bin - accuracy_in_bin) > mce:
                mce = torch.abs(avg_confidence_in_bin - accuracy_in_bin)


    return ece.item(), mce.item()


