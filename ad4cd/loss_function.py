import torch
import torch.nn as nn


class MyLossFunction(nn.Module):
    def __init__(self):
        super(MyLossFunction, self).__init__()
        self.bce = torch.nn.BCELoss()

    def forward(self, output, recon_loss, mu, logvar, labels):
        labels = labels.float()
        loss0 = self.bce(output, labels)
        loss1 = recon_loss.sum()
        loss2 = -0.5 * torch.mean(1 + logvar - mu ** 2 - torch.exp(logvar))
        return loss0 + 0.01*loss1 + loss2