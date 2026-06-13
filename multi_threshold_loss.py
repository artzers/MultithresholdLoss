
import torch
import torch.nn as nn
import numpy as np

class MultiThresholdCSILoss(torch.nn.Module):
    def __init__(self, thresholds=[10, 20, 30, 40],  k=10, eps=1e-6): 
        super().__init__()
        # self.thresholds = torch.tensor(thresholds) # useless
        self.k = k  
        self.eps = eps

    def forward(self, y_pred, y_true, threv): #, weights
        # Calculate the probability at each threshold
        p = torch.sigmoid(self.k * (y_pred - threv))  # [Batch, T,1,H,W]

        # Real label conversion to multi threshold binary classification
        y_true_t = (y_true >= threv).float()  # [Batch, T,1,H,W]

        # calculate TP, FP, FN
        tp = torch.sum(y_true_t * p)   # [T]
        fp = torch.sum((1 - y_true_t) * p)  # [T]
        fn = torch.sum(y_true_t * (1 - p))  # [T]

        # CSI loss
        csi_per_t = (tp + self.eps) / (tp + fp + fn + self.eps)
        loss_per_t = 1 - csi_per_t

        return loss_per_t 

class WeightedReflectivityLoss(nn.Module):
    def __init__(self,l1flag = False):
        super(WeightedReflectivityLoss, self).__init__()
        # Adjust the threshold according to the actual situation. Divided by 50 for normalization
        self.thresholds = torch.tensor([10, 20, 30, 40]) / 50. #0.2 0.4 0.6 0.8
        if l1flag:
            self.radarloss = nn.SmoothL1Loss(reduction='mean').cuda() #reduction='sum'
        else:
            self.radarloss = nn.MSELoss(reduction='mean').cuda() #reduction='sum'

    def forward(self, pred, target):
        # 初始化权重
        weights = torch.ones_like(target)  # only for batch size == 1
        weights[(target >= self.thresholds[0]) & (target < self.thresholds[1])] = 2  
        weights[(target >= self.thresholds[1]) & (target < self.thresholds[2])] = 5  
        weights[(target >= self.thresholds[2]) & (target < self.thresholds[3])] = 10
        weights[target >= self.thresholds[3]] = 20

        loss = self.radarloss(weights * pred, weights *target)
        return loss

class ZhouDDR(nn.Module):
    def __init__(self, l1flag = False):
        super(ZhouDDR, self).__init__()
        # 根据实际情况调整阈值 除以75是因为归一化了的
        # self.mseloss = torch.nn.MSELoss()
        self.weightradarloss = WeightedReflectivityLoss(l1flag)
        self.tau = 0.1
        self.eps = 1e-12
        self.alpha = 0.1
        self.loss_fn = MultiThresholdCSILoss()

    # TAU loss
    def DDRLoss(self, pred, target):
        B, T, C = pred.shape[:3]
        if T <= 2:
            return 0
        gap_pred_y = (pred[:, 1:] - pred[:, :-1]).reshape(B, T - 1, -1)
        gap_batch_y = (target[:, 1:] - target[:, :-1]).reshape(B, T - 1, -1)
        softmax_gap_p = nn.functional.softmax(gap_pred_y / self.tau, -1)
        softmax_gap_b = nn.functional.softmax(gap_batch_y / self.tau, -1)
        loss_gap = softmax_gap_p * torch.log(softmax_gap_p / (softmax_gap_b + self.eps) + self.eps)
        loss = loss_gap.mean()

        return loss

    def forward(self, pred, target, qpeFlag = True):
        loss = (self.weightradarloss(pred, target)
                + self.loss_fn(pred, target, 0.2) # 10 / 50  normalized threshold. Please adjust the threshold based on actual data
                + self.loss_fn(pred, target, 0.4) # 20 / 50  normalized threshold. Please adjust the threshold based on actual data
                + self.loss_fn(pred, target, 0.6) # 30 / 50  normalized threshold. Please adjust the threshold based on actual data
                + self.loss_fn(pred, target, 0.8) #+ self.alpha * self.DDRLoss(pred, target)
            )
        return loss

