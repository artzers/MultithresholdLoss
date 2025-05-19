
import torch
import torch.nn as nn
import numpy as np

class MultiThresholdCSILoss(torch.nn.Module):
    def __init__(self, thresholds=[10, 20, 30, 40],  k=10, eps=1e-6): #weights=[2,5,10,20],
        super().__init__()
        self.thresholds = torch.tensor(thresholds)
        # self.weights = weights if weights else torch.ones(len(thresholds))
        self.k = k  # Sigmoid斜率
        self.eps = eps

    def forward(self, y_pred, y_true, threv): #, weights
        # 回归输出缩放到0-60
        # y_pred = 60 * torch.sigmoid(y_pred_reg)

        # 计算每个阈值下的概率
        # t = self.thresholds.to(y_pred.device)
        p = torch.sigmoid(self.k * (y_pred - threv))  # [Batch, T,1,H,W]

        # 真实标签转换为多阈值二分类
        y_true_t = (y_true >= threv).float()  # [Batch, T,1,H,W]

        # 计算各阈值TP, FP, FN
        tp = torch.sum(y_true_t * p)   # [T]
        fp = torch.sum((1 - y_true_t) * p)  # [T]
        fn = torch.sum(y_true_t * (1 - p))  # [T]

        # 各阈值CSI损失
        csi_per_t = (tp + self.eps) / (tp + fp + fn + self.eps)
        loss_per_t = 1 - csi_per_t

        # 加权聚合
        # total_loss = torch.mean(loss_per_t * weights)
        return loss_per_t #total_loss

class WeightedReflectivityLoss(nn.Module):
    def __init__(self,l1flag = False):
        super(WeightedReflectivityLoss, self).__init__()
        # 根据实际情况调整阈值 除以50是因为归一化了的
        # self.thresholds = torch.tensor([15, 25, 35, 45, 50,100]) / 50.
        self.thresholds = torch.tensor([10, 20, 30, 40]) / 50.
        self.den = 15./50.#
        if l1flag:
            self.radarloss = nn.SmoothL1Loss(reduction='mean').cuda() #reduction='sum'
        else:
            self.radarloss = nn.MSELoss(reduction='mean').cuda() #reduction='sum'

    def forward(self, pred, target):
        # 初始化权重
        weights = torch.ones_like(target)  # only for batch size == 1
        # 根据阈值设置权重
        # weights += (target / self.den) **2 #/ 2.
        # weights = target / self.den
        weights[(target >= self.thresholds[0]) & (target < self.thresholds[1])] = 2  # *1.1**i
        weights[(target >= self.thresholds[1]) & (target < self.thresholds[2])] = 5  # *1.1**i
        weights[(target >= self.thresholds[2]) & (target < self.thresholds[3])] = 10
        weights[target >= self.thresholds[3]] = 20

        # 根据阈值设置权重
        # weights[target >= 0] = 1.
        # weights[target >= self.thresholds[3]] = 1.5
        # weights[target >= self.thresholds[3]] = 2
        #weights[(target >= self.thresholds[0]) & (target < self.thresholds[1])] = 1.  # *1.1**i
        #weights[(target >= self.thresholds[1]) & (target < self.thresholds[2])] = 1.5  # *1.1**i
        #weights[(target >= self.thresholds[2]) & (target < self.thresholds[3])] = 2.  # *1.1**i
        #weights[(target >= self.thresholds[3]) & (target < self.thresholds[4])] = 2.5  # *1.1**i
        #weights[target >= self.thresholds[4]] = 3
        # for k in range(0,weights.shape[1]):
            # i=weights.shape[1] - k-1
        #     weights[0, i, :] = 1.2 ** (i)
            # weights[0, i, target[0, i, :] > self.thresholds[0]] = 1.2
            # weights[0, i, target[0, i, :] > self.thresholds[0]] = 1.2 ** (i)
            # weights[0,i,target[0,i,:] >= self.thresholds[1]] = 1.2 ** (i)
            # weights[(target >= self.thresholds[0]) & (target < self.thresholds[1])] = 0.2
            # weights[0,i,(target[0,i,:] >= self.thresholds[1]) & (target[0,i,:] < self.thresholds[2])] = 3*(i+1)
            # weights[0,i,(target[0,i,:] >= self.thresholds[2]) & (target[0,i,:] < self.thresholds[3])] = 5*(i+1)
            # weights[0,i,target[0,i,:] >= self.thresholds[3]] = 10*(i+1)

        # den = torch.sum(weights>=self.thresholds[1])

        # loss = torch.mean(weights * ((pred - target) ** 2))
        loss = self.radarloss(weights * pred, weights *target)
        # loss = torch.sum(torch.abs(weights * (pred-  target)))/torch.sum(weights)
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

    # TAU差分散度正则化
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
        # loss = self.mseloss(pred, target) + self.alpha * self.DDRLoss(pred, target)
        # if qpeFlag:
        #     loss = ((self.weightradarloss(pred[:,:,:5,:], target[:,:,:5,:])
        #             + self.weightqpeloss(pred[:, :, 5, :], target[:, :, 5, :]))
        #             + self.alpha * self.DDRLoss(pred, target)
        #             )
        #     return loss
        # else:
        loss = (self.weightradarloss(pred, target)
                + self.alpha * self.DDRLoss(pred, target)
                # + self.loss_fn(pred, target, 0.2)
                # + self.loss_fn(pred, target, 0.4)
                # + self.loss_fn(pred, target, 0.6)
                # + self.loss_fn(pred, target, 0.8)
            )
        return loss

"""加权均方误差（MSE）作为损失函数"""
