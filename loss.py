import torch
from torch.autograd import Function
from torch import nn
import numpy as np
import math


# combined with cross entropy loss, instance level
class LossVariance(nn.Module):
    """ The instances in target should be labeled 
    """

    def __init__(self):
        super(LossVariance, self).__init__()

    def forward(self, input, target):
        
        B = input.size(0)

        loss = 0
        for k in range(B):
            unique_vals = target[k].unique()
            unique_vals = unique_vals[unique_vals != 0]

            sum_var = 0
            for val in unique_vals:
                instance = input[k][:, target[k] == val]
                if instance.size(1) > 1:
                    sum_var += instance.var(dim=1).sum()

            loss += sum_var / (len(unique_vals) + 1e-8)
        loss /= B
        return loss



class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, size_average=True, type="sigmoid"):
        super(FocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.type = type

    def forward(self, logit, target, class_weight=None):
        target = target.view(-1, 1).long()
        if self.type == 'sigmoid':
            if class_weight is None:
                class_weight = [1]*2

            prob = F.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif self.type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C

            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit,1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob*select).sum(1).view(-1,1)
        prob = torch.clamp(prob,1e-8,1-1e-8)
        batch_loss = - class_weight *(torch.pow((1-prob), self.gamma))*prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss

# Robust focal loss
class RobustFocalLoss2d(nn.Module):
    #assume top 10% is outliers
    def __init__(self, gamma=2, size_average=True, type="sigmoid"):
        super(RobustFocalLoss2d, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.type = type

    def forward(self, logit, target, class_weight=None):
        target = target.view(-1, 1).long()
        if self.type=='sigmoid':
            if class_weight is None:
                class_weight = [1]*2

            prob   = F.sigmoid(logit)
            prob   = prob.view(-1, 1)
            prob   = torch.cat((1-prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.)

        elif self.type=='softmax':
            B,C,H,W = logit.size()
            if class_weight is None:
                class_weight =[1]*C

            logit   = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob    = F.softmax(logit,1)
            select  = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1,1)
        class_weight = torch.gather(class_weight, 0, target)

        prob  = (prob*select).sum(1).view(-1,1)
        prob  = torch.clamp(prob,1e-8,1-1e-8)

        focus = torch.pow((1-prob), self.gamma)
        focus = torch.clamp(focus,0,2)

        batch_loss = - class_weight *focus*prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss
    


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = input_flat * target_flat

        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i], target[:, i])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss




class Weight_DiceLoss(nn.Module):
    def __init__(self):
        super(Weight_DiceLoss, self).__init__()

    def forward(self, input, target, weights):
        N = target.size(0)
        smooth = 1

        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        weights = weights.view(N, -1)

        intersection = input_flat * target_flat
        intersection = intersection * weights

        dice = 2 * (intersection.sum(1) + smooth) / ((input_flat * weights).sum(1) + (target_flat * weights).sum(1) + smooth)
        loss = 1 - dice.sum() / N

        return loss


class WeightMulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(WeightMulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # weights = torch.ones(C) #uniform weights for all classes
        # weights[0] = 3
        dice = DiceLoss()
        wdice = Weight_DiceLoss()
        totalLoss = 0

        for i in range(C):
            # diceLoss = dice(input[:, i], target[:, i])
            # diceLoss2 = 1 - wdice(input[:, i], target[:, i - 1])
            # diceLoss3 = 1 - wdice(input[:, i], target[:, i%(C-1) + 1])
            # diceLoss = diceLoss - diceLoss2 - diceLoss3

            # diceLoss = dice(input[:, i - 1] + input[:, i] + input[:, i%(C-1) + 1], target[:, i])
            ''''''
            if (i == 0):
                diceLoss = wdice(input[:, i], target[:, i], weights) * 2
            elif (i == 1):
                # diceLoss = dice(input[:, C - 1] + input[:, i] + input[:, i + 1], target[:, i])
                diceLoss = wdice(input[:, i], target[:, i], weights)
                diceLoss2 = 1 - wdice(input[:, i], target[:, C - 1], weights)
                diceLoss3 = 1 - wdice(input[:, i], target[:, i + 1], weights)
                diceLoss = diceLoss - diceLoss2 - diceLoss3

            elif (i == C - 1):
                # diceLoss = dice(input[:, i - 1] + input[:, i] + input[:, 1], target[:, i])
                diceLoss = wdice(input[:, i], target[:, i], weights)
                diceLoss2 = 1 - wdice(input[:, i], target[:, i - 1], weights)
                diceLoss3 = 1 - wdice(input[:, i], target[:, 1], weights)
                diceLoss = diceLoss - diceLoss2 - diceLoss3

            else:
                # diceLoss = dice(input[:, i - 1] + input[:, i] + input[:, i + 1], target[:, i])
                diceLoss = wdice(input[:, i], target[:, i], weights)
                diceLoss2 = 1 - wdice(input[:, i], target[:, i - 1], weights)
                diceLoss3 = 1 - wdice(input[:, i], target[:, i + 1], weights)
                diceLoss = diceLoss - diceLoss2 - diceLoss3

            #if weights is not None:
                #diceLoss *= weights[i]

            totalLoss += diceLoss
            avgLoss = totalLoss/C

        return avgLoss
    

    
    
    
class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=3, feat_dim=3, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            print(self.centers)
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, input_x, input_label):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        labels = input_label
        batch_size = input_x.size(0)
        channels = input_x.size(1)

        distmat = torch.pow(input_x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, input_x, self.centers.t()) # math:: out = beta * mat + alpha * (mat1_i @ mat2_i)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels2 = input_label.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels2.cuda().eq(classes.expand(batch_size, self.num_classes))  # eq() 想等返回1, 不相等返回0

        dist = distmat * mask.float()


        # torch.clamp(input, min, max, out=None) 将输入input张量每个元素的夹紧到区间 [min,max][min,max]，并返回结果到一个新张量

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss



import torch.nn as nn
import torch.nn.functional as F

def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    device = label.device
    one_hot_label = torch.eye(n_classes, device=device, requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label


class BoundaryLoss(nn.Module):
    """Boundary Loss proposed in:
    Alexey Bokhovkin et al., Boundary Loss for Remote Sensing Imagery Semantic Segmentation
    https://arxiv.org/abs/1905.07852
    """

    def __init__(self, theta0=3, theta=5):
        super().__init__()

        self.theta0 = theta0
        self.theta = theta

    def forward(self, pred_output, gt):
        """
        Input:
            - pred_output: the output from model (before softmax)
                    shape (N, C, H, W)
            - gt: ground truth map #这是原来的输入,最新输入为(N, C, H, W)
                    shape (N, H, w)
        Return:
            - boundary loss, averaged over mini-bathc
        """

        n, c, _, _ = pred_output.shape

        # softmax so that predicted map can be distributed in [0, 1]
        pred = torch.softmax(pred_output, dim=1)

        # one-hot vector of ground truth
        #one_hot_gt = one_hot(gt.long(), c) # 这是原来的输入,最新输入为(N, C, H, W)
        one_hot_gt = gt



        # boundary map
        gt_b = F.max_pool2d(1 - one_hot_gt, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        gt_b -= 1 - one_hot_gt

        pred_b = F.max_pool2d(1 - pred, kernel_size=self.theta0, stride=1, padding=(self.theta0 - 1) // 2)
        pred_b -= 1 - pred

        # extended boundary map
        gt_b_ext = F.max_pool2d(gt_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        pred_b_ext = F.max_pool2d(pred_b, kernel_size=self.theta, stride=1, padding=(self.theta - 1) // 2)

        # reshape
        gt_b = gt_b.view(n, c, -1)
        pred_b = pred_b.view(n, c, -1)
        gt_b_ext = gt_b_ext.view(n, c, -1)
        pred_b_ext = pred_b_ext.view(n, c, -1)

        # Precision, Recall
        P = torch.sum(pred_b * gt_b_ext, dim=2) / (torch.sum(pred_b, dim=2) + 1e-7)
        R = torch.sum(pred_b_ext * gt_b, dim=2) / (torch.sum(gt_b, dim=2) + 1e-7)

        # Boundary F1 Score
        BF1 = 2 * P * R / (P + R + 1e-7)

        # summing BF1 Score for each class and average over mini-batch
        loss = torch.mean(1 - BF1)

        return loss





def dice_loss(input, target, eps=1e-7, if_sigmoid=True):
    if if_sigmoid:
        input = F.sigmoid(input)
    b = input.shape[0]
    iflat = input.contiguous().view(b, -1)
    tflat = target.float().contiguous().view(b, -1)
    intersection = (iflat * tflat).sum(dim=1)
    L = (1 - ((2. * intersection + eps) / (iflat.pow(2).sum(dim=1) + tflat.pow(2).sum(dim=1) + eps))).mean()
    return L

def smooth_truncated_loss(p, t, ths=0.06, if_reduction=True, if_balance=True):
    n_log_pt = F.binary_cross_entropy_with_logits(p, t, reduction='none')
    pt = (-n_log_pt).exp()
    L = torch.where(pt>=ths, n_log_pt, -math.log(ths)+0.5*(1-pt.pow(2)/(ths**2)))
    if if_reduction:
        if if_balance:
            return 0.5*((L*t).sum()/t.sum().clamp(1) + (L*(1-t)).sum()/(1-t).sum().clamp(1))
        else:
            return L.mean()
    else:
        return L

def balance_bce_loss(input, target):
    L0 = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    return 0.5*((L0*target).sum()/target.sum().clamp(1)+(L0*(1-target)).sum()/(1-target).sum().clamp(1))

def compute_loss_list(loss_func, pred=[], target=[], **kwargs):
    losses = []
    for ipred, itarget in zip(pred, target):
        losses.append(loss_func(ipred, itarget, **kwargs))
    return losses














