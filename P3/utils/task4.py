import numpy as np
import torch
import torch.nn as nn

class RegressionLoss(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.SmoothL1Loss()

    def forward(self, pred, target, iou):
        '''
        Task 4.a
        We do not want to define the regression loss over the entire input space.
        While negative samples are necessary for the classification network, we
        only want to train our regression head using positive samples. Use 3D
        IoU ≥ 0.55 to determine positive samples and alter the RegressionLoss
        module such that only positive samples contribute to the loss.
        input
            pred (N,7) predicted bounding boxes
            target (N,7) target bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_reg_lb'] lower bound for positive samples
        '''
        pos_mask = iou >= self.config['positive_reg_lb']  # mask positive samples
        l_loc = self.loss(pred[pos_mask, :3], target[pos_mask, :3])  # location loss
        l_size = self.loss(pred[pos_mask, 3:6], target[pos_mask, 3:6])  # size loss
        l_rot = self.loss(pred[pos_mask, 6], target[pos_mask, 6])  # rotation loss
        return l_loc + 3*l_size + l_rot  # "The contribution to the loss of the three size parameters (h,w,l) should be multiplied by 3"

class ClassificationLoss(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.loss = nn.BCELoss()

    def forward(self, pred, iou):
        '''
        Task 4.b
        Extract the target scores depending on the IoU. For the training
        of the classification head we want to be more strict as we want to
        avoid incorrect training signals to supervise our network.  A proposal
        is considered as positive (class 1) if its maximum IoU with ground
        truth boxes is ≥ 0.6, and negative (class 0) if its maximum IoU ≤ 0.45.
            pred (N,7) predicted bounding boxes
            iou (N,) initial IoU of all paired proposal-targets
        useful config hyperparameters
            self.config['positive_cls_lb'] lower bound for positive samples
            self.config['negative_cls_ub'] upper bound for negative samples
        '''
        mid_mask = ~((iou < self.config['positive_cls_lb']) & (iou > self.config['negative_cls_ub']))  # we don't care about samples with iou between 0.45 and 0.6
        target = iou[mid_mask] >= self.config['positive_cls_lb']  # positive samples with iou >= 0.6
        return self.loss(pred[mid_mask].flatten(), target.type(torch.float))
