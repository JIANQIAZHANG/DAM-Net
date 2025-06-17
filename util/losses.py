import torch

import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        # smooth = 1e-5
        smooth = torch.full((score.shape[0], ), 1e-5).to(score.device)
        intersect = torch.sum(score * target, dim=[1,2])
        y_sum = torch.sum(target * target, dim=[1,2])
        z_sum = torch.sum(score * score, dim=[1,2])
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)          #
        if weight is None:
            # weight = [1] * self.n_classes
            weight = torch.zeros((self.n_classes, inputs.shape[0])).to(inputs.device)
            for i in range(0, self.n_classes):
                weight[i, :] = 1

        assert inputs.size() == target.size(), 'predict & target shape do not match'
        # class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            # class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes