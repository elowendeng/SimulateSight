# train/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class COD_Loss(nn.Module):
    def __init__(self, eps=1e-7, alpha=0.7, beta=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = eps
        self.alpha = alpha
        self.beta = beta

    def iou_loss(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pred_sigmoid = torch.clamp(pred_sigmoid, min=self.eps, max=1 - self.eps)

        intersection = (pred_sigmoid * target).sum(dim=(1, 2, 3))
        union = pred_sigmoid.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

        iou = (intersection + self.eps) / (union + self.eps)

        if torch.isnan(iou).any():
            return torch.tensor(1.0, device=pred.device, requires_grad=True)
        return 1 - iou.mean()

    def forward(self, pred, target):
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)

        bce_loss = self.bce(pred, target)
        iou_loss = self.iou_loss(pred, target)

        if torch.isnan(bce_loss) or torch.isnan(iou_loss):
            print("Warning: NaN loss detected, using BCE only")
            return bce_loss if not torch.isnan(bce_loss) else iou_loss

        return self.alpha * bce_loss + self.beta * iou_loss


class COD_Loss_With_Aux(nn.Module):
    """Loss function supporting auxiliary losses"""

    def __init__(self, eps=1e-7, aux_weight=0.3, alpha=0.7, beta=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.eps = eps
        self.aux_weight = aux_weight
        self.alpha = alpha
        self.beta = beta

    def iou_loss(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        pred_sigmoid = torch.clamp(pred_sigmoid, min=self.eps, max=1 - self.eps)

        intersection = (pred_sigmoid * target).sum(dim=(1, 2, 3))
        union = pred_sigmoid.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - intersection

        iou = (intersection + self.eps) / (union + self.eps)

        if torch.isnan(iou).any():
            return torch.tensor(1.0, device=pred.device, requires_grad=True)
        return 1 - iou.mean()

    def forward(self, pred, target):
        if pred.shape != target.shape:
            pred = F.interpolate(pred, size=target.shape[2:], mode='bilinear', align_corners=False)

        bce_loss = self.bce(pred, target)
        iou_loss = self.iou_loss(pred, target)

        if torch.isnan(bce_loss) or torch.isnan(iou_loss):
            return bce_loss if not torch.isnan(bce_loss) else iou_loss

        return self.alpha * bce_loss + self.beta * iou_loss

    def forward_with_aux(self, main_pred, aux_preds, targets):
        main_loss = self.forward(main_pred, targets)

        aux_losses = []
        for aux_pred in aux_preds:
            aux_loss = self.forward(aux_pred, targets)
            aux_losses.append(aux_loss)

        avg_aux_loss = sum(aux_losses) / len(aux_losses)
        total_loss = main_loss + self.aux_weight * avg_aux_loss

        return total_loss, main_loss, avg_aux_loss