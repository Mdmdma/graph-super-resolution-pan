import torch.nn.functional as f
import torch


def mse_loss_func(pred, gt, mask):
    if  mask.shape[1] != pred.shape[1]:
        pred = torch.mean(pred, 1, keepdim=True)
        gt = torch.mean(gt, 1, keepdim=True)
    return f.mse_loss(pred[mask == 1.], gt[mask == 1.])


def l1_loss_func(pred, gt, mask):
    if  mask.shape[1] != pred.shape[1]:
        pred = torch.mean(pred, 1, keepdim=True)
        gt = torch.mean(gt, 1, keepdim=True)
    return f.l1_loss(pred[mask == 1.], gt[mask == 1.])

