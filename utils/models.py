import torch.nn as nn
import numpy as np
import inferno
from inferno.extensions.layers import RemoveSingletonDimension
from inferno.extensions.layers import ConvReLU2D
from inferno.extensions import model as inf_model
import torch
import skimage

class SeqNet(nn.Module):
    def __init__(self, brunch_mdoel):
        super().__init__()
        self.branch = brunch_mdoel
        self.branch.train(True)

    def forward(self, img1, img2):
        out = []
        res1 = self.branch(img1)
        res2 = self.branch(img2)    
        return res1, res2
    
    def get_brunch(self):
        self.branch.train(False)
        return self.branch

def build_big_model(image_channels, pred_channels=1, no_sigm=False):
    if no_sigm:
        return torch.nn.Sequential(
            ConvReLU2D(in_channels=image_channels, out_channels=8, kernel_size=3),
            inf_model.ResBlockUNet(dim=2, in_channels=8, out_channels=pred_channels, activated=False),
        )
    else:
        return torch.nn.Sequential(
            ConvReLU2D(in_channels=image_channels, out_channels=8, kernel_size=3),
            inf_model.ResBlockUNet(dim=2, in_channels=8, out_channels=pred_channels, activated=False),
            torch.nn.Sigmoid()
        )

def build_standart_model(image_channels, pred_channels=1, no_sigm=False):
    if no_sigm:
        return torch.nn.Sequential(
            inf_model.ResBlockUNet(dim=2, in_channels=image_channels, out_channels=pred_channels, activated=False),
        )
    else:
        return torch.nn.Sequential(
            inf_model.ResBlockUNet(dim=2, in_channels=image_channels, out_channels=pred_channels, activated=False),
            torch.nn.Sigmoid()
        )

def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def calc_iou(prediction, ground_truth):
    n_images = len(prediction)
    intersection, union = 0, 0
    for i in range(n_images):
        intersection += np.logical_and(prediction[i] > 0, ground_truth[i] > 0).astype(np.float32).sum() 
        union += np.logical_or(prediction[i] > 0, ground_truth[i] > 0).astype(np.float32).sum()
    return float(intersection) / union

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def calc_ole(pred, labled, inter_trash=0.8):
    compnts_label = skimage.measure.label(labled)
    compnts_pred2 = skimage.measure.label(pred)
    
    used_pred_idx = []
    inter_trash = inter_trash

    for cur_label_idx in np.unique(compnts_label)[1:]:
        cur_max_inters = 0
        match_pred_idx = -1
        for cur_pred_idx in np.unique(compnts_pred2)[1:]:
            inter_rate = np.sum((compnts_label==cur_label_idx) & (compnts_pred2==cur_pred_idx)) / np.sum(compnts_label==cur_label_idx)
            if inter_rate > inter_trash and (cur_pred_idx not in used_pred_idx):
                match_pred_idx = cur_pred_idx

        used_pred_idx.append(match_pred_idx)
    
    true_pos = np.where(np.array(used_pred_idx) > 0)[0]+1
    false_neg = np.where(np.array(used_pred_idx) == -1)[0]+1
    false_pos = list(set(np.unique(compnts_pred2)[1:]) - set(used_pred_idx))
    
    plot_pred = compnts_pred2.copy()
    for i in false_pos:
        plot_pred[plot_pred==i] = 100

    for i in false_neg:
        plot_pred[compnts_label == i] = 300

    for i in used_pred_idx:
        plot_pred[plot_pred==i] = 200

    plot_pred = plot_pred/100
    
    return plot_pred, true_pos, false_pos, false_neg
