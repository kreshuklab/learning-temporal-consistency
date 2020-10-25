import torch.nn as nn
import torch
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import numpy as np

def compute_sdf(img_gt, out_shape):
    """
    compute the signed distance map of binary mask
    input: segmentation, shape = (batch_size,c, x, y, z)
    output: the Signed Distance Map (SDM) 
    sdf(x) = 0; x in segmentation boundary
             -inf|x-y|; x in segmentation
             +inf|x-y|; x out of segmentation
    normalize sdf to [-1,1]
    """

    img_gt = img_gt.astype(np.uint8)
    normalized_sdf = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                sdf[boundary==1] = 0
                normalized_sdf[b][c] = sdf
                assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

    return normalized_sdf

###

def compute_dice_loss(model, X_batch, y_batch):
    logits = model(X_batch)[:,0:1].contiguous()
    smooth = 1.

    iflat = logits.view(-1)
    tflat =  y_batch.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

def compute_bce_loss(model, X_batch, y_batch):
    logits = model(X_batch)[:,0:1].contiguous()
    
    masks_probs_flat = logits.view(-1)
    true_masks_flat = y_batch.view(-1)
    
    loss = nn.BCELoss()(masks_probs_flat, true_masks_flat)
    return loss.mean()

def compute_dist_loss(model, X_batch, y_batch):
    out_dis = model(X_batch)

    if torch.isnan(out_dis).any():
        print('net output has NAN!!!')
        exit()

    gt_dis = compute_sdf(y_batch.cpu().numpy(), out_dis.shape)
    gt_dis = torch.from_numpy(gt_dis).float().cuda()

    return torch.norm(out_dis - gt_dis, 2)/torch.numel(out_dis)


### SEQ LOSSED
def compute_loss_time_dice(model, one_batch):
    all_logits = []
    for cur_idx in range(one_batch.shape[1]):
        all_logits.append(model(one_batch[:,cur_idx]))

    all_loss = 0
    for idx in range(len(all_logits) - 1):
        all_loss += compute_one_dice(all_logits[idx], all_logits[idx+1], smooth=1)
    return all_loss

def compute_one_dice(logits1, logits2, smooth=1):
    iflat = logits1.view(-1)
    tflat =  logits2.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))

def compute_loss_time_bce(model, one_batch):
    all_logits = []
    for cur_idx in range(one_batch.shape[1]):
        all_logits.append(model(one_batch[:,cur_idx]))

    all_loss = 0
    for idx in range(len(all_logits) - 1):
        all_loss += compute_one_bce(all_logits[idx], all_logits[idx+1])
    return all_loss
    
def compute_one_bce(logits1, logits2):
    masks_probs_flat = logits1.view(-1)
    true_masks_flat = logits2.view(-1).detach()
    
    loss = nn.BCELoss()(masks_probs_flat, true_masks_flat)
    return loss.mean()
