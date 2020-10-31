import os
import numpy as np
import torch
from PIL import Image, ImageSequence
import sys

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from skimage import morphology

from skimage.filters import threshold_otsu, threshold_adaptive
from skimage import io, transform
from skimage import exposure
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import time
from tqdm import tqdm_notebook
from copy import copy
import torch.nn as nn
import torch.nn.functional as F
from IPython import display
import torchvision

import inferno
from inferno.extensions.layers import RemoveSingletonDimension
from inferno.extensions.layers import ConvReLU2D
from collections import OrderedDict

import nbimporter

from torch.utils.tensorboard import SummaryWriter

import argparse
from skimage.filters import sobel
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma
)

from collections import OrderedDict
from inferno.extensions import model as inf_model
from utils.loaders import SeqLoader, TrainLoader, BatchSampler
from utils.augmentation import data_norm, image_preproc, prediction_postprocessing, make_seq_transf, make_train_transf, mask_preprocc, image_mask_norm, d4_image2mask
from utils.visualize import make_merge_plot, print_images
from utils.models import calc_iou, build_standart_model, build_big_model, get_lr, update_lr
from data_config import create_dataset_info
from utils.losses import compute_dice_loss, compute_bce_loss, compute_dist_loss
from torch.utils.data import RandomSampler

parser = argparse.ArgumentParser()
parser.add_argument("-name", type=str, help='Name of model')
parser.add_argument("-batch_size", type=int, help='Batch size', default=10)
# parser.add_argument("-LOSS_TYPE", type=str, help='Type of loss func to use (BCE or DICE or DIST)')
parser.add_argument("-DIST_lambda", type=float, default = 1,help='Lambda coef for distance loss')

parser.add_argument("-EPOCHS", type=int, help='Number of epochs', default=300)
# parser.add_argument("-AUG_ON", action="store_true", help='Whether to use augmentation or not')
parser.add_argument("-NUM_CHAN", type=int, help='Number of channels to use', default=7)
# parser.add_argument("-ADD_CHAN", action="store_true", help='Whether to add addition channels to input data')
parser.add_argument("-BIG_MODEL", action="store_true", help='Whether to use a bigger model')
parser.add_argument("-lr", type=float, default = 1e-3,help='Learning Rate')
parser.add_argument("-cuda", type=int, default=7,help='Cuda device')
parser.add_argument("-use_npy_data", action="store_true", help='Whether to use preprocessed and saved data')
parser.add_argument("-DATA_TYPE", type=str, help='Data to train on (NUCL or TRITC or CHROM)')
parser.add_argument("-DEVIATE", action="store_true", help='Whether to deviate focused plane')


args = parser.parse_args()
batch_size = args.batch_size
NAME = args.name
BIG_MODEL = args.BIG_MODEL
# AUG_ON = args.AUG_ON
EPOCHS = args.EPOCHS
LR = args.lr
NUM_CHAN = args.NUM_CHAN
CUDA = args.cuda
use_npy_data = args.use_npy_data
DATA_TYPE = args.DATA_TYPE 
DEVIATE = args.DEVIATE
DIST_lambda = args.DIST_lambda

torch.cuda.set_device(CUDA)

PATH = f'runs/{NAME}/'
WEIGHTS_PATH = f'runs/{NAME}/weights/'
LOGS_PATH = f'runs/{NAME}/logs/'

os.makedirs(PATH, exist_ok=False)
os.makedirs(LOGS_PATH, exist_ok=False)
os.makedirs(WEIGHTS_PATH, exist_ok=False)

f = open(f'{PATH}/settings.txt', 'w')
f.write(' '.join(sys.argv))
f.close()
## Loading dataset info
data_list = create_dataset_info.assemble_dataset_from_py()
if DATA_TYPE == 'NUCL':
    data_list = [x for x in data_list if x['use_to_train'] == True]
elif DATA_TYPE == 'TRITC':
    data_list = [x for x in data_list if  x['train_tritc'] == True]
elif DATA_TYPE == 'CHROM':
    data_list = [x for x in data_list if  x['chrom'] == True]
else:
    raise ValueError

data_names =  np.hstack([x['absolute_data_names'] for x in data_list])
target_names = np.hstack([x['absolute_target_names'] for x in data_list])
focused_frame = np.hstack([ [x['name2focused_frame'][y] for y in x['data_names']]  for x in data_list])
if DATA_TYPE == 'NUCL' or DATA_TYPE == 'CHROM':
    mask_names = np.hstack([x['absolute_mask_names'] for x in data_list])
elif DATA_TYPE == 'TRITC':
    mask_names = np.hstack([x['absolute_tritc_names'] for x in data_list])

assert len(target_names) == len(data_names) == len(mask_names) == len(focused_frame)

train_idx = data_list[0]['train_idx']
val_idx = data_list[0]['val_idx']
test_idx = data_list[0]['test_idx']

for x in data_list[1:]:
    train_idx += (np.array(x['train_idx'])+len(train_idx)).tolist()
    val_idx += (np.array(x['val_idx'])+len(val_idx)).tolist()
    test_idx += (np.array(x['test_idx'])+len(test_idx)).tolist()

print("Number of train files: ", len(train_idx))
print('Number of val files ', len(val_idx))
print('Number of test files ', len(test_idx))

### Augmentations
transf_train = make_train_transf(image_preproc, mask_preprocc)

train_dataset = TrainLoader(
                        images_names = data_names[train_idx],
                        mask_names = mask_names[train_idx],
                        target_names = target_names[train_idx],
                        focused_frame = focused_frame[train_idx],
                        transform = transf_train,
                        num_channels = NUM_CHAN,
                        deviate = DEVIATE,
                        return_original = False,
                        chrom = DATA_TYPE=='CHROM',
                        use_npy_data=False)

val_dataset = TrainLoader(
                        images_names=data_names[val_idx],
                        mask_names = mask_names[val_idx],
                        target_names = target_names[val_idx],
                        focused_frame = focused_frame[val_idx],
                        transform = image_mask_norm,
                        num_channels = NUM_CHAN,
                        return_original = True,
                        chrom = DATA_TYPE=='CHROM',
                        use_npy_data=False)


train_batch_gen = torch.utils.data.DataLoader(train_dataset, 
                                              num_workers=6,
                                              sampler = RandomSampler(train_dataset, replacement=True, num_samples=1000),
                                              batch_size = batch_size,
                                              pin_memory=True)

val_batch_gen = torch.utils.data.DataLoader(val_dataset,
                                              num_workers=6,
                                              shuffle=True,
                                              batch_size = batch_size,
                                              pin_memory=True)

image_channels = NUM_CHAN
pred_channels = 1

### MODEL
if BIG_MODEL:
    res_unet = build_big_model(image_channels, pred_channels).cuda()
else:
    res_unet = build_standart_model(image_channels, pred_channels, no_sigm = True).cuda()

segment_pred = lambda x: torch.sigmoid(-1500*res_unet(x))

epochs = EPOCHS if EPOCHS else 2000

batch_size = batch_size
bes_val_loss = np.inf
wait = 5
no_update = 0

train_loss = []
val_iu = []
train_iu = []
val_loss = []

if DATA_TYPE == 'NUCL':
    log_train_name = 'Train loss'
    log_val_name = 'Val loss'
    lr_log_name = 'Segment_Learning_Rate'
elif DATA_TYPE == 'TRITC':
    log_train_name = 'Train TRITC loss'
    log_val_name = 'Val TRITC loss'
    lr_log_name = 'Segment_Learning_Rate'
elif DATA_TYPE == 'CHROM':
    log_train_name = 'Train CHROM loss'
    log_val_name = 'Val CHROM loss'
    lr_log_name = 'Segment_Learning_Rate'

opt = torch.optim.Adam(res_unet.parameters(), lr = LR)

writer = SummaryWriter(LOGS_PATH)

print("Start training")
for epoch in range(0, epochs+1):
    print('Training Stage')
    start_time = time.time()
    ep_loss = []
    res_unet.train(True)
    for cur_batch in train_batch_gen:
        X_batch = cur_batch['X'].cuda()
        y_batch = cur_batch['Y'].cuda()
        loss1 = compute_dice_loss(segment_pred, X_batch, y_batch)
        loss2 = compute_dist_loss(res_unet, X_batch, y_batch)
        loss = loss1 + DIST_lambda*loss2

        loss.backward()
        opt.step()
        opt.zero_grad()
        ep_loss.append(loss.detach().data.cpu().numpy())
        
    train_loss.append(np.mean(ep_loss))
    writer.add_scalar(log_train_name, train_loss[-1], global_step=epoch)
    
    print('Validation Stage')
    res_unet.train(False)
    ep_loss = []
    for cur_batch in val_batch_gen:
        X_batch = cur_batch['X'].cuda()
        y_batch = cur_batch['Y'].cuda()

        loss1 = compute_dice_loss(segment_pred, X_batch, y_batch)
        loss2 = compute_dist_loss(res_unet, X_batch, y_batch)
        loss = loss1 + DIST_lambda*loss2

        ep_loss.append(loss.detach().data.cpu().numpy())

    val_loss.append(np.mean(ep_loss))
    writer.add_scalar(log_val_name, val_loss[-1], global_step=epoch)

    cur_lr = get_lr(opt)
    writer.add_scalar(lr_log_name, cur_lr, global_step=epoch)

    no_update += 1
    #saving best model
    if val_loss[-1]<= bes_val_loss:
        bes_val_loss = val_loss[-1]
        torch.save(res_unet, f'{WEIGHTS_PATH}/{NAME}_{epoch}.tar')
        torch.save(res_unet, f'{PATH}/{NAME}.tar')
        no_update = 0
        
    if no_update >= wait:
        LR = LR*2/3
        update_lr(opt, LR)

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, epochs, time.time() - start_time))

    print("  training loss (in-iteration): \t{:.6f}".format(
        np.mean(train_loss[-1])))

    print("  val loss (in-iteration): \t{:.6f}".format(
        np.mean(val_loss[-1])))

    if epoch % 5 == 0:
        print('Visualization Stage')
        pred = segment_pred(X_batch)[0].data.cpu().numpy()
        input_data = X_batch.cpu()[0].data.numpy()
        truth = y_batch.cpu()[0].data.numpy()
        
        real_pred = copy(pred)
        
        pred = prediction_postprocessing(pred)
        fig_to_tens, axs = print_images([input_data[input_data.shape[0]//2], 
                    truth[0], real_pred[0], pred[0]], 
                    ['Origin data','Mask',  
                    'Prediction', 'Preprocessed Prediction'], 2, 2)

        writer.add_figure('Train Images', fig_to_tens, epoch)

writer.close()
