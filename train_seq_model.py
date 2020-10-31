import os
import numpy as np
import torch
from PIL import Image, ImageSequence


import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from skimage import morphology

from skimage.filters import threshold_otsu, threshold_adaptive
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import sklearn

import time
from tqdm import tqdm_notebook
from collections import Counter
from copy import copy
import torch.nn as nn
import skimage
import torch.nn.functional as F
from IPython import display
from skimage.exposure import histogram
import torchvision
from skimage.filters import sobel
import sys
import argparse

from skimage.exposure import histogram
from skimage import exposure
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score

from collections import OrderedDict
from inferno.extensions import model as inf_model
from utils.loaders import SeqLoader, TrainLoader, BatchSampler, DimLoader
from utils.augmentation import data_norm, image_preproc, prediction_postprocessing, make_seq_transf, make_train_transf, mask_preprocc, image_mask_norm
from utils.visualize import make_merge_plot, print_images
from utils.models import SeqNet, get_lr
from data_config import create_dataset_info
from utils.models import update_lr
import utils.losses as losses
from torch.utils.data import RandomSampler

parser = argparse.ArgumentParser()
parser.add_argument("-BASE_MODEL_NAME", type=str, help='Name of model', default = 'more_data' )
parser.add_argument("-EPOCHS", type=int, help='Number of epochs', default=500)
parser.add_argument("-TIME_LEN", type=int, help='Sequence len for temporal consistency', default=2)
parser.add_argument("-DIM_LEN", type=int, help='Sequence len for dimensional consistency', default=0)
parser.add_argument("-NUM_CHAN", type=int, help='Number channels per example in input z-stack', default=7)
parser.add_argument("-batch_size", type=int, help='Batch size', default=7)
parser.add_argument("-ADD_NAME", type=str, default='')
parser.add_argument("-INIT_NEW", default=False, action="store_true", help='Train new model (not pretrained one)')
parser.add_argument("-lr_time", type=float, default=1e-5, help='Learning rate for time consistency loss')
parser.add_argument("-lr_dim", type=float, default=1e-5, help='Learning rate for time dimensional loss')
parser.add_argument("-lr_seg", type=float, default=1e-4,help='Learning rate for semantic segmentation loss')
parser.add_argument("-cuda", type=int, default=7,help='Cuda device')
parser.add_argument("-TIME_LOSS", type=str, default='DICE')
parser.add_argument("-DIM_LOSS", type=str, default='DICE')
parser.add_argument("-DATA_TYPE", type=str, help='Data to train on (NUCL or TRITC)')

args = parser.parse_args()
NAME = args.BASE_MODEL_NAME
EPOCHS = args.EPOCHS
TIME_LEN = args.TIME_LEN
NUM_CHAN = args.NUM_CHAN
batch_size = args.batch_size
INIT_NEW = args.INIT_NEW
lr_time = args.lr_time
lr_seg = args.lr_seg
CUDA = args.cuda
TIME_LOSS = args.TIME_LOSS
lr_dim = args.lr_dim
DIM_LEN = args.DIM_LEN
DIM_LOSS = args.DIM_LOSS
DATA_TYPE = args.DATA_TYPE
TIME_LEN = args.TIME_LEN

NO_DIM = True if DIM_LEN == 0 else False
NO_TIME = True if TIME_LEN == 0 else False

torch.cuda.set_device(CUDA)

if not INIT_NEW:
    NEW_NAME = 'seq_'+args.ADD_NAME+'_'+NAME
else:
    NEW_NAME = 'seq_'+args.ADD_NAME+'_noinit'

PATH = f'runs/{NEW_NAME}/'
MODEL_PATH = f'runs/{NAME}/{NAME}.tar'
WEIGHTS_PATH = f'runs/{NEW_NAME}/weights/'
LOGS_PATH = f'runs/{NEW_NAME}/logs/'

os.makedirs(PATH, exist_ok=False) 
os.makedirs(LOGS_PATH, exist_ok=False)
os.makedirs(WEIGHTS_PATH, exist_ok=False)

f = open(f'{PATH}/settings.txt', 'w')
f.write(' '.join(sys.argv))
f.close()

###########################
#    PATHS DETAILS    #
###########################

all_data_list = create_dataset_info.assemble_dataset_from_py()

### SEQUENTAIL DATA
data_list = [x for x in all_data_list if x['name'] in ['movie-holo-overnight-day1', 'movie-holo-overnight-day2']]
seq_data_names_list =  np.array([x['absolute_data_names'] for x in data_list])
seq_focused_frame_list = np.array([ [x['name2focused_frame'][y] for y in x['data_names']]  for x in data_list])


### SEPARATE FRAMES DATA
data_list = create_dataset_info.assemble_dataset_from_py()
if DATA_TYPE == 'NUCL':
    data_list = [x for x in data_list if x['use_to_train'] == True]
elif DATA_TYPE == 'TRITC':
    data_list = [x for x in data_list if  x['train_tritc'] == True]
else:
    raise ValueError

data_names =  np.hstack([x['absolute_data_names'] for x in data_list])
target_names = np.hstack([x['absolute_target_names'] for x in data_list])
focused_frame = np.hstack([ [x['name2focused_frame'][y] for y in x['data_names']]  for x in data_list])
if DATA_TYPE == 'NUCL':
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

######################
#    Loaders init    #
######################

## Augmentations init
transf_seq = make_seq_transf(TIME_LEN, NUM_CHAN, image_preproc)
transf_train = make_train_transf(image_preproc, mask_preprocc)

## Sequential loader init
gen_seqs = [SeqLoader(x, y, seq_len = TIME_LEN, transform=transf_seq, use_npy_data=False) 
                    for x,y in zip(seq_data_names_list, seq_focused_frame_list)]
seq_batch_gens = [torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True, num_workers=15) for x in gen_seqs]
print("Number of sequences to train time consistency: ", len(seq_batch_gens))

data_list = create_dataset_info.assemble_dataset_from_py()
data_names_dim =  np.hstack([x['absolute_data_names'] for x in data_list])
focused_frame_dim = np.hstack([ [x['name2focused_frame'][y] for y in x['data_names']]  for x in data_list])

gen_dim = DimLoader(data_names_dim, focused_frame_dim, seq_range=DIM_LEN, num_channels=7, transform=image_mask_norm)
dim_batch_gen = torch.utils.data.DataLoader(gen_dim, batch_size=batch_size//3, shuffle=True, num_workers=10)

train_dataset = TrainLoader(
                        images_names = data_names[train_idx],
                        mask_names = mask_names[train_idx],
                        target_names = target_names[train_idx],
                        focused_frame = focused_frame[train_idx],
                        transform = transf_train,
                        num_channels = NUM_CHAN,
                        return_original = False,
                        use_npy_data=False)

val_dataset = TrainLoader(
                        images_names=data_names[val_idx],
                        mask_names = mask_names[val_idx],
                        target_names = target_names[val_idx],
                        focused_frame = focused_frame[val_idx],
                        transform = image_mask_norm,
                        num_channels = NUM_CHAN,
                        return_original = True,
                        use_npy_data=False)

train_batch_gen = torch.utils.data.DataLoader(train_dataset, 
                                              num_workers=15,
                                              sampler = RandomSampler(train_dataset, replacement=True, num_samples=250),
                                              batch_size=batch_size,
                                              pin_memory=True)

val_batch_gen = torch.utils.data.DataLoader(val_dataset,
                                              num_workers=15,
                                              shuffle=True,
                                              batch_size=batch_size,
                                              pin_memory=True)


######################
#    Models init    #
######################

if INIT_NEW:
    print("Init new model")
    base_model = res_unet = torch.nn.Sequential(
                            inf_model.ResBlockUNet(dim=2, in_channels=NUM_CHAN, out_channels=1, activated=False),
                            torch.nn.Sigmoid()
                            )
else:
    print("Init from model "+MODEL_PATH)
    base_model = torch.load(MODEL_PATH, map_location={'cuda:4': f'cuda:{CUDA}'})
to_check_model = torch.load(MODEL_PATH, map_location={'cuda:4': f'cuda:{CUDA}'})

opt_time = torch.optim.Adam(base_model.parameters(), lr = lr_time)
opt_seg = torch.optim.Adam(base_model.parameters(), lr = lr_seg)
opt_dim = torch.optim.Adam(base_model.parameters(), lr = lr_dim)

epochs=EPOCHS
wait = 5
no_update = 0
true_train_epochs = 0

time_losses_hist = []
dim_losses_hist = []
seg_loss_train = []
seg_loss_val = []

best_loss_time = np.inf
best_loss_dim = np.inf
best_loss_seg = np.inf

no_update_s = 0
no_update_t = 0
no_update_d = 0

wait_s = 10
wait_t = 10
wait_d = 10

val_loss = []

if DATA_TYPE == 'NUCL':
    log_train_name = 'Train loss'
    log_val_name = 'Val loss'
    lr_log_name = 'Segment_Learning_Rate'

elif DATA_TYPE == 'TRITC':
    log_train_name = 'Train TRITC loss'
    log_val_name = 'Val TRITC loss'
    lr_log_name = 'Segment_Learning_Rate'

if TIME_LOSS == 'DICE':
    time_loss = losses.compute_loss_time_dice

elif TIME_LOSS == 'BCE':
    time_loss = losses.compute_loss_time_bce
else:
    print("Time loss not found")
    assert False

if DIM_LOSS == 'DICE':
    dim_loss = losses.compute_loss_time_dice

elif TIME_LOSS == 'BCE':
    dim_loss = losses.compute_loss_time_bce

else:
    print("Time loss not found")
    assert False


seg_loss = losses.compute_dice_loss
writer = SummaryWriter(LOGS_PATH)

########################
#    Models training    #
########################

print("Start training")
for epoch in range(epochs):
    start_time = time.time()

    if not NO_TIME:
        ### Train Time Consistancy###
        print(" Train time consistency")
        ep_loss = []
        base_model.train(True)
        for idx, cur_seq_batch_gen in enumerate(seq_batch_gens):
            for one_batch in cur_seq_batch_gen:
                one_batch = one_batch.cuda()
                loss = time_loss(base_model, one_batch)
                loss.backward()
                opt_time.step()
                opt_time.zero_grad()
                ep_loss.append(loss.detach().data.cpu().numpy())

        time_losses_hist.append(np.mean(ep_loss))
        writer.add_scalar('Time Consistency Loss',  time_losses_hist[-1], global_step=epoch)

    if not NO_DIM:
        ### Train Dimensional Consistancy###
        print(" Train Dimensional consistency")
        ep_loss = []
        base_model.train(True)
        for one_batch in dim_batch_gen:
            one_batch = one_batch.cuda()
            loss = dim_loss(base_model, one_batch)
            loss.backward()
            opt_dim.step()
            opt_dim.zero_grad()
            ep_loss.append(loss.detach().data.cpu().numpy())

        dim_losses_hist.append(np.mean(ep_loss))
        writer.add_scalar('Dim Consistency Loss',  dim_losses_hist[-1], global_step=epoch)

    ### Train Semantic Segmentation###
    print(" Train Semantic Segmentation")
    train_ep_loss = []
    base_model.train(True)
    for cur_batch in train_batch_gen:
        X_batch = cur_batch['X'].cuda()
        y_batch = cur_batch['Y'].cuda()
        
        loss = seg_loss(base_model, X_batch, y_batch)
        loss.backward()
        opt_seg.step()
        opt_seg.zero_grad()
        train_ep_loss.append(loss.detach().data.cpu().numpy())
            
    writer.add_scalar(log_train_name, np.mean(train_ep_loss), global_step=epoch)

    cur_lr = get_lr(opt_seg)
    writer.add_scalar(lr_log_name, cur_lr, global_step=epoch)

    cur_lr = get_lr(opt_time)
    writer.add_scalar('Time_Learning_Rate', cur_lr, global_step=epoch)

    cur_lr = get_lr(opt_dim)
    writer.add_scalar('Dim_Learning_Rate', cur_lr, global_step=epoch)
    
    ### Train modifications ###
    if epoch %10 == 0:
        val_ep_loss = []
        val_ep_loss_tritc = []
        base_model.train(False)
        for cur_batch in val_batch_gen:
            X_batch = cur_batch['X'].cuda()
            y_batch = cur_batch['Y'].cuda()
            loss = seg_loss(base_model, X_batch, y_batch)
            val_ep_loss.append(loss.detach().data.cpu().numpy())
        
        if not NO_TIME and time_losses_hist[-1] < best_loss_time:
            best_loss_time = time_losses_hist[-1]
            no_update_t = 0
        
        if not NO_DIM and dim_losses_hist[-1] < best_loss_time:
            best_loss_dim = dim_losses_hist[-1]
            no_update_d = 0

        if np.mean(val_ep_loss) < best_loss_seg:
            best_loss_seg = np.mean(val_ep_loss)
            no_update_s = 0

        no_update_t += 1
        no_update_s += 1
        no_update_d += 1

        if no_update_s >= wait_s:
            lr_seg = lr_seg*2/3
            lr_time = lr_time/2
            lr_dim = lr_dim/2

            update_lr(opt_time, lr_time)
            update_lr(opt_seg, lr_seg)
            update_lr(opt_dim, lr_dim)

        writer.add_scalar(log_val_name, np.mean(val_ep_loss) , global_step=epoch)
                                    
        torch.save(base_model, f'{WEIGHTS_PATH}/{NEW_NAME}_{epoch}.tar')
        torch.save(base_model, f'{PATH}/{NEW_NAME}.tar')
        

        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, epochs, time.time() - start_time))

        print("  training loss: \t{:.6f}".format(
            np.mean(train_ep_loss) ))
        
        print("  val loss: \t{:.6f}".format(
            np.mean(val_ep_loss) ))
        
        ##### Visualization #####
        if epoch %50 == 0:
            base_model.train(False)
            
            pred1 = base_model(one_batch[:,0])
            pred2 = base_model(one_batch[:,1])
            pred1 = prediction_postprocessing(pred1[0,0].data.cpu().numpy())
            pred2 = prediction_postprocessing(pred2[0,0].data.cpu().numpy())
            
            pred_base1 = to_check_model(one_batch[:,0])
            pred_base1 = prediction_postprocessing(pred_base1[0,0].data.cpu().numpy())
            
            pred_base2 = to_check_model(one_batch[:,1])
            pred_base2 = prediction_postprocessing(pred_base2[0,0].data.cpu().numpy())
            
            input_data1 = one_batch[:,0].cpu()[0,0].data.numpy()
            input_data2 = one_batch[:,1].cpu()[0,0].data.numpy()
            one_batch = one_batch.data.cpu().numpy()

            fig_to_tens, axs  = print_images([
                1-input_data1, 1-input_data2, 1-input_data1, 1-input_data2,
                pred_base1, pred_base2, pred1, pred2,
                make_merge_plot(pred_base1, input_data1), make_merge_plot(pred_base2, input_data2), make_merge_plot(pred1, input_data1), make_merge_plot(pred2, input_data2)],
            [
                "One unet inp1", "One unet inp2", "Inp1", "Inp2", "One Unet Out1", "One Unet Out2", "Out1", "Out2", "One Unet Out", "One Unet Out", "Out1", "Out2"
            ], 3, 4, size=(15,15), 
            cmaps = [
                'Greys','Greys','Greys','Greys','','','','','Greys','Greys','Greys','Greys'
            ])

            writer.add_figure('Seq Train Images', fig_to_tens, epoch)
            
            print("Diff between images: ", (input_data1 - input_data2).sum())
            print("Diff between new way masks: ", (pred1 - pred2).sum())
            print("Diff between old way and new way masks", (pred1 - pred_base1).sum())
            print("Diff between old way masks", (pred_base2 - pred_base1).sum())