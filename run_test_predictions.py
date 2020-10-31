import os
import numpy as np
import torch
from PIL import Image, ImageSequence
import argparse

import matplotlib
import matplotlib.pyplot as plt

plt.switch_backend("agg")
from skimage import morphology

from skimage.filters import threshold_otsu, threshold_adaptive
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from skimage import morphology
from skimage import exposure
from skimage.filters import threshold_otsu, threshold_adaptive

import time
from tqdm import tqdm_notebook, tqdm
from copy import copy
import torch.nn as nn
import torch.nn.functional as F
from IPython import display
import torchvision
import skimage

import nbimporter
import os

from torch.utils.tensorboard import SummaryWriter
from skimage.filters import sobel

from collections import OrderedDict
from inferno.extensions import model as inf_model
from utils.loaders import SeqLoader, TrainLoader
from utils.augmentation import (
    data_norm,
    image_preproc,
    prediction_postprocessing,
    make_seq_transf,
    make_train_transf,
    mask_preprocc,
    image_mask_norm,
    d4_image2mask,
)
from utils.visualize import make_merge_plot, print_images
from utils.models import SeqNet
from data_config import create_dataset_info
from eval_model import eval_model
from utils.augmentation import image_mask_norm

parser = argparse.ArgumentParser()
parser.add_argument("-name", type=str, help="Name of model")
parser.add_argument("-batch_size", type=int, help="size of a batch", default=8)
parser.add_argument(
    "-MERGE_PLOT_PRED",
    action="store_true",
    help="Whether to plot mask and origin on the same plot",
)
parser.add_argument("-TTA", action="store_true", help="Whether to use TTA or not")
parser.add_argument("-NUM_CHAN", type=int, default=7, help="Number of channels")
parser.add_argument(
    "-TIME_FILT", action="store_true", help="Whether to use time filtering"
)
parser.add_argument("-COMPARE_RUN", type=str, help="Name of model to compare with")
parser.add_argument(
    "-DATA_TYPE", type=str, help="Data to train on (NUCL or TRITC or CHROM)"
)
parser.add_argument("-cuda", type=int, default=7, help="Cuda device")
parser.add_argument(
    "-USE_DST", action="store_true", help="Whether to use distance loss"
)


def make_cur_time_filtering(prev_mask, imarray2, next_mask):
    labeled_mask = skimage.measure.label(imarray2)
    res = np.zeros_like(imarray2)

    for compon_idx in np.unique(labeled_mask)[1:]:
        cur_comp = labeled_mask == compon_idx
        overlap_val = (
            cur_comp & prev_mask.astype("bool") | cur_comp & next_mask.astype("bool")
        ).sum()
        if overlap_val != 0:
            res += cur_comp.astype("float32")
    res = (res > 0).astype("float32")
    return res


def save_pure_preds(
    res_unet, final_test_batch_gen, test_files, cur_PURE_PRED_DIR, cur_filt
):
    all_preds = []

    for X_batch in tqdm(final_test_batch_gen):
        if TTA:
            pred_mask = d4_image2mask(res_unet, X_batch.cuda())
        else:
            pred_mask = res_unet(X_batch.cuda())
        for idx in range(pred_mask.shape[0]):
            pred = pred_mask[idx][0].data.cpu().numpy()
            pred = prediction_postprocessing(pred)
            all_preds.append(pred)

    for idx, cur_pred in enumerate(all_preds):
        if TIME_FILT and idx != 0 and idx != (len(all_preds) - 1) and cur_filt:
            prev_mask = all_preds[idx - 1]
            next_mask = all_preds[idx + 1]
            cur_pred = make_cur_time_filtering(prev_mask, cur_pred, next_mask)
        np.save(cur_PURE_PRED_DIR + test_files[idx][:-5] + ".npy", cur_pred)
        print(test_files[idx])


def run_test_pipeline(data_type):
    VIS_PRED_DIR = f"predictions/predictions_{NAME}/vis_preds_{data_type.lower()}/"
    RAW_PRED_DIR = f"predictions/predictions_{NAME}/raw_pred_{data_type.lower()}/"

    data_list = create_dataset_info.assemble_dataset_from_py()

    if data_type == "CHROM":
        data_list = [x for x in data_list if x["chrom"]]

    data_names = [x["absolute_data_names"] for x in data_list]
    target_names = [x["absolute_target_names"] for x in data_list]
    if data_type == "TRITC":
        mask_names = [x["absolute_tritc_names"] for x in data_list]
    else:
        mask_names = [x["absolute_mask_names"] for x in data_list]
    focused_frame = [
        [x["name2focused_frame"][y] for y in x["data_names"]] for x in data_list
    ]

    vis_pred_dirs = [VIS_PRED_DIR + x["name"] + "/" for x in data_list]
    raw_pred_dirs = [RAW_PRED_DIR + x["name"] + "/" for x in data_list]

    for cur_dir1, cur_dir2 in zip(vis_pred_dirs, raw_pred_dirs):
        os.makedirs(cur_dir1, exist_ok=True)
        os.makedirs(cur_dir2, exist_ok=True)

    time_filtering = [int(x["is_sequential"]) for x in data_list] 

    MODEL_PATH = f"runs/{NAME}/{NAME}.tar"

    torch.cuda.set_device(CUDA)
    res_unet = torch.load(MODEL_PATH, map_location=f"cuda:{CUDA}").cuda()
    res_unet.train(False)

    mask_predictor = res_unet

    for idx in range(len(data_list)):
        final_test_loader = TrainLoader(
            images_names=data_names[idx],
            mask_names=None,
            target_names=target_names[idx],
            focused_frame=focused_frame[idx],
            transform=image_mask_norm,
            num_channels=NUM_CHAN,
            return_original=False,
        )

        final_test_batch_gen = torch.utils.data.DataLoader(
            final_test_loader, batch_size=batch_size, shuffle=False, num_workers=10
        )

        save_pure_preds(
            mask_predictor,
            final_test_batch_gen,
            data_list[idx]["data_names"],
            raw_pred_dirs[idx],
            time_filtering[idx],
        )

    eval_model(data_list, NAME, data_type=data_type)

if __name__=='__main__':
    
    args = parser.parse_args()
    NAME = args.name
    MERGE_PLOT_PRED = args.MERGE_PLOT_PRED
    TTA = args.TTA
    NUM_CHAN = args.NUM_CHAN
    TIME_FILT = args.TIME_FILT
    COMPARE_RUN = args.COMPARE_RUN
    batch_size = args.batch_size
    DATA_TYPE = args.DATA_TYPE
    CUDA = args.cuda
    USE_DST = args.USE_DST

    run_test_pipeline(data_type=DATA_TYPE)

