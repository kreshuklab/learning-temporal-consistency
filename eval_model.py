import argparse
import os
import time
from collections import Counter
from copy import copy

import numpy as np
import skimage
import torch
from IPython import display
from PIL import Image, ImageSequence
from scipy import ndimage as ndi
from skimage import data, exposure, filters, io, morphology, transform
from skimage.exposure import histogram
from skimage.filters import threshold_adaptive, threshold_otsu
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, utils
from tqdm import tqdm_notebook
import glob

from data_config import create_dataset_info
from utils.models import calc_iou, calc_ole

def get_pred_paths(pred_names, label_names):
    res = []
    for y in label_names:
        for xx in pred_names:
            for x in xx:
                 if os.path.basename(y)[:-12] in x:
                        res.append(x)
    return res


def eval_model(data_list, model_name, data_type):
    LOGS_PATH = f'runs/{model_name}/logs/'

    if data_type=='TRITC':
        data_list = [x for x in data_list if x['train_tritc'] == True]
        mask_names = [x['absolute_tritc_names'] for x in data_list]
        pred_dirs = [f"predictions/predictions_{model_name}/raw_pred_{data_type.lower()}/{x['name']}/" for x in data_list]
    elif data_type =='NUCL':
        data_list = [x for x in data_list if x['use_to_train'] == True]
        mask_names = [x['absolute_mask_names'] for x in data_list]
        pred_dirs = [f"predictions/predictions_{model_name}/raw_pred_{data_type.lower()}/{x['name']}/" for x in data_list]
    elif data_type =='CHROM':
        data_list = [x for x in data_list if x['chrom'] == True]
        mask_names = [x['absolute_mask_names'] for x in data_list]
        pred_dirs = [f"predictions/predictions_{model_name}/raw_pred_{data_type.lower()}/{x['name']}/" for x in data_list]

    test_idx = [x['val_idx'] for x in data_list]
    pred_names = [[x+y for y in sorted(os.listdir(x))] for x in pred_dirs]
    print('Number of data packs with gt: ', len(pred_names))
    iou_scores = []
    for cur_gt_files, cur_pred_files, cur_test_idx in zip(mask_names, pred_names, test_idx):
        for cur_gt, cur_pred in zip(np.array(cur_gt_files)[cur_test_idx], np.array(cur_pred_files)[cur_test_idx]):
            gt = np.load(cur_gt)
            pred = np.load(cur_pred)
            cur_score = calc_iou(pred, gt)
            iou_scores.append(cur_score)
    
    writer = SummaryWriter(LOGS_PATH)
    writer.add_scalar(f'Mean test {data_type} IOU', np.mean(iou_scores))
    print(f"{data_type} IOU on test of model {model_name}: ", np.mean(iou_scores))

    ole_scores = []
    if data_type == 'NUCL':
        data_list = create_dataset_info.assemble_dataset_from_py()
        data_list = [x for x in data_list if x['is_sequential'] == True]
        data_names =  [x['absolute_data_names'] for x in data_list]

        pred_path = f'predictions/predictions_{model_name}/raw_pred_nucl/'
        pred_names = [[pred_path+x['name']+'/'+y for y in sorted(os.listdir(pred_path+x['name']))] for x in data_list]

        labeled_path = '/home/shabanov/docs/nuclei/labeled_for_ole/models_compar'
        label_names = glob.glob(f"{labeled_path}/*.npy")
        pred_with_label_names = get_pred_paths(pred_names, label_names)
        data_names_with_label = get_pred_paths(data_names, label_names)

        for cur_data, cur_pred, cur_label in zip(data_names_with_label, 
                                                      pred_with_label_names,
                                                        label_names):
            pred = np.load(cur_pred)
            label = np.load(cur_label)
            label = label[np.sum(label, axis=(1,2,3))!=0][0][:,:,0]>0
            
            _, true_pos1, false_pos1, false_neg1 = calc_ole(pred, label, 0.4)
            det_metr_1 = len(true_pos1)/(len(true_pos1)+len(false_pos1)+len(false_neg1))
            ole_scores.append(det_metr_1)
    

        writer.add_scalar(f'Mean test {data_type} OLE', np.mean(ole_scores))
        print(f"{data_type} OLE on test of model {model_name}: ", np.mean(ole_scores))
        writer.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-name", type=str, help='Name of model')
    parser.add_argument("-DATA_TYPE", type=str, help='Name of model')
    args = parser.parse_args()
    NAME = args.name
    DATA_TYPE = args.DATA_TYPE

    data_list = create_dataset_info.assemble_dataset_from_py()
    eval_model(data_list, NAME, DATA_TYPE)
