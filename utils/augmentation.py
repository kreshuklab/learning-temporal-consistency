import numpy as np
from skimage.exposure import histogram
from PIL import Image
from skimage import morphology
from skimage import exposure
from skimage.filters import threshold_otsu, threshold_adaptive
from collections import OrderedDict
from scipy import ndimage as ndi
import albumentations as A
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

def seq_transformer(aug, images, additional_targets, seq_len, image_preproc):
    if image_preproc is not None:
        data = {'image':image_preproc(images[0][0])}
        for idx, targ_name in enumerate(additional_targets.keys()):
            i = int(targ_name.split('_')[1])
            j = int(targ_name.split('_')[2])
            data[targ_name] = image_preproc(images[i][j])
    else:
        data = {'image':images[0][0]}
        for idx, targ_name in enumerate(additional_targets.keys()):
            i = int(targ_name.split('_')[1])
            j = int(targ_name.split('_')[2])
            data[targ_name] = images[i][j]

    augmented = aug(**data)

    to_stack_all = []
    
    for cur_len in range(seq_len):
        to_stack_one = []
        image_aug = augmented['image'] if cur_len==0 else augmented[f'image_{cur_len}_0']
        to_stack_one.append(image_aug)
        
        for targ_name in additional_targets.keys():
            i = int(targ_name.split('_')[1])
            j = int(targ_name.split('_')[2])
            if i == cur_len and j != 0:
                to_stack_one.append(augmented[targ_name])
                
        to_stack_all.append(np.stack(to_stack_one, axis = 2))
        
    return to_stack_all

def make_seq_transf(seq_len, NUM_CHAN, image_preproc=None):
    additional_targets = OrderedDict()
    for i in range(seq_len):
        for j in range(NUM_CHAN):
            if i == 0 and j == 0:
                continue
            additional_targets['image' + '_' + str(i) + '_' + str(j)] = 'image'

    aug = Compose([
        VerticalFlip(p=0.4),
        RandomRotate90(p=0.4),
        Transpose(p=0.4),
        RandomBrightnessContrast(p=0.5,
                                brightness_limit=0.3,
                                contrast_limit=0.3, 
                                ),
        ], additional_targets=additional_targets)
    
    return lambda imgs: seq_transformer(aug, imgs, additional_targets, seq_len, image_preproc)


# def train_transformer(aug, image, mask, additional_targets, image_proc=None, mask_proc=None):
#     if mask_proc is not None:
#         mask = mask_proc(mask)

#     if image_proc is not None:
#         data = {'image':image_proc(image[0]), 'mask':mask}
#         for idx, targ_name in enumerate(additional_targets.keys()):
#             data[targ_name] = image_proc(image[idx+1])
#     else:
#         data = {'image':image[0], 'mask':mask}
#         for idx, targ_name in enumerate(additional_targets.keys()):
#             data[targ_name] = image[idx+1]

#     augmented = aug(**data)

#     mask_aug = augmented['mask']
#     image_aug = augmented['image']
#     to_stack = []
#     to_stack.append(image_aug)
#     for targ_name in additional_targets.keys():
#         to_stack.append(augmented[targ_name])
#     image_aug = np.stack(to_stack, axis = 2)
#     return image_aug, mask_aug


# def make_train_transf(NUM_CHAN, image_proc=None, mask_proc=None):

#     additional_targets = OrderedDict()
#     for i, image in enumerate(range(NUM_CHAN)[1:]):
#         additional_targets['image' + str(i)] = 'image'

#     aug = Compose([
#         VerticalFlip(p=0.4),
#         RandomRotate90(p=0.4),
#         Transpose(p=0.4),
#         RandomBrightnessContrast(p=0.5,
#                                 brightness_limit=0.3,
#                                 contrast_limit=0.3, 
#                                 ),
#         OneOf([
#             ElasticTransform(p=0.9, alpha=90, sigma=120 * 0.05, alpha_affine=120 * 0.03),
#             # GridDistortion(p=0.9),
#         ], p=0.8)
#     ], additional_targets=additional_targets)

#     return lambda imgs, mask: train_transformer(aug, imgs, mask,  additional_targets, image_proc, mask_proc)

def make_train_transf(image_proc=None, mask_proc=None):
    aug = A.ReplayCompose([
        VerticalFlip(p=0.4),
        RandomRotate90(p=0.4),
        Transpose(p=0.4),
        RandomBrightnessContrast(p=0.5,
                                brightness_limit=0.3,
                                contrast_limit=0.3, 
                                ),
        OneOf([
            ElasticTransform(p=0.9, alpha=90, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        ], p=0.8)
    ])
    return lambda imgs, mask: apply_train_aug(aug, imgs, mask, image_proc, mask_proc) 

def apply_train_aug(aug, image, mask, image_proc=None, mask_proc=None):
    if image_proc is None:
        image_proc = lambda x: x
    if mask_proc is None:
        mask_proc = lambda x: x
    
    aug_imgs = []
    aug_mask = None
    aug_mask_tritc = None
    for idx in range(len(image)):
        if idx == 0:
            after_aug = aug(image=image_preproc(image[0]), mask=mask_preprocc(mask))
            aug_imgs.append(after_aug['image'])
            aug_mask = after_aug['mask']
        else:
            cur_img = A.ReplayCompose.replay(after_aug['replay'], image=image_preproc(image[idx]))['image']
            aug_imgs.append(cur_img)
            
    return np.stack(aug_imgs, axis = 2), aug_mask

data_norm = lambda x, cur_min, cur_max : (x - cur_min)/(cur_max - cur_min)

def image_mask_norm(image, mask=None):
    for idx in range(image.shape[0]):
        image[idx] = image_preproc(image[idx])

    image = image.transpose((1,2,0))

    if mask is not None:
        mask = np.array(mask).astype('float32')[:,:,np.newaxis]

    return image, mask

def image_preproc(image):
    p2, p98 = np.percentile(image, (2, 98))
    new_imarray = exposure.rescale_intensity(image, in_range=(p2, p98))
    new_imarray = data_norm(new_imarray, new_imarray.min(), new_imarray.max())
    return new_imarray

def mask_preprocc(mask):
    if mask is None:
        return None
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask)
    mask = morphology.binary_opening(mask)
    mask = np.array(mask).astype('float32')[:,:,np.newaxis]
    return mask

def prediction_postprocessing(all_pred, thr = None):
    if np.unique(all_pred).shape[0] == 1:
        return all_pred
    global_thresh = threshold_otsu(all_pred) if thr is None else thr
    all_pred = all_pred >= global_thresh
    all_pred = morphology.binary_opening(all_pred).astype('float32')
    return all_pred

def d4_image2mask(model, image):
    """Test-time augmentation for image classification that averages predictions
    of all D4 augmentations applied to input image.
    For segmentation we need to reverse the augmentation after making a prediction
    on augmented input.
    :param model: Model to use for making predictions.
    :param image: Model input.
    :return: Arithmetically averaged predictions
    """
    output = model(image).data.cpu()

    for aug, deaug in zip([torch_rot90, torch_rot180, torch_rot270],
                          [torch_rot270, torch_rot180, torch_rot90]):
        x = deaug(model(aug(image)).data.cpu())
        output = output + x

    image = torch_transpose(image)

    for aug, deaug in zip(
            [torch_none, torch_rot90, torch_rot180, torch_rot270],
            [torch_none, torch_rot270, torch_rot180, torch_rot90]):
        x = deaug(model(aug(image)).data.cpu())
        output = output + torch_transpose(x)

    one_over_8 = float(1.0 / 8.0)
    return output * one_over_8

def torch_none(x ):
    return x

def torch_rot90_(x ):
    return x.transpose_(2, 3).flip(2)


def torch_rot90(x ):
    return x.transpose(2, 3).flip(2)


def torch_rot180(x ):
    return x.flip(2).flip(3)


def torch_rot270(x ):
    return x.transpose(2, 3).flip(3)


def torch_flipud(x ):
    """
    Flip image tensor vertically
    :param x:
    :return:
    """
    return x.flip(2)


def torch_fliplr(x ):
    """
    Flip image tensor horizontally
    :param x:
    :return:
    """
    return x.flip(3)


def torch_transpose(x ):
    return x.transpose(2, 3)


def torch_transpose_(x ):
    return x.transpose_(2, 3)


def torch_transpose2(x ):
    return x.transpose(3, 2)