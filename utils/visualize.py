import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage.filters import sobel
plt.switch_backend('agg')

def make_merge_plot(mask, input_data):
    boarders = sobel(mask)
    boarders[boarders > 0] = 1
    merge_plot = 1-input_data/np.max(input_data) + ( boarders) * 0.3
    return merge_plot

def print_images(images,titles,rows, cols, size=(10,10), cmaps=''):
    if cmaps == '':
        cmaps = ['']*rows*cols
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=size)
    if cmaps == '':
        cmaps = ''*rows*cols
    for idx, (img, title, cmap) in enumerate(zip(images, titles, cmaps)):
        axs.flat[idx].set_title(title)
        if cmap != '':
            axs.flat[idx].imshow(img, cmap=cmap)
        else:
            axs.flat[idx].imshow(img)
    return fig, axs.flat