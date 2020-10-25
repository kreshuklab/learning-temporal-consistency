import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from PIL import Image
import torch
from torchvision import transforms, utils

class DimLoader(Dataset):
    def __init__(self, data_paths, name2frame, num_channels=7, seq_range=15, transform=None, shuffle=True):
        self.data_paths = data_paths
        self.name2frame = name2frame
        self.transform = transform
        self.num_channels = num_channels
        self.seq_range = seq_range
        self.shuffle = shuffle
        assert len(data_paths) == len(name2frame)
        assert seq_range >= 1

        flag = (np.array(self.name2frame) - self.num_channels//2 - self.seq_range)>=1
        self.data_paths = self.data_paths[flag]
        self.name2frame = self.name2frame[flag]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        res_imgs = []
        cur_img = Image.open(self.data_paths[idx])

        for cur_idx in list(range(-self.seq_range, self.seq_range+1, 5)):
            focused_frame_num = self.name2frame[idx] + cur_idx

            img_arr = []
            for chan_idx in range(focused_frame_num - self.num_channels//2, focused_frame_num + self.num_channels//2 + self.num_channels%2):
                cur_img.seek(chan_idx)
                img_arr.append(np.array(cur_img))
            img_arr = np.stack(img_arr, axis=0).astype('float32')
            
            img_arr, _ = self.transform(img_arr, None)

            res_imgs.append(img_arr)

        
        res_imgs = [transforms.ToTensor()(np.ascontiguousarray(res_imgs_curr))[None] for res_imgs_curr in res_imgs]
        return torch.cat(res_imgs, dim=0)

class SeqLoader(Dataset):
    def __init__(self, data_paths, name2frame, num_channels=7, seq_len=2, transform=None, use_npy_data=False):

        self.data_paths = data_paths
        self.name2frame = name2frame
        self.transform = transform
        self.num_channels = num_channels
        self.seq_len = seq_len
        self.use_ready = use_npy_data

    def __len__(self):
        return len(self.data_paths)-self.seq_len - 1

    def __getitem__(self, idx):
        res_imgs = []
        for cur_idx in range(self.seq_len):
            cur_img = Image.open(self.data_paths[idx+cur_idx])
            focused_frame_num = self.name2frame[idx+cur_idx]
            
            img_arr = []
            for chan_idx in range(focused_frame_num - self.num_channels//2, focused_frame_num + self.num_channels//2 + self.num_channels%2):
                cur_img.seek(chan_idx)
                img_arr.append(np.array(cur_img))
            img_arr = np.stack(img_arr, axis=0).astype('float32')

            res_imgs.append(img_arr)
        
        if self.transform:
            res_imgs = self.transform(res_imgs)
            
        res_imgs = [transforms.ToTensor()(np.ascontiguousarray(res_imgs_curr))[None] for res_imgs_curr in res_imgs]
        return torch.cat(res_imgs, dim=0)

class TrainLoader(Dataset):
    def __init__(self, images_names, mask_names=None, target_names=None, focused_frame=None, transform=None, num_channels=7, 
                return_original = False, use_npy_data=False, deviate=False, chrom = False, return_target = False):

        self.transform = transform
        self.images_names = images_names
        self.mask_names = mask_names
        self.return_original = return_original
        self.num_channels = num_channels
        self.target_names = target_names
        self.focused_frame = focused_frame
        self.use_ready = use_npy_data
        self.deviate = deviate
        self.chrom = chrom
        self.return_target = return_target
            
    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, idx):

        image = Image.open(self.images_names[idx])
        img_arr = []
        if self.deviate:
            shift = np.random.randint(-3, 3, size=1)[0]
            focused_frame_num = self.focused_frame[idx] + shift
        elif self.chrom:
            shift = 5
            focused_frame_num = self.focused_frame[idx] + shift
        else:
            focused_frame_num = self.focused_frame[idx]

        for chan_idx in range(focused_frame_num - self.num_channels//2, focused_frame_num + self.num_channels//2 + self.num_channels%2):
            image.seek(chan_idx)
            imarray_idx = np.array(image)
            img_arr.append(imarray_idx)
        image = np.stack(img_arr, axis=0).astype('float32')

        if self.mask_names is not None:
            if self.chrom:
                mask = np.load(self.mask_names[idx])[:,:,0]
            else:
                mask = np.load(self.mask_names[idx])
            mask = mask==1
        else:
            mask = None
            
        if self.return_original:
            original_mask = mask.copy()
            
        if self.transform:
            image, mask = self.transform(image, mask)

        image = transforms.ToTensor()( np.ascontiguousarray(image))
        if self.mask_names is not None:
            mask = transforms.ToTensor()( np.ascontiguousarray(mask))
        else:
            mask = [-1]
        
        if self.return_target:
            orig_tar = np.array(Image.open(self.target_names[idx]))
            orig_tar = np.array(orig_tar).astype('float32')[:,:,np.newaxis]
            orig_tar = transforms.ToTensor()( np.ascontiguousarray(orig_tar))
            return {
                'X':image, 
                'Y':mask, 
                'orig_tar':orig_tar
            }


        if self.return_original:
            original_mask = np.array(original_mask).astype('float32')[:,:,np.newaxis]
            original_mask = transforms.ToTensor()( np.ascontiguousarray(original_mask))
            
            return {
                'X':image, 
                'Y':mask, 
                'origin_mask':original_mask
            }

        if self.mask_names is not None:
            return {
                'X':image, 
                'Y':mask,
            }
        
        return image


class BatchSampler(Sampler):
    def __init__(self, data_source, batch_size):
        if  isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        self.data_source = data_source
        self.batch_size = batch_size
        self.range_idx = list(range(len(data_source)))
        self.flags = data_source.tritc_flags
        self.size = len(data_source)

        self.usual_pull = (np.array(self.range_idx)[self.flags==False]).tolist()
        self.tritc_pull = (np.array(self.range_idx)[self.flags==True]).tolist()
        self.two_pools = [self.usual_pull, self.tritc_pull]

    def __iter__(self):
        for _ in range(self.__len__()):
            cur_chance = np.random.randint(low=0, high=2, size=1)[0]
            cur_pool = self.two_pools[cur_chance]
            if len(cur_pool) == 0:
                cur_pool = self.two_pools[1-cur_chance]

            cur_idxs = np.random.choice(cur_pool, size = min(len(cur_pool), self.batch_size), replace=False).tolist()
            [cur_pool.remove(x) for x in cur_idxs]
            yield cur_idxs
        self.__init__(self.data_source, self.batch_size)

    def __len__(self):
        return self.one_len(len(self.usual_pull)) + self.one_len(len(self.tritc_pull))

    def one_len(self, cur_size):
        return (cur_size + self.batch_size - 1) // self.batch_size