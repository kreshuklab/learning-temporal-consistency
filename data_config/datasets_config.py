import os 

def data_train1():
    d = {
            'name': 'train_data_1', 
            'data_path':'/g/kreshuk/data/label-free/holographic imaging/RI_and_FITC-ground-truth/',
            'data_names':[],

            'raw_target_path':'/g/kreshuk/data/label-free/holographic imaging/RI_and_FITC-ground-truth/',
            'raw_target_names':[],

            'mask_path':'/g/kreshuk/shabanov/nuclei/GT/gt_masks_1/',
            'mask_names':[],

            'train_idx':[10, 20,  1,  3, 18, 17,  8, 24, 21,  4,  5, 16, 27,  7, 28],
            'val_idx':[11,  6, 15, 26, 23, 13, 14, 25, 12,  0, 22,  9, 19,  2],
            'test_idx':[],
            'use_to_train':True,

            'name2focused_frame':{}
        }
    file_names = sorted(os.listdir(d['data_path']))
    data_names = [x for x in file_names if 'RI' in x]
    target_names = [x for x in file_names if 'FITC' in x]
    d['data_names'] = data_names
    d['raw_target_names'] = target_names

    d['mask_names'] = [x for x in sorted(os.listdir(d['mask_path'])) if x.startswith('daja')]

    d['name2focused_frame'] = dict([(x, 45) for x in  d['data_names']])
    return d

def data_train2():
    d = {
            'name': 'train_data_2', 
            'data_path':'/g/kreshuk/data/label-free/holographic imaging/RI_and_FITC_TRITC_ground_truth/',
            'data_names':[],

            'raw_target_path':'/g/kreshuk/data/label-free/holographic imaging/RI_and_FITC_TRITC_ground_truth/',
            'raw_target_names':[],

            'mask_path':'/g/kreshuk/shabanov/nuclei/GT/gt_masks_2/',
            'mask_names':[],

            'tritc_path':'/home/shabanov/docs/nuclei/GT/gt_TRITC_2/',
            'tritc_names':[],
            'train_tritc':True,

            'train_idx':[31, 47, 28, 63, 30, 11, 57, 66, 39, 18, 59, 24, 32, 55, 62, 27,
                60,  8, 34, 38, 13, 29, 58, 43, 64, 54, 17, 49, 51, 14, 44,
                56, 16, 33, 37, 42, 65, 10,  7, 45,  4,  5, 21],
            'val_idx': [20,  6, 36, 48, 61, 40,  1, 52, 23, 41, 35, 46, 26, 15, 50,  3, 53, 25, 12,  0, 22,  9, 19,  2],
            'test_idx': [],
            'use_to_train':True,

            'name2focused_frame':{}
        }
    DATA_PATH_2 = d['data_path']
    data_names_2 = sorted([x for x in sorted(os.listdir(DATA_PATH_2)) if 'RI' in x and 'TRI' not in x])
    target_names_2 = sorted([x for x in sorted(os.listdir(DATA_PATH_2)) if 'FITC' in x])
    d['data_names'] = data_names_2
    d['raw_target_names'] = target_names_2

    d['mask_names'] = [x for x in sorted(os.listdir(d['mask_path'])) if x.startswith('c-36')]

    d['name2focused_frame'] = dict([(x, 45) for x in  d['data_names']])

    d['tritc_names'] = [x for x in sorted(os.listdir(d['tritc_path'])) if x.startswith('c-36')]
    return d

def test_1():
    d = {
            'name': 'without_gt1', 
            'data_path':'/g/kreshuk/data/label-free/holographic imaging/images_without_ground-truth/',
            'use_to_train':False,
            'name2focused_frame': {
                'daja-c1-actD-pos1_RI_frame01.tiff':20,
                'daja-c1-actD-pos2_RI_frame01.tiff':33,
                'daja-c1-actD-pos3_RI_frame01.tiff':40,
                'daja-c1-actD-pos4_RI_frame01.tiff':10,
                'daja-c1-actD-pos5_RI_frame01.tiff':40,
                'daja-c1-actD-pos6_RI_frame01.tiff':40,
                'daja-c1-actD-pos7_RI_frame01.tiff':40,
                'daja-c1-si4-pos1_RI_frame01.tiff':40,
                'daja-c1-si4-pos2_RI_frame01.tiff':40,
                'daja-c1-si4-pos3_RI_frame01.tiff':40,
                'daja-c1-si4-pos4_RI_frame01.tiff':40,
                'daja-c1-si4-pos5_RI_frame01.tiff':42,
                'daja-c1-si4-pos6_RI_frame01.tiff':47,
                'daja-c1-si4-sure-pos1_RI_frame01.tiff':45,
                'daja-c1-si4-sure-pos2_RI_frame01.tiff':42,
                'daja-c1-si4-sure-pos3_RI_frame01.tiff':41,
                'daja-c1-si4-sure-pos4_RI_frame01.tiff':40,
                "daja-c1-si4-sure-pos4'2_RI_frame01.tiff":30,
                'daja-c1-si11-pos1_RI_frame01.tiff':40,
                'daja-c1-si11-pos2_RI_frame01.tiff':40,
                'daja-c1-si11-pos3_RI_frame01.tiff':40,
                'daja-c1-si11-pos4_RI_frame01.tiff':32,
                'daja-c1-si11-pos5_RI_frame01.tiff':40,
                'daja-c1-si11-pos6_RI_frame01.tiff':36,
                'daja-c1-si11-pos8_RI_frame01.tiff':40,
                'daja-c1-si11-sure-pos1_RI_frame01.tiff':45,
                'daja-c1-si11-sure-pos2_RI_frame01.tiff':41
            }
        }

    d['data_names'] = [x for x in sorted(os.listdir(d['data_path'])) if x.startswith('daja')]
    return d

def test_2():

    import numpy as np
    def make_good_dict(cur_dict):
        for k in cur_dict.keys():
            cur_dict[k] = int(cur_dict[k])
        return cur_dict

    d = {
            'name': 'without_gt2', 
            'data_path':'/g/kreshuk/data/label-free/holographic imaging/images_without_ground-truth2/',
            'use_to_train':False,
            'name2focused_frame': {}
        }

    test_files_6 = [x for x in sorted(os.listdir(d['data_path'])) if 'sample' in x]
    d['data_names'] = test_files_6

    d['name2focused_frame'] = make_good_dict(dict(np.load('focused_frames/'+d['data_path'].split('/')[-2]+'.npy')))
    return d

def movie1():
    import numpy as np
    def make_good_dict(cur_dict):
        for k in cur_dict.keys():
            cur_dict[k] = int(cur_dict[k])
        return cur_dict
        
    d = {
            'name': 'movie-holo-overnight-day1', 
            'data_path': '/g/kreshuk/data/label-free/holographic imaging/images_without_ground-truth/movie-holo-overnight-day1/',
            'use_to_train':False,
            'use_time_filtering':True,
            'is_sequential':True,
            'name2focused_frame': [],
    }

    test_files_2 = np.array([x for x in sorted(os.listdir(d['data_path']))])
    file_numbers = [int(x.split('.')[-2].split('frame')[-1]) for x in test_files_2]
    test_files_2 = test_files_2[np.argsort(file_numbers)].tolist()
    d['data_names'] = test_files_2#[:30]

    d['name2focused_frame'] = make_good_dict(dict(np.load('focused_frames/'+d['data_path'].split('/')[-2]+'.npy')))
    return d

def movie2():

    import numpy as np
    def make_good_dict(cur_dict):
        for k in cur_dict.keys():
            cur_dict[k] = int(cur_dict[k])
        return cur_dict

    d = {
            'name': 'movie-holo-overnight-day2', 
            'data_path': '/g/kreshuk/data/label-free/holographic imaging/images_without_ground-truth/movie-holo-overnight-day2/',
            'use_to_train':False,
            'use_time_filtering':True,
            'is_sequential':True,
            'name2focused_frame':{}
        }

    test_files_3 = np.array([x for x in sorted(os.listdir(d['data_path']))])
    file_numbers = [int(x.split('.')[-2].split('frame')[-1]) for x in test_files_3]
    test_files_3 = test_files_3[np.argsort(file_numbers)].tolist()
    d['data_names'] = test_files_3#[:170]

    d['name2focused_frame'] = make_good_dict(dict(np.load('focused_frames/'+d['data_path'].split('/')[-2]+'.npy')))
    return d

def sample1_interhase():

    d = {
            'name': 'sample-1_c-36_untreated_interphases', 
            'data_path': '/g/kreshuk/data/label-free/holographic imaging/19-02-2020_ground_truth_for_3D_segm/02_raw_data/Sample-1_c-36_untreated_interphases/',
            'data_names':[],

            'raw_target_path':'/g/kreshuk/data/label-free/holographic imaging/19-02-2020_ground_truth_for_3D_segm/02_raw_data/Sample-1_c-36_untreated_interphases/',
            'raw_target_names':[],

            'mask_path':'/home/shabanov/docs/nuclei/GT/Sample-1_c-36_untreated_interphases/FITC/',
            'mask_names':[],

            'tritc_path':'/home/shabanov/docs/nuclei/GT/Sample-1_c-36_untreated_interphases/TRITC/',
            'tritc_names':[],
            'train_tritc':True,

            'train_idx':[ 7,  1,  5,  8, 12,  6,  2,  4],
            'val_idx': [0, 3, 9, 10, 11],
            'test_idx':[],

            'name2focused_frame':{},

            'absolute_data_names':[],

            'is_sequential':False,
            'use_to_train':True,
            'use_time_filtering':False
        }


    d['data_names'] = sorted([x for x in sorted(os.listdir(d['data_path'])) if 'RI' in x and 'TRI' not in x and 'infocus' in x])
    d['raw_target_names'] = sorted([x for x in sorted(os.listdir(d['data_path'])) if 'FITC' in x and 'infocus' in x])
    

    d['mask_names'] = [x for x in sorted(os.listdir(d['mask_path'])) if x.startswith('e-') and 'infocus' in x]
    d['tritc_names'] = [x for x in sorted(os.listdir(d['tritc_path'])) if x.startswith('e-') and 'infocus' in x]

    d['name2focused_frame'] = dict([(x, 45) for x in  d['data_names']])
    return d

def sample2_interhase():
    d = {
            'name': 'sample-2_c-36_ActD_interphases', 
            'data_path': '/g/kreshuk/data/label-free/holographic imaging/19-02-2020_ground_truth_for_3D_segm/02_raw_data/Sample-2_c-36_ActD_interphases/',
            'data_names':[],

            'raw_target_path':'/g/kreshuk/data/label-free/holographic imaging/19-02-2020_ground_truth_for_3D_segm/02_raw_data/Sample-2_c-36_ActD_interphases/',
            'raw_target_names':[],

            'mask_path':'/home/shabanov/docs/nuclei/GT/Sample-2_c-36_ActD_interphases/FITC/',
            'mask_names':[],

            'tritc_path':'/home/shabanov/docs/nuclei/GT/Sample-2_c-36_ActD_interphases/TRITC/',
            'tritc_names':[],
            'train_tritc':True,

            'train_idx':[ 1,2,4],
            'val_idx': [0, 3, 5],
            'test_idx':[],

            'name2focused_frame':{},

            'absolute_data_names':[],

            'is_sequential':False,
            'use_to_train':True,
            'use_time_filtering':False
        }


    d['data_names'] = sorted([x for x in sorted(os.listdir(d['data_path'])) if 'RI' in x and 'TRI' not in x and 'infocus' in x])
    d['raw_target_names'] = sorted([x for x in sorted(os.listdir(d['data_path'])) if 'FITC' in x and 'infocus' in x])
    

    d['mask_names'] = [x for x in sorted(os.listdir(d['mask_path'])) if x.startswith('e-') and 'infocus' in x]
    d['tritc_names'] = [x for x in sorted(os.listdir(d['tritc_path'])) if x.startswith('e-') and 'infocus' in x]

    d['name2focused_frame'] = dict([(x, 45) for x in  d['data_names']])
    return d

def sample1_metaphase():
    d = {
            'name': 'sample-1_c-36_untreated_metaphases', 
            'data_path': '/g/kreshuk/data/label-free/holographic imaging/19-02-2020_ground_truth_for_3D_segm/02_raw_data/Sample-1_c-36_untreated_metaphases/',
            'data_names':[],

            'raw_target_path':'/g/kreshuk/data/label-free/holographic imaging/19-02-2020_ground_truth_for_3D_segm/02_raw_data/Sample-1_c-36_untreated_metaphases/',
            'raw_target_names':[],

            'mask_path':'/home/shabanov/docs/nuclei/GT/Sample-1_c-36_untreated_metaphases/exported_data_CHROM/',
            'mask_names':[],

            'tritc_path':'',
            'tritc_names':[],
            'train_tritc':False,

            'train_idx': [0],
            'val_idx': [1,2],
            'test_idx':[],

            'name2focused_frame':{},

            'absolute_data_names':[],

            'is_sequential':False,
            'use_to_train':False,
            'use_time_filtering':False,
            'chrom':True,
        }

    d['mask_names'] = [x for x in sorted(os.listdir(d['mask_path'])) if 'oof' not in x and 'diff-focus' not in x]

    d['data_names'] = [x.replace('TRITC', 'RI').replace('npy', 'tiff') for x in d['mask_names']]
    d['raw_target_names'] = [x.replace('npy', 'tiff') for x in d['mask_names']]
    


    d['name2focused_frame'] = dict([(x, 45) for x in  d['data_names']])
    return d


def sample4_metaphase():
    d = {
            'name': 'sample-4_c-44_untreated_mitotic-cells', 
            'data_path': '/g/kreshuk/data/label-free/holographic imaging/19-02-2020_ground_truth_for_3D_segm/02_raw_data/Sample-4_c-44_untreated_mitotic-cells/',
            'data_names':[],

            'raw_target_path':'/g/kreshuk/data/label-free/holographic imaging/19-02-2020_ground_truth_for_3D_segm/02_raw_data/Sample-4_c-44_untreated_mitotic-cells/',
            'raw_target_names':[],

            'mask_path':'/home/shabanov/docs/nuclei/GT/Sample-4_c-44_untreated_mitotic-cells/exported_data_CHROM/',
            'mask_names':[],

            'tritc_path':'',
            'tritc_names':[],
            'train_tritc':False,

            'train_idx': [7, 1, 17, 12, 14, 11, 10, 4, 2, 13, 6, 0, 3, 15],
            'val_idx': [19, 16, 18, 8, 5, 9],
            'test_idx':[],

            'name2focused_frame':{},

            'absolute_data_names':[],

            'is_sequential':False,
            'use_to_train':False,
            'use_time_filtering':False,
            'chrom':True,
        }

    d['mask_names'] = [x for x in sorted(os.listdir(d['mask_path'])) if 'oof' not in x and 'diff-focus' not in x]

    d['data_names'] = [x.replace('FITC', 'RI').replace('npy', 'tiff') for x in d['mask_names']]
    d['raw_target_names'] = [x.replace('npy', 'tiff') for x in d['mask_names']]
    
    d['name2focused_frame'] = dict([(x, 45) for x in  d['data_names']])
    return d