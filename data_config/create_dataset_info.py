import os 

data_template = {
    'name': '', 
    'data_path':'',
    'data_names':[],

    'raw_target_path':'',
    'raw_target_names':[],

    'mask_path':'',
    'mask_names':[],

    'tritc_path':'',
    'tritc_names':[],

    'train_idx':[],
    'val_idx':[],
    'test_idx':[],

    'name2focused_frame':{},

    'absolute_data_names':[],

    'is_sequential':False,
    'train_tritc':False,
    'use_to_train':True,
    'chrom':False,
    'use_time_filtering':False
}

def assemble_dataset_from_func(func):
    temp = data_template.copy()
    temp.update(func())

    temp['absolute_data_names'] = [temp['data_path'] + '/' + x for x in temp['data_names']]
    temp['absolute_target_names'] = [temp['raw_target_path'] + '/' + x for x in temp['raw_target_names']]
    temp['absolute_mask_names'] = [temp['mask_path'] + '/' + x for x in temp['mask_names']]
    temp['absolute_tritc_names'] = [temp['tritc_path'] + '/' + x for x in temp['tritc_names']]
    return temp

def assemble_dataset_from_py():
    import importlib
    import inspect
    from inspect import getmembers, isfunction
    from . import datasets_config as dc

    all_data_info = []
    functions_list = [o for o in getmembers(dc) if isfunction(o[1])]
    for name, cur_func in functions_list:
        if name != 'assemble_dataset_info':
            assembled = assemble_dataset_from_func(cur_func)
            all_data_info.append(assembled)
    return all_data_info

if __name__=='__main__':
    assemble_dataset_from_py()