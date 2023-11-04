from data_process.data_transform import MaskTransformFn
import os
import yaml
import torch
from tqdm import tqdm
path = '/mnt/solid/tss/Project-2023/SCAGE2.0/data/pretrain_data/pubchem_demo/processed'
target_path = '/mnt/solid/tss/Project-2023/SCAGE2.0/data/pretrain_data/pubchem_demo4/processed'
config_path = '/mnt/8t/qjb/workspace/SCAGE2.0/config/config_pretrain_adcl.yaml'
config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

for file_name in tqdm(os.listdir(path)):
    if file_name.endswith('.pth') and file_name != 'pre_transform.pth' and file_name != 'pre_filter.pth':
        data_path = os.path.join(path, file_name)
        data = torch.load(data_path)
        pre_process = MaskTransformFn(config)
        raw_item, mask_item = pre_process(data)
        torch.save((raw_item, mask_item), os.path.join(target_path, file_name))


