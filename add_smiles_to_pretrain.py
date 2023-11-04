import os
import pandas as pd
import torch
from tqdm import tqdm


def extract_number(name):
    if name.lower().endswith('.pth'):
        return int(name.split('.')[0].split('_')[1])
    else:
        return -1

for idx in range(1):
    print(f'now processing {idx}')
    path = f'/mnt/sde/qjb/SCAGE/data/pubchem_split/pubchem_{idx}'
    # path = '/mnt/solid/tss/Project-2023/SCAGE/data/pretrain/pubchem_demo/processed'
    data = os.listdir(path)
    data = sorted(data, key=lambda x: extract_number(x))
    smi_dict = {}
    df = pd.read_csv(os.path.join(path, data[0]), header=None)
    for i in range(len(df)):
        smi_dict[df[0][i]] = df[1][i]
    # n = 0
    for pyg_file in tqdm(data[1:]):
        # print(n)
        # n = n + 1
        pyg_data = torch.load(os.path.join(path, pyg_file))
        number = int(pyg_file.split('.')[0].split('_')[1])
        pyg_data.smiles = smi_dict[number]
        torch.save(pyg_data, os.path.join(path, pyg_file))
