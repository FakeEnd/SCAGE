import io
import os
import pickle
from copy import deepcopy
import random
import lmdb

import torch
import numpy as np
import yaml
from tqdm import tqdm
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})
from data_process import algos
from utils import data_util

config_path = "config/config_pretrain_adcl_fg.yaml"
config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
config = config['model']

def transform(data):
    edge_index, x = data.edge_index, data.x
    N = x.size(0)
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    adj_w_sl = adj.clone()  # adj with self loop
    adj_w_sl[torch.arange(N), torch.arange(N)] = 1
    data.adj = adj_w_sl

    # print(adj_w_sl)
    in_degree = adj.long().sum(dim=1).view(-1)
    # print(in_degree)

    # if N < config['eig_pos_dim'] + 1:
    #     # print(adj_w_sl,x,idx)
    #     eig_pos_emb = torch.zeros(N, config['eig_pos_dim'])
    # else:
    #     eigval, eigvec = data_util.get_eig_dense(adj_w_sl)
    #     eig_idx = eigval.argsort()  # 返回由小到大的索引值
    #     eigval, eigvec = eigval[eig_idx], eigvec[:, eig_idx]
    #     eig_pos_emb = eigvec[:, 1:config['eig_pos_dim'] + 1]
    # # print(eig_pos_emb)
    #
    # if N < config['svd_pos_dim']:
    #     svd_pos_emb = torch.zeros(N, config['svd_pos_dim'] * 2)
    # else:
    #     pu, pv = data_util.get_svd_dense(adj_w_sl, config['svd_pos_dim'])
    #     svd_pos_emb = torch.cat([pu, pv], dim=-1)
    # print(svd_pos_emb)
    eig_pos_emb = 0
    svd_pos_emb = 0

    shortest_path_result, _ = algos.floyd_warshall(adj.numpy())
    spatial_pos = torch.from_numpy(shortest_path_result).long()
    data.spatial_pos = spatial_pos

    data.in_degree = in_degree
    data.out_degree = in_degree
    data.eig_pos_emb = eig_pos_emb
    data.svd_pos_emb = svd_pos_emb

    data.label = 0

    if config['use_super_node']:
        attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token
    else:
        attn_bias = torch.zeros([N, N], dtype=torch.float)
    data.attn_bias = attn_bias
    return data
def main():
    mask_rate=0.15
    # task_name_list = ['bace', 'bbbp', 'clintox', 'esol', 'freesolv', 'hiv', 'lipophilicity', 'muv', 'sider', 'tox21', 'toxcast']
    # task_name_list = ['bace']
    # task_name_list = ['CHEMBL204_Ki']
    folder_path = '/mnt/8t/qjb/workspace/SCAGE_DATA/cliff_data_split'
    file_names = os.listdir(folder_path)
    task_name_list = []
    for file_name in file_names:
        task_name_list.append(file_name)
    print(task_name_list)
    for task_name in task_name_list:
        print(f'now processing {task_name}')
        # path = f'/mnt/8t/qjb/workspace/SCAGE_DATA/finetune_data_split/{task_name}'
        path = f'/mnt/8t/qjb/workspace/SCAGE_DATA/cliff_data_split/{task_name}'
        all_file = os.listdir(path)
        for file in tqdm(all_file):
            if file.endswith('.pth') and file != 'pre_transform.pth' and file != 'pre_filter.pth':
                data_path = os.path.join(path, file)
                raw_data = torch.load(data_path)
                mask_data = deepcopy(raw_data)
                # print(data)
                raw_data = transform(raw_data)


                num_atoms = raw_data.x.size(0)
                sample_size = int(num_atoms * mask_rate + 1)
                masked_atom_indices = random.sample(range(num_atoms), sample_size)
                for atom_idx in masked_atom_indices:
                    mask_data.x[atom_idx] = torch.tensor([118, 0, 11, 11, 9, 5, 5, 2, 2])

                connected_edge_indices = []
                for bond_idx, (u, v) in enumerate(mask_data.edge_index.numpy().T):
                    for atom_idx in masked_atom_indices:
                        if atom_idx in {u, v} and bond_idx not in connected_edge_indices:
                            connected_edge_indices.append(bond_idx)
                if len(connected_edge_indices) > 0:
                    # modify the original bond features of the bonds connected to the mask atoms
                    for bond_idx in connected_edge_indices:
                        mask_data.edge_attr[bond_idx] = torch.tensor([4, 6, 2])


                # print(data)
                mask_data = transform(mask_data)
                torch.save((raw_data, mask_data), data_path)

def add_pretrain():
    mask_rate = 0.15
    path = f''
    all_file = os.listdir(path)
    for file in tqdm(all_file):
        if file.endswith('.pth') and file != 'pre_transform.pth' and file != 'pre_filter.pth':
            data_path = os.path.join(path, file)
            raw_data = torch.load(data_path)
            mask_data = deepcopy(raw_data)
            # print(data)
            raw_data = transform(raw_data)

            num_atoms = raw_data.x.size(0)
            sample_size = int(num_atoms * mask_rate + 1)
            masked_atom_indices = random.sample(range(num_atoms), sample_size)
            for atom_idx in masked_atom_indices:
                mask_data.x[atom_idx] = torch.tensor([118, 0, 11, 11, 9, 5, 5, 2, 2])

            connected_edge_indices = []
            for bond_idx, (u, v) in enumerate(mask_data.edge_index.numpy().T):
                for atom_idx in masked_atom_indices:
                    if atom_idx in {u, v} and bond_idx not in connected_edge_indices:
                        connected_edge_indices.append(bond_idx)
            if len(connected_edge_indices) > 0:
                # modify the original bond features of the bonds connected to the mask atoms
                for bond_idx in connected_edge_indices:
                    mask_data.edge_attr[bond_idx] = torch.tensor([4, 6, 2])

            # print(data)
            mask_data = transform(mask_data)
            torch.save((raw_data, mask_data), data_path)

def refresh_eig_svg():
    path = f'/mnt/8t/qjb/workspace/SCAGE_DATA/'
    all_file = os.listdir(path)
    for file in tqdm(all_file):
        if file.endswith('.pth') and file != 'pre_transform.pth' and file != 'pre_filter.pth':
            data_path = os.path.join(path, file)
            data = torch.load(data_path)
            data[0].eig_pos_emb = 0
            data[0].svd_pos_emb = 0
            data[1].eig_pos_emb = 0
            data[1].svd_pos_emb = 0
            torch.save((data[0], data[1]), data_path)

def lmdb_update():
    mask_rate = 0.15
    path = '/root/autodl-tmp/lmdb/demo'
    env = lmdb.open(path)
    txn = env.begin(write=True)
    length = int(txn.get(b'__len__').decode())
    for idx in tqdm(range(length)):
        raw_data = txn.get(str(idx).encode())
        buffer = io.BytesIO(raw_data)
        raw_data = torch.load(buffer)
        # print(raw_data)
        mask_data = deepcopy(raw_data)
        # print(data)
        raw_data = transform(raw_data)

        num_atoms = raw_data.x.size(0)
        sample_size = int(num_atoms * mask_rate + 1)
        masked_atom_indices = random.sample(range(num_atoms), sample_size)
        for atom_idx in masked_atom_indices:
            mask_data.x[atom_idx] = torch.tensor([118, 0, 11, 11, 9, 5, 5, 2, 2])

        connected_edge_indices = []
        for bond_idx, (u, v) in enumerate(mask_data.edge_index.numpy().T):
            for atom_idx in masked_atom_indices:
                if atom_idx in {u, v} and bond_idx not in connected_edge_indices:
                    connected_edge_indices.append(bond_idx)
        if len(connected_edge_indices) > 0:
            # modify the original bond features of the bonds connected to the mask atoms
            for bond_idx in connected_edge_indices:
                mask_data.edge_attr[bond_idx] = torch.tensor([4, 6, 2])

        # print(data)
        mask_data = transform(mask_data)
        data = (raw_data, mask_data)
        data = pickle.dumps(data)
        txn.put(str(idx).encode(), data)

    txn.commit()

def read_lmdb():
    path = '/root/autodl-tmp/lmdb/demo'
    env = lmdb.open(path)
    txn = env.begin()
    length = int(txn.get(b'__len__').decode())
    keys = pickle.loads(txn.get(b'__keys__'))
    # print(keys)
    for idx in tqdm(range(length)):
        data = txn.get(str(idx).encode())
        data = pickle.loads(data)
        print(data[0].x)
        print(data[0].fg)

def extra_env():
    source_path = '/root/autodl-tmp/lmdb/pubchem_1000_new'
    target_path = '/root/autodl-tmp/lmdb/demo'

    sou_env = lmdb.open(source_path)
    sou_txn = sou_env.begin()

    tar_env = lmdb.open(target_path)
    tar_txn = tar_env.begin(write=True)

    keys = []

    for idx in tqdm(range(10)):
        sou_data = sou_txn.get(str(idx).encode())

        keys.append(str(idx).encode())
        tar_txn.put(str(idx).encode(), sou_data)
    tar_txn.commit()

    with tar_env.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', str(len(keys)).encode())

# oriange=9941794  finetune=163045  cliff=48707
def lmdb_add_newdata():
    # data = torch.load('/root/autodl-tmp/finetune_data_sum/data_0.pth')
    # print(data)
    path = '/root/autodl-tmp/lmdb/pubchem_1000_new'
    data_path = '/root/autodl-tmp/cliff_data_sum/'
    env = lmdb.open(path, map_size=1024*1024*1024*1024)
    txn = env.begin(write=True)
    length = int(txn.get(b'__len__').decode())
    # length = 10
    print(length)
    keys = pickle.loads(txn.get(b'__keys__'))
    # print(keys)
    all_file = os.listdir(data_path)
    idx = length
    for file in tqdm(all_file):
        if file.endswith('.pth') and file != 'pre_transform.pth' and file != 'pre_filter.pth':
            # if idx < 20:
            data = os.path.join(data_path, file)
            data = torch.load(data)
            data = pickle.dumps(data)
            # file = open(data, 'rb')
            # print(file.read())
            keys.append(str(idx).encode())
            txn.put(str(idx).encode(), data)
            idx += 1
    txn.commit()
    with env.begin(write=True) as txn:
        txn.put(b'__keys__', pickle.dumps(keys))
        txn.put(b'__len__', str(len(keys)).encode())

def get_error_files():
    path = '/mnt/solid/tss/Project-2023/SCAGE2.0/data/pretrain_data/pubchem_1000/processed/'
    all_file = os.listdir(path)
    for file in tqdm(all_file):
        if file.endswith('.pth') and file != 'pre_transform.pth' and file != 'pre_filter.pth':
            data = os.path.join(path, file)
            data = torch.load(data)
            x = data.x
            fg = data.fg
            if len(x) != len(fg):
                print(file)


if __name__ == '__main__':
    get_error_files()
    # lmdb_add_newdata()
    # read_lmdb()
    # extra_env()
    # lmdb_update()
    # refresh_eig_svg()
    # edge_index, x = data.edge_index, data.x
    # N = x.size(0)
    # adj = torch.zeros([N, N], dtype=torch.bool)
    # adj[edge_index[0, :], edge_index[1, :]] = True
    #
    # adj_w_sl = adj.clone()  # adj with self loop
    # adj_w_sl[torch.arange(N), torch.arange(N)] = 1
    # if N < config['eig_pos_dim'] + 1:
    #     # print(adj_w_sl,x,idx)
    #     eig_pos_emb = torch.zeros(N, config['eig_pos_dim'])
    # else:
    #     eigval, eigvec = data_util.get_eig_dense(adj_w_sl)
    #     eig_idx = eigval.argsort()  # 返回由小到大的索引值
    #     eigval, eigvec = eigval[eig_idx], eigvec[:, eig_idx]
    #     eig_pos_emb = eigvec[:, 1:config['eig_pos_dim'] + 1]
    # # print(eig_pos_emb)
    #
    # if N < config['svd_pos_dim']:
    #     svd_pos_emb = torch.zeros(N, config['svd_pos_dim'] * 2)
    # else:
    #     pu, pv = data_util.get_svd_dense(adj_w_sl, config['svd_pos_dim'])
    #     svd_pos_emb = torch.cat([pu, pv], dim=-1)
    # data.eig_pos_emb = eig_pos_emb
    # data.svd_pos_emb = svd_pos_emb








