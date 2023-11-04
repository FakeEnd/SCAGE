from rdkit import Chem
from rdkit.Chem import AllChem
import random
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

from torch_geometric.utils import to_undirected
from torch_geometric.data import Data
# import sys
# sys.path.append('/home/tss/Project-2023/SCAGE2.0/utils')
from utils import data_util
from rdkit import RDLogger
import pyximport

# pyximport.install(setup_args={"include_dirs": np.get_include()})
pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos

RDLogger.DisableLog('rdApp.*')


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


class PreTransformFn(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, raw_data):
        edge_index, x = raw_data.edge_index, raw_data.x
        N = x.size(0)
        adj = torch.zeros([N, N], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True
        raw_data.list_adj = [adj]
        return raw_data


class TransformFn(object):
    def __init__(self, config):

        self.config = config['model']
        self.is_pretrain = config['is_pretrain']

    def __call__(self, raw_data):
        if hasattr(raw_data, 'label'):
            label = raw_data.label
        else:
            label = 0
        if hasattr(raw_data, 'atom_symbol'):
            atom_symbol = raw_data.atom_symbol
        else:
            atom_symbol = 0
        # if self.is_pretrain:
        #     smiles = 0
        # else:
        #     smiles = raw_data.smiles
        smiles = raw_data.smiles
        # fg = raw_data.fg
        edge_attr, edge_index, x = raw_data.edge_attr, raw_data.edge_index, raw_data.x
        N = x.size(0)
        adj = torch.zeros([N, N], dtype=torch.bool)
        adj[edge_index[0, :], edge_index[1, :]] = True

        adj_w_sl = adj.clone()  # adj with self loop
        adj_w_sl[torch.arange(N), torch.arange(N)] = 1

        # positional bias
        if 'degree' in self.config['node_level_modules']:
            in_degree = adj.long().sum(dim=1).view(-1)
        else:
            in_degree = 0

        if 'eig' in self.config['node_level_modules']:
            if N < self.config['eig_pos_dim'] + 1:
                # print(adj_w_sl,x,idx)
                eig_pos_emb = torch.zeros(N, self.config['eig_pos_dim'])
            else:
                eigval, eigvec = data_util.get_eig_dense(adj_w_sl)
                eig_idx = eigval.argsort()  # 返回由小到大的索引值
                eigval, eigvec = eigval[eig_idx], eigvec[:, eig_idx]
                eig_pos_emb = eigvec[:, 1:self.config['eig_pos_dim'] + 1]

        else:
            eig_pos_emb: int = 0

        if 'svd' in self.config['node_level_modules']:
            if N < self.config['svd_pos_dim']:
                svd_pos_emb = torch.zeros(N, self.config['svd_pos_dim'] * 2)
            else:
                pu, pv = data_util.get_svd_dense(adj_w_sl, self.config['svd_pos_dim'])
                svd_pos_emb = torch.cat([pu, pv], dim=-1)
        else:
            svd_pos_emb = 0

        shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        spatial_pos = torch.from_numpy(shortest_path_result).long()

        # super node
        if self.config['use_super_node']:
            attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token
        else:
            attn_bias = torch.zeros([N, N], dtype=torch.float)

        # ---------------------------------分割线----------------------------------- #
        item = Data(x=x,
                    edge_attr=edge_attr,
                    label=label,
                    attn_bias=attn_bias,
                    in_degree=in_degree,
                    out_degree=in_degree,
                    eig_pos_emb=eig_pos_emb,
                    svd_pos_emb=svd_pos_emb,
                    spatial_pos=spatial_pos,
                    adj=adj_w_sl,
                    edge_index=edge_index,
                    bond_length=raw_data.bond_length,
                    BondAngleGraph_index=raw_data.BondAngleGraph_index,
                    bond_angle=raw_data.bond_angle,
                    smiles=smiles,
                    atom_symbol=atom_symbol,
                    # fg=fg
                    )
        return item


class MaskTransformFn(object):
    def __init__(self, config, mask_rate=0.15):
        self.mask_rate = mask_rate
        self.config = config
        self.transform = TransformFn(config)

    def __call__(self, raw_data):
        mask_data = deepcopy(raw_data)
        raw_item = self.transform(raw_data)

        num_atoms = raw_data.x.size(0)
        sample_size = int(num_atoms * self.mask_rate + 1)
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
                # mask_data.bond_length[bond_idx] = -1
        # connected_angle_indices = []
        # for angle_inx, (u, v) in enumerate(mask_data.BondAngleGraph_index.numpy().T):
        #     for bond_idx in connected_edge_indices:
        #         if bond_idx in {u, v} and angle_inx not in connected_angle_indices:
        #             connected_angle_indices.append(angle_inx)
        # if len(connected_angle_indices) > 0:
        #     for angle_inx in connected_angle_indices:
        #         mask_data.bond_angle[angle_inx] = -1

        mask_item = self.transform(mask_data)

        return raw_item, mask_item
