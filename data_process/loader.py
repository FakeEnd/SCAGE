import os
import time
import csv
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pathlib
from itertools import repeat
from tqdm import tqdm
import lmdb
import io

# from torch_geometric.data import Data, Dataset
from torch_geometric.data import Data
from torch.utils.data import Dataset
from torch_geometric.data import InMemoryDataset
import torch_geometric.utils as tg_utils

from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from rdkit import Chem
from rdkit.Chem import AllChem

import pickle

full_atom_feature_dims = get_atom_feature_dims()
full_bond_feature_dims = get_bond_feature_dims()
import warnings
warnings.filterwarnings("ignore")


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_atom_feature_dims):
            emb = torch.nn.Embedding(dim + 1, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        for i, dim in enumerate(full_bond_feature_dims):
            emb = torch.nn.Embedding(dim+2, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding

class FGEncoder(torch.nn.Module):

    def __init__(self, vocab_size, emb_dim):
        super(FGEncoder, self).__init__()

        self.emb = torch.nn.Embedding(vocab_size, emb_dim)
        torch.nn.init.xavier_uniform_(self.emb.weight.data)

    def forward(self, fg):
        fg_embedding = self.emb(fg)

        return fg_embedding



# class PretrainDataset(Dataset):
#     def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
#         self.root = root
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
#
#         self.time_ = 0
#         self.record_n = 0
#         self.et = 0
#
#     @property
#     def raw_file_names(self):
#         file_name_list = os.listdir(self.raw_dir)
#         return file_name_list
#
#     @property
#     def processed_file_names(self):
#         data_list = []
#         for file_name in os.listdir(self.processed_dir):
#             if file_name.endswith('.pth') and file_name != 'pre_transform.pth' and file_name != 'pre_filter.pth':
#                 data_list.append(file_name)
#         return data_list
#
#     def download(self):
#         pass
#
#     def process(self):
#         data_smiles_list = []
#         smiles_list, rdkit_mol_objs = _load_pretrain_dataset(self.raw_paths[0])
#         for i in range(len(rdkit_mol_objs)):
#             print(i)
#             rdkit_mol = rdkit_mol_objs[i]
#             if rdkit_mol is not None:
#                 data = mol_to_graph_data(rdkit_mol)
#                 data.smiles = smiles_list[i]
#                 data_smiles_list.append(smiles_list[i])
#                 torch.save(data, os.path.join(self.processed_dir, 'data_{}.pth'.format(i)))
#         # write data_smiles_list in process paths
#         data_smiles_series = pd.Series(data_smiles_list)
#         data_smiles_series.to_csv(os.path.join(self.processed_dir, 'smiles.csv'), index=False, header=False)
#
#     def len(self):
#         return len(self.processed_file_names)
#
#     def get(self, idx):
#         # print(idx)
#         st = time.time()
#         print(st - self.et)
#         data = torch.load(os.path.join(self.processed_dir, 'data_{}.pth'.format(idx)))
#         self.et = time.time()
#         # print(self.et - st)
#         # self.record_n += 1
#         # print(self.time_ / self.record_n, self.time_, self.record_n)
#         return data

# 使用lmdb
class PretrainDataset(Dataset):
    def __init__(self, root):
        self.root = root

        self.env = lmdb.open(self.root,
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        self.txn = self.env.begin()

        self.length = int(self.txn.get(b'__len__').decode())

        self.time_ = 0
        self.record_n = 0
        self.et = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # print(idx)
        # st = time.time()
        # print(st - self.et)
        # data = torch.load(os.path.join(self.processed_dir, 'data_{}.pth'.format(idx)))
        data = self.txn.get(str(idx).encode())
        data = pickle.loads(data)
        # buffer = io.BytesIO(data)
        # data = torch.load(buffer)
        # self.et = time.time()
        # print(self.et - st)
        # self.record_n += 1
        # print(self.time_ / self.record_n, self.time_, self.record_n)
        return data

# class PretrainDataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
#         self.root = root
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
#         self.data = self.load_data()
#         print(len(self.data))
#
#     @property
#     def raw_file_names(self):
#         file_name_list = os.listdir(self.raw_dir)
#         return file_name_list
#
#     @property
#     def processed_file_names(self):
#         data_list = []
#         for file_name in os.listdir(self.processed_dir):
#             if file_name.endswith('.pth') and file_name != 'pre_transform.pth' and file_name != 'pre_filter.pth':
#                 data_list.append(file_name)
#         return data_list
#
#     def load_data(self):
#         loaded_data = []
#         for filename in tqdm(self.processed_file_names):
#             file_path = os.path.join(self.processed_dir, filename)
#             if os.path.isfile(file_path):
#                 try:
#                     data = torch.load(file_path)
#                     loaded_data.append(data)
#                 except Exception as e:
#                     print(f"Error loading {filename}: {str(e)}")
#
#         return loaded_data
#
#     def download(self):
#         pass
#
#     def process(self):
#         data_smiles_list = []
#         smiles_list, rdkit_mol_objs = _load_pretrain_dataset(self.raw_paths[0])
#         for i in range(len(rdkit_mol_objs)):
#             print(i)
#             rdkit_mol = rdkit_mol_objs[i]
#             if rdkit_mol is not None:
#                 data = mol_to_graph_data(rdkit_mol)
#                 data.smiles = smiles_list[i]
#                 data_smiles_list.append(smiles_list[i])
#                 torch.save(data, os.path.join(self.processed_dir, 'data_{}.pth'.format(i)))
#         # write data_smiles_list in process paths
#         data_smiles_series = pd.Series(data_smiles_list)
#         data_smiles_series.to_csv(os.path.join(self.processed_dir, 'smiles.csv'), index=False, header=False)
#
#     def len(self):
#         return len(self.processed_file_names)
#
#     def get(self, idx):
#         data = self.data[idx]
#         return data


class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 target=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 is_pretrain=False,
                 ):
        self.is_pretrain = is_pretrain
        self.dataset = root.split('/')[1]
        self.root = root
        self.target = target
        # self.pre_process()

        super(MoleculeDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])
        if not is_pretrain:
            self.smiles_list = pd.read_csv(os.path.join(self.processed_dir, 'smiles.csv'), header=None).to_numpy()[:, 0]

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        if not self.is_pretrain:
            data['smiles'] = self.smiles_list[idx]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        pass

    def process(self):
        data_list = []
        smiles_list, rdkit_mol_objs, labels = _load_finetune_dataset_zxx(self.raw_paths[0], self.target)
        for i in range(len(rdkit_mol_objs)):
            print(i)
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol is not None:
                data = mol_to_graph_data(rdkit_mol)
                data.id = torch.tensor([i])
                data.label = torch.tensor(labels[i])
                data.smiles = smiles_list[i]
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class FinetuneDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 split_type,
                 target=None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 is_pretrain=False,
                 ):
        self.is_pretrain = is_pretrain
        self.dataset = root.split('/')[1]
        self.root = root
        self.split_type = split_type
        assert split_type in ['train', 'valid', 'test']
        self.target = target
        # self.pre_process()

        super(FinetuneDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(
            os.path.join(self.processed_dir, '{}_data_processed.pt'.format(split_type)))

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        if self.split_type == 'train':
            return 'train_data_processed.pt'
        elif self.split_type == 'valid':
            return 'valid_data_processed.pt'
        else:
            return 'test_data_processed.pt'

    def download(self):
        pass

    def process(self):
        data_list = []
        smiles_list, rdkit_mol_objs, labels = _load_finetune_dataset(self.raw_paths[0], self.target)
        for i in range(len(rdkit_mol_objs)):
            print(i)
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol is not None:
                data = mol_to_graph_data(rdkit_mol)
                data.id = torch.tensor([i])
                data.label = torch.tensor(labels[i])
                data.smiles = smiles_list[i]
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class CliffDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 split_type,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 ):
        self.root = root
        self.split_type = split_type
        assert split_type in ['train', 'test']
        super(CliffDataset, self).__init__(root, transform, pre_transform, pre_filter)

        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        self.data, self.slices = torch.load(
            os.path.join(self.processed_dir, '{}_data_processed.pt'.format(split_type)))

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        if self.split_type == 'train':
            return 'train_data_processed.pt'
        elif self.split_type == 'test':
            return 'test_data_processed.pt'

    def download(self):
        pass

    def process(self):
        data_list = []
        smiles_list, rdkit_mol_objs, label_y, label_cliff = _load_cliff_dataset(self.raw_paths[0], self.split_type)
        for i in range(len(rdkit_mol_objs)):
            print(i)
            rdkit_mol = rdkit_mol_objs[i]
            if rdkit_mol is not None:
                data = mol_to_graph_data(rdkit_mol)
                data.id = torch.tensor([i])
                data.label_y = torch.tensor(label_y[i])
                data.label_cliff = torch.tensor(label_cliff[i])
                data.smiles = smiles_list[i]
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class PerturbDataset(InMemoryDataset):
    def __init__(self,
                 root,
                 target,
                 split,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 ):
        self.root = root
        self.target = target
        self.split = split
        super(PerturbDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if split == 'train':
            self.data, self.slices = torch.load(self.processed_paths[0])
            self.smiles_list = pd.read_csv(os.path.join(self.processed_dir, f'{split}_smiles.csv'),
                                           header=None).to_numpy()[:, 0]
        elif split == 'val':
            self.data, self.slices = torch.load(self.processed_paths[1])
            self.smiles_list = pd.read_csv(os.path.join(self.processed_dir, f'{split}_smiles.csv'),
                                           header=None).to_numpy()[:, 0]
        elif split == 'test':
            self.data, self.slices = torch.load(self.processed_paths[2])
            self.smiles_list = pd.read_csv(os.path.join(self.processed_dir, f'{split}_smiles.csv'),
                                           header=None).to_numpy()[:, 0]

    def get(self, idx):
        data = Data()
        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if torch.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]
        data['smiles'] = self.smiles_list[idx]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        pass

    @staticmethod
    def _load_dataset(input_path, target, split):
        input_df = pd.read_csv(input_path, sep=',')
        split_data = input_df[input_df.Label == split]
        smiles_list = split_data['SMILES']
        smiles_list = smiles_list.reset_index(drop=True)
        rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
        labels = split_data[target]

        assert len(smiles_list) == len(rdkit_mol_objs_list)
        assert len(smiles_list) == len(labels)
        return smiles_list, rdkit_mol_objs_list, labels.values

    def process(self):
        for s, split in enumerate(['train', 'val', 'test']):
            data_list = []
            data_smiles_list = []
            smiles_list, rdkit_mol_objs, labels = self._load_dataset(self.raw_paths[0], self.target, split)
            for i in range(len(rdkit_mol_objs)):
                print(i)
                rdkit_mol = rdkit_mol_objs[i]
                if rdkit_mol is not None:
                    data = mol_to_graph_data(rdkit_mol)
                    data.id = torch.tensor([i])
                    data.label = torch.tensor(labels[i])
                    data_list.append(data)
                    data_smiles_list.append(smiles_list[i])
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            data_smiles_series = pd.Series(data_smiles_list)
            data_smiles_series.to_csv(os.path.join(self.processed_dir, f'{split}_smiles.csv'), index=False, header=False)
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[s])


def _load_pretrain_dataset(input_path):
    df = pd.read_csv(input_path, header=None)
    smiles_data = list(df[0])
    rdkit_mol_list = [AllChem.MolFromSmiles(str(s)) for s in df[0]]
    print(len(smiles_data))
    assert len(smiles_data) == len(rdkit_mol_list)
    return smiles_data, rdkit_mol_list


def _load_finetune_dataset(input_path, target):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    labels = input_df[target]
    # convert 0 to -1
    labels = labels.replace(0, -1)
    # there are no nans
    labels = labels.fillna(0)

    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels.values

def _load_finetune_dataset_zxx(input_path, target):
    input_df = pd.read_csv(input_path, sep=',')
    smiles_list = input_df['smiles']
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    labels = input_df['label']

    labels_new = []
    for item in labels:
        if type(item) == str:
            item = item.split(' ')
            item = [float(x) for x in item]
            item = [-1. if x == 0. else 0. for x in item]
        else:
            item = [-1. if item == 0. else 0]
        labels_new.append(item)
        # item = torch.tensor(item)
        # print(len(item))
    # labels = torch.tensor(labels.values)
    # print(labels)
    labels = np.array(labels_new)
    # print(labels)

    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(labels)
    return smiles_list, rdkit_mol_objs_list, labels


def _load_cliff_dataset(input_path, split):
    input_df = pd.read_csv(input_path, sep=',')
    cliff_data = input_df[input_df['split'] == split]
    smiles_list = cliff_data['smiles'].values
    rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]

    label_y = cliff_data['y'].values
    label_cliff = cliff_data['cliff_mol'].values

    assert len(smiles_list) == len(rdkit_mol_objs_list)
    assert len(smiles_list) == len(label_y)
    assert len(smiles_list) == len(label_cliff)

    return smiles_list, rdkit_mol_objs_list, label_y, label_cliff


def mol_to_graph_data(mol):
    mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=5)
    # atoms
    atom_symbol = []
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
        atom_symbol.append(atom.GetSymbol())

    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 3
    if len(mol.GetBonds()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # pretrain_data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # pretrain_data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    bond_length = Compound3DKit.get_bond_lengths(edge_index, atom_poses)
    BondAngleGraph_edges, bond_angles = Compound3DKit.get_superedge_angles(edge_index, atom_poses)

    bond_length = torch.tensor(bond_length).reshape(-1, 1)
    BondAngleGraph_index = torch.tensor(BondAngleGraph_edges, dtype=torch.long).T
    bond_angles = torch.tensor(bond_angles).reshape(-1, 1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, bond_length=bond_length, atom_symbol=atom_symbol,
                BondAngleGraph_index=BondAngleGraph_index, bond_angle=bond_angles)
    return data


class Compound3DKit(object):
    @staticmethod
    def get_atom_poses(mol, conf):
        """tbd"""
        atom_poses = []
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() == 0:
                return [[0.0, 0.0, 0.0]] * len(mol.GetAtoms())
            pos = conf.GetAtomPosition(i)
            atom_poses.append([pos.x, pos.y, pos.z])
        atom_poses = torch.Tensor(atom_poses)
        return atom_poses

    @staticmethod
    def get_MMFF_atom_poses(mol, numConfs=None, return_energy=False):
        """the atoms of mol will be changed in some cases."""
        try:
            new_mol = Chem.AddHs(mol)
            AllChem.EmbedMultipleConfs(new_mol, numConfs=numConfs, numThreads=0, maxAttempts=10)
            res = AllChem.MMFFOptimizeMoleculeConfs(new_mol, numThreads=0)
            new_mol = Chem.RemoveHs(new_mol)
            index = np.argmin([x[1] for x in res])
            energy = res[index][1]
            conf = new_mol.GetConformer(id=int(index))
        except:
            new_mol = mol
            AllChem.Compute2DCoords(new_mol)
            energy = 0
            conf = new_mol.GetConformer()

        atom_poses = Compound3DKit.get_atom_poses(new_mol, conf)
        if return_energy:
            return new_mol, atom_poses, energy
        else:
            return new_mol, atom_poses

    @staticmethod
    def get_2d_atom_poses(mol):
        """get 2d atom poses"""
        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        atom_poses = Compound3DKit.get_atom_poses(mol, conf)
        return atom_poses

    @staticmethod
    def get_bond_lengths(edges, atom_poses):
        """get bond lengths"""
        bond_lengths = []
        edges = edges.T
        for src_node_i, tar_node_j in edges:
            bond_lengths.append(np.linalg.norm(atom_poses[tar_node_j] - atom_poses[src_node_i]))  # 2范数
        # bond_lengths = np.array(bond_lengths, 'float32')
        return bond_lengths

    @staticmethod
    def get_superedge_angles(edges, atom_poses, dir_type='HT'):
        """get superedge angles"""

        def _get_vec(atom_poses, edge):
            return atom_poses[edge[1]] - atom_poses[edge[0]]

        def _get_angle(vec1, vec2):
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            if norm1 == 0 or norm2 == 0:
                return 0
            vec1 = vec1 / (norm1 + 1e-5)  # 1e-5: prevent numerical errors
            vec2 = vec2 / (norm2 + 1e-5)
            angle = np.arccos(np.dot(vec1, vec2))
            return angle

        edges = edges.T
        E = len(edges)
        edge_indices = np.arange(E)
        super_edges = []
        bond_angles = []
        bond_angle_dirs = []
        for tar_edge_i in range(E):
            tar_edge = edges[tar_edge_i]
            if dir_type == 'HT':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[0]]
            elif dir_type == 'HH':
                src_edge_indices = edge_indices[edges[:, 1] == tar_edge[1]]
            else:
                raise ValueError(dir_type)
            for src_edge_i in src_edge_indices:
                if src_edge_i == tar_edge_i:
                    continue
                src_edge = edges[src_edge_i]
                src_vec = _get_vec(atom_poses, src_edge)
                tar_vec = _get_vec(atom_poses, tar_edge)
                super_edges.append([src_edge_i, tar_edge_i])
                angle = _get_angle(src_vec, tar_vec)
                bond_angles.append(angle)
                bond_angle_dirs.append(src_edge[1] == tar_edge[0])  # H -> H or H -> T

        if len(super_edges) == 0:
            super_edges = np.zeros([0, 2], 'int64')
            bond_angles = np.zeros([0, ], 'float32')
        else:
            super_edges = np.array(super_edges, 'int64')
            bond_angles = np.array(bond_angles, 'float32')
        return super_edges, bond_angles


class BondLengthRBF(torch.nn.Module):
    """
    Bond Length Encoder using Radial Basis Functions
    """

    def __init__(self, embed_dim, bond_length_names='bond_length', rbf_params=None):
        super(BondLengthRBF, self).__init__()
        self.bond_length_names = bond_length_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (np.arange(0, 2, 0.1), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        centers, gamma = self.rbf_params['bond_length']
        self.rbf = RBF(centers, gamma)
        self.linear = nn.Linear(len(centers), embed_dim)

    def forward(self, bond_length_features):
        """
        Args:
            bond_float_features(dict of tensor): bond float features.
        """
        rbf_x = self.rbf(bond_length_features)
        out_embed = self.linear(rbf_x)
        return out_embed


class BondAngleRBF(torch.nn.Module):
    """
    Bond Angle Float Encoder using Radial Basis Functions
    """

    def __init__(self, embed_dim, bond_angle_float_names='bond_angle', rbf_params=None):
        super(BondAngleRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (np.arange(0, np.pi, 0.1), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        centers, gamma = self.rbf_params['bond_angle']
        self.rbf = RBF(centers, gamma)
        self.linear = nn.Linear(len(centers), embed_dim)

    def forward(self, bond_angle_float_features):
        """
        Args:
            bond_angle_float_features(dict of tensor): bond angle float features.
        """

        rbf_x = self.rbf(bond_angle_float_features)
        out_embed = self.linear(rbf_x)
        return out_embed


class RBF(torch.nn.Module):
    """
    Radial Basis Function
    """

    def __init__(self, centers, gamma):
        super(RBF, self).__init__()
        # device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        self.centers = torch.reshape(torch.from_numpy(centers).float(), [1, -1])
        self.gamma = gamma

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = torch.reshape(x, [-1, 1])
        return torch.exp(-self.gamma * torch.square(x - self.centers.to(x.device)))
