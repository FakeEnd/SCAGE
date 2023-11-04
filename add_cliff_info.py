import os

import pandas as pd
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.data.collate import collate
from itertools import repeat
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from typing import Optional, List, Tuple, Dict
def _load_cliff_dataset(input_path):
    input_df = pd.read_csv(input_path, sep=',')
    cliff_data = input_df
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


def collate_(
        data_list: List[Data]) -> Tuple[Data, Optional[Dict[str, Tensor]]]:
    r"""Collates a Python list of :obj:`torch_geometric.data.Data` objects
    to the internal storage format of
    :class:`~torch_geometric.data.InMemoryDataset`."""
    if len(data_list) == 1:
        return data_list[0], None

    data, slices, _ = collate(
        data_list[0].__class__,
        data_list=data_list,
        increment=False,
        add_batch=False,
    )

    return data, slices

def add_data(source_path, target_path):
    directory = os.path.dirname(target_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    data_smiles_list = []
    smiles_list, rdkit_mol_objs, label_y, label_cliff = _load_cliff_dataset(source_path)
    for i in tqdm(range(len(rdkit_mol_objs))):
        rdkit_mol = rdkit_mol_objs[i]
        if rdkit_mol is not None:
            data = mol_to_graph_data(rdkit_mol)
            data.id = torch.tensor([i])
            data.label_y = torch.tensor(label_y[i])
            data.label_cliff = torch.tensor(label_cliff[i])
            data.smiles = smiles_list[i]
            torch.save(data, os.path.join(target_path, 'data_{}.pth'.format(i)))
    data_smiles_series = pd.Series(data_smiles_list)
    data_smiles_series.to_csv(os.path.join(target_path, 'smiles.csv'), index=False, header=False)



if __name__ == '__main__':
    # task_name = 'bace'
    folder_path = '/mnt/8t/qjb/workspace/SCAGE_DATA/cliff_data'  # 将'/your/folder/path'替换为实际文件夹路径
    file_names = os.listdir(folder_path)
    task_name_list = []
    for file_name in file_names:
        task_name_list.append(file_name)
    print(task_name_list)
    # task_name_list = ['CHEMBL204_Ki']
    for task_name in task_name_list:
        print(f'now processing {task_name}')
        source_path = f'/mnt/8t/qjb/workspace/SCAGE_DATA/cliff_data/{task_name}/raw/{task_name}.csv'
        target_path = f'/mnt/8t/qjb/workspace/SCAGE_DATA/cliff_data_split/{task_name}/'
        add_data(source_path, target_path)