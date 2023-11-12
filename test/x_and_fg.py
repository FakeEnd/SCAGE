import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import Chem

def _load_pretrain_dataset(input_path):
    df = pd.read_csv(input_path, header=None)
    smiles_data = list(df[0])
    rdkit_mol_list = [AllChem.MolFromSmiles(str(s)) for s in df[0]]
    print(len(smiles_data))
    assert len(smiles_data) == len(rdkit_mol_list)
    return smiles_data, rdkit_mol_list

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

# def atom_to_feature_vector(atom):
#     """
#     Converts rdkit atom object to feature list of indices
#     :param mol: rdkit atom object
#     :return: list
#     """
#     atom_feature = [
#             safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
#             allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
#             safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
#             safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
#             safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
#             safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
#             safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
#             allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
#             allowable_features['possible_is_in_ring_list'].index(atom.IsInRing()),
#             ]
#     return atom_feature

def f1():
    # path = '/mnt/sde/qjb/SCAGE/data/pubchem_split/pubchem_19/'
    # all_file = os.listdir(path)
    # for file in tqdm(all_file):
    #     if file.endswith('.pth') and file != 'pre_transform.pth' and file != 'pre_filter.pth':
    #         data = os.path.join(path, file)
    #         data = torch.load(data)
    #         x = data.x
    #
    #         # smiles = data.smiles
    #         # mol = Chem.MolFromSmiles(smiles)
    #         # # print(len(mol.GetAtoms()))
    #
    #         fg = data.fg
    #         if len(x) != len(fg):
    #             print(file)

    path = '/mnt/solid/tss/Project-2023/SCAGE2.0/data/pretrain_data/pubchem_1000/processed/data_882112.pth'
    data = torch.load(path)
    print(data)
    print(data.fg)
    # smiles = data.smiles
    # print(smiles)
    #
    # mol = Chem.MolFromSmiles(smiles)
    # print(len(mol.GetAtoms()))
    #
    # mol = AllChem.MolFromSmiles(str(smiles))
    # mol, atom_poses = Compound3DKit.get_MMFF_atom_poses(mol, numConfs=5)
    # mol_list = mol.GetAtoms()
    # print(len(mol_list))

    atom_symbol = []
    atom_features_list = []
    # for atom in mol.GetAtoms():
    #     atom_features_list.append(atom_to_feature_vector(atom))
    #     atom_symbol.append(atom.GetSymbol())
    #
    # x = torch.tensor(np.array(atom_features_list), dtype=torch.long)


    # input_path = '/mnt/solid/tss/Project-2023/SCAGE2.0/data/pretrain_data/pubchem_1000/raw/pubchem_smiles_1000w.txt'
    # df = pd.read_csv(input_path, header=None)
    # smiles_data = list(df[0])
    # print(smiles_data[882112])
    # rdkit_mol_list = [AllChem.MolFromSmiles(str(s)) for s in df[0]]

if __name__ == '__main__':
    f1()