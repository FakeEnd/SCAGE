import os

import torch
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures, Draw
from rdkit.Chem import Descriptors
from tqdm import tqdm
import pickle


def get_funcgroup():
    with open('/mnt/8t/qjb/workspace/SCAGE_DATA/funcgroup.txt', "r") as f:
        funcgroups = f.read().strip().split('\n')
        # name = [i.split()[0] for i in funcgroups]
        idx = [i for i in range(1, 83)]
        smart = [Chem.MolFromSmarts(i.split()[1]) for i in funcgroups]
        smart2name = dict(zip(smart, idx))
    return smart, smart2name

def match_fg(mol, smart, smart2name):
    n_atom = len(mol.GetAtoms())
    # print('n_atom=', n_atom)
    fg = [0 for _ in range(n_atom)]
    for sn in smart2name.keys():
        matches = mol.GetSubstructMatches(sn)
        for match in matches:
            for atom_idx in match:
                atom = mol.GetAtomWithIdx(atom_idx)
                # print(f"Atom {atom.GetSymbol()} at index {atom.GetIdx()} belongs to functional group: {Chem.MolToSmarts(sn)}")
                # print(
                #     f"Atom {atom.GetSymbol()} at index {atom.GetIdx()} belongs to functional group: {smart2name[sn]}")
                fg[atom.GetIdx()] = smart2name[sn]
    return fg

# 保存结果到文件
def save_results_to_file(results, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(results, file)

# 从txt文件中读取SMILES，并进行处理（添加tqdm进度条）
# def process_smiles_from_txt(file_path, save_every=10000, save_file='results_demo.pkl'):
#     smart, smart2name = get_funcgroup()
#     results = []  # 保存处理结果的列表
#     with open(file_path, 'r') as file:
#         num_lines = sum(1 for line in file)  # 计算文件中的行数，用于设置tqdm的total参数
#         file.seek(0)  # 将文件指针重置为文件开头
#         processed_count = 0
#         for line in tqdm(file, total=num_lines, desc="Processing SMILES"):
#             smiles = line.strip()  # 移除行末尾的换行符和空白字符
#             mol = Chem.MolFromSmiles(smiles)
#             if mol is not None:
#                 # print(f"SMILES: {smiles}")
#                 fg = match_fg(mol, smart, smart2name)
#                 result_dict = {'smiles': smiles, 'functional_groups': fg}
#                 results.append(result_dict)
#
#             processed_count += 1
#             if processed_count % save_every == 0:
#                 # print(f"Saving results for {processed_count} SMILES.")
#                 save_results_to_file(results, save_file)
#
#     # 最后处理剩余的结果并保存
#     if results:
#         print(f"Saving final results for {processed_count} SMILES.")
#         save_results_to_file(results, save_file)

def process_smiles_from_txt(file_path, save_every=10000, save_file='results_demo.pkl'):
    smart, smart2name = get_funcgroup()
    results = {}  # 保存处理结果的字典
    with open(file_path, 'r') as file:
        num_lines = sum(1 for line in file)  # 计算文件中的行数，用于设置tqdm的total参数
        file.seek(0)  # 将文件指针重置为文件开头
        processed_count = 0
        for line in tqdm(file, total=num_lines, desc="Processing SMILES"):
            smiles = line.strip()  # 移除行末尾的换行符和空白字符
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # print(f"SMILES: {smiles}")
                fg = match_fg(mol, smart, smart2name)
                results[smiles] = fg

            processed_count += 1
            if processed_count % save_every == 0:
                # print(f"Saving results for {processed_count} SMILES.")
                save_results_to_file(results, save_file)

    # 最后处理剩余的结果并保存
    if results:
        print(f"Saving final results for {processed_count} SMILES.")
        save_results_to_file(results, save_file)

def read_results_from_file(file_path):
    with open(file_path, 'rb') as file:
        results = pickle.load(file)
    return results

def adddim(path):
    data_list = [i for i in os.listdir(path) if
                 i.endswith('.pth') and i != 'pre_transform.pth' and i != 'pre_filter.pth']
    for name in tqdm(data_list):
        with open(os.path.join(path, name), 'rb') as rf:
            data = torch.load(rf)
            print(data)
            # fg = data['fg']
            # fg = fg.squeeze(0)
            # data['fg'] = fg
            # print(data)
            # torch.save(data, os.path.join(path, name))
            # print(fg)
            # smart, smart2name = get_funcgroup()
            # mol = Chem.MolFromSmiles(smiles)
            # if mol is not None:
            #     fg = match_fg(mol, smart, smart2name)
            #     fg = torch.tensor(fg)
            #     data['fg'] = fg
            # # print(data['fg'])
            # torch.save(data, os.path.join(path, name))


# 示例
if __name__ == "__main__":
    # file_path = '/mnt/solid/tss/Project-2023/SCAGE/data/pretrain/pubchem/raw/pubchem-10m-clean.txt'
    # file_path = '/mnt/solid/tss/Project-2023/SCAGE/data/pretrain/pubchem_demo/processed/smiles.csv'
    # process_smiles_from_txt(file_path)
    # f = read_results_from_file('/mnt/8t/qjb/workspace/SCAGE/data_process/results.pkl')
    # print(f)
    # smiles = "CC1(C)C2CCC(C2)C1C[P+]([Se-])(CO)CO"
    # smart, smart2name = get_funcgroup()
    # print(smart2name)
    # mol = Chem.MolFromSmiles(smiles)
    # # print(Chem.MolToSmiles(mol))
    # atoms = mol.GetAtoms()
    # # for atom in atoms:
    # #     print(atom.GetSymbol())
    # # for atom_idx, atom in enumerate(mol.GetAtoms()):
    # #     print(f'{atom_idx}  {atom.GetSymbol()}')
    # # print(atom.GetSymbol())
    # # print(smart)
    # # print(len(mol.GetAtoms()))
    # fg = match_fg(mol, smart, smart2name)
    # print(fg)
    # smiles = 'CCOC(=O)C1=C(C)N=c2sc(=Cc3ccc(c4ccc(Br)cc4[N+](=O)[O-])o3)c(=O)n2C1c1ccc2c(c1)OCO2'
    # mol = Chem.MolFromSmiles(smiles)
    # print(len(mol.GetAtoms()))
    # from tqdm import tqdm
    # max = 0
    # with open('/mnt/solid/tss/Project-2023/MGSSL-main/motif_based_pretrain/data/zinc/all.txt', 'r') as rf:
    #     num_lines = sum(1 for line in rf)
    #     rf.seek(0)
    #     for line in tqdm(rf, total=num_lines):
    #         line = line.strip()
    #         mol = Chem.MolFromSmiles(line)
    #         length = len(mol.GetAtoms())
    #         if length > max:
    #             max = length
    # print(max)
    # path = '/mnt/8t/qjb/workspace/SCAGE/data/pretrain/pubchem_demo/processed'
    # path = '/mnt/solid/tss/Project-2023/SCAGE/data/pretrain/pubchem_250/processed'
    # task_name_list = ['bace', 'bbbp', 'clintox', 'esol', 'freesolv', 'hiv', 'lipophilicity', 'muv', 'sider', 'tox21', 'toxcast']
    # task_name_list = ['bace']

    source_path = '/mnt/sde/qjb/SCAGE/data/pubchem_split'
    file_names = os.listdir(source_path)
    # task_name_list = []
    # for file_name in file_names:
    #     task_name_list.append(file_name)
    # print(task_name_list)
    task_name_list = ['pubchem_0']
    for idx in range(20):
        # for task_name in task_name_list:
        task_name = f'pubchem_{idx}'
        print(f'now processing {task_name}')
        # path = f'/mnt/8t/qjb/workspace/SCAGE_DATA/finetune_data_split/{task_name}/'
        path = f'{source_path}/{task_name}/'
        data_list = [i for i in os.listdir(path) if
                     i.endswith('.pth') and i != 'pre_transform.pth' and i != 'pre_filter.pth']
        # print(data_list)
        dic = {}
        for name in tqdm(data_list):
            with open(os.path.join(path, name), 'rb') as rf:
                data = torch.load(rf)
                # print(data)
                smiles = data['smiles']
                # print(smiles)
                smart, smart2name = get_funcgroup()
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    fg = match_fg(mol, smart, smart2name)
                    fg = torch.tensor(fg)
                    data['fg'] = fg
                # print(data['fg'])
                torch.save(data, os.path.join(path, name))
    # adddim(path)


