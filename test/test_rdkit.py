from matplotlib import pyplot as plt
from rdkit import Chem
import pickle
import numpy as np
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn

def test_mol():
    smiles = 'COc1ccc(CNC2[NH+]=C(NNC(=O)CO)c3cccc(C)c3N2C)c(OC)c1'
    mol = Chem.MolFromSmiles(smiles)
    print(len(mol.GetAtoms()))
    print(mol.GetNumBonds())

def plot_bindings(ax, val, chr, start, end, color='#17becf'):
    ax.fill_between(np.arange(val.shape[0]), 0, val, color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks(np.arange(0, val.shape[0], val.shape[0] // 5))
    ax.set_ylim(0, 1)
    ax.set_xticklabels(np.arange(start, end, (end - start) // 5))
    ax.margins(x=0)

def test():
    # path = '/mnt/sde/qjb/deeepbio2.0/DNAaccessibility/GM12878_dnase.pickle'
    # with open(path, 'rb') as file:
    #     data = pickle.load(file)
    #     print(data)
    #     print(data['2'])
    num_class = 3
    label_input = torch.Tensor(np.arange(num_class)).view(1, -1).long()
    print(label_input)
    label_inputs = label_input.repeat(8, 1)
    print(label_inputs)
    query_embed = nn.Embedding(num_class, 64)
    label_embed = query_embed(label_inputs)
    print(label_embed.sum(-1))
    print(label_embed.sum(-1).shape)
    out = torch.sigmoid(label_embed.sum(-1))
    out = torch.sigmoid(out).detach().cpu().numpy()
    print(out)
    print(out[:, 2])

    chrom, start, end = ['chr11', 46750000, 46750008]
    fig, ax = plt.subplots(figsize=(10, 1))
    val = out[:, 2]
    plot_bindings(ax, val, chrom, start, end)
    ax.set_title('test')
    ax.set_xlabel('chr%s:%s-%s' % (chrom, start, end))
    plt.show()

if __name__ == '__main__':
    test()