import os
import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from numpy.linalg import inv
import pickle
from torch_geometric.datasets import *
from torch_sparse.matmul import matmul
from torch_sparse import SparseTensor

c = 0.15
k = 5


def get_downstream_task_names(config):
    """
    Get task names of downstream dataset
    """
    if config['task_name'] == 'bace':
        target = ["Class"]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'bbbp':
        target = ["p_np"]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'clintox':
        target = ['CT_TOX', 'FDA_APPROVED']
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'hiv':
        target = ["HIV_active"]
        task = 'classification'
        loss_type = 'wb_bce'
    elif config['task_name'] == 'muv':
        target = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'sider':
        target = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues",
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders",
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders",
            "Reproductive system and breast disorders",
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "General disorders and administration site conditions", "Endocrine disorders",
            "Surgical and medical procedures", "Vascular disorders",
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders",
            "Congenital, familial and genetic disorders", "Infections and infestations",
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders",
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions",
            "Ear and labyrinth disorders", "Cardiac disorders",
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'tox21':
        target = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'toxcast':
        raw_path = os.path.join(config['root'], 'toxcast', 'raw')
        csv_file = os.listdir(raw_path)[0]
        input_df = pd.read_csv(os.path.join(raw_path, csv_file), sep=',')
        target = list(input_df.columns)[1:]
        task = 'classification'
        loss_type = 'bce'
    elif config['task_name'] == 'CYP3A4':
        target = ["label"]
        task = 'classification'
        loss_type = 'bce'

        # 回归数据集
    elif config['task_name'] == 'esol':
        target = ["measured log solubility in mols per litre"]
        task = 'regression'
        loss_type = 'mse'
    elif config['task_name'] == 'freesolv':
        target = ["expt"]
        task = 'regression'
        loss_type = 'mse'
    elif config['task_name'] == 'lipophilicity':
        target = ['exp']
        task = 'regression'
        loss_type = 'mse'
    elif config['task_name'] == 'qm7':
        target = ['u0_atom']
        task = 'regression'
        loss_type = 'l1'
    elif config['task_name'] == 'qm8':
        target = ['E1-CC2', 'E2-CC2', 'f1-CC2', 'f2-CC2',
                  'E1-PBE0', 'E2-PBE0', 'f1-PBE0', 'f2-PBE0',
                  'E1-CAM', 'E2-CAM', 'f1-CAM', 'f2-CAM']
        task = 'regression'
        loss_type = 'l1'
    elif config['task_name'] == 'qm9':
        target = ['homo', 'lumo', 'gap']
        task = 'regression'
        loss_type = 'l1'
    elif config['task_name'] == 'physprop_perturb':
        target = ['LogP']
        task = 'regression'
        loss_type = 'mse'
    else:
        raise ValueError('%s not supported' % config['task_name'])
    config['target'] = target
    config['task'] = task
    config['loss_type'] = loss_type
    config['DownstreamModel']['num_tasks'] = len(target)


def adj_normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx


def get_intimacy_matrix(edges, n):
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n),
                        dtype=np.float32)
    print('normalize')
    adj_norm = adj_normalize(adj)
    print('inverse')
    eigen_adj = c * inv((sp.eye(adj.shape[0]) - (1 - c) * adj_norm).toarray())

    return eigen_adj


def adj_normalize_sparse(mx):
    mx = mx.to(device)
    rowsum = mx.sum(1)
    r_inv = rowsum.pow(-0.5).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = SparseTensor(row=torch.arange(n).to(device), col=torch.arange(n).to(device), value=r_inv,
                             sparse_sizes=(n, n))
    nr_mx = matmul(matmul(r_mat_inv, mx), r_mat_inv)
    return nr_mx


def get_intimacy_matrix_sparse(edges, n):
    adj = SparseTensor(row=edges[0], col=edges[1], value=torch.ones(edges.shape[1]), sparse_sizes=(n, n))
    adj_norm = adj_normalize_sparse(adj)
    return adj_norm


def get_svd_dense(mx, q=3):
    mx = mx.float()
    u, s, v = torch.svd_lowrank(mx, q=q)
    s = torch.diag(s)
    pu = u @ s.pow(0.5)
    pv = v @ s.pow(0.5)
    return pu, pv


def unweighted_adj_normalize_dense_batch(adj):
    adj = (adj + adj.transpose(-1, -2)).bool().float()
    adj = adj.float()
    rowsum = adj.sum(-1)
    r_inv = rowsum.pow(-0.5)
    r_mat_inv = torch.diag_embed(r_inv)  # 指定值变成对角矩阵
    nr_adj = torch.matmul(torch.matmul(r_mat_inv, adj), r_mat_inv)
    return nr_adj


def get_eig_dense(adj):
    adj = adj.float()
    rowsum = adj.sum(1)
    r_inv = rowsum.pow(-0.5)
    r_mat_inv = torch.diag(r_inv)
    nr_adj = torch.matmul(torch.matmul(r_mat_inv, adj), r_mat_inv)
    graph_laplacian = torch.eye(adj.shape[0]) - nr_adj
    L, V = torch.eig(graph_laplacian, eigenvectors=True)
    return L.T[0], V


def check_checkpoints(output_dir):
    import os
    import shutil
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        for file in files:
            if 'checkpoint' in file:
                return True
        print('remove ', output_dir)
        shutil.rmtree(output_dir)
    return False
