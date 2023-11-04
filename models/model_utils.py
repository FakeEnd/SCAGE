import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from numpy.linalg import inv
import pickle
from torch_geometric.datasets import *
from torch_sparse.matmul import matmul
from torch_sparse import SparseTensor
from torch_geometric.nn import global_mean_pool

c = 0.15
k = 5


def AFPS(scores, dist, k, mask=None):
    # 合并所有的head，遍历head时间消耗大
    scores = torch.mean(scores, dim=0)

    if mask is not None:
        scores = scores[mask][:, mask]
        dist = dist[mask][:, mask]

    # initialize the first point
    scores = torch.sum(scores, dim=-2)

    # 初始化候选点和剩余点，从中心点开始
    remaining_points = [i for i in range(len(dist))]
    solution_set = [remaining_points.pop(0)]

    # incorporate the distance information
    dist = dist / torch.max(dist) + scores.unsqueeze(-1) / torch.max(scores) * 0.1

    while len(solution_set) < k:
        # 得到候选点和剩余点的距离矩阵
        distances = dist[remaining_points][:, solution_set]

        # 更新剩余点的距离，选最大的
        distances = torch.min(distances, dim=-1)[0]  # 先找到距离已解决点最近的点，然后在选择最远的点
        new_point = torch.argmax(distances).item()
        solution_set.append(remaining_points.pop(new_point))

    return solution_set


class GraphPool(nn.Module):
    def __init__(self, graph_pooling, hidden_dim, afps_k, use_super_node):
        super(GraphPool, self).__init__()

        self.graph_pooling = graph_pooling
        self.afps_norm = nn.LayerNorm(hidden_dim)
        self.afps_k = afps_k
        self.use_super_node = use_super_node

    def forward(self, x, batch_data, attn):
        n_graph, n_node = x.size()[:2]
        # print('n_graph=', n_graph)
        # print('n_node=', n_node)
        padding_mask = batch_data['x_mask']

        if self.use_super_node:
            # graph_rep = x[:, 0, :].squeeze()
            graph_rep = x[:, 0, :].squeeze(dim=1)
        elif self.graph_pooling == 'center_node':
            root_n_id = batch_data['root_n_id']
            root_idx = (torch.arange(n_graph, device=x.device) * n_node + root_n_id).long()
            graph_rep = x.reshape(-1, x.shape[-1])[root_idx].squeeze()
        elif self.graph_pooling == 'afps':
            graph_rep = []
            for i in range(n_graph):
                if torch.sum(padding_mask[i]) <= self.afps_k:
                    graph_rep.append(torch.mean(x[i][padding_mask[i].bool()], dim=0))
                else:
                    with torch.no_grad():
                        idx = AFPS(attn[i], batch_data['spatial_pos'][i], self.afps_k, padding_mask[i].bool())

                    graph_rep.append(torch.mean(x[i, idx], dim=0))

            graph_rep = self.afps_norm(torch.stack(graph_rep, dim=0))
        elif self.graph_pooling == 'mean':
            x_mask = padding_mask.bool().reshape(-1)
            x = x.reshape(-1, x.shape[-1])[x_mask]
            graph_rep = global_mean_pool(x, batch_data['batch'])
        else:
            raise NotImplementedError('please input correct pool methods')
        return graph_rep


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


if __name__ == '__main__':
    # just test

    device = torch.device('cuda', 0)

    data = Flickr('dataset/flickr')

    edges = data.data.edge_index
    n = data.data.x.shape[0]

    adj = SparseTensor(row=edges[0], col=edges[1], value=torch.ones(edges.shape[1]), sparse_sizes=(n, n))
    nr_adj = adj_normalize_sparse(adj)

    pu, pv = get_svd_dense(nr_adj.to_torch_sparse_coo_tensor(), q=10)

    adj = (torch.randn(10, 10) > 0).float()
    L, V = get_eig_dense(adj)
