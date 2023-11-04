# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch_geometric.data import Batch, Data
import time


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_pos_emb_unsqueeze(x, padlen):
    # print('x=', x)
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_adj_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)



# pretrain
def collator(items, config):
    max_node = config['max_node']
    spatial_pos_max = config['spatial_pos_max']

    filtered_items = []
    for item in items:
        if item is not None and item.x.size(0) <= max_node:
            filtered_items.append((
                item.label,
                item.attn_bias,
                item.spatial_pos,
                item.in_degree,
                item.out_degree,
                item.x,
                item.edge_attr,
                item.adj,
                item.eig_pos_emb,
                item.svd_pos_emb,
                item.edge_index,
                item.bond_length,
                item.BondAngleGraph_index,
                item.bond_angle,
                item.smiles,
                item.atom_symbol,
                item.fg
            ))

    (
        labels,
        attn_biases,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_attrs,
        adjs,
        eig_pos_embs,
        svd_pos_embs,
        edge_indexs,
        bond_lengths,
        BondAngleGraph_indexs,
        bond_angles,
        all_smiles,
        atom_symbols,
        fgs
    ) = zip(*filtered_items)

    for i, _ in enumerate(attn_biases):
        attn_biases[i][int(config['use_super_node']):, int(config['use_super_node']):][
            spatial_poses[i] >= spatial_pos_max] = float("-inf")

    max_node_num = max(i.size(0) for i in xs)
    # print('max_node_num=', max_node_num)
    ns = [x.size(0) for x in xs]
    x_mask = torch.zeros(len(xs), max_node_num)
    for i, n in enumerate(ns):
        x_mask[i, :n] = 1

    mol_batch = Batch.from_data_list([Data(x=ei, num_nodes=ns[i]) for i, ei in enumerate(xs)])

    label = torch.cat(labels).reshape(len(labels), -1) if not isinstance(labels[0], int) else None

    attn_bias = torch.cat([
        pad_attn_bias_unsqueeze(attn_bias, max_node_num + int(config['use_super_node']))
        for attn_bias in attn_biases
    ])

    in_degree = torch.cat([
        pad_1d_unsqueeze(in_degree, max_node_num)
        for in_degree in in_degrees
    ]) if not isinstance(in_degrees[0], int) else None

    adj = torch.cat([
        pad_adj_unsqueeze(adj, max_node_num)
        for adj in adjs
    ])

    spatial_pos = torch.cat([
        pad_spatial_pos_unsqueeze(spatial_pos, max_node_num)
        for spatial_pos in spatial_poses
    ]) if not isinstance(spatial_poses[0], int) else None

    batch_2d_index = Batch.from_data_list([
        Data(edge_index=edge_index, num_nodes=ns[i])
        for i, edge_index in enumerate(edge_indexs)
    ]).edge_index

    batch_2d_attr = Batch.from_data_list([
        Data(edge_attr=edge_attr)
        for edge_attr in edge_attrs
    ]).edge_attr

    eig_pos_embs = torch.cat([
        pad_pos_emb_unsqueeze(eig_pos_emb, max_node_num)
        for eig_pos_emb in eig_pos_embs
    ]) if not isinstance(eig_pos_embs[0], int) else None

    svd_pos_embs = torch.cat([
        pad_pos_emb_unsqueeze(svd_pos_emb, max_node_num)
        for svd_pos_emb in svd_pos_embs
    ]) if not isinstance(svd_pos_embs[0], int) else None

    batch_bond_length = Batch.from_data_list([
        Data(bond_length=bond_length)
        for bond_length in bond_lengths
    ]).bond_length

    max_bond_num = max(i.size(0) for i in bond_lengths)
    num_bond_length = [x.size(0) for x in bond_lengths]
    bond_length_mask = torch.zeros(len(bond_lengths), max_bond_num)
    for i, n in enumerate(num_bond_length):
        bond_length_mask[i, :n] = 1

    batch_bond_angle = Batch.from_data_list([
        Data(bond_angle=bond_angle)
        for bond_angle in bond_angles
    ]).bond_angle

    batch_3d_index = Batch.from_data_list([
        Data(BondAngleGraph_index=BondAngleGraph_index, num_nodes=num_bond_length[i])
        for i, BondAngleGraph_index in enumerate(BondAngleGraph_indexs)
    ]).BondAngleGraph_index

    fgs = Batch.from_data_list([
        Data(fg=fg)
        for fg in fgs
    ]).fg

    data_dict = dict(
        label=label,
        attn_bias=attn_bias,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,
        x=mol_batch.x,
        batch=mol_batch.batch,
        edge_attr=batch_2d_attr,
        edge_index=batch_2d_index,
        x_mask=x_mask,
        ns=torch.LongTensor(ns),
        adj=adj,
        eig_pos_emb=eig_pos_embs,
        svd_pos_emb=svd_pos_embs,
        x_3d=batch_bond_length,
        x_mask_3d=bond_length_mask,
        edge_index_3d=batch_3d_index,
        edge_attr_3d=batch_bond_angle,
        smiles=list(all_smiles),
        atom_symbols=list(atom_symbols),
        edge_weight=None,
        fg=fgs
    )

    return data_dict

# finetune
def collator_finetune(items, config):
    max_node = config['max_node']
    spatial_pos_max = config['spatial_pos_max']

    filtered_items = []
    for item in items:
        if item is not None and item.x.size(0) <= max_node:
            filtered_items.append((
                item.label,
                item.attn_bias,
                item.spatial_pos,
                item.in_degree,
                item.out_degree,
                item.x,
                item.edge_attr,
                item.adj,
                item.eig_pos_emb,
                item.svd_pos_emb,
                item.edge_index,
                item.bond_length,
                item.BondAngleGraph_index,
                item.bond_angle,
                item.smiles,
                item.atom_symbol,
            ))

    (
        labels,
        attn_biases,
        spatial_poses,
        in_degrees,
        out_degrees,
        xs,
        edge_attrs,
        adjs,
        eig_pos_embs,
        svd_pos_embs,
        edge_indexs,
        bond_lengths,
        BondAngleGraph_indexs,
        bond_angles,
        all_smiles,
        atom_symbols,
    ) = zip(*filtered_items)

    for i, _ in enumerate(attn_biases):
        attn_biases[i][int(config['use_super_node']):, int(config['use_super_node']):][
            spatial_poses[i] >= spatial_pos_max] = float("-inf")

    max_node_num = max(i.size(0) for i in xs)
    # print('max_node_num=', max_node_num)
    ns = [x.size(0) for x in xs]
    x_mask = torch.zeros(len(xs), max_node_num)
    for i, n in enumerate(ns):
        x_mask[i, :n] = 1

    mol_batch = Batch.from_data_list([Data(x=ei, num_nodes=ns[i]) for i, ei in enumerate(xs)])

    label = torch.cat(labels).reshape(len(labels), -1) if not isinstance(labels[0], int) else None

    attn_bias = torch.cat([
        pad_attn_bias_unsqueeze(attn_bias, max_node_num + int(config['use_super_node']))
        for attn_bias in attn_biases
    ])

    in_degree = torch.cat([
        pad_1d_unsqueeze(in_degree, max_node_num)
        for in_degree in in_degrees
    ]) if not isinstance(in_degrees[0], int) else None

    adj = torch.cat([
        pad_adj_unsqueeze(adj, max_node_num)
        for adj in adjs
    ])

    spatial_pos = torch.cat([
        pad_spatial_pos_unsqueeze(spatial_pos, max_node_num)
        for spatial_pos in spatial_poses
    ]) if not isinstance(spatial_poses[0], int) else None

    batch_2d_index = Batch.from_data_list([
        Data(edge_index=edge_index, num_nodes=ns[i])
        for i, edge_index in enumerate(edge_indexs)
    ]).edge_index

    batch_2d_attr = Batch.from_data_list([
        Data(edge_attr=edge_attr)
        for edge_attr in edge_attrs
    ]).edge_attr

    eig_pos_embs = torch.cat([
        pad_pos_emb_unsqueeze(eig_pos_emb, max_node_num)
        for eig_pos_emb in eig_pos_embs
    ]) if not isinstance(eig_pos_embs[0], int) else None

    svd_pos_embs = torch.cat([
        pad_pos_emb_unsqueeze(svd_pos_emb, max_node_num)
        for svd_pos_emb in svd_pos_embs
    ]) if not isinstance(svd_pos_embs[0], int) else None

    batch_bond_length = Batch.from_data_list([
        Data(bond_length=bond_length)
        for bond_length in bond_lengths
    ]).bond_length

    max_bond_num = max(i.size(0) for i in bond_lengths)
    num_bond_length = [x.size(0) for x in bond_lengths]
    bond_length_mask = torch.zeros(len(bond_lengths), max_bond_num)
    for i, n in enumerate(num_bond_length):
        bond_length_mask[i, :n] = 1

    batch_bond_angle = Batch.from_data_list([
        Data(bond_angle=bond_angle)
        for bond_angle in bond_angles
    ]).bond_angle

    batch_3d_index = Batch.from_data_list([
        Data(BondAngleGraph_index=BondAngleGraph_index, num_nodes=num_bond_length[i])
        for i, BondAngleGraph_index in enumerate(BondAngleGraph_indexs)
    ]).BondAngleGraph_index


    data_dict = dict(
        label=label,
        attn_bias=attn_bias,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=in_degree,
        x=mol_batch.x,
        batch=mol_batch.batch,
        edge_attr=batch_2d_attr,
        edge_index=batch_2d_index,
        x_mask=x_mask,
        ns=torch.LongTensor(ns),
        adj=adj,
        eig_pos_emb=eig_pos_embs,
        svd_pos_emb=svd_pos_embs,
        x_3d=batch_bond_length,
        x_mask_3d=bond_length_mask,
        edge_index_3d=batch_3d_index,
        edge_attr_3d=batch_bond_angle,
        smiles=list(all_smiles),
        atom_symbols=list(atom_symbols),
        edge_weight=None,
    )

    return data_dict

# def collator(items, config):
#     max_node = config['max_node']
#     spatial_pos_max = config['spatial_pos_max']
#
#     items = [item for item in items if item is not None and item.x.size(0) <= max_node]
#     items = [
#         (
#             item.label,
#             item.attn_bias,
#             item.spatial_pos,
#             item.in_degree,
#             item.out_degree,
#             item.x,
#             item.edge_attr,
#             item.adj,  # 带有自环的邻接矩阵
#             item.eig_pos_emb,
#             item.svd_pos_emb,
#             item.edge_index,
#             item.bond_length,
#             item.BondAngleGraph_index,
#             item.bond_angle,
#             item.smiles,
#             item.atom_symbol,
#             item.fg
#         )
#         for item in items
#     ]
#     (
#         labels,
#         attn_biases,
#         spatial_poses,
#         in_degrees,
#         out_degrees,
#         xs,
#         edge_attrs,
#         adjs,
#         eig_pos_embs,
#         svd_pos_embs,
#         edge_indexs,
#         bond_lengths,
#         BondAngleGraph_indexs,
#         bond_angles,
#         all_smiles,
#         atom_symbols,
#         fgs
#     ) = zip(*items)
#
#     for i, _ in enumerate(attn_biases):
#         attn_biases[i][int(config['use_super_node']):, int(config['use_super_node']):][
#             spatial_poses[i] >= spatial_pos_max] = float("-inf")
#     max_node_num = max(i.size(0) for i in xs)
#     ns = [x.size(0) for x in xs]
#     x_mask = torch.zeros(len(xs), max_node_num)
#     for i, n in enumerate(ns):
#         x_mask[i, :n] = 1
#
#     mol_batch = Batch.from_data_list([Data(x=ei, num_nodes=ns[i]) for i, ei in enumerate(xs)])
#
#     label = torch.cat(labels).reshape(len(labels), -1) if not isinstance(labels[0], int) else None
#
#     attn_bias = torch.cat(
#         [pad_attn_bias_unsqueeze(i, max_node_num + int(config['use_super_node'])) for i in attn_biases]
#     )
#
#     in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees]) if not isinstance(in_degrees[0],
#                                                                                                      int) else None
#     adj = torch.cat([pad_adj_unsqueeze(a, max_node_num) for a in adjs])
#
#     spatial_pos = torch.cat(
#         [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
#     ) if not isinstance(spatial_poses[0], int) else None
#
#     batch_2d_index = Batch.from_data_list([Data(edge_index=ei, num_nodes=ns[i]) for i, ei in
#                                            enumerate(edge_indexs)]).edge_index
#     batch_2d_attr = Batch.from_data_list([Data(edge_attr=ei) for i, ei in enumerate(edge_attrs)]).edge_attr
#
#     eig_pos_embs = torch.cat([pad_pos_emb_unsqueeze(i, max_node_num) for i in eig_pos_embs]) if not isinstance(
#         eig_pos_embs[0], int) else None
#
#     svd_pos_embs = torch.cat([pad_pos_emb_unsqueeze(i, max_node_num) for i in svd_pos_embs]) if not isinstance(
#         svd_pos_embs[0], int) else None
#
#     # 3D pretrain_data for model
#     batch_bond_length = Batch.from_data_list([Data(bond_length=ei) for i, ei in enumerate(bond_lengths)]).bond_length
#
#     max_bond_num = max(i.size(0) for i in bond_lengths)
#     num_bond_length = [x.size(0) for x in bond_lengths]
#     bond_length_mask = torch.zeros(len(bond_lengths), max_bond_num)
#     for i, n in enumerate(num_bond_length):
#         bond_length_mask[i, :n] = 1
#
#     batch_bond_angle = Batch.from_data_list([Data(bond_angle=ei) for i, ei in enumerate(bond_angles)]).bond_angle
#     batch_3d_index = Batch.from_data_list([Data(BondAngleGraph_index=ei, num_nodes=num_bond_length[i]) for i, ei in
#                                            enumerate(BondAngleGraph_indexs)]).BondAngleGraph_index
#
#
#     fgs = Batch.from_data_list([Data(fg=ei) for i, ei in enumerate(fgs)]).fg
#
#     data_dict = dict(
#         label=label,
#         attn_bias=attn_bias,
#         spatial_pos=spatial_pos,
#         in_degree=in_degree,
#         out_degree=in_degree,  # for undirected graph
#         x=mol_batch.x,
#         batch=mol_batch.batch,
#         edge_attr=batch_2d_attr,
#         edge_index=batch_2d_index,
#         x_mask=x_mask,
#         ns=torch.LongTensor(ns),  # node number in each graph
#         adj=adj,
#         eig_pos_emb=eig_pos_embs,
#         svd_pos_emb=svd_pos_embs,
#         x_3d=batch_bond_length,
#         x_mask_3d=bond_length_mask,
#         edge_index_3d=batch_3d_index,
#         edge_attr_3d=batch_bond_angle,
#         smiles=list(all_smiles),
#         atom_symbols=list(atom_symbols),
#         edge_weight=None,
#         fg=fgs
#     )
#
#     return data_dict


def pretrain_collator(items, config):
    # st = time.time()
    # print(items)
    batch_raw, batch_mask = zip(*items)
    dict_raw = collator(batch_raw, config)
    dict_mask = collator(batch_mask, config)

    # et = time.time()
    # cost = et - st
    # print(f'collator花费的时间: {cost:.5f}')
    return dict_raw, dict_mask
