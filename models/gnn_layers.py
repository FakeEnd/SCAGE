import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, softmax
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros
from data_process.loader import BondEncoder, BondLengthRBF, BondAngleRBF


class GINConv(MessagePassing):

    def __init__(self, emb_dim):
        super(GINConv, self).__init__(aggr="add")
        self.bond_encoder = BondEncoder(emb_dim)
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                       torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr, edge_weight=None):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 3)
        self_loop_attr[:, 0] = 5  # bond type for self-loop edge
        self_loop_attr[:, 1] = 7  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        if edge_weight is not None:
            self_loop_weights = torch.ones(self_loop_attr.shape[0], dtype=edge_weight.dtype, device=edge_weight.device)
            edge_weight = torch.cat((edge_weight, self_loop_weights), dim=0)

        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x,
                                                           edge_attr=edge_embedding,
                                                           edge_weight=edge_weight))
        return out

    def message(self, x_j, edge_attr, edge_weight):
        # return F.relu(x_j + edge_attr)
        return F.relu(x_j + edge_attr) if edge_weight is None else F.relu(x_j + edge_attr) * edge_weight.view(-1, 1)

    def update(self, aggr_out):
        return aggr_out


class GINConvBondAngle(MessagePassing):
    def __init__(self, emb_dim):
        super(GINConvBondAngle, self).__init__(aggr="add")
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2 * emb_dim),
                                       torch.nn.BatchNorm1d(2 * emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2 * emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

    def forward(self, x, edge_index, edge_attr):
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))
        return out

    def message(self, x_j, edge_attr):
        # return F.relu(x_j + edge_attr)
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


class GNNBlock(torch.nn.Module):

    def __init__(self, embed_dim, dropout_rate, last_act=False, residual=False):
        super(GNNBlock, self).__init__()

        self.embed_dim = embed_dim
        self.last_act = last_act
        self.drop_rate = dropout_rate
        self.residual = residual
        self.gnn = GINConv(embed_dim)
        self.norm = torch.nn.BatchNorm1d(embed_dim)

    def forward(self, x, edge_index, edge_attr, edge_weight=None):

        out = self.gnn(x, edge_index, edge_attr, edge_weight)
        out = self.norm(out)
        if self.last_act:
            out = F.relu(out)
        out = F.dropout(out, self.drop_rate)
        if self.residual:
            out += x
        return out


class GeometryGraphAttnBias(torch.nn.Module):
    def __init__(self,
                 num_layer,
                 num_heads,
                 JK='last',
                 drop_ratio=0,
                 use_super_node=True,
                 residual=False
                 ):
        super(GeometryGraphAttnBias, self).__init__()
        self.use_super_node = use_super_node
        self.residual = residual
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.bond_length_encoder = BondLengthRBF(num_heads)
        self.bond_angle_encoder = BondAngleRBF(num_heads)
        self.layer_norm = torch.nn.LayerNorm(num_heads)

        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConvBondAngle(num_heads))

        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(num_heads))

    def forward(self, x, batch):
        edge_index = batch['edge_index_3d']
        x = self.bond_length_encoder(batch['x_3d'])
        edge_attr = self.bond_angle_encoder(batch['edge_attr_3d'])
        h_x = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_x[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer != self.num_layer - 1:
                h = F.relu(h)
            h = F.dropout(h, self.drop_ratio)
            if self.residual:
                h += h_x[layer]
            h_x.append(h)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_x, dim=1)
        elif self.JK == "last":
            node_representation = h_x[-1]
        elif self.JK == "max":
            h_x = [h.unsqueeze_(0) for h in h_x]
            node_representation = torch.max(torch.cat(h_x, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_x = [h.unsqueeze_(0) for h in h_x]
            node_representation = torch.sum(torch.cat(h_x, dim=0), dim=0)
        else:
            raise ValueError(self.JK)
        node_representation = self.layer_norm(node_representation)
        n_graph, n_node = batch['x_mask'].size()
        adj = batch['adj'].clone()
        adj[:, torch.arange(n_node), torch.arange(n_node)] = False
        edge_mask = adj.reshape(-1)
        edge_bias_zeros = torch.zeros([n_graph*n_node*n_node, node_representation.size(-1)],
                                      device=node_representation.device)
        edge_bias_zeros[edge_mask] = node_representation
        edge_bias = edge_bias_zeros.reshape(n_graph, n_node, n_node, node_representation.shape[-1]).permute(0, 3, 1, 2)

        return edge_bias


class AtomBondGraph(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, JK="last", drop_ratio=0, use_super_node=True):
        super(AtomBondGraph, self).__init__()
        self.use_super_node = use_super_node
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.atom_bond_block = torch.nn.ModuleList()
        for layer_id in range(self.num_layer):
            self.atom_bond_block.append(
                GNNBlock(emb_dim, self.drop_ratio, last_act=(layer_id != self.num_layer - 1)))

    def forward(self, batch, x):
        x_mask = batch['x_mask'].bool().reshape(-1)
        res = x.clone()
        x = x[:, int(self.use_super_node):, :].reshape(-1, x.shape[-1])
        x_zeros = torch.zeros(x.shape, device=x.device)
        x = x[x_mask]
        h_x = [x]
        for i in range(self.num_layer):
            node_hidden = self.atom_bond_block[i](h_x[i], batch['edge_index'], batch['edge_attr'], batch['edge_weight'])
            h_x.append(node_hidden)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_x, dim=1)
        elif self.JK == "last":
            node_representation = h_x[-1]
        elif self.JK == "max":
            h_x = [h.unsqueeze_(0) for h in h_x]
            node_representation = torch.max(torch.cat(h_x, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_x = [h.unsqueeze_(0) for h in h_x]
            node_representation = torch.sum(torch.cat(h_x, dim=0), dim=0)
        else:
            raise ValueError(self.JK)

        x_zeros[x_mask] = node_representation
        res[:, int(self.use_super_node):, :] += x_zeros.reshape(res.shape[0], -1, x.shape[-1])

        return res
