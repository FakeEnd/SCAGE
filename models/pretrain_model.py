from data_process.loader import FGEncoder
from .layers import *
from models.architectures import get_model
from torch_scatter import scatter


def reparame_trick(edge_logit):
    temperature = 1.0
    bias = 0.0 + 0.0001  # If bias is 0, we run into problems
    eps = (bias - (1 - bias)) * torch.rand(edge_logit.size()) + (1 - bias)
    gate_inputs = torch.log(eps) - torch.log(1 - eps)
    gate_inputs = gate_inputs.to('cuda')
    gate_inputs = (gate_inputs + edge_logit) / temperature
    batch_aug_edge_weight = torch.sigmoid(gate_inputs).squeeze()
    return batch_aug_edge_weight


def regular_trick(batch, batch_aug_edge_weight):
    row, col = batch['edge_index']
    edge_batch = batch['batch'][row]
    edge_drop_out_prob = 1 - batch_aug_edge_weight

    uni, edge_batch_num = edge_batch.unique(return_counts=True)
    sum_pe = scatter(edge_drop_out_prob, edge_batch, reduce="sum")
    batch_size = batch['batch'][-1].item() + 1
    reg = []
    for b_id in range(batch_size):
        if b_id in uni:
            num_edges = edge_batch_num[uni.tolist().index(b_id)]
            reg.append(sum_pe[b_id] / num_edges)
        else:
            # means no edges in that graph. So don't include.
            pass
    reg = torch.stack(reg)
    reg = reg.mean()
    return reg


class MolGraphCL(nn.Module):
    def __init__(self, config):
        super(MolGraphCL, self).__init__()
        self.mol_rep = get_model(config)
        hidden_dim = config['hidden_dim']
        self.projection_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, hidden_dim))
        self.fg_layer = nn.Linear(hidden_dim, 83)
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    # def forward(self, batch):
    #     mol_rep, transformer_rep, atom_x = self.mol_rep(batch)
    #     # mol_rep, transformer_rep, atom_x, fg_rep = self.mol_rep(batch)
    #     # mol_rep, _, atom_x = self.mol_rep(batch)
    #     # print('mol_rep=', mol_rep.size())
    #     # print('atom_x=', atom_x.size())
    #     # print('fg_rep=', fg_rep.size())
    #     out = self.projection_head(mol_rep)
    #     # print('out=', out.size())
    #     # fg_out = self.fg_layer(atom_x)
    #     # fg_out = self.fg_layer(fg_rep)
    #     # print('fg_out=', fg_out.size())
    #     return out, atom_x
    #     # return out, atom_x

    def forward(self, batch):
        mol_rep, transformer_rep, atom_x = self.mol_rep(batch)
        # mol_rep, transformer_rep, atom_x, fg_rep = self.mol_rep(batch)
        # mol_rep, _, atom_x = self.mol_rep(batch)
        # print('mol_rep=', mol_rep.size())
        # print('atom_x=', atom_x.size())
        # print('fg_rep=', fg_rep.size())
        out = self.projection_head(mol_rep)
        # print('out=', out.size())
        fg_out = self.fg_layer(atom_x)
        # fg_out = self.fg_layer(fg_rep)
        # print('fg_out=', fg_out.size())
        return out, atom_x, fg_out
        # return out, atom_x

    def loss_fg(self, outs, labels):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outs, labels)
        return loss

    def loss_cl(self, x1, x2):
        T = 0.2
        # print('x1.size()=', x1.size())
        # print('x2.size()=', x2.size())
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


class ViewLearner(nn.Module):
    def __init__(self, config):
        super(ViewLearner, self).__init__()
        self.mol_rep = get_model(config)
        hidden_dim = config['hidden_dim']
        self.projection_head = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim),
                                             nn.ReLU(),
                                             nn.Linear(hidden_dim, 1))
        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch, node_rep):
        # _, node_rep, _ = self.mol_rep.get_atom_rep(batch)
        edge_index = batch['edge_index']
        src, dst = edge_index[0], edge_index[1]
        emb_src = node_rep[src]
        emb_dst = node_rep[dst]

        edge_emb = torch.cat([emb_src, emb_dst], 1)
        edge_logit = self.projection_head(edge_emb)

        return edge_logit

# def attention(query, key, value, mask, dropout=None):
#     """Compute 'Scaled Dot Product Attention'"""
#     d_k = query.size(-1)
#     scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
#     if mask is not None:
#         scores = scores.masked_fill(mask == 0, -1e9)
#
#     p_attn = F.softmax(scores, dim=-1)
#     if dropout is not None:
#         p_attn = dropout(p_attn)
#     return torch.matmul(p_attn, value), p_attn
#
# class AttentionLayer(nn.Module):
#     def __init__(self, hidden_dim):
#         super(AttentionLayer, self).__init__()
#         self.hidden_dim = hidden_dim
#         self.w_q = nn.Linear(self.hidden_dim, 32)
#         self.w_k = nn.Linear(self.hidden_dim, 32)
#         self.w_v = nn.Linear(self.hidden_dim, 32)
#
#         self.dense = nn.Linear(32, self.hidden_dim)
#         self.LayerNorm = nn.LayerNorm(self.hidden_dim, eps=1e-6)
#         self.dropout = nn.Dropout(0.1)
#
#     def forward(self, fg_hiddens, init_hiddens):
#         query = self.w_q(fg_hiddens)
#         key = self.w_k(fg_hiddens)
#         value = self.w_v(fg_hiddens)
#
#         padding_mask = (init_hiddens != 0) + 0.0
#         mask = torch.matmul(padding_mask, padding_mask.transpose(-2, -1))
#         x, attn = attention(query, key, value, mask)
#
#         hidden_states = self.dense(x)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states + fg_hiddens)
#
#         return hidden_states
#
# def group_node_rep(node_rep, batch_index, batch_size):
#     group = []
#     count = 0
#     for i in range(batch_size):
#         num = sum(batch_index == i)
#         group.append(node_rep[count:count + num])
#         count += num
#     return group
#
# class FGPred(nn.Module):
#     def __init__(self, batch_size, hidden_dim):
#         super(FGPred, self).__init__()
#         self.batch_size = batch_size
#         self.hidden_dim = hidden_dim
#         self.vocab_size = 83
#
#         self.fg_feature_encoder = FGEncoder(self.vocab_size, self.hidden_dim)
#
#         self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
#         self.alpha.data.fill_(0.1)
#         self.attention_layer_1 = AttentionLayer(hidden_dim)
#         self.attention_layer_2 = AttentionLayer(hidden_dim)
#         self.norm = nn.LayerNorm(self.hidden_dim)
#
#         self.fg_layer = nn.Linear(hidden_dim, 83)
#
#     def forward(self, batch_data):
#
#         # node_rep = group_node_rep(node_rep, batch_data['batch'].cpu().numpy(), self.batch_size)
#
#         fg = self.fg_feature_encoder(batch_data['fg'])
#
#         # print('fg.size=', fg.size())
#         # print('fg=', fg)
#
#         hidden_states = self.attention_layer_1(fg, fg)
#         hidden_states = self.attention_layer_2(hidden_states, fg)
#
#         # print('hidden_states.size=', hidden_states.size())
#         # print('hidden_states=', hidden_states)
#
#         fg = self.norm(hidden_states)
#
#         # print('fg.size=', fg.size())
#         # print('fg=', fg)
#
#         fg = self.fg_layer(fg)
#
#         # print('fg.size=', fg.size())
#         # print('fg=', fg)
#
#         criterion = nn.CrossEntropyLoss()
#         loss = criterion(fg, batch_data['fg'])
#         # print('loss=', loss)
#
#         return fg, loss









