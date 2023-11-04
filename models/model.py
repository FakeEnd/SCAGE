from .gnn_layers import GeometryGraphAttnBias, AtomBondGraph
from .gt_layers import *
from .layers import *
from .multi_scale_transformer import MultiScaleTransformer
from torch_geometric.nn import global_mean_pool
from data_process.loader import AtomEncoder, FGEncoder
from .model_utils import GraphPool

def init_graphormer_params(module):
    def normal_(data):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class GraphTransformer(nn.Module):
    def __init__(
            self,
            use_sms: bool = True,
            num_encoder_layers: int = 6,
            hidden_dim: int = 80,
            ffn_hidden_dim: int = 80,
            num_attn_heads: int = 8,
            emb_dropout: float = 0,
            dropout: float = 0.1,
            attn_dropout: float = 0.1,
            dist_bar: list = None,
            encoder_normalize_before: bool = False,
            apply_graphormer_init: bool = True,
            activation_fn: str = "gelu",
            n_trans_layers_to_freeze: int = 0,

            use_super_node: bool = True,
            graph_pooling: str = None,
            afps_k: int = 10,

            node_level_modules: tuple = ('degree', 'eig'),
            attn_mask_modules: str = None,

            num_in_degree: int = None,
            num_out_degree: int = None,
            eig_pos_dim: int = None,
            svd_pos_dim: int = None,
            use_gnn_layers: bool = False,
            residual: bool = False,
            num_gnn_layers: int = 1,
            JK: str = 'last',
            gnn_dropout: float = 0.5
    ):
        super(GraphTransformer, self).__init__()
        self.use_sms = use_sms
        self.dist_bar = dist_bar
        self.padding_mask = None
        self.attn = None
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.hidden_dim = hidden_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.use_super_node = use_super_node
        self.use_gnn_layers = use_gnn_layers
        self.num_attn_heads = num_attn_heads
        self.attn_mask_modules = attn_mask_modules
        self.graph_pooling = GraphPool(graph_pooling=graph_pooling,
                                       hidden_dim=hidden_dim,
                                       afps_k=hidden_dim,
                                       use_super_node=hidden_dim)

        self.afps_norm = nn.LayerNorm(hidden_dim)

        if encoder_normalize_before:
            self.emb_layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.emb_layer_norm = None

        self.node_feature_encoder = AtomEncoder(emb_dim=hidden_dim)

        self.fg_feature_encoder = FGEncoder(83, self.hidden_dim)

        if use_super_node:
            self.add_super_node = AddSuperNode(hidden_dim=hidden_dim)

        self.node_level_layers = nn.ModuleList([])
        if node_level_modules != 'None':
            for module_name in node_level_modules:
                if module_name == 'degree':
                    layer = DegreeEncoder(num_in_degree=num_in_degree,
                                          num_out_degree=num_out_degree,
                                          hidden_dim=hidden_dim,
                                          n_layers=num_encoder_layers)
                elif module_name == 'eig':
                    layer = Eig_Embedding(eig_dim=eig_pos_dim, hidden_dim=hidden_dim)
                elif module_name == 'svd':
                    layer = SVD_Embedding(svd_dim=svd_pos_dim, hidden_dim=hidden_dim)
                else:
                    raise ValueError('node level module error!')
                self.node_level_layers.append(layer)

        self.geometry_layer = GeometryGraphAttnBias(num_layer=num_gnn_layers,
                                                    num_heads=num_attn_heads,
                                                    JK=JK,
                                                    drop_ratio=gnn_dropout,
                                                    use_super_node=use_super_node,
                                                    residual=residual)

        # gnn layers
        if use_gnn_layers:
            self.gnn_layers = AtomBondGraph(
                num_layer=num_gnn_layers,
                emb_dim=hidden_dim,
                JK=JK,
                drop_ratio=gnn_dropout,
                use_super_node=use_super_node,
            )

        # transformer layers
        self.transformer_layers = nn.ModuleList([
            Transformer_Layer(
                num_heads=num_attn_heads,
                hidden_dim=hidden_dim,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
                temperature=1,
                activation_fn=activation_fn
            ) for _ in range(num_encoder_layers)
        ])

        self.multi_scale_transformer_layers = nn.ModuleList([
            MultiScaleTransformer(
                num_heads=num_attn_heads,
                hidden_dim=hidden_dim,
                dist_bar=dist_bar,
                ffn_hidden_dim=ffn_hidden_dim,
                dropout=dropout,
                attn_dropout=attn_dropout,
                temperature=0.1,
                activation_fn=activation_fn,
                use_super_node=use_super_node
            ) for _ in range(num_encoder_layers)
        ])

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

        # 冻结部分transformer层
        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        for layer in range(n_trans_layers_to_freeze):
            if use_sms:
                freeze_module_params(self.multi_scale_transformer_layers[layer])
            else:
                freeze_module_params(self.transformer_layers[layer])

        self.fg_layer = nn.Linear(hidden_dim, 83)

    def get_atom_rep(self, batch_data):
        # calculate attention padding mask # B x T x T / B x T+1 x T+1
        self.padding_mask = batch_data['x_mask']
        n_graph, n_node = self.padding_mask.size()
        # print('n_graph=', n_graph)
        # print('n_node=', n_node)
        if self.use_super_node:
            padding_mask_cls = torch.ones(n_graph, 1, device=self.padding_mask.device, dtype=self.padding_mask.dtype)
            self.padding_mask = torch.cat((padding_mask_cls, self.padding_mask), dim=1).float()
        attn_mask = torch.matmul(self.padding_mask.unsqueeze(-1), self.padding_mask.unsqueeze(1)).long()

        # x feature encode
        x = self.node_feature_encoder(batch_data['x'])  # B x T x C
        # print('---------------------------')
        # print('x1.size=', x.size())
        x_zeros = torch.zeros([n_graph * n_node, x.shape[-1]], device=x.device)
        x_mask = batch_data['x_mask'].bool().reshape(-1)
        # print('x_mask.size=', x_mask.size())
        # print('x2.size=', x.size())
        # print('x_zeros.size=', x_zeros.size())
        # print('self.padding_mask.device=', self.padding_mask.device)
        # print('x_mask.device=', x_mask.device)
        # print('x.device=', x.device)
        # print('x_zeros.device=', x_zeros.device)
        x_zeros[x_mask] = x
        x = x_zeros.reshape(n_graph, n_node, x.shape[-1])
        # print('x3.size=', x.size())
        # print('---------------------------')

        for nl_layer in self.node_level_layers:
            node_bias = nl_layer(batch_data)
            x += node_bias
        # add the super node
        if self.use_super_node:
            x = self.add_super_node(x)  # B x T+1 x C

        # print('batch x.size=', batch_data['x'].size())
        # attention bias computation,  B x H x (T+1) x (T+1)  or B x H x T x T
        attn_bias = torch.zeros(n_graph, self.num_attn_heads, n_node + int(self.use_super_node),
                                n_node + int(self.use_super_node)).to(x.device)
        bias = self.geometry_layer(x, batch_data)
        if bias.shape[-1] == attn_bias.shape[-1]:
            attn_bias += bias
        elif bias.shape[-1] == attn_bias.shape[-1] - 1:
            attn_bias[:, :, int(self.use_super_node):, int(self.use_super_node):] = attn_bias[:, :,
                                                                                    int(self.use_super_node):,
                                                                                    int(self.use_super_node):] + bias
        else:
            raise ValueError('attention calculation error')

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        # gnn layers
        # print('x1.size=', x.size())
        x = self.gnn_layers(batch_data, x)
        # print('x2.size=', x.size())

        # transformer layers
        if self.use_sms:
            transformer_layer = self.multi_scale_transformer_layers
        else:
            transformer_layer = self.transformer_layers

        # graph transformer layers
        for i, layer in enumerate(transformer_layer):
            if self.use_sms:
                x, self.attn, attn_list = layer.attention(
                    x=x,
                    dist=batch_data['spatial_pos'],
                    mask=attn_mask,
                    attn_bias=attn_bias
                )
            else:
                x, self.attn = layer.attention(x=x, mask=attn_mask, attn_bias=attn_bias)
                attn_list = None

            # FFN layer
            x = layer.ffn_layer(x)
        transformer_x = x
        # print('int(self.use_super_node)=', int(self.use_super_node))
        # print('transformer_x=', transformer_x.size())
        z = x[:, int(self.use_super_node):, :].reshape(-1, x.shape[-1])
        # print('z=', z.size())
        graph_x = z[x_mask]
        # print('x_mask=', x_mask)
        # print('graph_x=', graph_x.size())

        return transformer_x, graph_x, attn_list

    def get_fg_rep(self, batch_data):
        # calculate attention padding mask # B x T x T / B x T+1 x T+1
        self.padding_mask = batch_data['x_mask']
        n_graph, n_node = self.padding_mask.size()
        # print('n_graph=', n_graph)
        # print('n_node=', n_node)
        if self.use_super_node:
            padding_mask_cls = torch.ones(n_graph, 1, device=self.padding_mask.device, dtype=self.padding_mask.dtype)
            self.padding_mask = torch.cat((padding_mask_cls, self.padding_mask), dim=1).float()
        attn_mask = torch.matmul(self.padding_mask.unsqueeze(-1), self.padding_mask.unsqueeze(1)).long()

        # x feature encode
        # print('---------------------------')
        x = self.fg_feature_encoder(batch_data['fg'])  # B x T x C
        # print('x1.size=', x.size())
        # x = x.unsqueeze()
        # print('x2.size=', x.size())
        # print('x1.size=', x.size())
        x_zeros = torch.zeros([n_graph * n_node, x.shape[-1]], device=x.device)
        x_mask = batch_data['x_mask'].bool().reshape(-1)
        # print('x_mask.size=', x_mask.size())
        # print('x2.size=', x.size())
        # print('x_zeros.size=', x_zeros.size())
        x_zeros[x_mask] = x
        x = x_zeros.reshape(n_graph, n_node, x.shape[-1])
        # print('x2.size=', x.size())
        # print('---------------------------')

        for nl_layer in self.node_level_layers:
            node_bias = nl_layer(batch_data)
            x += node_bias
        # add the super node
        if self.use_super_node:
            x = self.add_super_node(x)  # B x T+1 x C

        # print('batch x.size=', batch_data['x'].size())
        # attention bias computation,  B x H x (T+1) x (T+1)  or B x H x T x T
        attn_bias = torch.zeros(n_graph, self.num_attn_heads, n_node + int(self.use_super_node),
                                n_node + int(self.use_super_node)).to(x.device)
        bias = self.geometry_layer(x, batch_data)
        if bias.shape[-1] == attn_bias.shape[-1]:
            attn_bias += bias
        elif bias.shape[-1] == attn_bias.shape[-1] - 1:
            attn_bias[:, :, int(self.use_super_node):, int(self.use_super_node):] = attn_bias[:, :,
                                                                                    int(self.use_super_node):,
                                                                                    int(self.use_super_node):] + bias
        else:
            raise ValueError('attention calculation error')

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        # gnn layers
        # print('x1.size=', x.size())
        x = self.gnn_layers(batch_data, x)
        # print('x2.size=', x.size())

        # transformer layers
        if self.use_sms:
            transformer_layer = self.multi_scale_transformer_layers
        else:
            transformer_layer = self.transformer_layers

        # graph transformer layers
        for i, layer in enumerate(transformer_layer):
            if self.use_sms:
                x, self.attn, attn_list = layer.attention(
                    x=x,
                    dist=batch_data['spatial_pos'],
                    mask=attn_mask,
                    attn_bias=attn_bias
                )
            else:
                x, self.attn = layer.attention(x=x, mask=attn_mask, attn_bias=attn_bias)
                attn_list = None

            # FFN layer
            x = layer.ffn_layer(x)
        transformer_x = x
        z = x[:, int(self.use_super_node):, :].reshape(-1, x.shape[-1])
        graph_x = z[x_mask]

        return transformer_x, graph_x, attn_list

    def forward(self, batch_data):
        transformer_rep, atom_rep, _ = self.get_atom_rep(batch_data)
        graph_rep = self.graph_pooling(transformer_rep, batch_data, self.attn)
        # transformer_fg_rep, fg_rep, _ = self.get_fg_rep(batch_data)
        # print('fg_rep.size=', fg_rep.size())
        # print('transformer_rep.size=', transformer_rep.size())
        # print('graph_rep.size=', graph_rep.size())
        # return graph_rep, transformer_rep, atom_rep, fg_rep
        return graph_rep, transformer_rep, atom_rep

    def load_model_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            name = name.split('.', 1)[1]
            # print(name)
            if name.split('.', 1)[0] == 'multi_scale_transformer_layers' and name.split('.', 4)[
                -2] == 'scale_linear' and len(self.dist_bar) < 4:
                continue
            if name not in own_state:
                continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)
