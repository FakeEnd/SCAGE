import torch.nn as nn
from collections import OrderedDict
from models.architectures import get_model


class MLP(nn.Module):
    def __init__(self, inp_dim, hidden_dim, num_layers, num_tasks, batch_norm=False, dropout=0.):
        super(MLP, self).__init__()
        layer_list = OrderedDict()
        in_dim = inp_dim
        for l in range(num_layers):
            if l < num_layers - 1:
                layer_list['fc{}'.format(l)] = nn.Linear(in_dim, hidden_dim)
                if batch_norm:
                    layer_list['norm{}'.format(l)] = nn.BatchNorm1d(num_features=hidden_dim)
                layer_list['relu{}'.format(l)] = nn.ReLU()
                if dropout > 0:
                    layer_list['drop{}'.format(l)] = nn.Dropout(p=dropout)
            else:
                layer_list['fc{}'.format(l)] = nn.Linear(in_dim, num_tasks)

            in_dim = hidden_dim
        if num_layers > 0:
            self.network = nn.Sequential(layer_list)
        else:
            self.network = nn.Identity()

    def forward(self, emb):
        out = self.network(emb)
        return out


class DownstreamModel(nn.Module):
    def __init__(self, config):
        super(DownstreamModel, self).__init__()
        self.model = get_model(config['model'])
        self.mlp_proj = MLP(**config['DownstreamModel'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):
        view, _, _ = self.model(batch)
        pred = self.mlp_proj(view)
        return pred

