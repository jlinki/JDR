import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter, Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import MessagePassing, APPNP


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        if Init == 'SGC':
            self.alpha = int(alpha)
        else:
            self.alpha = float(alpha)

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like, note that in this case, alpha has to be a integer. It means where the peak at when initializing GPR weights.
            TEMP = 0.0*np.ones(K+1)
            TEMP[self.alpha] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = self.alpha*(1-self.alpha)**np.arange(K+1)
            TEMP[-1] = (1-self.alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (self.alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):


        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        if args.dataset.lower() in ['tolokers', 'minesweeper', 'questions']:
            self.lin2 = Linear(args.hidden, 1)
        else:
            self.lin2 = Linear(args.hidden, dataset.num_classes)

        if args.ppnp == 'PPNP':
            self.prop1 = APPNP(args.K, args.alpha)
        elif args.ppnp == 'GPR_prop':
            self.prop1 = GPR_prop(args.K, args.alpha, args.Init, args.Gamma)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout
        self.args = args

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index, edge_weight=data.edge_attr)
            if self.args.dataset.lower() in ['tolokers', 'minesweeper', 'questions']:
                return x.squeeze(1)
            else:
                return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index, edge_weight=data.edge_attr)
            if self.args.dataset.lower() in ['tolokers', 'minesweeper', 'questions']:
                return x.squeeze(1)
            else:
                return F.log_softmax(x, dim=1)


class GCN_Net(nn.Module):
    def __init__(self, dataset, args):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight=data.edge_attr))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=data.edge_attr)
        return F.log_softmax(x, dim=1)


class GCN_large(torch.nn.Module):
    '''
    GCN from Lim et al. (2021)
    '''
    def __init__(self, dataset, args):
        super(GCN_large, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(dataset.num_features, args.hidden, cached=True))
        self.bns = torch.nn.ModuleList()
        if args.dataset.lower() in ['penn94']:
            self.bns.append(torch.nn.Identity())
        else:
            self.bns.append(torch.nn.BatchNorm1d(args.hidden))
        for _ in range(args.num_layers - 2):
            self.convs.append(
                GCNConv(args.hidden, args.hidden, cached=True))
            if args.dataset.lower() in ['penn94']:
                self.bns.append(torch.nn.Identity())
            else:
                self.bns.append(torch.nn.BatchNorm1d(args.hidden))
        self.convs.append(GCNConv(args.hidden, dataset.num_classes, cached=True))

        self.dropout = args.dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight=data.edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight=data.edge_attr)
        return x.log_softmax(dim=-1)


class ChebNet(nn.Module):
    def __init__(self, dataset, args):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(dataset.num_features, 32, K=2)
        self.conv2 = ChebConv(32, dataset.num_classes, K=2)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT_Net(nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNP_Net(nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN_JKNet(nn.Module):
    def __init__(self, dataset, args):
        in_channels = dataset.num_features
        out_channels = dataset.num_classes

        super(GCN_JKNet, self).__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, 16)
        self.lin1 = nn.Linear(16, out_channels)
        self.one_step = APPNP(K=1, alpha=0)
        self.JK = JumpingKnowledge(mode='lstm',
                                   channels=16,
                                   num_layers=4
                                   )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x1 = F.relu(self.conv1(x, edge_index))
        x1 = F.dropout(x1, p=0.5, training=self.training)

        x2 = F.relu(self.conv2(x1, edge_index))
        x2 = F.dropout(x2, p=0.5, training=self.training)

        x = self.JK([x1, x2])
        x = self.one_step(x, edge_index)
        x = self.lin1(x)
        return F.log_softmax(x, dim=1)


class MLP(nn.Module):
    def __init__(self, dataset, args):
        super(MLP, self).__init__()

        layers = []
        layers.append(nn.Linear(dataset.num_features, args.hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=args.dropout))

        for _ in range(args.num_layers - 2):
            layers.append(nn.Linear(args.hidden, args.hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=args.dropout))

        layers.append(nn.Linear(args.hidden, args.hidden))

        self.mlp = nn.Sequential(*layers)

    def forward(self, data):
        x = data.x
        return F.log_softmax(self.mlp(x), dim=1)


# HeteroGNN from Platonov et al. (2023)

class ResidualModuleWrapper(nn.Module):
    def __init__(self, module, normalization, dim, **kwargs):
        super().__init__()
        self.normalization = normalization(dim)
        self.module = module(dim=dim, **kwargs)

    def forward(self, data, x):
        x_res = self.normalization(x)
        x_res = self.module(data, x_res)
        x = x + x_res
        return x


class FeedForwardModule(nn.Module):
    def __init__(self, dim, hidden_dim_multiplier, dropout, input_dim_multiplier=1, **kwargs):
        super().__init__()
        input_dim = int(dim * input_dim_multiplier)
        hidden_dim = int(dim * hidden_dim_multiplier)
        self.linear_1 = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=dim)
        self.dropout_2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)

        return x


class GCNModule(MessagePassing):
    def __init__(self, dim, hidden_dim_multiplier, dropout, **kwargs):
        super(GCNModule, self).__init__(aggr='add', **kwargs)  # 'add' aggregation
        self.feed_forward_module = FeedForwardModule(dim=dim,
                                                     hidden_dim_multiplier=hidden_dim_multiplier,
                                                     dropout=dropout)
        self.conv = GCNConv(dim, dim)

    def forward(self, data, x):
        x = self.conv(x, data.edge_index, data.edge_attr)
        x = self.feed_forward_module(x)
        return x

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

MODULES = {
    'ResNet': [FeedForwardModule],
    'GCN': [GCNModule],
}


NORMALIZATION = {
    'None': nn.Identity,
    'LayerNorm': nn.LayerNorm,
    'BatchNorm': nn.BatchNorm1d
}


class HeteroGCN(nn.Module):
    def __init__(self, dataset, args):

        super().__init__()
        model_name = 'GCN'
        num_layers = args.num_layers
        input_dim = dataset.num_features
        hidden_dim = args.hidden
        if args.dataset.lower() in ['tolokers', 'minesweeper', 'questions']:
            output_dim = 1
        else:
            output_dim = dataset.num_classes
        hidden_dim_multiplier = 1
        num_heads = args.heads
        normalization = 'LayerNorm'
        dropout = args.dropout


        normalization = NORMALIZATION[normalization]

        self.input_linear = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.GELU()

        self.residual_modules = nn.ModuleList()
        for _ in range(num_layers):
            for module in MODULES[model_name]:
                residual_module = ResidualModuleWrapper(module=module,
                                                        normalization=normalization,
                                                        dim=hidden_dim,
                                                        hidden_dim_multiplier=hidden_dim_multiplier,
                                                        dropout=dropout)

                self.residual_modules.append(residual_module)

        self.output_normalization = normalization(hidden_dim)
        self.output_linear = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.args = args

    def forward(self, data):
        x = self.input_linear(data.x)
        x = self.dropout(x)
        x = self.act(x)

        for residual_module in self.residual_modules:
            x = residual_module(data, x)

        x = self.output_normalization(x)
        x = self.output_linear(x)

        if self.args.dataset.lower() in ['tolokers', 'minesweeper', 'questions']:
            return x.squeeze(1)
        else:
            return F.log_softmax(x, dim=1)