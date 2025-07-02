import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, LayerNorm, Sequential
from torch_geometric.data import Data
from torch_sparse.matmul import spmm_max, spmm_mean, spmm_add
from functools import partial
from torch_sparse import SparseTensor    

from src.utils import DropAdj, DropEdge

class BGRL_GCN(nn.Module):
    def __init__(self, in_channels: int, param):
        super().__init__()
        layer_sizes=param['layer_sizes']
        batchnorm=param['batch_layer_norm']
        batchnorm_mm=param['batchnorm_mm']
        layernorm=not param['batch_layer_norm']
        weight_standardization=param['weight_standardization']
        # print(batchnorm, layernorm)
        assert batchnorm != layernorm
        assert len(layer_sizes) >= 1
        self.input_size, self.representation_size = in_channels, layer_sizes[-1]
        layer_sizes.insert(0, in_channels)
        self.weight_standardization = weight_standardization

        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append((GCNConv(in_dim, out_dim), 'x, edge_index -> x'))

            if batchnorm:
                layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append(LayerNorm(out_dim))

            layers.append(nn.PReLU())
        print(layers)
        self.model = Sequential('x, edge_index', layers)

    def forward(self, x, edge_index):
        if self.weight_standardization:
            self.standardize_weights()
        return self.model(x, edge_index)

    def reset_parameters(self):
        self.model.reset_parameters()

    def standardize_weights(self):
        skipped_first_conv = False
        for m in self.model.modules():
            if isinstance(m, GCNConv):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight


class TBGRL_GCN(nn.Module):
    """Basic GCN encoder.
    This is based off of the official BGRL encoder implementation.
    """

    def __init__(
        self,
        layer_sizes,
        batchnorm=False,
        batchnorm_mm=0.99,
        layernorm=False,
        weight_standardization=False,
        use_feat=True,
        n_nodes=0,
        batched=False,
    ):
        super().__init__()

        assert batchnorm != layernorm
        assert len(layer_sizes) >= 2
        self.n_layers = len(layer_sizes)
        self.batched = batched
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization

        layers = []
        relus = []
        batchnorms = []

        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            if batched:
                layers.append(GCNConv(in_dim, out_dim))
                relus.append(nn.PReLU())
                if batchnorm:
                    batchnorms.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append(
                    (GCNConv(in_dim, out_dim), 'x, edge_index -> x'),
                )

                if batchnorm:
                    layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
                else:
                    layers.append(LayerNorm(out_dim))

                layers.append(nn.PReLU())

        if batched:
            self.convs = nn.ModuleList(layers)
            self.relus = nn.ModuleList(relus)
            self.batchnorms = nn.ModuleList(batchnorms)
        else:
            self.model = Sequential('x, edge_index', layers)

        self.use_feat = use_feat
        if not self.use_feat:
            self.node_feats = nn.Embedding(n_nodes, layer_sizes[1])

    def split_forward(self, x, edge_index):
        """Convenience function to perform a forward pass on a feature matrix
        and edge index separately without needing to create a Data object.
        """
        return self(Data(x, edge_index))

    def forward(self, x, edge_index):
        if not self.batched:
            if self.weight_standardization:
                self.standardize_weights()
            if self.use_feat:
                return self.model(x, edge_index)
            return self.model(self.node_feats.weight.data.clone(), edge_index)
        # otherwise, batched
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.relus[i](x)
            x = self.batchnorms[i](x)
        return x

    def reset_parameters(self):
        self.model.reset_parameters()

    def standardize_weights(self):
        skipped_first_conv = False
        for m in self.model.modules():
            if isinstance(m, GCNConv):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight

    def get_node_feats(self):
        if hasattr(self, 'node_feats'):
            return self.node_feats
        return None

    @property
    def num_layers(self):
        return self.n_layers


class GRACE_GCN(nn.Module):
    def __init__(self, in_channels: int, param):
        super(GRACE_GCN, self).__init__()
        out_channels: int = param['hidden']
        switch = {
            'identity': nn.Identity(),
            'relu': F.relu,
            'prelu': nn.PReLU(),
        }
        activation = switch[param['activation']]
        base_model=GCNConv
        k = param['n_layers']
        skip=param['skip']
        self.base_model = base_model
        self.out_channels = out_channels

        assert k >= 2
        self.k = k
        self.skip = skip
        if not self.skip:
            self.conv = [base_model(in_channels, 2 * out_channels).jittable()]
            for _ in range(1, k - 1):
                self.conv.append(base_model(2 * out_channels, 2 * out_channels))
            self.conv.append(base_model(2 * out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation
        else:
            self.fc_skip = nn.Linear(in_channels, out_channels)
            self.conv = [base_model(in_channels, out_channels)]
            for _ in range(1, k):
                self.conv.append(base_model(out_channels, out_channels))
            self.conv = nn.ModuleList(self.conv)

            self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        if not self.skip:
            for i in range(self.k):
                x = self.activation(self.conv[i](x, edge_index))
            return x
        else:
            h = self.activation(self.conv[0](x, edge_index))
            hs = [self.fc_skip(x), h]
            for i in range(1, self.k):
                u = sum(hs)
                hs.append(self.activation(self.conv[i](u, edge_index)))
            return hs[-1]
    
    def reset_parameters(self):
        for conv in self.conv:
            conv.reset_parameters()
        if self.skip:
            self.fc_skip.reset_parameters()

class NCN_GCN(nn.Module):
    
    def __init__(self,
                 in_channels,
                 param):
        super().__init__()

        hidden_channels = param['hidden']
        out_channels = param['hidden']
        num_layers = param['n_layers']
        dropout = param['gnn_dp']
        ln = param['layer_norm']
        res = param['res']
        jk = param['jk']
        edrop = param['edrop']
        xdropout = param['xdropout']
        taildropout = param['taildropout']
        

        self.eidrop = DropEdge(edrop)
        self.adjdrop = DropAdj(edrop)
        
    
        self.xemb = nn.Sequential(nn.Dropout(xdropout)) #nn.Identity()
        if ("pure" in param['conv_fn'] or num_layers==0):
            self.xemb.append(nn.Linear(in_channels, hidden_channels))
            self.xemb.append(nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity())
        
        self.res = res
        self.jk = jk
        if jk:
            self.register_parameter("jkparams", nn.Parameter(torch.randn((num_layers,))))

        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        if num_layers == 1:
            hidden_channels = out_channels

        convfn = GCNConv

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        if "pure" in param['conv_fn']:
            self.convs.append(convfn(hidden_channels, hidden_channels))
            for i in range(num_layers-1):
                self.lins.append(nn.Identity())
                self.convs.append(convfn(hidden_channels, hidden_channels))
            self.lins.append(nn.Dropout(taildropout, True))
        else:
            self.convs.append(convfn(in_channels, hidden_channels))
            self.lins.append(
                nn.Sequential(lnfn(hidden_channels, ln), nn.Dropout(dropout, True),
                            nn.ReLU(True)))
            for i in range(num_layers - 1):
                self.convs.append(
                    convfn(
                        hidden_channels,
                        hidden_channels if i == num_layers - 2 else out_channels))
                if i < num_layers - 2:
                    self.lins.append(
                        nn.Sequential(
                            lnfn(
                                hidden_channels if i == num_layers -
                                2 else out_channels, ln),
                            nn.Dropout(dropout, True), nn.ReLU(True)))
                else:
                    self.lins.append(nn.Identity())
        

    def forward(self, x, adj_t):
        if isinstance(adj_t, torch.Tensor):
            drop = self.eidrop
        elif isinstance(adj_t, SparseTensor):
            drop = self.adjdrop
        x = self.xemb(x)
        jkx = []
        for i, conv in enumerate(self.convs):
            x1 = self.lins[i](conv(x, drop(adj_t)))
            if self.res and x1.shape[-1] == x.shape[-1]: # residual connection
                x = x1 + x
            else:
                x = x1
            if self.jk:
                jkx.append(x)
        if self.jk: # JumpingKnowledge Connection
            jkx = torch.stack(jkx, dim=0)
            sftmax = self.jkparams.reshape(-1, 1, 1)
            x = torch.sum(jkx*sftmax, dim=0)
        return x

    def reset_parameters(self):
        for layer in self.xemb:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        for conv in self.convs:
            if hasattr(conv, 'reset_parameters'):
                conv.reset_parameters()

        for lin in self.lins:
            if hasattr(lin, 'reset_parameters'):
                lin.reset_parameters()
