import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, LayerNorm, Sequential
from torch_sparse import SparseTensor

from src.utils import DropAdj, DropEdge


class BGRL_GCN(nn.Module):
    """
    Encoder from BGRL
    """
    def __init__(self, in_channels: int, param: dict):
        super().__init__()
        layer_sizes = copy.deepcopy(param["layer_sizes"])
        batchnorm = param["batch_layer_norm"]
        batchnorm_mm = param["batchnorm_mm"]
        layernorm = not param["batch_layer_norm"]
        weight_standardization = param["weight_standardization"]
        assert batchnorm != layernorm
        assert len(layer_sizes) >= 1
        self.input_size, self.representation_size = in_channels, layer_sizes[-1]
        layer_sizes.insert(0, in_channels)
        self.weight_standardization = weight_standardization

        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append((GCNConv(in_dim, out_dim), "x, edge_index -> x"))

            if batchnorm:
                layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append(LayerNorm(out_dim))

            layers.append(nn.PReLU())
        self.model = Sequential("x, edge_index", layers)

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


class MLP_Head_BGRL(nn.Module):
    r"""MLP used for projection head in BGRL. The MLP has one hidden layer.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """

    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True),
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()


class GRACE_GCN(nn.Module):
    """
    Encoder from GRACE
    """
    def __init__(self, in_channels: int, param):
        super(GRACE_GCN, self).__init__()
        out_channels: int = param["hidden"]
        switch = {
            "identity": nn.Identity(),
            "relu": F.relu,
            "prelu": nn.PReLU(),
        }
        activation = switch[param["activation"]]
        base_model = GCNConv
        k = param["n_layers"]
        skip = param["skip"]
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
    """
    Encoder from NCN
    """
    def __init__(self, in_channels, param):
        super().__init__()

        hidden_channels = param["hidden"]
        out_channels = param["hidden"]
        num_layers = param["n_layers"]
        dropout = param["gnn_dp"]
        ln = param["layer_norm"]
        res = param["res"]
        jk = param["jk"]
        edrop = param["edrop"]
        xdropout = param["xdropout"]
        taildropout = param["taildropout"]

        self.eidrop = DropEdge(edrop)
        self.adjdrop = DropAdj(edrop)

        self.xemb = nn.Sequential(nn.Dropout(xdropout))  # nn.Identity()
        if "pure" in param["conv_fn"] or num_layers == 0:
            self.xemb.append(nn.Linear(in_channels, hidden_channels))
            self.xemb.append(
                nn.Dropout(dropout, inplace=True) if dropout > 1e-6 else nn.Identity()
            )

        self.res = res
        self.jk = jk
        if jk:
            self.register_parameter(
                "jkparams", nn.Parameter(torch.randn((num_layers,)))
            )

        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        if num_layers == 1:
            hidden_channels = out_channels

        convfn = GCNConv

        self.convs = nn.ModuleList()
        self.lins = nn.ModuleList()
        if "pure" in param["conv_fn"]:
            self.convs.append(convfn(hidden_channels, hidden_channels))
            for i in range(num_layers - 1):
                self.lins.append(nn.Identity())
                self.convs.append(convfn(hidden_channels, hidden_channels))
            self.lins.append(nn.Dropout(taildropout, True))
        else:
            self.convs.append(convfn(in_channels, hidden_channels))
            self.lins.append(
                nn.Sequential(
                    lnfn(hidden_channels, ln), nn.Dropout(dropout, True), nn.ReLU(True)
                )
            )
            for i in range(num_layers - 1):
                self.convs.append(
                    convfn(
                        hidden_channels,
                        hidden_channels if i == num_layers - 2 else out_channels,
                    )
                )
                if i < num_layers - 2:
                    self.lins.append(
                        nn.Sequential(
                            lnfn(
                                hidden_channels
                                if i == num_layers - 2
                                else out_channels,
                                ln,
                            ),
                            nn.Dropout(dropout, True),
                            nn.ReLU(True),
                        )
                    )
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
            if self.res and x1.shape[-1] == x.shape[-1]:  # residual connection
                x = x1 + x
            else:
                x = x1
            if self.jk:
                jkx.append(x)
        if self.jk:  # JumpingKnowledge Connection
            jkx = torch.stack(jkx, dim=0)
            sftmax = self.jkparams.reshape(-1, 1, 1)
            x = torch.sum(jkx * sftmax, dim=0)
        return x

    def reset_parameters(self):
        for layer in self.xemb:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

        for lin in self.lins:
            if hasattr(lin, "reset_parameters"):
                lin.reset_parameters()
