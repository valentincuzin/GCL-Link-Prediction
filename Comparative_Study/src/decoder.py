import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse.matmul import spmm_add
from src.utils import adjoverlap, DropAdj, NodeLabel

class MlpProdDecoder(torch.nn.Module):
    """Hadamard-product-based MLP link predictor."""

    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.net = nn.Sequential(
            nn.Linear(embedding_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def multidomainforward(
        self, x, adj, tar_ei, filled1: bool = False, cndropprobs: list[float] = []
    ):
        x1 = x[tar_ei[0]]
        x2 = x[tar_ei[1]]
        return self.net(x1 * x2)

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        mdforward = self.multidomainforward(x, adj, tar_ei)
        return torch.cat([torch.sigmoid(mdforward)], dim=-1)

class CNLinkPredictor(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        edrop=0.0,
        ln=False,
        cndeg=-1,
        use_xlin=False,
        tailact=False,
        twolayerlin=False,
        beta=1.0,
    ):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(beta * torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = (
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.Dropout(dropout, inplace=True),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, hidden_channels),
                lnfn(hidden_channels, ln),
                nn.Dropout(dropout, inplace=True),
                nn.ReLU(inplace=True),
            )
            if use_xlin
            else lambda x: 0
        )

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels)
            if not tailact
            else nn.Identity(),
        )
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels)
            if not tailact
            else nn.Identity(),
        )
        self.lin = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels)
            if twolayerlin
            else nn.Identity(),
            lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
            nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
            nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.cndeg = cndeg

    def multidomainforward(
        self, x, adj, tar_ei, filled1: bool = False, cndropprobs: list[float] = []
    ):
        adj = self.dropadj(adj)
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        x = x + self.xlin(x)
        cn = adjoverlap(adj, adj, tar_ei, filled1, cnsampledeg=self.cndeg)
        xcns = [spmm_add(cn, x)]
        xij = self.xijlin(xi * xj)

        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns], dim=-1
        )
        return xs

    def forward(self, x, adj, tar_ei, filled1: bool = False):
        return self.multidomainforward(x, adj, tar_ei, filled1, [])


class MLP(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        hidden_dim,
        output_dim,
        dropout_ratio,
        norm_type="none",
        tailnormactdrop=False,
        affine=True,
    ):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        self.norm_type = norm_type
        self.tailnormactdrop = tailnormactdrop
        self.affine = affine # the affine in batchnorm
        self.layers = []
        if num_layers == 1:
            self.layers.append(nn.Linear(input_dim, output_dim))
            if tailnormactdrop:
                self.__build_normactdrop(self.layers, output_dim, dropout_ratio)
        else:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.__build_normactdrop(self.layers, hidden_dim, dropout_ratio)
            for i in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.__build_normactdrop(self.layers, hidden_dim, dropout_ratio)
            self.layers.append(nn.Linear(hidden_dim, output_dim))
            if tailnormactdrop:
                self.__build_normactdrop(self.layers, hidden_dim, dropout_ratio)
        self.layers = nn.Sequential(*self.layers)

    def __build_normactdrop(self, layers, dim, dropout):
        if self.norm_type == "batch":
            layers.append(nn.BatchNorm1d(dim, affine=self.affine))
        elif self.norm_type == "layer":
            layers.append(nn.LayerNorm(dim))
        layers.append(nn.Dropout(dropout, inplace=True))
        layers.append(nn.ReLU(inplace=True))

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()

    def forward(self, feats, adj_t=None):
        return self.layers(feats)

class MPLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers,
                 feat_dropout, label_dropout, num_hops=2, prop_type='combine', signature_sampling='torchhd', use_degree='none',
                 signature_dim=1024, minimum_degree_onehot=-1, batchnorm_affine=True,
                 feature_combine="hadamard",adj2=False):
        super(MPLP, self).__init__()

        self.in_channels = in_channels
        self.feat_dropout = feat_dropout
        self.label_dropout = label_dropout
        self.num_hops = num_hops
        self.prop_type = prop_type # "MPLP+exactly","MPLP+prop_only","MPLP+combine"
        self.signature_sampling=signature_sampling
        self.use_degree = use_degree
        self.feature_combine = feature_combine
        self.adj2 = adj2
        if self.use_degree == 'mlp':
            self.node_weight_encode = MLP(2, in_channels + 1, 32, 1, feat_dropout, norm_type="batch", affine=batchnorm_affine)
        if self.prop_type in ['prop_only', 'precompute']:
            struct_dim = 8
        elif self.prop_type == 'exact':
            struct_dim = 5
        elif self.prop_type == 'combine':
            struct_dim = 15
        self.nodelabel = NodeLabel(signature_dim, signature_sampling=self.signature_sampling, prop_type=self.prop_type,
                               minimum_degree_onehot= minimum_degree_onehot)
        self.struct_encode = MLP(1, struct_dim, struct_dim, struct_dim, self.label_dropout, "batch", tailnormactdrop=True, affine=batchnorm_affine)

        dense_dim = struct_dim + in_channels
        if in_channels > 0:
            if feature_combine == "hadamard":
                feat_encode_input_dim = in_channels
            elif feature_combine == "plus_minus":
                feat_encode_input_dim = in_channels * 2
            self.feat_encode = MLP(2, feat_encode_input_dim, in_channels, in_channels, self.feat_dropout, "batch", tailnormactdrop=True, affine=batchnorm_affine)
        self.classifier = nn.Linear(dense_dim, 1)


    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm1d):
                m.reset_parameters()
    def multidomainforward(
        self, x, adj, edges, filled1: bool = False, cndropprobs: list[float] = []
    ):
        """
        Args:
            x: [N, in_channels] node embedding after GNN
            adj: [N, N] adjacency matrix
            edges: [2, E] target edges
            fast_inference: bool. If True, only caching the message-passing without calculating the structural features
        """
        if self.use_degree == 'none':
            node_weight = None
        elif self.use_degree == 'mlp': # 'mlp' for now
            xs = []
            if self.in_channels > 0:
                xs.append(x)
            degree = adj.sum(dim=1).view(-1,1).to(adj.device())
            xs.append(degree)
            node_weight_feat = torch.cat(xs, dim=1)
            node_weight = self.node_weight_encode(node_weight_feat).squeeze(-1) + 1 # like residual, can be learned as 0 if needed
        else:
            # AA or RA
            degree = adj.sum(dim=1).view(-1,1).to(adj.device()).squeeze(-1) + 1 # degree at least 1. then log(degree) > 0.
            if self.use_degree == 'AA':
                node_weight = torch.sqrt(torch.reciprocal(torch.log(degree)))
            elif self.use_degree == 'RA':
                node_weight = torch.sqrt(torch.reciprocal(degree))
            node_weight = torch.nan_to_num(node_weight, nan=0.0, posinf=0.0, neginf=0.0)

        
        propped = self.nodelabel(edges, adj, node_weight=node_weight)
        propped_stack = torch.stack([*propped], dim=1)
        out = self.struct_encode(propped_stack)

        if self.in_channels > 0:
            x_i = x[edges[0]]
            x_j = x[edges[1]]
            if self.feature_combine == "hadamard":
                x_ij = x_i * x_j
            elif self.feature_combine == "plus_minus":
                x_ij = torch.cat([x_i+x_j, torch.abs(x_i-x_j)], dim=1)
            x_ij = self.feat_encode(x_ij)
            x = torch.cat([x_ij, out], dim=1)
        else:
            x = out
        logit = self.classifier(x)
        return logit
    
    def forward(self, x, adj, edges, filled1: bool = False):
        return self.multidomainforward(self, x, adj, edges, filled1, [])
