import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse.matmul import spmm_add
from utils import adjoverlap, DropAdj

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
