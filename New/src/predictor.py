import torch
import torch.nn as nn
from src.utils import DropAdj
from torch_sparse.matmul import spmm_add
from torch_sparse import SparseTensor

def get_predictor(predictor_name: str, hp):
    if predictor_name == 'inner':
        predictor = InnerProd()
    elif predictor_name == "mlp":
        predictor = MlpProdDecoder(hp['hidden'], hp['hidden'])
    elif predictor_name == "ncn":
        predictor = CNLinkPredictor(hp['hidden'], hp['hidden'], hp['predp'], hp['preedp'], hp['lnnn'], hp['use_xlin'], hp['tailact'], hp['twolayerlin'])
    elif predictor_name == "mplp":
        predictor = MPLP(hp['hidden'], hp['feat_dropout'], hp['label_dropout'], hp['prop_type'], hp['use_degree'], hp['signature_dim'], hp['minimum_degree_onehot'], hp['batchnorm_affine'])
    return predictor

class InnerProd:
    def __call__(self, h, u, v, adj=None):
        h_u = h[u]
        h_v = h[v]
        h = h_u * h_v
        out = torch.sum(h, dim=-1)
        return out

    def predict(self, h, u, v, adj=None):
        return self.__call__(h, u, v)

class MlpProdDecoder(nn.Module):
    """Hadamard-product-based MLP link predictor."""
    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.net = nn.Sequential(
            nn.Linear(embedding_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, h, u, v, adj=None):
        h_u = h[u]
        h_v = h[v]
        return self.net(h_u * h_v)

    def predict(self, h, u, v, adj=None):
        forward_res = self.forward(h, u, v)
        res = torch.cat([torch.sigmoid(forward_res)], dim=-1)
        return res

class ProbDecoder:
    def __init__(self, probs, block):
        self.probs = probs
        self.block = block

    def __call__(self, u, v, adj=None):
        u = u.cpu().numpy()
        v = v.cpu().numpy()
        b1 = self.block[u]
        b2 = self.block[v]
        res = []
        for x,y in zip(b1, b2):
            res.append(self.probs[int(x), int(y)])
        res = torch.tensor(res)
        if res.numel() <= 1:
            res = res.unsqueeze()
        return res

    def predict(self, h, u, v, adj=None):
        return self.__call__(u, v)

# NCN predictor
class CNLinkPredictor(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 dropout,
                 edrop=0.0,
                 ln=False,
                 use_xlin=False,
                 tailact=False,
                 twolayerlin=False):
        super().__init__()

        self.register_parameter("beta", nn.Parameter(torch.ones((1))))
        self.dropadj = DropAdj(edrop)
        lnfn = lambda dim, ln: nn.LayerNorm(dim) if ln else nn.Identity()

        self.xlin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True)) if use_xlin else lambda x: 0

        self.xcnlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels),
            lnfn(hidden_channels, ln), nn.Dropout(dropout, inplace=True),
            nn.ReLU(inplace=True), nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.xijlin = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), lnfn(hidden_channels, ln),
            nn.Dropout(dropout, inplace=True), nn.ReLU(inplace=True),
            nn.Linear(hidden_channels, hidden_channels) if not tailact else nn.Identity())
        self.lin = nn.Sequential(nn.Linear(hidden_channels, hidden_channels),
                                 lnfn(hidden_channels, ln),
                                 nn.Dropout(dropout, inplace=True),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(hidden_channels, hidden_channels) if twolayerlin else nn.Identity(),
                                 lnfn(hidden_channels, ln) if twolayerlin else nn.Identity(),
                                 nn.Dropout(dropout, inplace=True) if twolayerlin else nn.Identity(),
                                 nn.ReLU(inplace=True) if twolayerlin else nn.Identity(),
                                 nn.Linear(hidden_channels, 1))

    def _elem2spm(self, element, sizes: list[int]):
        # Convert adjacency matrix to a 1-d vector
        col = torch.bitwise_and(element, 0xffffffff)
        row = torch.bitwise_right_shift(element, 32)
        return SparseTensor(row=row, col=col, sparse_sizes=sizes).to_device(
            element.device).fill_value_(1.0)


    def _spm2elem(self, spm):
        # Convert 1-d vector to an adjacency matrix
        elem = torch.bitwise_left_shift(spm.storage.row(),
                                        32).add_(spm.storage.col())
        return elem

    def _spmoverlap(self, adj1, adj2):
        '''
        Compute the overlap of neighbors (rows in adj). The returned matrix is similar to the hadamard product of adj1 and adj2
        '''
        assert adj1.sizes() == adj2.sizes()
        element1 = self._spm2elem(adj1)
        element2 = self._spm2elem(adj2)

        if element2.shape[0] > element1.shape[0]:
            element1, element2 = element2, element1

        idx = torch.searchsorted(element1[:-1], element2)
        mask = (element1[idx] == element2)
        retelem = element2[mask]
        return self._elem2spm(retelem, adj1.sizes())

    def _adjoverlap(self,
                   adj1,
                   adj2,
                   u, 
                   v):
        # a wrapper for functions above.
        adj1 = adj1[u]
        adj2 = adj2[v]
        adjoverlap = self._spmoverlap(adj1, adj2)
        return adjoverlap

    def forward(self, h, u, v, adj=None):
        adj = self.dropadj(adj)
        xi = h[u]
        xj = h[v]
        h = h + self.xlin(h)
        cn = self._adjoverlap(adj, adj, u, v)
        xcns = [spmm_add(cn, h)]
        xij = self.xijlin(xi * xj)
        
        xs = torch.cat(
            [self.lin(self.xcnlin(xcn) * self.beta + xij) for xcn in xcns],
            dim=-1)
        return xs
    
    def predict(self, h, u, v, adj=None):
        return self.forward(h, u, v, adj)
