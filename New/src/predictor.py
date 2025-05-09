import torch
import torch.nn as nn

def get_predictor(predictor_name: str, hp):
    if predictor_name == 'inner':
        predictor = InnerProd()
    elif predictor_name == "mlp":
        predictor = MlpProdDecoder(hp['hidden'], hp['hidden'])
    return predictor

class InnerProd:
    def __call__(self, h, u, v):
        h_u = h[u]
        h_v = h[v]
        h = h_u * h_v
        out = torch.sum(h, dim=-1)
        return out

    def predict(self, h, u, v):
        return self.__call__(h, u, v)

class MlpProdDecoder(nn.Module):
    """Hadamard-product-based MLP link predictor."""
    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.embedding_size = embedding_size
        self.net = nn.Sequential(
            nn.Linear(embedding_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, h, u, v):
        h_u = h[u]
        h_v = h[v]
        return self.net(h_u * h_v)

    def predict(self, h, u, v):
        forward_res = self.forward(h, u, v)
        res = torch.cat([torch.sigmoid(forward_res)], dim=-1)
        return res

class ProbDecoder:
    def __init__(self, probs, block):
        self.probs = probs
        self.block = block

    def __call__(self, u, v):
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

    def predict(self, h, u, v):
        return self.__call__(u, v)