class inner_prod:

    def __init__(self):
        pass

    def __call__(self, h, u, v):
        x_i = x[tar_ei[0]]
        x_j = x[tar_ei[1]]
        x = x_i * x_j
        out = torch.sum(x, dim=-1)
        return out
