import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, LayerNorm, Sequential

def get_model(model_name: str, data, hp: dict):
    device = data.x.device
    _encoder = ENCODER_GRACE(data.num_features, hp['hidden'], nn.Identity()).to(device)
    if model_name == "baseline":
        return _encoder
    elif model_name == "grace":
        model = GRACE(_encoder, hp['hidden'], hp['proj_hidden']).to(device)
    elif model_name == "lgrace":
        model = LinkGRACE(_encoder, hp['hidden'], hp['hidden']).to(device)
    elif model_name in  ["agrace", "ândgrace", "âorgrace",   "extagrace"]:
        model = AGRACE(_encoder, hp['hidden'], hp['proj_hidden']).to(device)
    elif model_name in  "a2grace":
        model = A2GRACE(_encoder, hp['hidden'], hp['proj_hidden']).to(device)
    elif model_name == "csgcl":
        model = CSGCL(_encoder,
                        hp['hidden'],
                        hp['proj_hidden'],
                        hp['tau']).to(device)
    elif model_name in "bgrl":
        _predictor = MLP_Head_BGRL(hp['hidden'], hp['hidden']).to(device)
        model = BGRL(_encoder, _predictor).to(device)
    elif model_name in ["abgrl", "âorbgrl", "extabgrl"]:
        _predictor = MLP_Head_BGRL(hp['hidden'], hp['hidden']).to(device)
        model = ABGRL(_encoder, _predictor).to(device)
    elif model_name in "a2bgrl":
        _predictor = MLP_Head_BGRL(hp['hidden'], hp['hidden']).to(device)
        model = A2BGRL(_encoder, _predictor).to(device)
    return model

###############################################
# Code from GRACE

class ENCODER_GRACE(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation, base_model=GCNConv, k: int = 2, skip=False):
        super(ENCODER_GRACE, self).__init__()
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


class GRACE(nn.Module):
    def __init__(self, encoder: ENCODER_GRACE, hidden: int, proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder: ENCODER_GRACE = encoder
        self.tau: float = tau

        self.fc1 = nn.Linear(hidden, proj_hidden)
        self.fc2 = nn.Linear(proj_hidden, hidden)

        self.hidden = hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

class LinkGRACE(nn.Module):
    def __init__(self, encoder: ENCODER_GRACE, hidden: int, proj_hidden: int, tau: float = 0.5):
        super(LinkGRACE, self).__init__()
        self.encoder: ENCODER_GRACE = encoder
        self.tau: float = tau

        self.fc1 = nn.Linear(hidden, proj_hidden)
        self.fc2 = nn.Linear(proj_hidden, hidden)

        self.hidden = hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1_uv, neg_z1_uv, z2_uv, neg_z2_uv):
        f = lambda x: torch.exp(x / self.tau)

        pos_refl_sim = f(self.sim(z1_uv, z1_uv))
        pos_between_sim = f(self.sim(z1_uv, z2_uv))

        neg_refl_sim = f(self.sim(neg_z1_uv, neg_z1_uv))
        neg_between_sim = f(self.sim(neg_z1_uv, neg_z2_uv))

        loss = -torch.log((pos_between_sim.diag())/(neg_refl_sim.sum(1) + neg_between_sim.sum(1) - neg_refl_sim.diag()))
        return loss

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, edge, neg_edge, mean: bool = True, batch_size: int = None):
        z1_uv = self.projection(z1[edge[0]] * z1[edge[1]])
        neg_z1_uv = self.projection(z1[neg_edge[0]] * z1[neg_edge[1]])
        z2_uv = self.projection(z2[edge[0]] * z2[edge[1]])
        neg_z2_uv = self.projection(z2[neg_edge[0]] * z2[neg_edge[1]])

        if batch_size is None:
            l1 = self.semi_loss(z1_uv, neg_z1_uv, z2_uv, neg_z2_uv)
            l2 = self.semi_loss(z2_uv, neg_z2_uv, z1_uv, neg_z1_uv)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

class AGRACE(nn.Module):
    def __init__(self, encoder: ENCODER_GRACE, hidden: int, proj_hidden: int, tau: float = 0.5):
        super(AGRACE, self).__init__()
        self.encoder: ENCODER_GRACE = encoder
        self.tau: float = tau

        self.fc1 = nn.Linear(hidden, proj_hidden)
        self.fc2 = nn.Linear(proj_hidden, hidden)

        self.hidden = hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, adjacence: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        nb_neight = adjacence.to_dense().sum(1).unsqueeze(1)
        Az1 = torch.mm(adjacence, z1)
        Az1_mean = Az1 / nb_neight
        Az2 = torch.mm(adjacence, z2)
        Az2_mean = Az2 / nb_neight
        refl_sim = f(self.sim(z1, Az1_mean))
        between_sim = f(self.sim(z1, Az2_mean))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, adjacence: torch.Tensor, mean: bool = True, batch_size: int = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2, adjacence)
            l2 = self.semi_loss(h2, h1, adjacence)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret


class A2GRACE(nn.Module):
    def __init__(self, encoder: ENCODER_GRACE, hidden: int, proj_hidden: int, tau: float = 0.5):
        super(A2GRACE, self).__init__()
        self.encoder: ENCODER_GRACE = encoder
        self.tau: float = tau

        self.fc1 = nn.Linear(hidden, proj_hidden)
        self.fc2 = nn.Linear(proj_hidden, hidden)

        self.hidden = hidden

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, adjacence_1: torch.Tensor, adjacence_2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        Az1 = torch.mm(adjacence_1, z1)
        Az1_mean = Az1 / adjacence_1.sum(1).unsqueeze(1)
        Az2 = torch.mm(adjacence_2, z2)
        Az2_mean = Az2 / adjacence_2.sum(1).unsqueeze(1)
        refl_sim = f(self.sim(z1, Az1_mean))
        between_sim = f(self.sim(z1, Az2_mean))

        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor, batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor, adjacence_1: torch.Tensor, adjacence_2: torch.Tensor, mean: bool = True, batch_size: int = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2, adjacence_1, adjacence_2)
            l2 = self.semi_loss(h2, h1, adjacence_1, adjacence_2)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

###############################################
# Code from CSGCL

class CSGCL(torch.nn.Module):
    def __init__(self,
                 encoder: ENCODER_GRACE,
                 hidden: int,
                 proj_hidden: int,
                 tau: float = 0.5):

        super(CSGCL, self).__init__()
        self.encoder = encoder
        self.tau = tau
        self.fc1 = torch.nn.Linear(hidden, proj_hidden)
        self.fc2 = torch.nn.Linear(proj_hidden, hidden)
        self.hidden = hidden

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self,
                   z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def _sim(self,
             z1: torch.Tensor,
             z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def _infonce(self,
                  z1: torch.Tensor,
                  z2: torch.Tensor) -> torch.Tensor:

        temp = lambda x: torch.exp(x / self.tau)
        refl_sim = temp(self._sim(z1, z1))
        between_sim = temp(self._sim(z1, z2))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def _batched_infonce(self,
                          z1: torch.Tensor,
                          z2: torch.Tensor,
                          batch_size: int) -> torch.Tensor:
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []
        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self._sim(z1[mask], z1))
            between_sim = f(self._sim(z1[mask], z2))
            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
        return torch.cat(losses)
        
    def _team_up(self,
                 z1: torch.Tensor,
                 z2: torch.Tensor,
                 cs: torch.Tensor,
                 current_ep: int,
                 t0: int,
                 gamma_max: int) -> torch.Tensor:
        gamma = min(max(0, (current_ep - t0) / 100), gamma_max)
        temp = lambda x: torch.exp(x / self.tau)
        refl_sim = temp(self._sim(z1, z1) + gamma * cs + gamma * cs.unsqueeze(dim=1))
        between_sim = temp(self._sim(z1, z2) + gamma * cs + gamma * cs.unsqueeze(dim=1))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def _batched_team_up(self,
                         z1: torch.Tensor,
                         z2: torch.Tensor,
                         cs: torch.Tensor,
                         current_ep: int,
                         t0: int,
                         gamma_max: int,
                         batch_size: int) -> torch.Tensor:
        gamma = min(max(0, (current_ep - t0) / 100), gamma_max)
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        temp = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = temp(self._sim(z1[mask], z1) + gamma * cs + gamma * cs.unsqueeze(dim=1)[mask])
            between_sim = temp(self._sim(z1[mask], z2) + gamma * cs + gamma * cs.unsqueeze(dim=1)[mask])

            losses.append(-torch.log(between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                                     / (refl_sim.sum(1) + between_sim.sum(1)
                                        - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def infonce(self,
                z1: torch.Tensor,
                z2: torch.Tensor,
                mean: bool = True,
                batch_size: int = None) -> torch.Tensor:
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self._infonce(h1, h2)
            l2 = self._infonce(h2, h1)
        else:
            l1 = self._batched_infonce(h1, h2, batch_size)
            l2 = self._batched_infonce(h2, h1, batch_size)

        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret

    def team_up_loss(self,
                     z1: torch.Tensor,
                     z2: torch.Tensor,
                     cs: np.ndarray,
                     current_ep: int,
                     t0: int = 0,
                     gamma_max: int = 1,
                     mean: bool = True,
                     batch_size: int = None) -> torch.Tensor:

        h1 = self.projection(z1)
        h2 = self.projection(z2)
        cs = torch.from_numpy(cs).to(h1.device)
        if batch_size is None:
            l1 = self._team_up(h1, h2, cs, current_ep, t0, gamma_max)
            l2 = self._team_up(h2, h1, cs, current_ep, t0, gamma_max)
        else:
            l1 = self._batched_team_up(h1, h2, cs, current_ep, t0, gamma_max, batch_size)
            l2 = self._batched_team_up(h2, h1, cs, current_ep, t0, gamma_max, batch_size)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret


###############################################
# Code from BGRL

class ENCODER_BGRL(nn.Module):
    def __init__(self, layer_sizes, batchnorm=False, batchnorm_mm=0.99, layernorm=False, weight_standardization=False):
        super(ENCODER_BGRL, self).__init__()

        assert batchnorm != layernorm
        assert len(layer_sizes) >= 2
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization

        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append((GCNConv(in_dim, out_dim), 'x, edge_index -> x'),)

            if batchnorm:
                layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append(LayerNorm(out_dim))

            layers.append(nn.PReLU())

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

class MLP_Head_BGRL(nn.Module):
    r"""MLP used for predictor in BGRL. The MLP has one hidden layer.

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
            nn.Linear(hidden_size, output_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

class BGRL(torch.nn.Module):
    r"""BGRL architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, encoder, predictor):
        super().__init__()
        # online network
        self.online_encoder = encoder
        self.predictor = predictor

        # target network
        self.target_encoder = copy.deepcopy(encoder)

        # reinitialize weights
        self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)
            # mm c'est le poids de la target ~= param_k.data[i] = param_k.data[i] * mm + param_q.data[i] * (1 - mm)

    def train_forward(self, online_x, target_x):
        # forward online network
        online_y = self.online_encoder(online_x[0], online_x[1])

        # prediction
        online_q = self.predictor(online_y)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x[0], target_x[1]).detach()
        return online_q, target_y

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # forward online network
        online_y = self.online_encoder(x, edge_index)
        return online_y

    def loss(self, z1, z2, y1, y2):
        loss = 2 - F.cosine_similarity(z1, y2.detach(), dim=-1).mean() - F.cosine_similarity(z2, y1.detach(), dim=-1).mean()
        return loss


class ABGRL(torch.nn.Module):
    r"""BGRL architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, encoder, predictor):
        super().__init__()
        # online network
        self.online_encoder = encoder
        self.predictor = predictor

        # target network
        self.target_encoder = copy.deepcopy(encoder)

        # reinitialize weights
        self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)
            # mm c'est le poids de la target ~= param_k.data[i] = param_k.data[i] * mm + param_q.data[i] * (1 - mm)

    def train_forward(self, online_x, target_x):
        # forward online network
        online_y = self.online_encoder(online_x[0], online_x[1])

        # prediction
        online_q = self.predictor(online_y)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x[0], target_x[1]).detach()
        return online_q, target_y

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # forward online network
        online_y = self.online_encoder(x, edge_index)
        return online_y

    def loss(self, z1, z2, y1, y2, adjacence):
        nb_neight = adjacence.sum(1).unsqueeze(1)
        Ay1 = torch.mm(adjacence, y1)
        Ay1_mean = Ay1 / nb_neight
        Ay2 = torch.mm(adjacence, y2)
        Ay2_mean = Ay2 / nb_neight
        loss = 2 - F.cosine_similarity(z1, Ay1_mean.detach(), dim=-1).mean() - F.cosine_similarity(z2, Ay2_mean.detach(), dim=-1).mean()
        return loss


class A2BGRL(torch.nn.Module):
    r"""BGRL architecture for Graph representation learning.

    Args:
        encoder (torch.nn.Module): Encoder network to be duplicated and used in both online and target networks.
        predictor (torch.nn.Module): Predictor network used to predict the target projection from the online projection.

    .. note::
        `encoder` must have a `reset_parameters` method, as the weights of the target network will be initialized
        differently from the online network.
    """
    def __init__(self, encoder, predictor):
        super().__init__()
        # online network
        self.online_encoder = encoder
        self.predictor = predictor

        # target network
        self.target_encoder = copy.deepcopy(encoder)

        # reinitialize weights
        self.target_encoder.reset_parameters()
        # stop gradient
        for param in self.target_encoder.parameters():
            param.requires_grad = False

    def trainable_parameters(self):
        r"""Returns the parameters that will be updated via an optimizer."""
        return list(self.online_encoder.parameters()) + list(self.predictor.parameters())

    @torch.no_grad()
    def update_target_network(self, mm):
        r"""Performs a momentum update of the target network's weights.

        Args:
            mm (float): Momentum used in moving average update.
        """
        assert 0.0 <= mm <= 1.0, "Momentum needs to be between 0.0 and 1.0, got %.5f" % mm
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)
            # mm c'est le poids de la target ~= param_k.data[i] = param_k.data[i] * mm + param_q.data[i] * (1 - mm)

    def train_forward(self, online_x, target_x):
        # forward online network
        online_y = self.online_encoder(online_x[0], online_x[1])

        # prediction
        online_q = self.predictor(online_y)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x[0], target_x[1]).detach()
        return online_q, target_y

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # forward online network
        online_y = self.online_encoder(x, edge_index)
        return online_y

    def loss(self, z1, z2, y1, y2, adjacence_1, adjacence_2):
        Ay1 = torch.mm(adjacence_1, y1)
        Ay1_mean = Ay1 / adjacence_1.sum(1).unsqueeze(1)
        Ay2 = torch.mm(adjacence_2, y2)
        Ay2_mean = Ay2 / adjacence_2.sum(1).unsqueeze(1)
        loss = 2 - F.cosine_similarity(z1, Ay1_mean.detach(), dim=-1).mean() - F.cosine_similarity(z2, Ay2_mean.detach(), dim=-1).mean()
        return loss
