import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from encoder import ENCODER_GRACE, ENCODER_CSGCL

class GRACE(nn.Module):
    def __init__(self, encoder: ENCODER_GRACE, num_hidden: int, num_proj_hidden: int, tau: float = 0.5):
        super(GRACE, self).__init__()
        self.encoder: ENCODER_GRACE = encoder
        self.tau: float = tau

        self.fc1 = nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = nn.Linear(num_proj_hidden, num_hidden)

        self.num_hidden = num_hidden

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
        online_y = self.online_encoder(online_x.x, online_x.edge_index)

        # prediction
        online_q = self.predictor(online_y)

        # forward target network
        with torch.no_grad():
            target_y = self.target_encoder(target_x.x, target_x.edge_index).detach()
        return online_q, target_y

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        # forward online network
        online_y = self.online_encoder(x, edge_index)
        return online_y


class CSGCL(torch.nn.Module):
    def __init__(self,
                 encoder: ENCODER_CSGCL,
                 num_hidden: int,
                 num_proj_hidden: int,
                 tau: float = 0.5):

        super(CSGCL, self).__init__()
        self.encoder = encoder
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.num_hidden = num_hidden

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

