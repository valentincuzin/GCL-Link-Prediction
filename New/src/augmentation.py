import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
from networkx.generators.community import stochastic_block_model
from torch_geometric.utils import degree, to_undirected, to_networkx, dropout_adj, from_networkx
from torch_scatter import scatter
from functools import partial

from src.utils import get_commu_strength


class Aug:
    def __init__(self, data, param, type: str = 'random'):
        self.data = data
        self.device = self.data.x.device
        feature_weights = None
        drop_weights = None
        if type in ['deg', 'pr', 'evc']:
            compute_weight = {
                "deg": self.degree,
                "pr": self.page_rank,
                "evc": self.eigenvector
            }
            feature_weights, drop_weights = compute_weight[type]()
        elif type == 'scom':
            feature_weights, drop_weights = self.commu_strength()
        elif type == 'sbm':
            self.commu_repartition()
        self.types = {
            'random': self.random,
            'deg': partial(self.gca, feature_weights, drop_weights),
            'pr': partial(self.gca, feature_weights, drop_weights),
            'evc': partial(self.gca, feature_weights, drop_weights),
            'scom': partial(self.csgcl, feature_weights, drop_weights),
            'sbm': self.sbm
        }
        self.param = param
        self.get = self.types[type]
        self.type = type

    def __call__(self):
        return self.get()

    def random(self):
        edge_index_1 = dropout_adj(self.data.edge_index, p=self.param[f'drop_edge_rate_{1}'])[0]
        edge_index_2 = dropout_adj(self.data.edge_index, p=self.param[f'drop_edge_rate_{2}'])[0]
        x_1 = _drop_feature(self.data.x, self.param['drop_feature_rate_1'])
        x_2 = _drop_feature(self.data.x, self.param['drop_feature_rate_2'])
        return x_1, edge_index_1, x_2, edge_index_2

    def degree(self):
        def degree_drop_weights(edge_index):
            edge_index_ = to_undirected(edge_index)
            deg = degree(edge_index_[1])
            deg_col = deg[edge_index[1]].to(torch.float32)
            s_col = torch.log(deg_col)
            weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
            return weights
        drop_weights = degree_drop_weights(self.data.edge_index).to(self.device)
        edge_index_ = to_undirected(self.data.edge_index)
        node_deg = degree(edge_index_[1], num_nodes=self.data.x.size(0))
        feature_weights = _feature_drop_weights(self.data.x, node_c=node_deg).to(self.device)
        return feature_weights, drop_weights

    def page_rank(self):
        def compute_pr(data, damp: float = 0.85, k: int = 10):
            num_nodes = data.edge_index.max().item() + 1
            deg_out = degree(data.edge_index[0], num_nodes=num_nodes)
            x = torch.ones((num_nodes, )).to(data.edge_index.device).to(torch.float32)
            for _ in range(k):
                edge_msg = x[data.edge_index[0]] / deg_out[data.edge_index[0]]
                agg_msg = scatter(edge_msg, data.edge_index[1], reduce='sum')
                x = (1 - damp) * x + damp * agg_msg
            return x

        def pr_drop_weights(data, aggr: str = 'sink', k: int = 10):
            pv = compute_pr(data, k=k)
            pv_row = pv[data.edge_index[0]].to(torch.float32)
            pv_col = pv[data.edge_index[1]].to(torch.float32)
            s_row = torch.log(pv_row)
            s_col = torch.log(pv_col)
            if aggr == 'sink':
                s = s_col
            elif aggr == 'source':
                s = s_row
            elif aggr == 'mean':
                s = (s_col + s_row) * 0.5
            else:
                s = s_col
            weights = (s.max() - s) / (s.max() - s.mean())
            return weights
        
        drop_weights = pr_drop_weights(self.data, aggr='sink', k=200).to(self.device)
        node_pr = compute_pr(self.data)
        feature_weights = _feature_drop_weights(self.data.x, node_c=node_pr).to(self.device)
        return feature_weights, drop_weights

    def eigenvector(self):
        def eigenvector_centrality(data):
            graph = to_networkx(data)
            x = nx.eigenvector_centrality(graph, max_iter=200)
            x = [x[i] for i in range(data.num_nodes)]
            return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)
        def evc_drop_weights(data):
            evc = eigenvector_centrality(data)
            evc = evc.where(evc > 0, torch.zeros_like(evc))
            evc = evc + 1e-8
            s = evc.log()
            edge_index = data.edge_index
            s_row, s_col = s[edge_index[0]], s[edge_index[1]]
            s = s_col
            return (s.max() - s) / (s.max() - s.mean())

        drop_weights = evc_drop_weights(self.data).to(self.device)
        node_evc = eigenvector_centrality(self.data)
        feature_weights = _feature_drop_weights(self.data.x, node_c=node_evc).to(self.device)
        return feature_weights, drop_weights

    def gca(self, feature_weights, drop_weights):
        edge_index_1 = _drop_edge_weighted(self.data.edge_index, drop_weights, p=self.param[f'drop_edge_rate_{1}'], threshold=0.7)
        edge_index_2 = _drop_edge_weighted(self.data.edge_index, drop_weights, p=self.param[f'drop_edge_rate_{2}'], threshold=0.7)
        x_1 = _drop_feature_weighted(self.data.x, feature_weights, self.param['drop_feature_rate_1'])
        x_2 = _drop_feature_weighted(self.data.x, feature_weights, self.param['drop_feature_rate_2'])
        return x_1, edge_index_1, x_2, edge_index_2

    def commu_strength(self):
        def transition(communities,
                    num_nodes: int) -> np.ndarray:
            classes = np.full(num_nodes, -1)
            for i, node_list in enumerate(communities):
                classes[np.asarray(node_list)] = i
            return classes

        def get_edge_weight(edge_index: torch.Tensor,
                            com: np.ndarray,
                            com_cs: np.ndarray) -> torch.Tensor:
            edge_mod = lambda x: com_cs[x[0]] if x[0] == x[1] else -(float(com_cs[x[0]]) + float(com_cs[x[1]]))
            normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
            edge_weight = np.asarray([edge_mod([com[u.item()], com[v.item()]]) for u, v in edge_index.T])
            edge_weight = normalize(edge_weight)
            return torch.from_numpy(edge_weight).to(edge_index.device)

        communities, com_cs, node_cs = get_commu_strength(self.data)
        com = transition(communities, self.data.num_nodes)
        edge_weight = get_edge_weight(self.data.edge_index, com, com_cs)
        return node_cs, edge_weight

    def csgcl(self, feature_weights, node_cs):
        def ced(edge_index: torch.Tensor,
                edge_weight: torch.Tensor,
                p: float,
                threshold: float = 1.) -> torch.Tensor:
            edge_weight = edge_weight / edge_weight.mean() * (1. - p)
            edge_weight = edge_weight.where(edge_weight > (1. - threshold), torch.ones_like(edge_weight) * (1. - threshold))
            edge_weight = edge_weight.where(edge_weight < 1, torch.ones_like(edge_weight) * 1)
            sel_mask = torch.bernoulli(edge_weight).to(torch.bool)
            return edge_index[:, sel_mask]

        def cav(feature: torch.Tensor,
                node_cs: np.ndarray,
                p: float,
                max_threshold: float = 0.7) -> torch.Tensor:
            x = feature.abs()
            device = feature.device
            w = x.t() @ torch.tensor(node_cs).to(device)
            w[torch.nonzero(w == 0)] = w.max()  # for redundant attributes of Cora
            w = w.log()
            w = (w.max() - w) / (w.max() - w.min())
            w = w / w.mean() * p
            w = w.where(w < max_threshold, max_threshold * torch.ones(1).to(device))
            w = w.where(w > 0, torch.zeros(1).to(device))
            drop_mask = torch.bernoulli(w).to(torch.bool)
            feature = feature.clone()
            feature[:, drop_mask] = 0.
            return feature

        edge_index_1 = ced(self.data.edge_index, node_cs, p=self.param['drop_edge_rate_1'])
        edge_index_2 = ced(self.data.edge_index, node_cs, p=self.param['drop_edge_rate_2'])
        x_1 = cav(self.data.x, feature_weights, self.param["drop_feature_rate_1"])
        x_2 = cav(self.data.x, feature_weights, self.param['drop_feature_rate_2'])
        return x_1, edge_index_1, x_2, edge_index_2

    def commu_repartition(self):
        G = to_networkx(self.data)
        communities = nx.community.louvain_communities(G, resolution=0.5)
        probs = np.zeros((len(communities), len(communities)))
        sizes = []
        for idx, c in enumerate(communities):
            sizes.append(len(c))
            for n in c:
                G.nodes[n]["com"] = idx # get com label
        for u, v in zip(self.data.edge_index[0], self.data.edge_index[1]): # count number of edge per com
            u = float(u)
            v = float(v)
            probs[G.nodes[u]["com"], G.nodes[v]["com"]] += 1
        for x in range(len(probs)): # make the probs
            for y in range(len(probs)):
                if x == y:
                    probs[x,x] = probs[x,x]/(sizes[x]*(sizes[x]-1))/2 if sizes[x] > 1 else probs[x,x]
                else:
                    probs[x,y] /= ((sizes[x]+sizes[y])*(sizes[x]+sizes[y]-1))/2 #complete graph formula
        probs /= 2 # undirected graph
        self.data.community = communities
        self.data.probs = probs
        print("probs, ", probs)
        self.data.sizes = sizes
        print("sizes, ", sizes)

    def sbm(self):
        def gen_sbm(sizes, probs):
            G = stochastic_block_model(sizes, probs)
            G.remove_edges_from(nx.selfloop_edges(G)) # remove self loops
            data = from_networkx(G)
            data.num_nodes = sum(sizes)
            data.sizes = sizes
            data.probs = probs
            data.num_features = data.num_nodes
            data.x = self.data.x # F.one_hot(torch.arange(0, data.num_nodes)).float()
            data = data.to(self.device)
            return data

        sizes, probs = self.data.sizes, self.data.probs
        data_1 = gen_sbm(sizes, probs)
        data_1.x = _drop_feature(data_1.x, self.param['drop_feature_rate_1'])
        data_2 = gen_sbm(sizes, probs)
        data_2.x = _drop_feature(data_2.x, self.param['drop_feature_rate_2'])
        return data_1.x, data_1.edge_index, data_2.x, data_2.edge_index


def _drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x

def _feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())
    return s

def _drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
    return edge_index[:, sel_mask]

def _drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_mask = torch.bernoulli(w).to(torch.bool)
    x = x.clone()
    x[:, drop_mask] = 0.
    return x