import copy
import random

from networkx import subgraph
import torch
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import degree, to_undirected, to_networkx, dropout_adj, from_networkx, negative_sampling, k_hop_subgraph, to_dense_adj
from torch_scatter import scatter
from functools import partial

from src.utils import get_commu_strength, gen_sbm, commu_repartition, gen_sgf, gen_ba, gen_deg


class Aug:
    def __init__(self, data, split_edge, param, type: str = 'random'):
        self.data = copy.deepcopy(data)
        self.split_edge = split_edge
        self.device = self.data.x.device
        self.param = param
        self.type = type
        self.data = self.data.to(self.device)
        self.train_mode = True
        if '+' in type:
            type1, type2 = type.split('+')
            self.data1, aug_fct1 = self.precompute(self.data, type1)
            self.data2, aug_fct2 = self.precompute(self.data, type2)
            self.get = partial(self.mix, aug_fct1, aug_fct2)
        else:
            self.data, aug_fct = self.precompute(self.data, type)
            self.get = aug_fct

    def precompute(self, data, type: str):
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
        elif 'sbm' in type:
            cd_algo = None
            if self.param['commu_detect']:
                cd_algo = self.param['commu_detect']
            print(cd_algo, 'detection...')
            data = commu_repartition(data, cd_algo).to(self.device)
        elif '_d' in type:
            type, delta = type.split('_d')
            print(type, " with variance: ", delta)
            self.perturb_commu(delta)
        elif type in ['rjc', 'raa', 'rra']:
            data = self.pred(data, type)
            type = 'reconstruct'
        # elif type == 'ctri':
        #     data.edge_index = _close_triangle(data.edge_index)
        types = {
            'random': self.random,
            'reconstruct': self.reconstruct,
            # 'raa': self.raa,
            # 'rra': self.rra,
            'deg': partial(self.gca, feature_weights, drop_weights),
            'pr': partial(self.gca, feature_weights, drop_weights),
            'evc': partial(self.gca, feature_weights, drop_weights),
            'scom': partial(self.csgcl, feature_weights, drop_weights),
            'sbm': self.sbm,
            'sbm2': self.sbm_2,
            'sgf': self.sgf,
            'ba': self.ba,
            'deg_seq': self.config_model
        }
        return data, types[type]

    # def train(self):
    #     if not self.train_mode:
    #         self.data.edge_index = to_undirected(self.split_edge["train"]["edge"].t()).to(self.device)
    #         self.train_mode = True

    # def eval(self):
    #     if self.train_mode:
    #         self.data.edge_index = to_undirected(self.split_edge["valid"]["edge"].t()).to(self.device)
    #         self.train_mode = False


    def __call__(self):
        return self.get() if self.train_mode else self.get_val()
    
    def mix(self, aug_fct1, aug_fct2):
        self.tmp = copy.deepcopy(self.data)
        self.data = self.data1
        x_1, edge_index_1, _, _ = aug_fct1()
        self.data = self.data2
        _, _, x_2, edge_index_2 = aug_fct2()
        self.data = self.tmp
        return x_1, edge_index_1, x_2, edge_index_2

    def random(self):
        edge_attr = self.data.edge_attr if 'edge_attr' in self.data else None
        edge_index_1 = dropout_adj(self.data.edge_index, edge_attr, p=self.param[f'drop_edge_rate_{1}'], force_undirected=True)[0].to(self.device)
        edge_index_2 = dropout_adj(self.data.edge_index, edge_attr, p=self.param[f'drop_edge_rate_{2}'], force_undirected=True)[0].to(self.device)
        x_1 = _drop_feature(self.data.x, self.param['drop_feature_rate_1']).to(self.device)
        x_2 = _drop_feature(self.data.x, self.param['drop_feature_rate_2']).to(self.device)
        return x_1, edge_index_1, x_2, edge_index_2

    def degree(self):
        def degree_drop_weights(edge_index):
            edge_index_ = to_undirected(edge_index)
            deg = degree(edge_index_[1])
            print('deg', deg)
            deg_col = deg[edge_index[1]].to(torch.float32)
            print('deg_col', deg_col)
            s_col = torch.log(deg_col)
            print('s_col', s_col)
            weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
            print('weights', weights)
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
            graph = to_networkx(data, to_undirected=True)
            x = nx.eigenvector_centrality(graph, max_iter=1000)
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

    def pred(self, data_param, type):
        data = copy.deepcopy(data_param)
        G = to_networkx(data, to_undirected=True)
        switch = {
            'rjc': nx.jaccard_coefficient,
            'raa': nx.adamic_adar_index,
            'rra': nx.resource_allocation_index
        }
        preds = switch[type](G)
        
        nb_add = 0
        values = []
        values_norm = []
        probs = []
        min_p = float('inf')
        max_p = float('-inf')
        mean_p = []
        for u, v, p in preds:
            mean_p.append(p)
            values.append((u, v, p))
            if p < min_p:
                min_p = p
            if p > max_p:
                max_p = p
        mean_p = np.mean(mean_p)
        for u, v, p in values:
            if max_p != min_p:
                p = float((p-min_p)/(max_p-min_p))
            # p = float((max_p-p)/(max_p-mean_p))
            probs.append(p)
            values_norm.append((u, v, p))

        print('mean pred: ', mean_p)

        for u, v in G.edges():
            G[u][v]['weight'] = 1.0 # not sure to put 1 if their is a link

        to_reconstruct = int(data.edge_index.shape[1]*self.param['reconstruction_rate'])
        values_norm = sorted(values_norm, key=lambda x: x[2], reverse=True)
        for u, v, p in values_norm:
            if p >= mean_p:
                G.add_edge(u, v, weight=p)
                G.add_edge(v, u, weight=p)
                nb_add += 1
                if nb_add >= to_reconstruct:
                    print('min de prob added: ', p)
                    break
        print('add', nb_add, 'edges')

        tmp = from_networkx(G).to(self.device)
        data.edge_index = tmp.edge_index
        data.weight = tmp.weight
        print(tmp.weight)
        return data

    def gca(self, feature_weights, drop_weights):
        edge_index_1 = _drop_edge_weighted(self.data.edge_index, drop_weights, p=self.param[f'drop_edge_rate_{1}'], threshold=0.7).to(self.device)
        edge_index_2 = _drop_edge_weighted(self.data.edge_index, drop_weights, p=self.param[f'drop_edge_rate_{2}'], threshold=0.7).to(self.device)
        x_1 = _drop_feature_weighted(self.data.x, feature_weights, self.param['drop_feature_rate_1']).to(self.device)
        x_2 = _drop_feature_weighted(self.data.x, feature_weights, self.param['drop_feature_rate_2']).to(self.device)
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
            return edge_index[:, sel_mask].to(self.device)

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
            return feature.to(self.device)

        edge_index_1 = ced(self.data.edge_index, node_cs, p=self.param['drop_edge_rate_1'])
        edge_index_2 = ced(self.data.edge_index, node_cs, p=self.param['drop_edge_rate_2'])
        x_1 = cav(self.data.x, feature_weights, self.param["drop_feature_rate_1"])
        x_2 = cav(self.data.x, feature_weights, self.param['drop_feature_rate_2'])
        return x_1, edge_index_1, x_2, edge_index_2


    def perturb_commu(self, delta):
        if not (hasattr(self.data, "probs") and hasattr(self.data, "sizes")):
            print("no community...")
            return
        delta = float(delta)
        self.data.sizes = _permute_nodes(self.data.sizes, delta)
        self.data.probs = _perturb_matrix(self.data.probs, delta)

    def sbm(self):
        sizes, probs = self.data.sizes, self.data.probs
        data_1 = gen_sbm(sizes, probs).to(self.device)
        data_1.x = self.data.x
        data_1.x = _drop_feature(data_1.x, self.param['drop_feature_rate_1'])
        data_2 = self.data
        data_2.x = _drop_feature(data_2.x, self.param['drop_feature_rate_2'])
        return data_1.x, data_1.edge_index, data_2.x, data_2.edge_index

    def sbm_2(self):
        sizes, probs = self.data.sizes, self.data.probs
        data_1 = gen_sbm(sizes, probs).to(self.device)
        data_1.x = self.data.x
        data_1.x = _drop_feature(data_1.x, self.param['drop_feature_rate_1'])
        data_2 = gen_sbm(sizes, probs).to(self.device)
        data_2.x = self.data.x
        data_2.x = _drop_feature(data_2.x, self.param['drop_feature_rate_2'])
        return data_1.x, data_1.edge_index, data_2.x, data_2.edge_index
    
    def sgf(self):
        data_1 = gen_sgf(self.data, 0.5).to(self.device)
        data_1.x = self.data.x
        data_1.x = _drop_feature(data_1.x, self.param['drop_feature_rate_1'])
        data_2 = gen_sgf(self.data, 0.5).to(self.device)
        data_2.x = self.data.x
        data_2.x = _drop_feature(data_2.x, self.param['drop_feature_rate_2'])
        return data_1.x, data_1.edge_index, data_2.x, data_2.edge_index
    
    def ba(self):
        data_1 = gen_ba(self.data).to(self.device)
        data_1.x = self.data.x
        data_1.x = _drop_feature(data_1.x, self.param['drop_feature_rate_1'])
        data_2 = gen_ba(self.data).to(self.device)
        data_2.x = self.data.x
        data_2.x = _drop_feature(data_2.x, self.param['drop_feature_rate_2'])
        return data_1.x, data_1.edge_index, data_2.x, data_2.edge_index

    def config_model(self):
        data_1 = gen_deg(self.data).to(self.device)
        data_1.x = self.data.x
        data_1.x = _drop_feature(data_1.x, self.param['drop_feature_rate_1'])
        data_2 = gen_deg(self.data).to(self.device)
        data_2.x = self.data.x
        data_2.x = _drop_feature(data_2.x, self.param['drop_feature_rate_2'])
        return data_1.x, data_1.edge_index, data_2.x, data_2.edge_index


    def reconstruct(self):
        edge_index_1 = _drop_edge_weighted(self.data.edge_index, self.data.weight, self.param['drop_edge_rate_1']).to(self.device)
        edge_index_2 = _drop_edge_weighted(self.data.edge_index, self.data.weight, self.param['drop_edge_rate_2']).to(self.device)
        x_1 = _drop_feature(self.data.x, self.param['drop_feature_rate_1']).to(self.device)
        x_2 = _drop_feature(self.data.x, self.param['drop_feature_rate_2']).to(self.device)
        return x_1, edge_index_1, x_2, edge_index_2


    """ predictor augmentation...
    def rjc(self):
        x_1, edge_index_1, x_2, edge_index_2 = self.random()
        neg_edge = negative_sampling(self.data.edge_index).T.tolist()
        
        data_1 = Data(x=x_1, edge_index=edge_index_1)
        G1 = to_networkx(data_1, to_undirected=True)
        G1 = _reconstruction(G1, nx.jaccard_coefficient, 0.9, neg_edge)
        data_2 = Data(x=x_2, edge_index=edge_index_2)
        G2 = to_networkx(data_2, to_undirected=True)
        G2 = _reconstruction(G2, nx.jaccard_coefficient, 0.9, neg_edge)

        data_1 = from_networkx(G1).to(self.device)
        data_2 = from_networkx(G2).to(self.device)
        return x_1, data_1.edge_index, x_2, data_2.edge_index

    def rjc2(self):
        # x_1, edge_index_1, x_2, edge_index_2 = self.random()
        # neg_edge = negative_sampling(self.data.edge_index).T.tolist()
        
        # data_1 = Data(x=x_1, edge_index=edge_index_1)
        # G1 = to_networkx(data_1, to_undirected=True)
        # G1 = _reconstruction(G1, nx.jaccard_coefficient, 0.9, neg_edge)
        # data_2 = Data(x=x_2, edge_index=edge_index_2)
        # G2 = to_networkx(data_2, to_undirected=True)
        # G2 = _reconstruction(G2, nx.jaccard_coefficient, 0.9, neg_edge)

        # data_1 = from_networkx(G1).to(self.device)
        # data_2 = from_networkx(G2).to(self.device)
        edge_index_1 = _drop_edge_weighted(self.data.edge_index, self.data.weight, self.param['drop_edge_rate_1']).to(self.device)
        edge_index_2 = _drop_edge_weighted(self.data.edge_index, self.data.weight, self.param['drop_edge_rate_2']).to(self.device)
        x_1 = _drop_feature(self.data.x, self.param['drop_feature_rate_1']).to(self.device)
        x_2 = _drop_feature(self.data.x, self.param['drop_feature_rate_2']).to(self.device)
        return x_1, edge_index_1, x_2, edge_index_2

    def raa(self):
        x_1, edge_index_1, x_2, edge_index_2 = self.random()
        neg_edge = negative_sampling(self.data.edge_index).T.tolist()

        data_1 = Data(x=x_1, edge_index=edge_index_1)
        G1 = to_networkx(data_1, to_undirected=True)
        G1 = _reconstruction(G1, nx.adamic_adar_index, 0.9, neg_edge)
        data_2 = Data(x=x_2, edge_index=edge_index_2)
        G2 = to_networkx(data_2, to_undirected=True)
        G2 = _reconstruction(G2, nx.adamic_adar_index, 0.9, neg_edge)

        data_1 = from_networkx(G1).to(self.device)
        data_2 = from_networkx(G2).to(self.device)
        return x_1, data_1.edge_index, x_2, data_2.edge_index

    def rra(self):
        x_1, edge_index_1, x_2, edge_index_2 = self.random()
        data_1 = Data(x=x_1, edge_index=edge_index_1)
        G1 = to_networkx(data_1, to_undirected=True)
        neg_edge = negative_sampling(edge_index_1).T.tolist()
        G1 = _reconstruction(G1, nx.resource_allocation_index, 0.9, neg_edge)    
        data_2 = Data(x=x_2, edge_index=edge_index_2)
        G2 = to_networkx(data_2, to_undirected=True)
        neg_edge = negative_sampling(edge_index_2).T.tolist()
        G2 = _reconstruction(G2, nx.resource_allocation_index, 0.9, neg_edge)

        data_1 = from_networkx(G1).to(self.device)
        data_2 = from_networkx(G2).to(self.device)
        return x_1, data_1.edge_index, x_2, data_2.edge_index
    
    def close_triangle(self):
        def exist_neight(u, v, edge_index):
            mask = edge_index[0] == u
            n_u = edge_index[1][mask]
            mask = edge_index[0] == v
            n_v = edge_index[1][mask]
            mask = torch.eq(n_v[:, None], n_u[None, :])
            if not mask.any():
                # print('mask', mask)
                return False
            cn_uv = n_v[torch.nonzero(mask[:, 0])[:, 0]]
            if cn_uv.shape[0] != 0:
                # print('cn_uv', cn_uv)
                return True
            else:
                return False

        x_1, edge_index_1, x_2, edge_index_2 = self.random()
        nb_sample = int(0.3*self.data.edge_index.size(1))
        edge_index_1_close = copy.copy(edge_index_1)
        edge_index_2_close = copy.copy(edge_index_2)
        neg_edges = torch.cat((self.split_edge['test']['edge'], self.split_edge['test']['edge_neg'])) # to_undirected(negative_sampling(self.data.edge_index, num_neg_samples=nb_sample)).T
        # neg_edges_1 = to_undirected(negative_sampling(edge_index_1, num_neg_samples=nb_sample)).T
        nb_l1_add = 0
        nb_l2_add = 0
        # print('neg_edges_1', neg_edges_1.shape)
        for neg_edge in neg_edges:
            if exist_neight(neg_edge[0], neg_edge[1], edge_index_1):
                nb_l1_add += 1
                neg_edge_to_add = to_undirected(neg_edge.unsqueeze(dim=-1))
                neg_edge_to_add = neg_edge_to_add.to(self.device)
                edge_index_1_close = torch.cat((edge_index_1_close, neg_edge_to_add), dim=1)
            if exist_neight(neg_edge[0], neg_edge[1], edge_index_2):
                nb_l2_add += 1
                neg_edge_to_add = to_undirected(neg_edge.unsqueeze(dim=-1))
                neg_edge_to_add = neg_edge_to_add.to(self.device)
                edge_index_2_close = torch.cat((edge_index_2_close, neg_edge_to_add), dim=1)
        # neg_edges_2 = to_undirected(negative_sampling(edge_index_2, num_neg_samples=nb_sample)).T
        # for neg_edge in neg_edges_2:
        #     if exist_neight(neg_edge[0], neg_edge[1], edge_index_2):
        #         nb_l2_add += 1
        #         neg_edge = to_undirected(neg_edge.unsqueeze(dim=-1))
        #         edge_index_2_close = torch.cat((edge_index_2_close, neg_edge), dim=1)
        if nb_l1_add+nb_l2_add > 0:
            print([nb_l1_add, nb_l2_add])
        return x_1, edge_index_1_close, x_2, edge_index_2_close"""


    def random_prob(self):
        weight = self.data.weight if 'weight' in self.data else None
        edge_index_1 = _droupout_adj_prob(self.data.edge_index, weight, p=self.param[f'drop_edge_rate_{1}'], force_undirected=True)[0]
        edge_index_2 = _droupout_adj_prob(self.data.edge_index, weight, p=self.param[f'drop_edge_rate_{2}'], force_undirected=True)[0]
        x_1 = _drop_feature(self.data.x, self.param['drop_feature_rate_1'])
        x_2 = _drop_feature(self.data.x, self.param['drop_feature_rate_2'])
        return x_1, edge_index_1, x_2, edge_index_2

def _droupout_adj_prob(edge_index, edge_weight, p: float = 0.5, 
                       force_undirected: bool = False, num_nodes: int = None,training: bool = True):
    def filter_adj(row, col, edge_weight, mask):
        return row[mask], col[mask], None if edge_weight is None else edge_weight[mask]
    if p < 0. or p > 1.:
        raise ValueError(f'Dropout probability has to be between 0 and 1 '
                            f'(got {p}')
    if not training or p == 0.0:
        return edge_index, edge_weight
    row, col = edge_index
    mask = torch.rand(row.size(0), device=edge_index.device) >= p*edge_weight
    if force_undirected:
        mask[row > col] = False
    row, col, edge_weight = filter_adj(row, col, edge_weight, mask)
    if force_undirected:
        edge_index = torch.stack(
            [torch.cat([row, col], dim=0),
                torch.cat([col, row], dim=0)], dim=0)
        if edge_weight is not None:
            edge_weight = torch.cat([edge_weight, edge_weight], dim=0)
    else:
        edge_index = torch.stack([row, col], dim=0)
    return edge_index, edge_weight

"""
def _reconstruction(G, algo: callable, threshold, edge_list = None):
    new_G = nx.Graph()
    new_G.add_node(G)
    preds = algo(G, [tuple(pair) for pair in edge_list])
    retains = []
    for u, v, p in preds:
        if p >= threshold:
            retains.append((u, v))
    new_G.add_edges_from(retains)
    return new_G"""

""" ### optimisation
def _close_triangle(edge_index):
    edge_index = to_undirected(edge_index)
    link_to_add = [edge_index]
    for u in edge_index[0].unique():
        subset, sub_edge_index, _, _ = k_hop_subgraph(int(u), 2, edge_index)
        n = len(subset)
        num_samples = int((n*(n-1)/2)-len(sub_edge_index[0]))
        if num_samples <= 0:
            continue
        neg_sampl = to_undirected(negative_sampling(sub_edge_index, num_neg_samples=num_samples))
        link_to_add.append(neg_sampl)
    return torch.cat(link_to_add, dim=1)
"""

def _permute_nodes(sizes, perm_rate):
        sizes = copy.copy(sizes)
        total_nodes = sum(sizes)
        num_nodes_to_permute = int(total_nodes * perm_rate) 
        for _ in range(num_nodes_to_permute):
            orig_community = random.randint(0, len(sizes) - 1)
            dest_community = random.randint(0, len(sizes) - 1)
            while dest_community == orig_community:
                dest_community = random.randint(0, len(sizes) - 1)
            sizes[orig_community] -= 1
            sizes[dest_community] += 1
        return sizes

def _perturb_matrix(matrix, delta):
    perturbed_matrix = matrix.copy()
    difference = 0.0
    while difference < delta:
        i = np.random.randint(0, matrix.shape[0])
        j = np.random.randint(i, matrix.shape[1])
        epsilon = np.random.uniform(-0.1, 0.1)
        perturbed_matrix[i, j] = np.clip(perturbed_matrix[i, j] + epsilon, 0, 1)
        difference += np.abs(matrix[i, j] - perturbed_matrix[i, j])
    upper_triangular = np.triu(perturbed_matrix)
    perturbed_matrix = upper_triangular + upper_triangular.T - np.diag(np.diag(upper_triangular))
    return perturbed_matrix

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
    # print(edge_weights)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)
    return edge_index[:, sel_mask]

def _drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_mask = torch.bernoulli(w).to(torch.bool)
    x = x.clone()
    x[:, drop_mask] = 0.
    return x
