import copy

import torch
import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import (
    degree,
    to_undirected,
    to_networkx,
    dropout_adj,
    from_networkx,
)
from torch_scatter import scatter
from functools import partial

from tqdm import tqdm

from src.utils import (
    get_commu_strength,
    gen_sbm,
    gen_sbm_fast,
    commu_distrib,
    commu_distrib_old,
    to_graph_tool,
    to_graph_tool_bug
)
from graph_tool.inference import BlockState

sbm_bank = []
ct_epoch = 0

class Aug:
    def __init__(
        self,
        data: Data,
        split_edge: list,
        param: dict,
        type: str = "random",
        run: int = 0,
    ) -> None:
        global ct_epoch
        ct_epoch = 0
        self.data = copy.deepcopy(data)
        self.split_edge = split_edge
        self.device = data.x.device
        self.param = param
        self.run = run
        if run == 0:
            self.bank = True
        self.type = type
        self.data = self.data.to(self.device)
        if "+" in type:
            type1, type2 = type.split("+")
            self.data1, aug_fct1 = self.precompute(self.data, type1)
            self.data2, aug_fct2 = self.precompute(self.data, type2)
            self.get = partial(self.mix, aug_fct1, aug_fct2)
        else:
            self.data, aug_fct = self.precompute(self.data, type)
            self.get = aug_fct

    def precompute(self, data: Data, type: str):
        feature_weights = None
        drop_weights = None
        if type in ["deg", "pr", "evc"]:
            compute_weight = {
                "deg": self.degree,
                "pr": self.page_rank,
                "evc": self.eigenvector,
            }
            feature_weights, drop_weights = compute_weight[type]()
        elif type == "scom":
            feature_weights, drop_weights = self.commu_strength()
        elif "sbm" in type:
            cd_algo = None
            global sbm_bank
            if len(sbm_bank) != 0 and "fast" not in type and self.run == 0:
                type = "sbm_bank"
            elif self.param["commu_detect"]:
                cd_algo = self.param["commu_detect"]
                print(cd_algo, "detection...")
                if type == "sbm_old":
                    data = commu_distrib_old(data, cd_algo).to(self.device)
                else:
                    data = commu_distrib(data, cd_algo).to(self.device)
                if "fix" in type:
                    data.node_list = [j for sub in data.communities for j in sub]
            if "fast_bug" in type:
                gtG, block_map = to_graph_tool_bug(data)
                data.state = BlockState(gtG, block_map, deg_corr=False)
            elif "fast_test" in type:
                data.sizes = []
                nb_grp = 0
                comm_size = int(data.num_nodes/len(data.communities))
                for i in range(0, data.num_nodes, comm_size):
                    data.sizes.append(comm_size)
                    nb_grp +=1
                print("graph density", data.num_nodes/len(data.edge_index[0]))
                print(data.sizes)
                gtG, block_map = to_graph_tool_bug(data)
                data.state = BlockState(gtG, block_map, deg_corr=False)
            elif "fast" in type:
                gtG, block_map = to_graph_tool(data)
                data.state = BlockState(gtG, block_map, deg_corr=False)
        elif type in ["rjc", "raa", "rra"]:
            data = self.pred(data, type)
            type = "reconstruct"
        types = {
            "random": self.random,
            "random2": self.random_2,
            "reconstruct": self.reconstruct,
            "deg": partial(self.gca, feature_weights, drop_weights),
            "pr": partial(self.gca, feature_weights, drop_weights),
            "evc": partial(self.gca, feature_weights, drop_weights),
            "scom": partial(self.csgcl, feature_weights, drop_weights),
            "sbm_bank": self.sbm_bank,
            "sbm_fix": self.sbm_fix,
            "sbm": self.sbm,
            "sbm_old": self.sbm,
            "sbm2": self.sbm_2,
            "sbm_fast": self.sbm_fast,
            "sbm_fast_bug": self.sbm_fast,
            "sbm_fast_test": self.sbm_fast,
            "sbm_fast2": self.sbm_fast_2,
        }
        return data, types[type]

    def __call__(self):
        return self.get()

    def mix(self, aug_fct1, aug_fct2):
        tmp = copy.deepcopy(self.data)
        self.data = self.data1
        x_1, edge_index_1, _, _ = aug_fct1()
        self.data = self.data2
        _, _, x_2, edge_index_2 = aug_fct2()
        self.data = tmp
        return x_1, edge_index_1, x_2, edge_index_2

    def random(self):
        edge_attr = self.data.edge_attr if "edge_attr" in self.data else None
        edge_index_1 = dropout_adj(
            self.data.edge_index,
            edge_attr,
            p=self.param[f"drop_edge_rate_{1}"],
            force_undirected=True,
        )[0].to(self.device)
        edge_index_2 = dropout_adj(
            self.data.edge_index,
            edge_attr,
            p=self.param[f"drop_edge_rate_{2}"],
            force_undirected=True,
        )[0].to(self.device)
        x_1 = _drop_feature(self.data.x, self.param["drop_feature_rate_1"]).to(
            self.device
        )
        x_2 = _drop_feature(self.data.x, self.param["drop_feature_rate_2"]).to(
            self.device
        )
        return x_1, edge_index_1, x_2, edge_index_2

    def random_2(self):
        edge_attr = self.data.edge_attr if "edge_attr" in self.data else None
        edge_index_1 = dropout_adj(
            self.data.edge_index,
            edge_attr,
            p=self.param[f"drop_edge_rate_{1}"],
            force_undirected=True,
        )[0].to(self.device)
        edge_index_2 = dropout_adj(
            self.data.edge_index,
            edge_attr,
            p=self.param[f"drop_edge_rate_{2}"],
            force_undirected=True,
        )[0].to(self.device)
        x_1 = _drop_feature(self.data.x, self.param["drop_feature_rate_1"]).to(
            self.device
        )
        x_2 = _drop_feature(self.data.x, self.param["drop_feature_rate_2"]).to(
            self.device
        )
        return x_1, edge_index_1, x_2, edge_index_2

    def degree(self):
        def degree_drop_weights(edge_index):
            edge_index_ = to_undirected(edge_index)
            deg = degree(edge_index_[1])
            # print('deg', deg)
            deg_col = deg[edge_index[1]].to(torch.float32)
            # print('deg_col', deg_col)
            s_col = torch.log(deg_col)
            # print('s_col', s_col)
            weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())
            # print('weights', weights)
            return weights

        drop_weights = degree_drop_weights(self.data.edge_index).to(self.device)
        edge_index_ = to_undirected(self.data.edge_index)
        node_deg = degree(edge_index_[1], num_nodes=self.data.x.size(0))
        feature_weights = _feature_drop_weights(self.data.x, node_c=node_deg).to(
            self.device
        )
        return feature_weights, drop_weights

    def page_rank(self):
        def compute_pr(data, damp: float = 0.85, k: int = 10):
            num_nodes = data.edge_index.max().item() + 1
            deg_out = degree(data.edge_index[0], num_nodes=num_nodes)
            x = torch.ones((num_nodes,)).to(data.edge_index.device).to(torch.float32)
            for _ in range(k):
                edge_msg = x[data.edge_index[0]] / deg_out[data.edge_index[0]]
                agg_msg = scatter(edge_msg, data.edge_index[1], reduce="sum")
                x = (1 - damp) * x + damp * agg_msg
            return x

        def pr_drop_weights(data, aggr: str = "sink", k: int = 10):
            pv = compute_pr(data, k=k)
            pv_row = pv[data.edge_index[0]].to(torch.float32)
            pv_col = pv[data.edge_index[1]].to(torch.float32)
            s_row = torch.log(pv_row)
            s_col = torch.log(pv_col)
            if aggr == "sink":
                s = s_col
            elif aggr == "source":
                s = s_row
            elif aggr == "mean":
                s = (s_col + s_row) * 0.5
            else:
                s = s_col
            weights = (s.max() - s) / (s.max() - s.mean())
            return weights

        drop_weights = pr_drop_weights(self.data, aggr="sink", k=200).to(self.device)
        node_pr = compute_pr(self.data)
        feature_weights = _feature_drop_weights(self.data.x, node_c=node_pr).to(
            self.device
        )
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
        feature_weights = _feature_drop_weights(self.data.x, node_c=node_evc).to(
            self.device
        )
        return feature_weights, drop_weights

    def pred(self, data_param, type):
        data = copy.deepcopy(data_param)
        G = to_networkx(data, to_undirected=True)
        switch = {
            "rjc": nx.jaccard_coefficient,
            "raa": nx.adamic_adar_index,
            "rra": nx.resource_allocation_index,
        }
        preds = switch[type](G)

        nb_add = 0
        values = []
        values_norm = []
        probs = []
        min_p = float("inf")
        max_p = float("-inf")
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
                p = float((p - min_p) / (max_p - min_p))
            # p = float((max_p-p)/(max_p-mean_p))
            probs.append(p)
            values_norm.append((u, v, p))

        print("mean pred: ", mean_p)

        for u, v in G.edges():
            G[u][v]["weight"] = 1.0  # not sure to put 1 if their is a link

        to_reconstruct = int(
            data.edge_index.shape[1] * self.param["reconstruction_rate"]
        )
        values_norm = sorted(values_norm, key=lambda x: x[2], reverse=True)
        for u, v, p in values_norm:
            if p >= mean_p:
                G.add_edge(u, v, weight=p)
                G.add_edge(v, u, weight=p)
                nb_add += 1
                if nb_add >= to_reconstruct:
                    print("min prob added: ", p)
                    break
        print("add", nb_add, "edges")

        tmp = from_networkx(G).to(self.device)
        data.edge_index = tmp.edge_index
        data.weight = tmp.weight
        # print(tmp.weight)
        return data

    def gca(self, feature_weights, drop_weights):
        edge_index_1 = _drop_edge_weighted(
            self.data.edge_index,
            drop_weights,
            p=self.param[f"drop_edge_rate_{1}"],
            threshold=0.7,
        ).to(self.device)
        edge_index_2 = _drop_edge_weighted(
            self.data.edge_index,
            drop_weights,
            p=self.param[f"drop_edge_rate_{2}"],
            threshold=0.7,
        ).to(self.device)
        x_1 = _drop_feature_weighted(
            self.data.x, feature_weights, self.param["drop_feature_rate_1"]
        ).to(self.device)
        x_2 = _drop_feature_weighted(
            self.data.x, feature_weights, self.param["drop_feature_rate_2"]
        ).to(self.device)
        return x_1, edge_index_1, x_2, edge_index_2

    def commu_strength(self):
        def transition(communities, num_nodes: int) -> np.ndarray:
            classes = np.full(num_nodes, -1)
            for i, node_list in enumerate(communities):
                classes[np.asarray(node_list)] = i
            return classes

        def get_edge_weight(
            edge_index: torch.Tensor, com: np.ndarray, com_cs: np.ndarray
        ) -> torch.Tensor:
            edge_mod = (
                lambda x: com_cs[x[0]]
                if x[0] == x[1]
                else -(float(com_cs[x[0]]) + float(com_cs[x[1]]))
            )
            normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
            edge_weight = np.asarray(
                [edge_mod([com[u.item()], com[v.item()]]) for u, v in edge_index.T]
            )
            edge_weight = normalize(edge_weight)
            return torch.from_numpy(edge_weight).to(edge_index.device)

        communities, com_cs, node_cs = get_commu_strength(self.data)
        com = transition(communities, self.data.num_nodes)
        edge_weight = get_edge_weight(self.data.edge_index, com, com_cs)
        return node_cs, edge_weight

    def csgcl(self, feature_weights, node_cs):
        def ced(
            edge_index: torch.Tensor,
            edge_weight: torch.Tensor,
            p: float,
            threshold: float = 1.0,
        ) -> torch.Tensor:
            edge_weight = edge_weight / edge_weight.mean() * (1.0 - p)
            edge_weight = edge_weight.where(
                edge_weight > (1.0 - threshold),
                torch.ones_like(edge_weight) * (1.0 - threshold),
            )
            edge_weight = edge_weight.where(
                edge_weight < 1, torch.ones_like(edge_weight) * 1
            )
            sel_mask = torch.bernoulli(edge_weight).to(torch.bool)
            return edge_index[:, sel_mask].to(self.device)

        def cav(
            feature: torch.Tensor,
            node_cs: np.ndarray,
            p: float,
            max_threshold: float = 0.7,
        ) -> torch.Tensor:
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
            feature[:, drop_mask] = 0.0
            return feature.to(self.device)

        edge_index_1 = ced(
            self.data.edge_index, node_cs, p=self.param["drop_edge_rate_1"]
        )
        edge_index_2 = ced(
            self.data.edge_index, node_cs, p=self.param["drop_edge_rate_2"]
        )
        x_1 = cav(self.data.x, feature_weights, self.param["drop_feature_rate_1"])
        x_2 = cav(self.data.x, feature_weights, self.param["drop_feature_rate_2"])
        return x_1, edge_index_1, x_2, edge_index_2

    def sbm(self):
        sizes, probs = self.data.sizes, self.data.probs
        data_1 = gen_sbm(sizes, probs).to(self.device)
        data_1.x = self.data.x
        data_1.x = _drop_feature(data_1.x, self.param['drop_feature_rate_1'])
        data_2 = self.data
        data_2.x = _drop_feature(data_2.x, self.param['drop_feature_rate_2'])
        return data_1.x, data_1.edge_index, data_2.x, data_2.edge_index

    def sbm_fix(self):
        sizes, probs = self.data.sizes, self.data.probs
        node_list = self.data.node_list
        data_1 = gen_sbm(sizes, probs, node_list).to(self.device)
        data_1.x = self.data.x
        data_1.x = _drop_feature(data_1.x, self.param['drop_feature_rate_1'])
        data_2 = self.data
        data_2.x = _drop_feature(data_2.x, self.param['drop_feature_rate_2'])
        return data_1.x, data_1.edge_index, data_2.x, data_2.edge_index
    
    def sbm_bank(self):
        global sbm_bank
        global ct_epoch
        data_1 = sbm_bank[ct_epoch].to(self.device)
        data_1.x = self.data.x
        data_1.x = _drop_feature(data_1.x, self.param['drop_feature_rate_1'])
        data_2 = self.data
        data_2.x = _drop_feature(data_2.x, self.param['drop_feature_rate_2'])
        
        ct_epoch += 1
        return data_1.x, data_1.edge_index, data_2.x, data_2.edge_index

    def sbm_2(self):
        sizes, probs = self.data.sizes, self.data.probs
        data_1 = gen_sbm(sizes, probs).to(self.device)
        data_1.x = self.data.x
        data_1.x = _drop_feature(data_1.x, self.param["drop_feature_rate_1"])
        data_2 = gen_sbm(sizes, probs).to(self.device)
        data_2.x = self.data.x
        data_2.x = _drop_feature(data_2.x, self.param["drop_feature_rate_2"])
        return data_1.x, data_1.edge_index, data_2.x, data_2.edge_index

    def sbm_fast(self):
        data_1 = gen_sbm_fast(self.data, self.data.state).to(self.device)
        data_1.x = self.data.x
        data_1.x = _drop_feature(data_1.x, self.param["drop_feature_rate_1"])
        data_2 = self.data
        data_2.x = _drop_feature(data_2.x, self.param["drop_feature_rate_2"])
        return data_1.x, data_1.edge_index, data_2.x, data_2.edge_index

    def sbm_fast_2(self):
        data_1 = gen_sbm_fast(self.data, self.data.state).to(self.device)
        data_1.x = self.data.x
        data_1.x = _drop_feature(data_1.x, self.param["drop_feature_rate_1"])
        data_2 = gen_sbm_fast(self.data, self.data.state).to(self.device)
        data_2.x = _drop_feature(data_2.x, self.param["drop_feature_rate_2"])
        return data_1.x, data_1.edge_index, data_2.x, data_2.edge_index

    def reconstruct(self):
        edge_index_1 = _drop_edge_weighted(
            self.data.edge_index, self.data.weight, self.param["drop_edge_rate_1"]
        ).to(self.device)
        edge_index_2 = _drop_edge_weighted(
            self.data.edge_index, self.data.weight, self.param["drop_edge_rate_2"]
        ).to(self.device)
        x_1 = _drop_feature(self.data.x, self.param["drop_feature_rate_1"]).to(
            self.device
        )
        x_2 = _drop_feature(self.data.x, self.param["drop_feature_rate_2"]).to(
            self.device
        )
        return x_1, edge_index_1, x_2, edge_index_2

def _drop_feature(x, drop_prob):
    drop_mask = (
        torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1)
        < drop_prob
    )
    x = x.clone()
    x[:, drop_mask] = 0
    return x


def _feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())
    return s


def _drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.0):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(
        edge_weights < threshold, torch.ones_like(edge_weights) * threshold
    )
    # print(edge_weights)
    sel_mask = torch.bernoulli(1.0 - edge_weights).to(torch.bool)
    return edge_index[:, sel_mask]


def _drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_mask = torch.bernoulli(w).to(torch.bool)
    x = x.clone()
    x[:, drop_mask] = 0.0
    return x


def gen_sbm_bank(data_split):
    global sbm_bank
    data, _ = data_split.get(0)
    data = commu_distrib(data).to(data.x.device)
    sizes, probs = data.sizes, data.probs
    for _ in tqdm(range(5000), desc="sbm bank hp"):
        data_new = gen_sbm(sizes, probs).to(data.x.device)
        sbm_bank.append(data_new)
