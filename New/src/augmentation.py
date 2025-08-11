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
)
from torch_scatter import scatter
from functools import partial

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.utils import (
    get_commu_strength,
    gen_sbm,
    gen_sbm_fast,
    commu_distrib,
    to_graph_tool,
    to_graph_tool_bug
)
from graph_tool.inference import BlockState

class Aug:
    def __init__(
        self,
        data: Data,
        split_edge: list,
        param: dict,
        type: str = "random",
    ) -> None:
        """
        init the Aug class

        Args:
            data (Data): 
            split_edge (list): 
            param (dict): 
            type (str, optional): type of augmentation, can be combined augmentation (eg. random+deg). Defaults to "random".
        """
        self.split_edge = split_edge
        self.device = data.x.device
        self.data = copy.deepcopy(data).to(self.device)
        self.param = param
        self.type = type
        if "+" in type:
            type1, type2 = type.split("+")
            self.data1, aug_fct1 = self.precompute(self.data, type1)
            self.data2, aug_fct2 = self.precompute(self.data, type2)
            self.get = partial(self.mix, aug_fct1, aug_fct2)
        else:
            self.data, aug_fct = self.precompute(self.data, type)
            self.get = aug_fct

    def precompute(self, data: Data, type: str) -> tuple:
        """
        pre compute usfull thing one time for augmentations

        Args:
            data (Data): 
            type (str): name of augmentation

        Returns:
            tuple: data and an augmentation function
        """
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
            cd_algo = self.param["commu_detect"]
            if "_cheat" in type:
                full_split = to_undirected(torch.cat((self.split_edge['train']['edge'], 
                                                     self.split_edge['valid']['edge'], 
                                                     self.split_edge['test']['edge'])).T)
                data = commu_distrib(data, cd_algo, full_split=full_split).to(self.device)
                type = type.replace('_cheat', '')
            else:
                data = commu_distrib(data, cd_algo).to(self.device)
            
            if "_fix" in type:
                data.node_list = [j for sub in data.communities for j in sub]
            elif "_fast_bug" in type:
                gtG, block_map = to_graph_tool_bug(data)
                data.state = BlockState(gtG, block_map, deg_corr=False)
            elif "_fast" in type:
                gtG, block_map = to_graph_tool(data)
                data.state = BlockState(gtG, block_map, deg_corr=False)

        elif "kmeans" in type:
            if hasattr(data, "_pos"):
                pos = []
                for p in data._pos:
                    x, y = p.split(", ")
                    pos.append([float(x), float(y)])
            else:
                assert("No Kmeans without pos!!")

            pos = np.array(pos)
            
            switch = {
                'crime': 49,
                'euroroad': 46,
                'netscience': 57,
                'power': 169,
                'wiki_science': 21,
                'yeast': 48,
            } # find with KMEANS.ipynb
            n_clusters = switch[self.data.name]
            kmeans = KMeans(n_clusters=n_clusters)
            clusters = kmeans.fit_predict(pos)
            print(clusters)
            data.block = clusters
            _, counts = np.unique(clusters, return_counts=True)
            data.sizes = np.sort(counts)[::-1]
            print(data.sizes)

            gtG, block_map = to_graph_tool(data)
            data.state = BlockState(gtG, block_map, deg_corr=False)
        types = {
            "random": self.random,
            "deg": partial(self.gca, feature_weights, drop_weights),
            "pr": partial(self.gca, feature_weights, drop_weights),
            "evc": partial(self.gca, feature_weights, drop_weights),
            "scom": partial(self.csgcl, feature_weights, drop_weights),
            "sbm": self.sbm,
            "sbm2": self.sbm_2,
            "sbm_fix": self.sbm_fix,
            "sbm_fast": self.sbm_fast,
            "sbm_fast2": self.sbm_fast_2,
            "sbm_fast_bug": self.sbm_fast,
            "kmeans": self.sbm_fast,
            "kmeans2": self.sbm_fast_2
        }
        return data, types[type]

    def __call__(self) -> tuple:
        """
        wrapper to get the augmentation

        Returns:
            tuple: x_1, edge_index_1, x_2, edge_index_2
        """
        return self.get()

    def mix(self, aug_fct1, aug_fct2) -> tuple:
        """
        mix 2 function of augmentation

        Args:
            aug_fct1 (func): augmentation function 1
            aug_fct2 (func): augmentationt function 2

        Returns:
            tuple: x_1, edge_index_1, x_2, edge_index_2
        """
        tmp = copy.deepcopy(self.data)
        self.data = self.data1
        x_1, edge_index_1, _, _ = aug_fct1()
        self.data = self.data2
        _, _, x_2, edge_index_2 = aug_fct2()
        self.data = tmp
        return x_1, edge_index_1, x_2, edge_index_2

    def random(self) -> tuple:
        """
        random drop edge and attributs in X

        Returns:
            tuple: x_1, edge_index_1, x_2, edge_index_2
        """
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

    def degree(self) -> tuple:
        """
        compute feature and edge drop weights based on degree of nodes

        Returns:
            tuple: feature_weights, drop_weights
        """
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
        feature_weights = _feature_drop_weights(self.data.x, node_c=node_deg).to(
            self.device
        )
        return feature_weights, drop_weights

    def page_rank(self) -> tuple:
        """
        compute feature and edge drop weights based on page rank of nodes

        Returns:
            tuple: feature_weights, drop_weights
        """
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

    def eigenvector(self) -> tuple:
        """
        compute feature and edge drop weights based on eigenvector of nodes

        Returns:
            tuple: feature_weights, drop_weights
        """
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

    def gca(self, feature_weights: torch.Tensor, drop_weights: torch.Tensor) -> tuple:
        """
        adaptative augmentation depending of the centrality mesure (deg, pr, evc) choosed.

        Args:
            feature_weights (torch.Tensor):
            drop_weights (torch.Tensor):

        Returns:
            tuple: x_1, edge_index_1, x_2, edge_index_2
        """
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

    def commu_strength(self) -> tuple:
        """
        compute node and edge weight based on communities strength using leiden algo

        Returns:
            tuple: node_cs, edge_weight
        """
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

    def csgcl(self, feature_weights, node_cs) -> tuple:
        """
        augmentation based on communities strength (cav, ced)

        Args:
            feature_weights (torch.Tensor):
            node_cs (np.ndarray):

        Returns:
            tuple: x_1, edge_index_1, x_2, edge_index_2
        """
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

    def sbm(self) -> tuple:
        """
        networkx sbm augmentation without fix

        Returns:
            tuple: x_1, edge_index_1, x_2, edge_index_2
        """
        sizes, probs = self.data.sizes, self.data.probs
        data_1 = gen_sbm(sizes, probs).to(self.device)
        data_2 = self.data
        x_1 = _drop_feature(self.data.x, self.param["drop_feature_rate_1"]).to(self.device)
        x_2 = _drop_feature(self.data.x, self.param["drop_feature_rate_2"]).to(self.device)
        return x_1, data_1.edge_index, x_2, data_2.edge_index

    def sbm_2(self) -> tuple:
        """
        double networkx sbm augmentation without fix

        Returns:
            tuple: x_1, edge_index_1, x_2, edge_index_2
        """
        sizes, probs = self.data.sizes, self.data.probs
        data_1 = gen_sbm(sizes, probs).to(self.device)
        data_2 = gen_sbm(sizes, probs).to(self.device)
        x_1 = _drop_feature(self.data.x, self.param["drop_feature_rate_1"]).to(self.device)
        x_2 = _drop_feature(self.data.x, self.param["drop_feature_rate_2"]).to(self.device)
        return x_1, data_1.edge_index, x_2, data_2.edge_index
    
    def sbm_fix(self) -> tuple:
        """
        networkx sbm augmentation with fix

        Returns:
            tuple: x_1, edge_index_1, x_2, edge_index_2
        """
        sizes, probs = self.data.sizes, self.data.probs
        node_list = self.data.node_list
        edge_index_1 = gen_sbm(sizes, probs, node_list).to(self.device)
        x_1 = _drop_feature(self.data.x, self.param["drop_feature_rate_1"]).to(self.device)
        x_2 = _drop_feature(self.data.x, self.param["drop_feature_rate_2"]).to(self.device)
        return x_1, edge_index_1, x_2, self.data.edge_index

    def sbm_fix_2(self) -> tuple:
        """
        double networkx sbm augmentation with fix

        Returns:
            tuple: x_1, edge_index_1, x_2, edge_index_2
        """
        sizes, probs = self.data.sizes, self.data.probs
        node_list = self.data.node_list
        edge_index_1 = gen_sbm(sizes, probs, node_list).to(self.device)
        edge_index_2 = gen_sbm(sizes, probs, node_list).to(self.device)
        x_1 = _drop_feature(self.data.x, self.param["drop_feature_rate_1"]).to(self.device)
        x_2 = _drop_feature(self.data.x, self.param["drop_feature_rate_2"]).to(self.device)
        return x_1, edge_index_1, x_2, edge_index_2

    def sbm_fast(self) -> tuple:
        """
        graph-tools sbm augmentation

        Returns:
            tuple: x_1, edge_index_1, x_2, edge_index_2
        """
        edge_index_1 = gen_sbm_fast(self.data.state).to(self.device)
        x_1 = _drop_feature(self.data.x, self.param["drop_feature_rate_1"]).to(self.device)
        x_2 = _drop_feature(self.data.x, self.param["drop_feature_rate_2"]).to(self.device)
        return x_1, edge_index_1, x_2, self.data.edge_index

    def sbm_fast_2(self) -> tuple:
        """
        double graph-tools sbm augmentation

        Returns:
            tuple: x_1, edge_index_1, x_2, edge_index_2
        """
        edge_index_1 = gen_sbm_fast(self.data.state).to(self.device)
        edge_index_2 = gen_sbm_fast(self.data.state).to(self.device)
        x_1 = _drop_feature(self.data.x, self.param["drop_feature_rate_1"]).to(self.device)
        x_2 = _drop_feature(self.data.x, self.param["drop_feature_rate_2"]).to(self.device)
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
    sel_mask = torch.bernoulli(1.0 - edge_weights).to(torch.bool)
    return edge_index[:, sel_mask]


def _drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_mask = torch.bernoulli(w).to(torch.bool)
    x = x.clone()
    x[:, drop_mask] = 0.0
    return x
