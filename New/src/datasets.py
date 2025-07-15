import time
from tqdm import tqdm
from copy import deepcopy
import torch
import numpy as np
import igraph as ig
import scipy.io as sio
import networkx as nx
import torch.nn.functional as F
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_sparse import SparseTensor
from sklearn.decomposition import PCA
from networkx.generators.community import LFR_benchmark_graph
from torch_geometric import seed_everything
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.data import Data
from torch_geometric.utils import (
    to_undirected,
    add_self_loops,
    from_networkx,
    to_networkx,
    from_scipy_sparse_matrix,
)
from torch_geometric.transforms import RandomLinkSplit

from src.utils import gen_sbm, average_precision, removerepeated


def randomsplit(data: Data, val_ratio: float = 0.10, test_ratio: float = 0.2):
    """
    random split on edges in 3 sets train,val,test

    Args:
        dataset (list[Data]):
        val_ratio (float, optional): . Defaults to 0.10.
        test_ratio (float, optional): . Defaults to 0.2.
    """

    def split_pos_neg(data):
        pos_mask = (data.edge_label == 1).bool()
        neg_mask = (data.edge_label == 0).bool()
        pos_edge_index = data.edge_label_index[:, pos_mask]
        neg_edge_index = data.edge_label_index[:, neg_mask]
        return pos_edge_index, neg_edge_index

    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True,
    )
    train_data, val_data, test_data = transform(data)
    val_data_pos, val_data_neg = split_pos_neg(val_data)
    test_data_pos, test_data_neg = split_pos_neg(test_data)

    split_edge = {
        "train": {"edge": removerepeated(train_data.edge_index).t()},
        "valid": {
            "edge": removerepeated(val_data_pos).t(),
            "edge_neg": removerepeated(val_data_neg).t(),
        },
        "test": {
            "edge": removerepeated(test_data_pos).t(),
            "edge_neg": removerepeated(test_data_neg).t(),
        },
    }
    return split_edge


def loaddataset(
    name: str | list,
    reduce_feature: int | None = None,
    only_feature: bool = False,
) -> tuple[Data, dict]:
    """
    implement several way to load many dataset

    Args:
        name (str | list): name of the graph
        reduce_feature (int | None, optional): use PCA. Defaults to None.
        only_feature (bool, optional): destroy structure of the graph. Defaults to False.

    Returns:
        tuple[Data, dict]: return the Data loaded and is split associated
    """

    if isinstance(name, list):  # already laoded, only need to format
        data = name[0]
        data.edge_index = to_undirected(
            removerepeated(data.edge_index)
        )
        data.num_nodes = data.x.shape[0]
        split_edge = randomsplit(data)

    elif name in [
        "facebook_friends",
        "wiki_science",
        "crime",
        "power",
        "unicodelang",
        "euroroad",
        "escort",
        "tips",
        "pol_kato",
        "pol_robertson",
        "yeast",
        "netscience",
    ]:  # dataset from https://networks.skewed.de/
        igG = ig.Graph.Read_GML(f"./small_gml/{name}.gml")  # read gml file
        G = igG.to_networkx()
        data = from_networkx(G)
        data.edge_index = to_undirected(
            removerepeated(data.edge_index)
        )
        data.x = F.one_hot(torch.arange(0, len(G.nodes))).float()
        data.num_nodes = data.x.shape[0]
        split_edge = randomsplit(data)

    elif name in ["USAir", "NS", "PB", "Yeast", "Celegans", "Power", "Router", "Ecoli"]:
        data_dir = f"./small_mat/{name}.mat"
        net = sio.loadmat(data_dir)
        edge_index, _ = from_scipy_sparse_matrix(net["net"])
        edge_index = to_undirected(
            removerepeated(edge_index)
        )
        data = Data(edge_index=edge_index, num_nodes=torch.max(edge_index).item() + 1)
        data.x = F.one_hot(torch.arange(0, data.num_nodes)).float()
        split_edge = randomsplit(data)

    elif name in ["collab"]:
        dataset = PygLinkPropPredDataset(name=f"ogbl-{name}")
        data = dataset[0]
        split_edge = dataset.get_edge_split()

    else:
        if name in ["cora", "citeseer", "pubmed"]:
            dataset = Planetoid(root="dataset", name=name)
        elif name in ["cs", "physics"]:
            dataset = Coauthor(root="dataset", name=name)
        elif name in ["computers", "photo"]:
            dataset = Amazon(root="dataset", name=name)
        data = dataset[0]
        data.num_nodes = data.x.shape[0]
        split_edge = randomsplit(data)

    data.edge_index = to_undirected(
        split_edge["train"]["edge"].t()
    )

    if data.edge_index.max().item() + 1 < data.num_nodes:
        data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]

    if reduce_feature is not None:
        if reduce_feature == 0:
            data.x = F.one_hot(torch.arange(0, len(data.x))).float()
        else:
            reduce_node_features(data, reduce_feature)
    if only_feature:
        data.edge_index = torch.tensor([[], []], dtype=torch.long)
        data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]

    data.edge_weight = None
    data.max_x = -1
    data.adj_t = SparseTensor.from_edge_index(
        data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)
    )
    data.adj_t = data.adj_t.to_symmetric().coalesce()

    val_edge_index = split_edge["valid"]["edge"].t()
    full_edge_index = torch.cat([data.edge_index, val_edge_index], dim=-1)

    data.full_adj_t = SparseTensor.from_edge_index(
        full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)
    ).coalesce()
    data.full_adj_t = data.full_adj_t.to_symmetric()
    return data, split_edge


def reduce_node_features(data: Data, nb_features: int) -> Data:
    """
    reduce size of nodes features

    Args:
        data (Data):
        nb_features (int):

    Returns:
        Data: data reducted
    """
    data_np = data.x.numpy()
    pca = PCA(n_components=nb_features)
    data_reduced = pca.fit_transform(data_np)
    data.x = torch.tensor(data_reduced, dtype=torch.float)
    print("reduce node features: ", data.x.shape)
    return data


class DataSplit:
    def __init__(
        self,
        dataset: str | list,
        device: str,
        runs: int,
        reduce_feature: int | None = None,
        only_feature: bool = False,
    ) -> None:
        """
        init DataSplit class

        Args:
            dataset (str | list): the graph choosed
            device (str): cpu or cuda device
            runs (int): number of runs
            reduce_feature (int | None, optional): PCA apply on matrix X to reduce feature. Defaults to None.
            only_feature (bool, optional): destroy structure of the graph to base prediction only on feature. Defaults to False.
        """
        print(f"{runs} split from the dataset {dataset}")
        if dataset in ["synthetic_1", "synthetic_2", "synthetic_3"]:
            if dataset == "synthetic_1":
                data = _LFR_gen(400, 4, 3, 0.2, average_degree=10, min_community=75)
            elif dataset == "synthetic_2":
                data = _LFR_gen(
                    400,
                    2.5,
                    2.5,
                    0.2,
                    average_degree=10,
                    min_community=200,
                    max_community=200,
                )
            elif dataset == "synthetic_3":
                dataset = Planetoid(root="dataset", name="cora")
                data = dataset[0]
                data.edge_index = to_undirected(data.edge_index)
                G = to_networkx(data, to_undirected=True)
                data.communities = nx.community.louvain_communities(G, resolution=0.5)
                data.sizes, data.probs = _get_sizes_probs(data, G, data.communities)
            data = gen_sbm(data.sizes, data.probs)
            data.num_nodes = sum(data.sizes)
            data.x = F.one_hot(torch.arange(0, data.num_nodes)).float()
            dataset = [data]

        self.device = device
        self.runs = runs
        self.data_runs: dict[int, tuple[any, dict]] = {}
        t1 = time.time()
        for r in tqdm(range(runs)):
            seed_everything(r)
            dataset_tmp = deepcopy(dataset)
            data, split_edge = loaddataset(dataset_tmp, reduce_feature, only_feature)
            data = data.to(device)
            self.data_runs[r] = data, split_edge
        self.info_time = round(time.time() - t1, 2)
        self.info()

    def get(self, r: int) -> tuple[Data, dict]:
        """
        get the data and split on run r

        Args:
            r (int): current run

        Returns:
            tuple[Data, dict]: the data and the split_edge
        """
        data, split_edge = self.data_runs[r]
        return data.to(self.device), split_edge

    def info(self) -> None:
        """
        just print usefull info on the DataSplit
        """
        print("split time: ", self.info_time, " s")
        data, split_edge = self.data_runs[0]
        print("data 0: ", data)
        print("split 0:")
        for key1 in split_edge:
            for key2 in split_edge[key1]:
                print(key1, key2, split_edge[key1][key2].shape[0])


def get_evaluator(dataset: str = "ogbl-ppa") -> Evaluator:
    """
    return the Evaluator of link prediction

    Args:
        dataset (str, optional): name of the dataset. Defaults to "ogbl-ppa".

    Returns:
        Evaluator: usefull to evaluate
    """
    if dataset in ["collab", "citation2", "wikikg2", "ddi", "biokg", "vessel"]:
        evaluator = Evaluator(name=f"ogbl-{dataset}")
    else:
        evaluator = Evaluator(name="ogbl-ppa")
    return evaluator


def full_eval(
    evaluator: Evaluator, pos_pred: torch.tensor, neg_pred: torch.tensor
) -> dict:
    """
    return full results of the evaluation of the predicted sets

    Args:
        evaluator (Evaluator):
        pos_pred (torch.tensor): prediction on True links set
        neg_pred (torch.tensor): prediction on False links set

    Returns:
        dict: full results
    """
    results = {}
    evaluator.eval_metric = "hits@k"
    for K in [10, 20, 50, 100]:
        evaluator.K = K
        hits = evaluator.eval(
            {
                "y_pred_pos": pos_pred,
                "y_pred_neg": neg_pred,
            }
        )[f"hits@{K}"]
        results[f"Hits@{K}"] = hits
    evaluator.eval_metric = "rocauc"
    auc = evaluator.eval(
        {
            "y_pred_pos": pos_pred,
            "y_pred_neg": neg_pred,
        }
    )["rocauc"]
    results["ROCAUC"] = auc

    results["AP"] = average_precision(pos_pred, neg_pred)
    return results


def _LFR_gen(
    n: int,
    tau1: float,
    tau2: float,
    mu: float,
    average_degree: int,
    min_community: int,
    max_community: int | None = None,
) -> Data:
    """
    Generate a synthetic graph using LFR

    Args:
        n (int):
        tau1 (float):
        tau2 (float):
        mu (float):
        average_degree (int):
        min_community (int):
        max_community (int | None, optional):. Defaults to None.

    Returns:
        Data: the Graph generated
    """
    G = LFR_benchmark_graph(
        n,
        tau1,
        tau2,
        mu,
        average_degree=average_degree,
        min_community=min_community,
        max_community=max_community,
        seed=10,
    )
    G.remove_edges_from(nx.selfloop_edges(G))  # remove self loops
    data = from_networkx(G)
    data.edge_index = to_undirected(data.edge_index)
    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    data.community = communities
    sizes, probs = _get_sizes_probs(data, G, communities)
    data.sizes = sizes
    data.probs = probs
    data.num_features = data.num_nodes
    data.x = F.one_hot(torch.arange(0, data.num_nodes)).float()
    return data


def _get_sizes_probs(
    data: Data, G: nx.Graph, communities: np.ndarray
) -> tuple[list, np.ndarray]:
    """
    get communities sizes and densities of the graph

    Args:
        data (Data):
        G (nx.Graph):
        communities (np.ndarray):

    Returns:
        tuple[list, np.ndarray]: size and probs of link between com
    """
    probs = np.zeros((len(communities), len(communities)))
    sizes = []
    for idx, c in enumerate(communities):
        sizes.append(len(c))
        for n in c:
            G.nodes[n]["com"] = idx  # get com label
    for u, v in zip(
        data.edge_index[0], data.edge_index[1]
    ):  # count number of edge per com
        u = float(u)
        v = float(v)
        probs[G.nodes[u]["com"], G.nodes[v]["com"]] += 1
    for x in range(len(probs)):  # make the probs
        for y in range(len(probs)):
            if x == y:
                if sizes[x] > 1:
                    probs[x, x] = probs[x, x] / ((sizes[x] * (sizes[x] - 1)) / 2)
            else:
                probs[x, y] = probs[x, y] / (
                    ((sizes[x] + sizes[y]) * (sizes[x] + sizes[y] - 1)) / 2
                )
    probs /= 2  # undirected graph
    return sizes, probs
