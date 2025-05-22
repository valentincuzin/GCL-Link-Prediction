import time
from tqdm import tqdm
from copy import deepcopy
import torch
import numpy as np
import igraph as ig
import networkx as nx
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_sparse import SparseTensor
from sklearn.decomposition import PCA
from networkx.generators.community import LFR_benchmark_graph
from torch_geometric import seed_everything
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected, add_self_loops, from_networkx, to_networkx
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from numpy.core.multiarray import _reconstruct

torch.serialization.add_safe_globals([_reconstruct, DataEdgeAttr, DataTensorAttr, GlobalStorage])

from src.utils import gen_sbm, average_precision

def randomsplit(dataset, val_ratio: float = 0.10, test_ratio: float = 0.2):
    def removerepeated(ei):
        ei = to_undirected(ei)

        ei = ei[:, ei[0] < ei[1]]
        return ei
    def split_pos_neg(data):
        pos_mask = (data.edge_label == 1).bool()
        neg_mask = (data.edge_label == 0).bool()
        pos_edge_index = data.edge_label_index[:, pos_mask]
        neg_edge_index = data.edge_label_index[:, neg_mask]
        return pos_edge_index, neg_edge_index

    data = dataset[0]
    data.num_nodes = data.x.shape[0]
    transform = RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        add_negative_train_samples=True
    )
    train_data, val_data, test_data = transform(data)
    val_data_pos, val_data_neg = split_pos_neg(val_data)
    test_data_pos, test_data_neg = split_pos_neg(test_data)
    
    split_edge = {'train': {'edge': removerepeated(train_data.edge_index).t()},
                  'valid': {'edge': removerepeated(val_data_pos).t(),
                            'edge_neg': removerepeated(val_data_neg).t()}, 
                  'test': {'edge': removerepeated(test_data_pos).t(),
                           'edge_neg': removerepeated(test_data_neg).t()}}    
    return split_edge

def loaddataset(name: str|list, use_valedges_as_input: bool, reduce_feature: int|None = None, only_feature: bool = False, load=None):
    if isinstance(name,list):
        split_edge = randomsplit(name)
        data = name[0]
        if reduce_feature is not None:
            if reduce_feature == 0:
                data.x = F.one_hot(torch.arange(0, len(data.x))).float()
            else:
                reduce_node_features(data, reduce_feature)
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        if only_feature:
            data.edge_index = torch.tensor([[], []], dtype=torch.long)
            data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]

        if data.edge_index.max().item() + 1 < data.num_nodes:
            data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
    elif name in ["cora", "citeseer", "pubmed"]:
        dataset = Planetoid(root="dataset", name=name)
        split_edge = randomsplit(dataset)
        data = dataset[0]
        if reduce_feature is not None:
            if reduce_feature == 0:
                data.x = F.one_hot(torch.arange(0, len(data.x))).float()
            else:
                reduce_node_features(data, reduce_feature)
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        if only_feature:
            data.edge_index = torch.tensor([[], []], dtype=torch.long)
            data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]

        if data.edge_index.max().item() + 1 < data.num_nodes:
            data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
    elif name in ["facebook_friends", "wiki_science", "crime", 
                  "power", "unicodelang", "euroroad",
                  "escort", "tips", "pol_kato", "pol_robertson"]:
        igG = ig.Graph.Read_GML(f'./small_gml/{name}.gml')
        G = igG.to_networkx()
        data = from_networkx(G)
        data.x = F.one_hot(torch.arange(0, len(G.nodes))).float()
        data.edge_index = to_undirected(data.edge_index)
        split_edge = randomsplit([data])
        data.edge_index = to_undirected(split_edge["train"]["edge"].t())
        edge_index = data.edge_index
        data.num_nodes = data.x.shape[0]

        if data.edge_index.max().item() + 1 < data.num_nodes:
            data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
    else:
        dataset = PygLinkPropPredDataset(name=f'ogbl-{name}')
        split_edge = dataset.get_edge_split()
        data = dataset[0]
        if reduce_feature is not None:
            if reduce_feature == 0:
                data.x = F.one_hot(torch.arange(0, len(data.x))).float()
            else:
                reduce_node_features(data, reduce_feature)
        edge_index = data.edge_index
        if only_feature:
            data.edge_index = torch.tensor([[], []], dtype=torch.long)
            data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]

        if data.edge_index.max().item() + 1 < data.num_nodes:
            data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]

    data.edge_weight = None 
    data.adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.max_x = -1
    if name == "ppa":
        data.x = torch.argmax(data.x, dim=-1)
        data.max_x = torch.max(data.x).item()
    elif name == "ddi":
        data.x = torch.arange(data.num_nodes)
        data.max_x = data.num_nodes
    if load is not None:
        data.x = torch.load(load, map_location="cpu")
        data.max_x = -1

    # Use training + validation edges for inference on test set.
    if use_valedges_as_input:
        val_edge_index = split_edge['valid']['edge'].t()
        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).coalesce()
        data.full_adj_t = data.full_adj_t.to_symmetric()
    else:
        data.full_adj_t = data.adj_t
    return data, split_edge

def reduce_node_features(data, nb_features):
    # reduce features size per nodes
    data_np = data.x.numpy()
    pca = PCA(n_components=nb_features)
    data_reduced = pca.fit_transform(data_np)
    data.x = torch.tensor(data_reduced, dtype=torch.float)
    print('reduce node features: ',data.x.shape)
    return data

def _LFR_gen(n, tau1, tau2, mu, average_degree, min_community, max_community = None):
    G = LFR_benchmark_graph(n, tau1, tau2, mu, 
                            average_degree=average_degree, min_community=min_community, 
                            max_community=max_community, seed=10)
    G.remove_edges_from(nx.selfloop_edges(G)) # remove self loops
    data = from_networkx(G)
    data.edge_index = to_undirected(data.edge_index)
    communities = {frozenset(G.nodes[v]["community"]) for v in G}
    data.community = communities
    sizes, probs = _get_sizes_probs(data, G, communities)
    data.sizes = sizes
    data.probs = probs
    data.num_features = data.num_nodes
    data.x = F.one_hot(torch.arange(0, data.num_nodes)).float()
    print(np.round(probs, 5))
    return data

def _get_sizes_probs(data, G, communities):
    probs = np.zeros((len(communities), len(communities)))
    sizes = []
    for idx, c in enumerate(communities):
        sizes.append(len(c))
        for n in c:
            G.nodes[n]["com"] = idx # get com label
    for u, v in zip(data.edge_index[0], data.edge_index[1]): # count number of edge per com
        u = float(u)
        v = float(v)
        probs[G.nodes[u]["com"], G.nodes[v]["com"]] += 1
    for x in range(len(probs)): # make the probs
        for y in range(len(probs)):
            if x == y:
                probs[x,x] /= (sizes[x]*(sizes[x]-1))/2
            else:
                probs[x,y] /= ((sizes[x]+sizes[y])*(sizes[x]+sizes[y]-1))/2
    probs /= 2 # undirected graph
    return sizes, probs

class DataSplit:
    def __init__(self, dataset: str|list, device: str, runs: int, use_valedges_as_input: bool = False, reduce_feature: int|None = None, only_feature: bool = False):
        print(f"{runs} split from the dataset {dataset}")
        if dataset in ["synthetic_1", "synthetic_2", "synthetic_3"]:
            if dataset == "synthetic_1":
                data = _LFR_gen(400, 4, 3, 0.2, average_degree=10, min_community=75)
            elif dataset == "synthetic_2":
                data = _LFR_gen(400, 2.5, 2.5, 0.2, average_degree=10, min_community=200, max_community=200)
            elif dataset == "synthetic_3":
                dataset = Planetoid(root="dataset", name="cora")
                data = dataset[0]
                data.edge_index = to_undirected(data.edge_index)
                G = to_networkx(data, to_undirected=True)
                data.communities = nx.community.louvain_communities(G, resolution=0.5)
                data.sizes, data.probs = _get_sizes_probs(data, G, data.communities)
            data = gen_sbm(data.sizes, data.probs)
            dataset = [data]
        self.device = device
        self.runs = runs
        self.data_runs: dict[int, tuple[any, dict]] = {}
        t1 = time.time()
        for r in tqdm(range(runs)):
            seed_everything(r)
            dataset_tmp = deepcopy(dataset)
            data, split_edge = loaddataset(dataset_tmp, use_valedges_as_input, reduce_feature, only_feature)
            data = data.to(device)
            self.data_runs[r] = data, split_edge
        self.info_time = round(time.time()-t1, 2)
        self.info()

    def get(self, r):
        data, split_edge = self.data_runs[r]
        return data.to(self.device), split_edge

    def info(self):
        print("split time: ", self.info_time, " s")
        data, split_edge = self.data_runs[0]
        print("dataset split ")
        for key1 in split_edge:
            for key2  in split_edge[key1]:
                print(key1, key2, split_edge[key1][key2].shape[0])

def get_evaluator(dataset: str = 'ogbl-ppa'):
    if dataset in ["collab", "citation2", "wikikg2", "ddi", "biokg", "vessel"]:
        evaluator = Evaluator(name=f'ogbl-{dataset}')
    else:
        evaluator = Evaluator(name='ogbl-ppa')
    return evaluator

def full_eval(evaluator, pos_pred, neg_pred):
    results = {}
    evaluator.eval_metric = 'hits@k'
    for K in [10, 20, 50, 100]:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']
        results[f'Hits@{K}'] = hits
    evaluator.eval_metric = 'rocauc'
    auc = evaluator.eval({
        'y_pred_pos': pos_pred,
        'y_pred_neg': neg_pred,
    })['rocauc']
    results['ROCAUC'] = auc
    

    
    results['AP'] = average_precision(pos_pred, neg_pred)
    return results
