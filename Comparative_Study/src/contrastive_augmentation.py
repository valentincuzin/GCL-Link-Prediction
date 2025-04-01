import torch
import random
import networkx as nx
import numpy as np
from torch_geometric.utils import degree, to_undirected, to_networkx
from torch_scatter import scatter
from cdlib import algorithms
from cdlib.utils import convert_graph_formats


def compute_pr(data, damp: float = 0.85, k: int = 10):
    num_nodes = data.edge_index.max().item() + 1
    print(num_nodes, data.x.size(0))
    deg_out = degree(data.edge_index[0], num_nodes=num_nodes)
    x = torch.ones((num_nodes, )).to(data.edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[data.edge_index[0]] / deg_out[data.edge_index[0]]
        agg_msg = scatter(edge_msg, data.edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x

def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality(graph, max_iter=200)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)


def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x

def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w

    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)

    x = x.clone()
    x[:, drop_mask] = 0.

    return x

def feature_drop_weights(x, node_c):
    x = x.to(torch.bool).to(torch.float32)
    w = x.t() @ node_c
    w = w.log()
    s = (w.max() - w) / (w.max() - w.mean())

    return s

def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
    edge_weights = edge_weights / edge_weights.mean() * p
    edge_weights = edge_weights.where(edge_weights < threshold, torch.ones_like(edge_weights) * threshold)
    sel_mask = torch.bernoulli(1. - edge_weights).to(torch.bool)

    return edge_index[:, sel_mask]

def degree_drop_weights(edge_index):
    edge_index_ = to_undirected(edge_index)
    deg = degree(edge_index_[1])
    deg_col = deg[edge_index[1]].to(torch.float32)
    s_col = torch.log(deg_col)
    weights = (s_col.max() - s_col) / (s_col.max() - s_col.mean())

    return weights

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

def evc_drop_weights(data):
    evc = eigenvector_centrality(data)
    evc = evc.where(evc > 0, torch.zeros_like(evc))
    evc = evc + 1e-8
    s = evc.log()

    edge_index = data.edge_index
    s_row, s_col = s[edge_index[0]], s[edge_index[1]]
    s = s_col

    return (s.max() - s) / (s.max() - s.mean())



def community_detection(name):
    algs = {
        # non-overlapping algorithms
        'louvain': algorithms.louvain,
        'combo': algorithms.pycombo,
        'leiden': algorithms.leiden,
        'ilouvain': algorithms.ilouvain,
        #'edmot': algorithms.edmot,
        'eigenvector': algorithms.eigenvector,
        'girvan_newman': algorithms.girvan_newman,
        # overlapping algorithms
        'demon': algorithms.demon,
        'lemon': algorithms.lemon,
        #'ego-splitting': algorithms.egonet_splitter,
        #'nnsed': algorithms.nnsed,
        'lpanni': algorithms.lpanni,
    }
    return algs[name]


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


def community_strength(graph: nx.Graph,
                       communities) -> (np.ndarray, np.ndarray):
    graph = convert_graph_formats(graph, nx.Graph)
    coms = {}
    for cid, com in enumerate(communities):
        for node in com:
            coms[node] = cid
    inc, deg = {}, {}
    links = graph.size(weight="weight")
    assert links > 0, "A graph without link has no communities."
    for node in graph:
        try:
            com = coms[node]
            deg[com] = deg.get(com, 0.0) + graph.degree(node, weight="weight")
            for neighbor, dt in graph[node].items():
                weight = dt.get("weight", 1)
                if coms[neighbor] == com:
                    if neighbor == node:
                        inc[com] = inc.get(com, 0.0) + float(weight)
                    else:
                        inc[com] = inc.get(com, 0.0) + float(weight) / 2.0
        except:
            pass
    com_cs = []
    for idx, com in enumerate(set(coms.values())):
        com_cs.append((inc.get(com, 0.0) / links) - (deg.get(com, 0.0) / (2.0 * links)) ** 2)
    com_cs = np.asarray(com_cs)
    node_cs = np.zeros(graph.number_of_nodes(), dtype=np.float32)
    for i, w in enumerate(com_cs):
        for j in communities[i]:
            node_cs[j] = com_cs[i]
    return com_cs, node_cs


# generator of graph
def gen_sbm(sizes, probs, device=None, seed=random.randint(1, 10000), draw=False):
    G = stochastic_block_model(sizes, probs, seed=seed)
    G.remove_edges_from(nx.selfloop_edges(G)) # remove self loops
    if draw:
        nx.draw(G)
    #plt.savefig(f'{random.randint(0, 10000)}sbm.png')
    data = from_networkx(G)
    data.num_nodes = sum(sizes)
    data.sizes = sizes
    data.probs = probs
    data.num_features = data.num_nodes
    data.x = F.one_hot(torch.arange(0, sum(sizes))).float()
    if device is not None:
        data = data.to(device)
    return data
