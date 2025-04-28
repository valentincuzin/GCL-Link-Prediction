import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from cdlib import algorithms
from cdlib.utils import convert_graph_formats
from torch_geometric.utils import to_networkx, from_networkx, to_undirected
from networkx.generators.community import stochastic_block_model

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

def get_commu_strength(data):
    g = to_networkx(data, to_undirected=True)
    communities = community_detection('leiden')(g).communities
    com_cs, node_cs = community_strength(g, communities)
    return communities, com_cs, node_cs

def gen_sbm(sizes, probs):
    G = stochastic_block_model(sizes, probs)
    G.remove_edges_from(nx.selfloop_edges(G)) # remove self loops
    data = from_networkx(G)
    data.edge_index = to_undirected(data.edge_index)
    data.num_nodes = sum(sizes)
    data.sizes = sizes
    data.probs = probs
    data.num_features = data.num_nodes
    data.x = F.one_hot(torch.arange(0, data.num_nodes)).float()
    return data

class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps # augmentation de plus en plus grande
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2 # décroit de façon lisse et progressive.
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))

def store_res(test_res: dict[float], res_dict: dict[list[float]]):
    for key, result in test_res.items():
        if key in res_dict.keys():
            res_dict[key].append(result)
    return res_dict

def compute_table(res_dict: dict[str, list | float], name: str):
    # Compute the mean and std from a dict return tab and latex table
    new_tab = []
    for key, result in res_dict.items():
        if key == "test_pred":
            new_tab.append({"metrics": key, name: result})
        elif isinstance(result, list):
            result = np.array(result)
            unit = 100 if key != 'pretrain_time' else 1
            mean = round(unit * np.mean(result), 2)
            std = round(unit * np.std(result), 2)
            new_tab.append({"metrics": key, name: fr"{mean}$\pm${std}"})
    df = pd.DataFrame(data=new_tab)
    df.set_index('metrics')
    res_latex = df.to_latex(
        index=False, formatters={"name": str.upper}, float_format="{:.1f}".format
    )
    return df, res_latex

def full_output(full_res: list):
    # concat full result to one dataframe, then print with latex
    full_res = pd.concat(full_res, axis=1)
    full_res = full_res.loc[:, ~full_res.columns.duplicated()]
    full_res.set_index('metrics', inplace=True)
    full_latex = full_res.to_latex(
        index=True, formatters={"name": str.upper}, float_format="{:.1f}".format
    )
    return full_res, full_latex