import numpy as np
import pandas as pd
import networkx as nx
import torch
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

import torch_sparse
import torch.nn as nn
import torch.nn.functional as F
from cdlib import algorithms
from cdlib.utils import convert_graph_formats
from torch_geometric.utils import (
    to_networkx,
    from_networkx,
    to_undirected,
    remove_self_loops,
)
from torch_geometric.data import Data
from networkx.generators.community import stochastic_block_model
from graph_tool.generation import (
    generate_sbm,
    generate_maxent_sbm,
    solve_sbm_fugacities,
)
from graph_tool.inference import BlockState
import graph_tool.all as gt


def community_detection(name):
    algs = {
        # non-overlapping algorithms
        "louvain": algorithms.louvain,
        "leiden": algorithms.leiden,
        "combo": algorithms.pycombo,
        "infomap": algorithms.infomap,
        # overlapping algorithms
        # 'demon': algorithms.demon,
        # 'lemon': algorithms.lemon,
    }
    return algs[name]


def commu_repartition(data, cd_algo: str = None):
    data.edge_index = to_undirected(removerepeated(data.edge_index))
    if hasattr(data, "probs") and hasattr(data, "sizes") and cd_algo is None:
        print("already communities")
        return data
    if cd_algo == None:
        cd_algo = "louvain"
    G = to_networkx(data, to_undirected=True)
    communities = community_detection(cd_algo)(G).communities
    probs = np.zeros((len(communities), len(communities)))
    sizes = []
    block = np.empty((len(G.nodes)))

    # fill block list and com attr
    for idx, c in enumerate(communities):
        sizes.append(len(c))
        for n in c:
            block[n] = idx
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
    data.block = block
    data.communities = communities
    data.probs = probs
    # print("probs, ", probs)
    data.sizes = sizes
    # print("sizes, ", sizes)
    return data

def removerepeated(ei):
    ei = to_undirected(ei)

    ei = ei[:, ei[0] < ei[1]]
    return ei

def average_precision(y_pred_pos, y_pred_neg):
    if isinstance(y_pred_pos, torch.Tensor):
        y_pred_pos_np = y_pred_pos.cpu().numpy()
        y_pred_neg_np = y_pred_neg.cpu().numpy()
    else:
        y_pred_pos_np = y_pred_pos
        y_pred_neg_np = y_pred_neg

    y_true = np.concatenate(
        [np.ones(len(y_pred_pos_np)), np.zeros(len(y_pred_neg_np))]
    ).astype(np.int32)
    y_pred = np.concatenate([y_pred_pos_np, y_pred_neg_np])
    return average_precision_score(y_true, y_pred)


def community_strength(graph: nx.Graph, communities) -> (np.ndarray, np.ndarray):
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
        com_cs.append(
            (inc.get(com, 0.0) / links) - (deg.get(com, 0.0) / (2.0 * links)) ** 2
        )
    com_cs = np.asarray(com_cs)
    node_cs = np.zeros(graph.number_of_nodes(), dtype=np.float32)
    for i, w in enumerate(com_cs):
        for j in communities[i]:
            node_cs[j] = com_cs[i]
    return com_cs, node_cs


def get_commu_strength(data):
    g = to_networkx(data, to_undirected=True)
    communities = community_detection("leiden")(g).communities
    com_cs, node_cs = community_strength(g, communities)
    return communities, com_cs, node_cs


def gen_sbm(sizes, probs, block=None):
    # probs_2 = probs * (np.eye(probs.shape[0]) * 2)
    if block is not None:
        G = stochastic_block_model(sizes, probs, block)
    else:
        G = stochastic_block_model(sizes, probs)
    G.remove_edges_from(nx.selfloop_edges(G))  # remove self loops
    data = from_networkx(G)
    data.edge_index = to_undirected(data.edge_index)
    data.num_nodes = sum(sizes)
    data.sizes = sizes
    data.probs = probs
    data.num_features = data.num_nodes
    return data


def to_graph_tool(data):
    ei = to_undirected(data.edge_index)
    ei = ei[:, ei[0] < ei[1]]

    G_gt = gt.Graph(directed=False)

    node_map = G_gt.new_vertex_property(
        "int"
    )
    block_map = G_gt.new_vertex_property(
        "int"
    )
    for i in range(data.num_nodes):
        v = G_gt.add_vertex()
        node_map[v] = (
            i
        )
        block_map[v] = data.block[i]

    for edge in ei.t().tolist():
        G_gt.add_edge(G_gt.vertex(node_map[edge[0]]), G_gt.vertex(node_map[edge[1]]))

    return G_gt, block_map


def gen_sbm_fast(data):
    gtG = generate_sbm(data.block, data.probs)
    edge_index = torch.from_numpy(gtG.get_edges().T)
    new_data = Data(edge_index=edge_index).to(data.edge_index.device)
    new_data.edge_index = to_undirected(remove_self_loops(new_data.edge_index)[0])
    new_data.num_nodes = sum(data.sizes)
    new_data.sizes = data.sizes
    new_data.probs = data.probs
    new_data.num_features = data.num_nodes
    return new_data

def gen_sbm_fast_2(data, state):
    gtG = state.sample_graph(self_loops=False, multigraph=False)
    edge_index = torch.from_numpy(gtG.get_edges().T)
    new_data = Data(edge_index=edge_index).to(data.edge_index.device)
    new_data.edge_index = to_undirected(new_data.edge_index)
    new_data.num_nodes = sum(data.sizes)
    new_data.sizes = data.sizes
    new_data.probs = data.probs
    new_data.num_features = data.num_nodes
    return new_data


def gen_sgf(data, alpha, transformation="identity"):
    G = to_networkx(data, to_undirected=True)
    new_G = nx.spectral_graph_forge(G, alpha, transformation)
    data = from_networkx(new_G)
    data.edge_index = to_undirected(data.edge_index)
    data.num_nodes = len(G.nodes)
    data.num_features = data.num_nodes
    data.x = F.one_hot(torch.arange(0, data.num_nodes)).float()
    return data


def gen_ba(data):
    G = to_networkx(data, to_undirected=True)
    new_G = nx.barabasi_albert_graph(data.num_nodes, 1, initial_graph=G)
    data = from_networkx(new_G)
    data.edge_index = to_undirected(data.edge_index)
    data.num_nodes = len(G.nodes)
    data.num_features = data.num_nodes
    data.x = F.one_hot(torch.arange(0, data.num_nodes)).float()
    return data


def gen_deg(data):
    G = to_networkx(data, to_undirected=True)
    deg = [d for v, d in nx.degree(G)]
    new_G = nx.expected_degree_graph(deg, False)
    data = from_networkx(new_G)
    data.edge_index = to_undirected(data.edge_index)
    data.num_nodes = len(G.nodes)
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
            return (
                self.max_val * step / self.warmup_steps
            )  # augmentation de plus en plus grande
        elif self.warmup_steps <= step <= self.total_steps:
            return (
                self.max_val
                * (
                    1
                    + np.cos(
                        (step - self.warmup_steps)
                        * np.pi
                        / (self.total_steps - self.warmup_steps)
                    )
                )
                / 2
            )  # décroit de façon lisse et progressive.
        else:
            raise ValueError(
                "Step ({}) > total number of steps ({}).".format(step, self.total_steps)
            )


# Edge dropout with adjacency matrix as input
class DropAdj(nn.Module):
    doscale: bool  # whether to rescale edge weight

    def __init__(self, dp: float = 0.0, doscale=True) -> None:
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1 / (1 - dp)))
        self.doscale = doscale

    def forward(self, adj):
        if self.dp < 1e-6 or not self.training:
            return adj
        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = torch_sparse.masked_select_nnz(adj, mask, layout="coo")
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value() * self.ratio, layout="coo")
            else:
                adj.fill_value_(1 / (1 - self.dp), dtype=torch.float)
        return adj


# Edge dropout
class DropEdge(nn.Module):
    def __init__(self, dp: float = 0.0) -> None:
        super().__init__()
        self.dp = dp

    def forward(self, edge_index):
        if self.dp == 0:
            return edge_index
        mask = torch.rand_like(edge_index[0], dtype=torch.float) > self.dp
        return edge_index[:, mask]


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
            unit = 100 if key != "pretrain_time" else 1
            mean = round(unit * np.mean(result), 2)
            std = round(unit * np.std(result), 2)
            new_tab.append(
                {"metrics": key, name + "_mean": rf"{mean}$\pm${std}", name: result}
            )
    df = pd.DataFrame(data=new_tab)
    df.set_index("metrics")
    res_latex = df.to_latex(
        index=False, formatters={"name": str.upper}, float_format="{:.1f}".format
    )
    return df, res_latex


def full_output(full_res: list):
    # concat full result to one dataframe, then print with latex
    full_res = pd.concat(full_res, axis=1)
    full_res = full_res.loc[:, ~full_res.columns.duplicated()]
    full_res.set_index("metrics", inplace=True)
    full_latex = full_res.to_latex(
        index=True, formatters={"name": str.upper}, float_format="{:.1f}".format
    )
    return full_res, full_latex


def visu_tsne(h, partition=None, name=None):  # TODO color by groups
    tsne = TSNE(n_components=2, verbose=1, random_state=0, perplexity=40, n_iter=300)
    h = h.cpu()
    tsne_results = tsne.fit_transform(h)
    df = pd.DataFrame()
    df["tsne-2d-one"] = tsne_results[:, 0]
    df["tsne-2d-two"] = tsne_results[:, 1]
    labels = [-1] * len(df)
    if partition is not None:
        for group_id, group_nodes in enumerate(partition):
            for node in group_nodes:
                if node < len(labels):
                    labels[node] = group_id
                else:
                    print(f"index {node} doesn't exist in h.")
    else:
        partition = []
    df["group"] = labels
    markers_list = ["o", "s", "D", "v", "^", "<", ">", "P", "X", "*"]
    df["marker"] = df["group"].apply(lambda g: markers_list[g % 10] if g >= 0 else "o")
    fig, ax = plt.subplots(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="group" if len(partition) > 0 else None,
        style="marker",
        palette=sns.color_palette("hls", len(partition)),
        data=df,
        legend=False,
        alpha=0.5,
        ax=ax,
    )
    if name is not None:
        ax.set_title(name)
        plt.savefig(name, bbox_inches="tight", dpi=300)
    plt.show()
