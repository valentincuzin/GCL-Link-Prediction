from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, dropout_edge
from torch_geometric.transforms import LineGraph
import torch
import numpy as np
import random
import nni

from pGRACE.model import Encoder, GRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from pGRACE.dataset import get_dataset
from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.utils import degree, to_undirected
from pGRACE.functional import degree_drop_weights, feature_drop_weights


attr_graph_dataset = AttributedGraphDataset("./dataset", "Cora")
cora = attr_graph_dataset[0]
print(cora)
# nx.draw(corax, node_size=30)

G = cora
dataset = attr_graph_dataset


def to_linegraph(graph: Data):
    graph = graph.clone()
    num_edges = graph.num_edges
    # on construit le line graph avec 50% de negative edge et 50% de positive edge
    # on essaie de garder le nombre total de edge identique
    neg_edge_index = negative_sampling(graph.edge_index, num_neg_samples=int(num_edges/2))
    pos_edge_index, _ = dropout_edge(graph.edge_index)
    graph.edge_index = torch.cat((pos_edge_index, neg_edge_index), -1)
    num_edges = graph.num_edges
    src_nodes = graph.edge_index[0]
    target_nodes = graph.edge_index[1]
    edge_attr = (
        graph.x[src_nodes] + graph.x[target_nodes]
    )  # aggregation des attributs de noeuds
    graph.edge_attr = edge_attr
    linegraph = LineGraph()(graph)
    linegraph.y = torch.tensor(
        np.concatenate((np.ones(int(num_edges/2)), np.zeros(int(num_edges/2))))
    )  # label la moitier son vrai, l'autre neg
    print(linegraph)
    return linegraph

LG = to_linegraph(G)
param = {
    "learning_rate": 0.01,
    "num_hidden": 256,
    "num_proj_hidden": 32,
    "activation": "prelu",
    "base_model": "GCNConv",
    "num_layers": 2,
    "drop_edge_rate_1": 0.3,
    "drop_edge_rate_2": 0.4,
    "drop_feature_rate_1": 0.1,
    "drop_feature_rate_2": 0.0,
    "tau": 0.4,
    "num_epochs": 3000,
    "weight_decay": 1e-5,
    "drop_scheme": "degree",
}
device = 'cuda:0'
torch.manual_seed(12345)
torch.device(device)
random.seed(12345)

data = LG.to(device)

split = generate_split(
    LG.num_nodes, train_ratio=0.1, val_ratio=0.1
)  # generic train test split

encoder = Encoder(
    dataset.num_features,
    param["num_hidden"],
    get_activation(param["activation"]),
    base_model=get_base_model(param["base_model"]),
    k=param["num_layers"],
).to(device)
model = GRACE(
    encoder, param["num_hidden"], param["num_proj_hidden"], param["tau"]
).to(device)  # init the model
optimizer = torch.optim.Adam(
    model.parameters(), lr=param["learning_rate"], weight_decay=param["weight_decay"]
)


def train():
    model.train()
    optimizer.zero_grad()
    drop_weights = degree_drop_weights(G.edge_index).to(device)
    edge_index_ = to_undirected(G.edge_index)
    node_deg = degree(edge_index_[1])
    feature_weights = feature_drop_weights(G.x, node_c=node_deg).to(device)
    def drop_edge(idx: int):
        
        if param['drop_scheme'] == 'uniform':
            return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')

    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)
    x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
    x_2 = drop_feature(data.x, param['drop_feature_rate_2'])

    if param['drop_scheme'] in ['pr', 'degree', 'evc']:
        x_1 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted_2(data.x, feature_weights, param['drop_feature_rate_2'])

    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2)
    loss.backward()
    optimizer.step()

    return loss.item()

# data_test = to_complete_linegraph(G).to(device)

def test(final=False):
    model.eval()
    
    z = model(data.x, data.edge_index)

    evaluator = MulticlassEvaluator()
    acc = log_regression(z, data, evaluator, split='rand:0.1', num_epochs=3000, preload_split=split)['acc']

    if final:
        nni.report_final_result(acc)
    else:
        nni.report_intermediate_result(acc)

    return acc

for epoch in range(1, param["num_epochs"] + 1):
    loss = train()
    if epoch % 1 == 0:
        print(f"(T) | Epoch={epoch:03d}, loss={loss:.4f}")

    if epoch % 100 == 0:
        acc = test()
        print(f"(E) | Epoch={epoch:04d}, avg_acc = {acc}")

acc = test()
print(f"{acc}")
