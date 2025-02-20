from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling, dropout_edge
from torch_geometric.transforms import LineGraph
import torch
import numpy as np
import random

from torch_geometric.datasets import AttributedGraphDataset
from torch_geometric.utils import degree, to_undirected
from pGRACE.model import Encoder, GRACE
from pGRACE.functional import degree_drop_weights, feature_drop_weights
from pGRACE.utils import get_base_model, get_activation, generate_split
from train import train, test


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
    print(len(neg_edge_index))
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
    "num_epochs": 201,
    "weight_decay": 1e-5,
    "drop_scheme": "degree",
}
torch.manual_seed(12345)
random.seed(12345)

split = generate_split(
    LG.num_nodes, train_ratio=0.1, val_ratio=0.1
)  # generic train test split

encoder = Encoder(
    dataset.num_features,
    param["num_hidden"],
    get_activation(param["activation"]),
    base_model=get_base_model(param["base_model"]),
    k=param["num_layers"],
)
model = GRACE(
    encoder, param["num_hidden"], param["num_proj_hidden"], param["tau"]
)  # init the model
optimizer = torch.optim.Adam(
    model.parameters(), lr=param["learning_rate"], weight_decay=param["weight_decay"]
)
drop_weights = degree_drop_weights(LG.edge_index)
edge_index_ = to_undirected(LG.edge_index)
node_deg = degree(edge_index_[1])
feature_weights = feature_drop_weights(LG.x, node_c=node_deg)

for epoch in range(1, param["num_epochs"] + 1):
    loss = train(model, optimizer, LG, drop_weights, feature_weights, param)
    if epoch % 1 == 0:
        print(f"(T) | Epoch={epoch:03d}, loss={loss:.4f}")

    if epoch % 100 == 0:
        acc = test(model, LG, dataset)
        print(f"(E) | Epoch={epoch:04d}, avg_acc = {acc}")

acc = test(model, LG, dataset, final=True)
print(f"{acc}")
