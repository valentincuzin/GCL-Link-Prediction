import time
from tqdm import tqdm
import torch
from torch_geometric.utils import negative_sampling, to_networkx, degree, to_edge_index, to_dense_adj, add_self_loops, remove_self_loops
from torch_geometric.data import Data
from torch_sparse import SparseTensor
import torch_sparse as spar
import numpy as np
import networkx as nx
from src.utils import CosineDecayScheduler
from src.utils import get_commu_strength

### CONTRASTIVE FRAMEWORK ###

def pretrain(model_name, model, aug, param):
    switch = {"grace": pretrain_grace,
              "lgrace": pretrain_lgrace,
              "agrace": pretrain_agrace,
              "ândgrace":pretrain_and_grace,
              "âorgrace": pretrain_aor_grace,
              "extagrace": pretrain_extend_agrace,
              "a2grace": pretrain_a2grace,
              "csgcl": pretrain_csgcl,
              "bgrl": pretrain_bgrl,
              "lbgrl": pretrain_lbgrl,
              "abgrl": pretrain_abgrl,
              "âorbgrl": pretrain_or_bgrl,
              "extabgrl": pretrain_extend_abgrl,
              "a2bgrl": pretrain_a2bgrl,
              }
    return switch[model_name](model, aug, param)

def pretrain_grace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        weight_decay=param['weight_decay']
    )
    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        loss = model.loss(z1, z2)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time

def pretrain_lgrace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        weight_decay=param['weight_decay']
    )
    loss_res = []
    t1 = time.time()
    device = aug.data.adj_t.device()

    total_loss = []
    for epoch in tqdm(range(1, 1 + param["ct_epochs"])):
        model.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        edge_index_1, _ = add_self_loops(edge_index_1, num_nodes=aug.data.num_nodes)
        edge_index_2, _ = add_self_loops(edge_index_2, num_nodes=aug.data.num_nodes)
        a_index_set = {tuple(edge_index_1[:, i].tolist()) for i in range(edge_index_1.size(1))}
        b_index_set = {tuple(edge_index_2[:, i].tolist()) for i in range(edge_index_2.size(1))}
        common = list(a_index_set.intersection(b_index_set))
        union = list(a_index_set.union(b_index_set))
        and_edge_index = torch.tensor(common, device=aug.device).t()
        and_edge_index, _ = remove_self_loops(and_edge_index)
        or_edge_index =  torch.tensor(union, device=aug.device).t()
        or_edge_index, _ = remove_self_loops(or_edge_index)
        if and_edge_index.size(1) == 0:
            print()
            continue
        neg_edge = negative_sampling(or_edge_index, num_neg_samples=and_edge_index.size(1))
        h1 = model(x_1, edge_index_1).to(aug.device)
        h2 = model(x_2, edge_index_2).to(aug.device)
        loss = model.loss(h1, h2, and_edge_index, neg_edge)
        loss.backward()
        optimizer.step()
        total_loss.append(loss)
        if epoch % 10 == 0:
            _loss = np.average([_.item() for _ in total_loss])
            loss_res.append(round(float(_loss), 2))
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time

def pretrain_agrace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        weight_decay=param['weight_decay']
    )
    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        adj_t = SparseTensor.from_edge_index(edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
        adj_t = adj_t.to_symmetric().coalesce()
        A_hat = adj_t.to_dense()+torch.eye(aug.data.x.shape[0]).to(aug.device)
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        loss = model.loss(z1, z2, A_hat)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time

def pretrain_aor_grace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        weight_decay=param['weight_decay']
    )
    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        adj_1 = SparseTensor.from_edge_index(edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes)).to_torch_sparse_coo_tensor().coalesce()
        adj_2 = SparseTensor.from_edge_index(edge_index_2, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes)).to_torch_sparse_coo_tensor().coalesce()
        hat_indices, hat_values = spar.eye(aug.data.x.shape[0])
        A_indices = torch.cat([adj_1.indices(), adj_2.indices(), hat_indices.to(aug.device)], dim=-1)
        A_value = torch.cat([adj_1.values(), adj_2.values(), hat_values.to(aug.device)])
        A_hat = torch.sparse_coo_tensor(A_indices, A_value).coalesce()
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        loss = model.loss(z1, z2, A_hat)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time

def pretrain_and_grace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        weight_decay=param['weight_decay']
    )
    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        adj_1 = SparseTensor.from_edge_index(edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
        adj_1 = adj_1.to_symmetric().coalesce().to_torch_sparse_csr_tensor()
        adj_2 = SparseTensor.from_edge_index(edge_index_2, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
        adj_2 = adj_2.to_symmetric().coalesce().to_torch_sparse_csr_tensor()
        hat = torch.eye(aug.data.x.shape[0]).to_sparse().to(aug.device)
        A_hat = torch.sparse.mm(adj_1, adj_2)+hat
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        loss = model.loss(z1, z2, A_hat)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time

def pretrain_extend_agrace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        weight_decay=param['weight_decay']
    )
    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        data = Data(x_1, edge_index_1)
        G = to_networkx(data, to_undirected=True)
        distance_matrix = dict(nx.all_pairs_shortest_path_length(G))
        adj_matrix = np.zeros((len(G.nodes), len(G.nodes)))
        for source, distances in distance_matrix.items():
            for target, distance in distances.items():
                adj_matrix[source, target] = distance
        adj_matrix[adj_matrix!=0] = 1/adj_matrix[adj_matrix!=0]
        adj = torch.tensor(adj_matrix, dtype=torch.float).to(aug.device)
        adj += torch.eye(aug.data.x.shape[0]).to(aug.device)
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        loss = model.loss(z1, z2, adj)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time

def pretrain_a2grace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        weight_decay=param['weight_decay']
    )
    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        adj_t = SparseTensor.from_edge_index(edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
        adj_t = adj_t.to_symmetric().coalesce()
        A_hat_1 = adj_t.to_dense()+torch.eye(aug.data.x.shape[0]).to(aug.device)

        adj_t = SparseTensor.from_edge_index(edge_index_2, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
        adj_t = adj_t.to_symmetric().coalesce()
        A_hat_2 = adj_t.to_dense()+torch.eye(aug.data.x.shape[0]).to(aug.device)

        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        loss = model.loss(z1, z2, A_hat_1, A_hat_2)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time

def pretrain_csgcl(model, aug, param):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=param['gnn_lr'],
                                 weight_decay=param['weight_decay'])
    t1 = time.time()
    loss_res = []
    _, _, node_cs = get_commu_strength(aug.data)
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        loss = model.team_up_loss(z1, z2,
                                  cs=node_cs,
                                  current_ep=epoch)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res, ' s')
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time


def pretrain_bgrl(model, aug, param):
    # optimizer
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=param['gnn_lr'], weight_decay=param['weight_decay'])

    # scheduler
    lr_scheduler = CosineDecayScheduler(param['gnn_lr'], 1000, param['ct_epochs'])
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, param['ct_epochs'])

    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()

        lr = lr_scheduler.get(epoch)
        mm = 1 - mm_scheduler.get(epoch)


        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        z1, y2 = model.train_forward((x_1, edge_index_1), (x_2, edge_index_2))
        z2, y1 = model.train_forward((x_2, edge_index_2), (x_1, edge_index_1))

        loss = model.loss(z1, z2, y1, y2)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res, ' s')
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time

def pretrain_lbgrl(model, aug, param):
    # optimizer
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=param['gnn_lr'], weight_decay=param['weight_decay'])

    # scheduler
    lr_scheduler = CosineDecayScheduler(param['gnn_lr'], 1000, param['ct_epochs'])
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, param['ct_epochs'])

    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()

        lr = lr_scheduler.get(epoch)
        mm = 1 - mm_scheduler.get(epoch)


        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        edge_index_1, _ = add_self_loops(edge_index_1, num_nodes=aug.data.num_nodes)
        edge_index_2, _ = add_self_loops(edge_index_2, num_nodes=aug.data.num_nodes)
        a_index_set = {tuple(edge_index_1[:, i].tolist()) for i in range(edge_index_1.size(1))}
        b_index_set = {tuple(edge_index_2[:, i].tolist()) for i in range(edge_index_2.size(1))}
        common = list(a_index_set.intersection(b_index_set))
        and_edge_index = torch.tensor(common, device=aug.device).t()
        and_edge_index, _ = remove_self_loops(and_edge_index)
        if and_edge_index.size(1) == 0:
            print()
            continue

        z1, y2 = model.train_forward((x_1, edge_index_1), (x_2, edge_index_2), and_edge_index)
        z2, y1 = model.train_forward((x_2, edge_index_2), (x_1, edge_index_1), and_edge_index)

        loss = model.loss(z1, z2, y1, y2)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res, ' s')
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time


def pretrain_abgrl(model, aug, param):
    # optimizer
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=param['gnn_lr'], weight_decay=param['weight_decay'])

    # scheduler
    lr_scheduler = CosineDecayScheduler(param['gnn_lr'], 1000, param['ct_epochs'])
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, param['ct_epochs'])

    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()

        lr = lr_scheduler.get(epoch)
        mm = 1 - mm_scheduler.get(epoch)


        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        adj_t = SparseTensor.from_edge_index(edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
        adj_t = adj_t.to_symmetric().coalesce()
        A_hat = adj_t.to_dense()+torch.eye(aug.data.x.shape[0]).to(aug.device)

        z1, y2 = model.train_forward((x_1, edge_index_1), (x_2, edge_index_2))
        z2, y1 = model.train_forward((x_2, edge_index_2), (x_1, edge_index_1))

        loss = model.loss(z1, z2, y1, y2, A_hat)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res, ' s')
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time

def pretrain_or_bgrl(model, aug, param):
    # optimizer
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=param['gnn_lr'], weight_decay=param['weight_decay'])

    # scheduler
    lr_scheduler = CosineDecayScheduler(param['gnn_lr'], 1000, param['ct_epochs'])
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, param['ct_epochs'])

    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()

        lr = lr_scheduler.get(epoch)
        mm = 1 - mm_scheduler.get(epoch)


        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        adj_1 = SparseTensor.from_edge_index(edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes)).to_torch_sparse_coo_tensor().coalesce()
        adj_2 = SparseTensor.from_edge_index(edge_index_2, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes)).to_torch_sparse_coo_tensor().coalesce()
        hat_indices, hat_values = spar.eye(aug.data.x.shape[0])
        A_indices = torch.cat([adj_1.indices(), adj_2.indices(), hat_indices.to(aug.device)], dim=-1)
        A_value = torch.cat([adj_1.values(), adj_2.values(), hat_values.to(aug.device)])
        A_hat = torch.sparse_coo_tensor(A_indices, A_value).coalesce()

        z1, y2 = model.train_forward((x_1, edge_index_1), (x_2, edge_index_2))
        z2, y1 = model.train_forward((x_2, edge_index_2), (x_1, edge_index_1))

        loss = model.loss(z1, z2, y1, y2, A_hat)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res, ' s')
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time

def pretrain_extend_abgrl(model, aug, param):
    # optimizer
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=param['gnn_lr'], weight_decay=param['weight_decay'])

    # scheduler
    lr_scheduler = CosineDecayScheduler(param['gnn_lr'], 1000, param['ct_epochs'])
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, param['ct_epochs'])

    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()

        lr = lr_scheduler.get(epoch)
        mm = 1 - mm_scheduler.get(epoch)


        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        data = Data(x_1, edge_index_1)
        G = to_networkx(data, to_undirected=True)
        distance_matrix = dict(nx.all_pairs_shortest_path_length(G))
        adj_matrix = np.zeros((len(G.nodes), len(G.nodes)))
        for source, distances in distance_matrix.items():
            for target, distance in distances.items():
                adj_matrix[source, target] = distance
        adj_matrix[adj_matrix!=0] = 1/adj_matrix[adj_matrix!=0]
        adj = torch.tensor(adj_matrix, dtype=torch.float).to(aug.device)
        adj += torch.eye(aug.data.x.shape[0]).to(aug.device)

        z1, y2 = model.train_forward((x_1, edge_index_1), (x_2, edge_index_2))
        z2, y1 = model.train_forward((x_2, edge_index_2), (x_1, edge_index_1))

        loss = model.loss(z1, z2, y1, y2, adj)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res, ' s')
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time

def pretrain_a2bgrl(model, aug, param):
    # optimizer
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=param['gnn_lr'], weight_decay=param['weight_decay'])

    # scheduler
    lr_scheduler = CosineDecayScheduler(param['gnn_lr'], 1000, param['ct_epochs'])
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, param['ct_epochs'])

    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()

        lr = lr_scheduler.get(epoch)
        mm = 1 - mm_scheduler.get(epoch)


        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        adj_t = SparseTensor.from_edge_index(edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
        adj_t = adj_t.to_symmetric().coalesce()
        A_hat_1 = adj_t.to_dense()+torch.eye(aug.data.x.shape[0]).to(aug.device)

        adj_t = SparseTensor.from_edge_index(edge_index_2, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
        adj_t = adj_t.to_symmetric().coalesce()
        A_hat_2 = adj_t.to_dense()+torch.eye(aug.data.x.shape[0]).to(aug.device)

        z1, y2 = model.train_forward((x_1, edge_index_1), (x_2, edge_index_2))
        z2, y1 = model.train_forward((x_2, edge_index_2), (x_1, edge_index_1))

        loss = model.loss(z1, z2, y1, y2, A_hat_1, A_hat_2)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res, ' s')
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time
