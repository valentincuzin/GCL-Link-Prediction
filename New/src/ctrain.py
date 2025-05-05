import time
from tqdm import tqdm
import torch
from torch_geometric.utils import negative_sampling, to_networkx, degree, to_edge_index, to_dense_adj, add_self_loops, remove_self_loops
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from torch.utils.tensorboard import SummaryWriter
import torch_sparse as spar
import numpy as np
import networkx as nx
from src.utils import CosineDecayScheduler
from src.utils import get_commu_strength

writer = SummaryWriter()
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
        aug.train()
        aug.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        loss = model.loss(z1, z2)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
        # valid part
        with torch.no_grad():
            model.eval()
            aug.eval()
            x_1, edge_index_1, x_2, edge_index_2 = aug()
            z1 = model(x_1, edge_index_1)
            z2 = model(x_2, edge_index_2)
            val_loss = model.loss(z1, z2)
        writer.add_scalars("grace", {'train':loss, 'val': val_loss}, epoch)
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
    return pre_time

def pretrain_lgrace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        weight_decay=param['weight_decay']
    )
    loss_res = []
    t1 = time.time()

    total_loss = []
    nb_jump = 0
    for epoch in tqdm(range(1, 1 + param["ct_epochs"])):
        model.train()
        aug.train()
        aug.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        edge_index_1 = edge_index_1.T
        edge_index_2 = edge_index_2.T
        eq = torch.eq(edge_index_1[:, None], edge_index_2[None, :]).all(dim=2)
        intersection_idx = torch.nonzero(eq)
        and_edge_index = edge_index_1[intersection_idx[:, 0]].T
        or_edge_index = torch.unique(torch.cat((edge_index_1, edge_index_2)), dim=0).T
        edge_index_1 = edge_index_1.T
        edge_index_2 = edge_index_2.T
        if and_edge_index.size(1) == 0:
            nb_jump += 1
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
        
        # valid part
        with torch.no_grad():
            model.eval()
            aug.eval()
            x_1, edge_index_1, x_2, edge_index_2 = aug()
            edge_index_1 = edge_index_1.T
            edge_index_2 = edge_index_2.T
            eq = torch.eq(edge_index_1[:, None], edge_index_2[None, :]).all(dim=2)
            intersection_idx = torch.nonzero(eq)
            and_edge_index = edge_index_1[intersection_idx[:, 0]].T
            or_edge_index = torch.unique(torch.cat((edge_index_1, edge_index_2)), dim=0).T
            edge_index_1 = edge_index_1.T
            edge_index_2 = edge_index_2.T
            if and_edge_index.size(1) == 0:
                nb_jump += 1
                continue
            neg_edge = negative_sampling(or_edge_index, num_neg_samples=and_edge_index.size(1))
            h1 = model(x_1, edge_index_1).to(aug.device)
            h2 = model(x_2, edge_index_2).to(aug.device)
            val_loss = model.loss(h1, h2, and_edge_index, neg_edge)
        writer.add_scalars("lgrace", {'train':loss, 'val': val_loss}, epoch)
    print('real epochs: ', param['ct_epochs']-nb_jump)
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
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
        aug.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        adj_t = SparseTensor.from_edge_index(edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
        adj_t = adj_t.to_symmetric().coalesce()
        A_hat = adj_t.to_dense()+torch.eye(aug.data.x.shape[0]).to(aug.device)
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        loss = model.loss(z1, z2, A_hat)
        writer.add_scalar("Loss/pretrain_agrace", loss, epoch)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
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
        aug.train()
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
        writer.add_scalar("Loss/pretrain_aor_grace", loss, epoch)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
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
        aug.train()
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
        writer.add_scalar("Loss/pretrain_and_grace", loss, epoch)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
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
        aug.train()
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
        writer.add_scalar("Loss/pretrain_exta_grace", loss, epoch)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
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
        aug.train()
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
        writer.add_scalar("Loss/pretrain_a2grace", loss, epoch)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
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
        aug.train()
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
    writer.flush()
    return pre_time


def pretrain_bgrl(model, aug, param):
    # optimizer
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=param['gnn_lr'], weight_decay=param['weight_decay'])

    # scheduler
    lr_scheduler = CosineDecayScheduler(param['gnn_lr'], int(param['ct_epochs']/10), param['ct_epochs'])
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, param['ct_epochs'])

    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        aug.train()

        lr = lr_scheduler.get(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
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
        # valid part
        with torch.no_grad():
            model.eval()
            aug.eval()
            x_1, edge_index_1, x_2, edge_index_2 = aug()
            z1, y2 = model.train_forward((x_1, edge_index_1), (x_2, edge_index_2))
            z2, y1 = model.train_forward((x_2, edge_index_2), (x_1, edge_index_1))

            val_loss = model.loss(z1, z2, y1, y2)
        writer.add_scalars("bgrl", {'train':loss, 'val': val_loss}, epoch)
    print('pretrain loss: ', loss_res, ' s')
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
    return pre_time

def pretrain_lbgrl(model, aug, param):
    # optimizer
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=param['gnn_lr'], weight_decay=param['weight_decay'])

    # scheduler
    lr_scheduler = CosineDecayScheduler(param['gnn_lr'],  int(param['ct_epochs']/10), param['ct_epochs'])
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, param['ct_epochs'])

    t1 = time.time()
    loss_res = []
    nb_jump = 0

    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        aug.train()

        lr = lr_scheduler.get(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        mm = 1 - mm_scheduler.get(epoch)


        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        edge_index_1 = edge_index_1.T
        edge_index_2 = edge_index_2.T
        eq = torch.eq(edge_index_1[:, None], edge_index_2[None, :]).all(dim=2)
        intersection_idx = torch.nonzero(eq)
        and_edge_index = edge_index_1[intersection_idx[:, 0]].T
        edge_index_1 = edge_index_1.T
        edge_index_2 = edge_index_2.T
        if and_edge_index.size(1) == 0:
            nb_jump += 1
            continue

        z1, y2 = model.train_forward((x_1, edge_index_1), (x_2, edge_index_2), and_edge_index)
        z2, y1 = model.train_forward((x_2, edge_index_2), (x_1, edge_index_1), and_edge_index)

        loss = model.loss(z1, z2, y1, y2)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
        # valid part
        with torch.no_grad():
            model.eval()
            aug.eval()
            x_1, edge_index_1, x_2, edge_index_2 = aug()
            edge_index_1 = edge_index_1.T
            edge_index_2 = edge_index_2.T
            eq = torch.eq(edge_index_1[:, None], edge_index_2[None, :]).all(dim=2)
            intersection_idx = torch.nonzero(eq)
            and_edge_index = edge_index_1[intersection_idx[:, 0]].T
            edge_index_1 = edge_index_1.T
            edge_index_2 = edge_index_2.T
            if and_edge_index.size(1) == 0:
                nb_jump += 1
                continue
            z1, y2 = model.train_forward((x_1, edge_index_1), (x_2, edge_index_2), and_edge_index)
            z2, y1 = model.train_forward((x_2, edge_index_2), (x_1, edge_index_1), and_edge_index)

            val_loss = model.loss(z1, z2, y1, y2)
        writer.add_scalars("lbgrl", {'train':loss, 'val': val_loss}, epoch)
    print('real epochs: ', param['ct_epochs']-nb_jump)
    print('pretrain loss: ', loss_res, ' s')
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
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
        aug.train()

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
        writer.add_scalar("Loss/pretrain_abgrl", loss, epoch)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res, ' s')
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
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
        aug.train()

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
        writer.add_scalar("Loss/pretrain_or_bgrl", loss, epoch)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res, ' s')
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
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
        aug.train()

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
        writer.add_scalar("Loss/pretrain_ext_bgrl", loss, epoch)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res, ' s')
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
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
        aug.train()

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
        writer.add_scalar("Loss/pretrain_a2bgrl", loss, epoch)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res, ' s')
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
    return pre_time
