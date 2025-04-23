from functools import partial
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, to_networkx, degree, to_edge_index, to_dense_adj, add_self_loops, remove_self_loops
from torch_geometric.data import Data
from torch_sparse import SparseTensor
import numpy as np
import networkx as nx
from src.utils import CosineDecayScheduler
from src.utils import get_commu_strength

@torch.no_grad()
def test(encoder: nn.Module, predictor: nn.Module, data, split_edge: dict, hp: dict):
    if isinstance(encoder, nn.Module):
        encoder.eval()
    if isinstance(predictor, nn.Module):
        predictor.eval()
    device = data.adj_t.device()
    adj_t = data.adj_t
    h = encoder(data.x, adj_t)
    def test_split(split):
        # pred positive edges and negatives edges for nodes in the split
        pos_test_edge = split_edge[split]['edge'].to(device)
        neg_test_edge = split_edge[split]['edge_neg'].to(device)
        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), hp['batch_size']):
            edge = pos_test_edge[perm].t()
            out = predictor.predict(h, edge[0], edge[1])
            pos_test_preds += [out.squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)
        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), hp['batch_size']):
            edge = neg_test_edge[perm].t()
            out = predictor.predict(h, edge[0], edge[1])
            neg_test_preds += [out.squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)
        return pos_test_pred, neg_test_pred

    pos_valid_pred, neg_valid_pred = test_split('valid')
    if hp['use_valedges_as_input']:
        adj_t = data.full_adj_t
        h = encoder(data.x, adj_t)
    pos_test_pred, neg_test_pred = test_split('test')
    return pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred

def pred_train(encoder: nn.Module,
          predictor: nn.Module,
          data,
          split_edge: dict,
          loss_name: str,
          hp: dict):
    if not hp['freeze']:
        optimizer = torch.optim.Adam(
            [
                {"params": encoder.parameters(), "lr": hp["gnn_lr"]},
                {"params": predictor.parameters(), "lr": hp["pre_lr"]},
            ]
        )
    elif hp['freeze']:
        optimizer = torch.optim.Adam(params=predictor.parameters(), lr=hp["pre_lr"])
    encoder.train()
    predictor.train()
    return _train(encoder, predictor, data, split_edge, optimizer, hp, loss_name)

def baseline_train(encoder: nn.Module,
          predictor: nn.Module,
          data,
          split_edge: dict,
          loss_name: str,
          hp: dict):
    if isinstance(predictor, nn.Module):
        predictor = predictor.to(data.x.device)
        optimizer = torch.optim.Adam(
            [
                {"params": encoder.parameters(), "lr": hp["gnn_lr"],  "weight_decay": hp['weight_decay']},
                {"params": predictor.parameters(), "lr": hp["pre_lr"]},
            ]
        )
        predictor.train()
    else:
        optimizer = torch.optim.Adam(params=encoder.parameters(), lr=hp["gnn_lr"], weight_decay=hp['weight_decay'])
    encoder.train()
    return _train(encoder, predictor, data, split_edge, optimizer, hp, loss_name)

def _train(encoder, predictor, data, split_edge, optimizer, hp, loss_name):
    loss_res = []
    t1 = time.time()
    device = data.adj_t.device()

    pos_train_edge = split_edge['train']['edge'].to(device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    for epoch in tqdm(range(1, 1 + hp["epochs"])):
        negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
        for perm in DataLoader(range(adjmask.size(0)), hp['batch_size'], shuffle=True):
            optimizer.zero_grad()
            if hp['mask_input']:
                adjmask[perm] = 0
                tei = pos_train_edge[:, adjmask]
                adj = SparseTensor.from_edge_index(tei,
                                sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                    pos_train_edge.device, non_blocking=True)
                adjmask[perm] = 1
                adj = adj.to_symmetric()
            else:
                adj = data.adj_t
            h = encoder(data.x, adj)
            edge = pos_train_edge[:, perm]
            pos_outs = predictor(h, edge[0], edge[1])

            edge = negedge[:, perm]
            neg_outs = predictor(h, edge[0], edge[1])
            loss = get_loss(loss_name, pos_outs, neg_outs)
            loss.backward()
            optimizer.step()
            total_loss.append(loss)
        if epoch % 10 == 0:
            _loss = np.average([_.item() for _ in total_loss])
            loss_res.append(round(float(_loss), 2))
    print('train loss: ', loss_res)
    print(f"train time: {time.time()-t1:.2f} s")
    return total_loss

def get_loss(loss_name: str, pos_outs, neg_outs):
    switch = {
        'log_sig': partial(log_sig_loss, pos_outs, neg_outs),
        'bce': partial(bce_loss, pos_outs, neg_outs),
        'auc': partial(auc_loss, pos_outs, neg_outs),
        'hinge_auc': partial(hinge_auc_loss, pos_outs, neg_outs),
    }
    return switch[loss_name]()

def log_sig_loss(pos_outs, neg_outs):
    pos_losss = -F.logsigmoid(pos_outs).mean()
    neg_losss = -F.logsigmoid(-neg_outs).mean()
    return neg_losss + pos_losss

def bce_loss(pos_out, neg_out):
    out = torch.cat((pos_out, neg_out), dim=0).to(pos_out.device)
    label = torch.cat((torch.ones(pos_out.size()), torch.zeros(neg_out.size())), dim=0).to(pos_out.device)
    return F.binary_cross_entropy_with_logits(out, label, reduction="mean")

def auc_loss(pos_out, neg_out):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, 1))
    return torch.square(1 - (pos_out - neg_out)).sum()

def hinge_auc_loss(pos_out, neg_out):
    pos_out = torch.reshape(pos_out, (-1, 1))
    neg_out = torch.reshape(neg_out, (-1, 1))
    return (torch.square(torch.clamp(1 - (pos_out - neg_out), min=0))).sum()








### CONTRASTIVE FRAMEWORK ###


def pretrain(model_name, model, aug, param):
    switch = {"grace": pretrain_grace,
              "lgrace": pretrain_lgrace,
              "agrace": pretrain_agrace,
              "ândgrace":pretrain_and_grace,
              "extagrace": pretrain_extend_agrace,
              "a2grace": pretrain_a2grace,
              "csgcl": pretrain_csgcl,
              "bgrl": pretrain_bgrl,
              "abgrl": pretrain_abgrl,
              "ândbgrl": pretrain_and_bgrl,
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
        adj_1 = to_dense_adj(edge_index_1)
        adj_2 = to_dense_adj(edge_index_2)
        adj_or = (adj_1 + adj_2).squeeze()
        adj_and = (adj_1 * adj_2).squeeze()
        and_edge_index = to_edge_index(adj_and.to_sparse())[0].to(aug.device)
        and_edge_index, _ = remove_self_loops(and_edge_index)
        or_edge_index = to_edge_index(adj_or.to_sparse())[0].to(aug.device)
        or_edge_index, _ = remove_self_loops(or_edge_index)
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
        A_hat = torch.sparse.mm(adj_1, adj_2)+torch.eye(aug.data.x.shape[0]).to_sparse_csr().to(aug.device)
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

def pretrain_and_bgrl(model, aug, param):
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
