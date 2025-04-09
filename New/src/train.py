import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling, to_networkx
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
          loss_compute: callable,
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
    return _train(encoder, predictor, data, split_edge, optimizer, hp, loss_compute)

def baseline_train(encoder: nn.Module,
          predictor: nn.Module,
          data,
          split_edge: dict,
          loss_compute: callable,
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
    return _train(encoder, predictor, data, split_edge, optimizer, hp, loss_compute)


def _train(encoder, predictor, data, split_edge, optimizer, hp, loss_compute):
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
            loss = loss_compute(pos_outs, neg_outs)
            loss.backward()
            optimizer.step()
            total_loss.append(loss)
        if epoch % 10 == 0:
            _loss = np.average([_.item() for _ in total_loss])
            loss_res.append(round(float(_loss), 2))
    print('train loss: ', loss_res)
    print(f"train time: {time.time()-t1:.2f} s")
    return total_loss

def ncn_loss(pos_outs, neg_outs):
    pos_losss = -F.logsigmoid(pos_outs).mean()
    neg_losss = -F.logsigmoid(-neg_outs).mean()
    return neg_losss + pos_losss

def bce_loss(pos_out, neg_out):
    out = torch.cat((pos_out, neg_out), dim=-1).to(pos_out.device())
    label = torch.cat((torch.ones(pos_out.size()[1]), torch.zeros(neg_out.size()[1])), dim=0).to(pos_out.device())
    return F.binary_cross_entropy_with_logits(out, label, reduction="mean")



def pretrain(model_name, model, aug, param):
    switch = {"grace": pretrain_grace,
              "agrace": pretrain_agrace,
              "cgrace": pretrain_cgrace,
              "csgcl": pretrain_csgcl,
              "bgrl": pretrain_bgrl,
              "bgrl2": pretrain_bgrl_2}
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

def pretrain_agrace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        weight_decay=param['weight_decay']
    )
    A_hat = aug.data.adj_t.to_dense()+torch.eye(aug.data.x.shape[0]).to(aug.device)
    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
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

def pretrain_cgrace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        weight_decay=param['weight_decay']
    )
    G = to_networkx(aug.data, to_undirected=True)
    L = torch.tensor(nx.laplacian_matrix(G).toarray()).to(aug.device)
    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        loss = model.loss(z1, z2, L)
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

def pretrain_bgrl_2(model, aug, param):
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
        h1 = model(x_1, edge_index_1)
        h2 = model(x_2, edge_index_2)
        S = z1 @ z2.T
        A_hat = aug.data.adj_t.to_dense()+torch.eye(aug.data.x.shape[0]).to(aug.device)
        target = torch.tensor(A_hat).to(A_hat.device)
        loss = F.mse_loss(S, target)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res, ' s')
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time
