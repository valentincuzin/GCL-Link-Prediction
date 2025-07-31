import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch_geometric.utils import negative_sampling
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import torch_sparse as spar
import numpy as np

from src.predictor import InnerProd
from src.utils import (
    CosineDecayScheduler,
    get_commu_strength,
)
from src.datasets import get_evaluator
from src.augmentation import Aug

def valid_hits50(model: nn.Module, data: Data, split_edge: dict, param: dict, split="valid") -> tuple[float, float]:
    """
    validation with Inner predictor, return Hit@50 score for valid and test

    Args:
        model (nn.Module): 
        data (Data): 
        split_edge (dict): 
        param (dict): 
        split (str, optional):. Defaults to "valid".

    Returns:
        tuple[float, float]: valid and test Hit@50 scores
    """
    if isinstance(model, nn.Module):
        model.eval()
    device = data.adj_t.device()
    adj_t = data.adj_t
    h = None if model is None else model(data.x, adj_t)
    predictor = InnerProd()

    def test_split(split):
        # pred positive edges and negatives edges for nodes in the split
        pos_test_edge = split_edge[split]["edge"].to(device)
        neg_test_edge = split_edge[split]["edge_neg"].to(device)
        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), param["batch_size"]):
            edge = pos_test_edge[perm].t()
            out = predictor.predict(h, edge[0], edge[1], data.adj_t)
            pos_test_preds += [out.squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)
        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), param["batch_size"]):
            edge = neg_test_edge[perm].t()
            out = predictor.predict(h, edge[0], edge[1], data.adj_t)
            neg_test_preds += [out.squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)
        return pos_test_pred, neg_test_pred

    pos_valid_pred, neg_valid_pred = test_split("valid")
    pos_test_pred, neg_test_pred = test_split("test")

    evaluator = get_evaluator()
    evaluator.eval_metric = "hits@k"
    evaluator.K = 50
    return evaluator.eval(
        {
            "y_pred_pos": pos_valid_pred,
            "y_pred_neg": neg_valid_pred,
        }
    )["hits@50"], evaluator.eval(
        {
            "y_pred_pos": pos_test_pred,
            "y_pred_neg": neg_test_pred,
        }
    )["hits@50"]


### CONTRASTIVE FRAMEWORK ###


def pretrain(model_name: str, model: nn.Module, aug: Aug, param: dict) -> float:
    """
    pretrain the model

    Args:
        model_name (str): _description_
        model (nn.Module): _description_
        aug (Aug): _description_
        param (dict): _description_

    Returns:
        float: the pretrain time
    """
    switch = {
        "grace": pretrain_grace,
        "lgrace": pretrain_lgrace,
        "csgcl": pretrain_csgcl,
        "bgrl": pretrain_bgrl,
        "lbgrl": pretrain_lbgrl,
        "agrace": pretrain_agrace,
        "abgrl": pretrain_abgrl,
    }
    return switch[model_name](model, aug, param)


def pretrain_grace(model: nn.Module, aug: Aug, param: dict)-> float:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param["gnn_lr"],
    )
    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param["ct_epochs"] + 1)):
        model.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        loss = model.loss(z1, z2)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            loss_res.append(round(float(loss), 2))

    print("pretrain loss: ", loss_res)
    pre_time = time.time() - t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time


def pretrain_csgcl(model: nn.Module, aug: Aug, param: dict)-> float:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param["gnn_lr"],
    )
    t1 = time.time()
    loss_res = []
    _, _, node_cs = get_commu_strength(aug.data)
    for epoch in tqdm(range(1, param["ct_epochs"] + 1)):
        model.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)
        loss = model.team_up_loss(z1, z2, cs=node_cs, current_ep=epoch)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            loss_res.append(round(float(loss), 2))

    print("pretrain loss: ", loss_res)
    pre_time = time.time() - t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time


def pretrain_bgrl(model: nn.Module, aug: Aug, param: dict)-> float:
    # optimizer
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=param["gnn_lr"],
        weight_decay=param["weight_decay"],
    )

    # scheduler
    lr_scheduler = CosineDecayScheduler(
        param["gnn_lr"], int(param["ct_epochs"] / 10), param["ct_epochs"]
    )
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, param["ct_epochs"])

    t1 = time.time()
    loss_res = []

    for epoch in tqdm(range(1, param["ct_epochs"] + 1)):
        model.train()

        lr = lr_scheduler.get(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        mm = 1 - mm_scheduler.get(epoch)

        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        z1, y2 = model.train_forward((x_1, edge_index_1), (x_2, edge_index_2))
        z2, y1 = model.train_forward((x_2, edge_index_2), (x_1, edge_index_1))

        loss = model.loss(z1, z2, y1, y2)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 50 == 0:
            loss_res.append(round(float(loss), 2))

    print("pretrain loss: ", loss_res)
    pre_time = time.time() - t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time


def pretrain_lgrace(model: nn.Module, aug: Aug, param: dict)-> float:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param["gnn_lr"],
    )
    loss_res = []
    t1 = time.time()

    total_loss = []
    nb_jump = 0
    for epoch in tqdm(range(1, 1 + param["ct_epochs"])):
        model.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        combined_edges = torch.cat([edge_index_1, edge_index_2], dim=1)
        unique_edges, counts = combined_edges.unique(dim=1, return_counts=True)
        intersection_mask = counts > 1
        and_edge_index = unique_edges[:, intersection_mask]
        if and_edge_index.size(1) == 0:
            nb_jump += 1
            continue
        neg_edge = negative_sampling(
            unique_edges, num_neg_samples=and_edge_index.size(1)
        )
        h1 = model(x_1, edge_index_1).to(aug.device)
        h2 = model(x_2, edge_index_2).to(aug.device)
        loss = model.loss(h1, h2, and_edge_index, neg_edge)
        loss.backward()
        optimizer.step()
        total_loss.append(loss)
        if epoch % 50 == 0:
            _loss = np.average([_.item() for _ in total_loss])
            loss_res.append(round(float(_loss), 2))

    print("real epochs: ", param["ct_epochs"] - nb_jump)
    print("pretrain loss: ", loss_res)
    pre_time = time.time() - t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time



def pretrain_lbgrl(model: nn.Module, aug: Aug, param: dict)-> float:
    # optimizer
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=param["gnn_lr"],
        weight_decay=param["weight_decay"],
    )

    # scheduler
    lr_scheduler = CosineDecayScheduler(
        param["gnn_lr"], int(param["ct_epochs"] / 10), param["ct_epochs"]
    )
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, param["ct_epochs"])

    t1 = time.time()
    loss_res = []
    nb_jump = 0

    for epoch in tqdm(range(1, param["ct_epochs"] + 1)):
        model.train()

        lr = lr_scheduler.get(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        mm = 1 - mm_scheduler.get(epoch)

        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        combined_edges = torch.cat([edge_index_1, edge_index_2], dim=1)
        unique_edges, counts = combined_edges.unique(dim=1, return_counts=True)
        intersection_mask = counts > 1
        and_edge_index = unique_edges[:, intersection_mask]
        if and_edge_index.size(1) == 0:
            nb_jump += 1
            continue

        z1, y2 = model.train_forward(
            (x_1, edge_index_1), (x_2, edge_index_2), and_edge_index
        )
        z2, y1 = model.train_forward(
            (x_2, edge_index_2), (x_1, edge_index_1), and_edge_index
        )

        loss = model.loss(z1, z2, y1, y2)
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 50 == 0:
            loss_res.append(round(float(loss), 2))

    print("real epochs: ", param["ct_epochs"] - nb_jump)
    print("pretrain loss: ", loss_res)
    pre_time = time.time() - t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time



def pretrain_agrace(model: nn.Module, aug: Aug, param: dict)-> float:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param["gnn_lr"],
    )
    t1 = time.time()
    loss_res = []

    for epoch in tqdm(range(1, param["ct_epochs"] + 1)):
        model.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        adj_t = spar.SparseTensor.from_edge_index(
            edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes)
        )
        adj_t = adj_t.to_symmetric().coalesce()
        A_hat_1 = adj_t.to_dense() + torch.eye(aug.data.x.shape[0]).to(aug.device)

        adj_t = spar.SparseTensor.from_edge_index(
            edge_index_2, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes)
        )
        adj_t = adj_t.to_symmetric().coalesce()
        A_hat_2 = adj_t.to_dense() + torch.eye(aug.data.x.shape[0]).to(aug.device)

        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        loss = model.loss(z1, z2, A_hat_1, A_hat_2)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            loss_res.append(round(float(loss), 2))


    print("pretrain loss: ", loss_res)
    pre_time = time.time() - t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time


def pretrain_abgrl(model: nn.Module, aug: Aug, param: dict)-> float:
    # optimizer
    optimizer = torch.optim.AdamW(
        model.trainable_parameters(),
        lr=param["gnn_lr"],
        weight_decay=param["weight_decay"],
    )

    # scheduler
    lr_scheduler = CosineDecayScheduler(
        param["gnn_lr"], int(param["ct_epochs"] / 10), param["ct_epochs"]
    )
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, param["ct_epochs"])

    t1 = time.time()
    loss_res = []

    for epoch in tqdm(range(1, param["ct_epochs"] + 1)):
        model.train()

        lr = lr_scheduler.get(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        mm = 1 - mm_scheduler.get(epoch)

        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        adj_t = spar.SparseTensor.from_edge_index(
            edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes)
        )
        adj_t = adj_t.to_symmetric().coalesce()
        A_hat_1 = adj_t.to_dense() + torch.eye(aug.data.x.shape[0]).to(aug.device)

        adj_t = spar.SparseTensor.from_edge_index(
            edge_index_2, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes)
        )
        adj_t = adj_t.to_symmetric().coalesce()
        A_hat_2 = adj_t.to_dense() + torch.eye(aug.data.x.shape[0]).to(aug.device)

        z1, y2 = model.train_forward((x_1, edge_index_1), (x_2, edge_index_2))
        z2, y1 = model.train_forward((x_2, edge_index_2), (x_1, edge_index_1))

        loss = model.loss(z1, z2, y1, y2, A_hat_1, A_hat_2)

        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 10 == 0:
            loss_res.append(round(float(loss), 2))

    print("pretrain loss: ", loss_res)
    pre_time = time.time() - t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time
