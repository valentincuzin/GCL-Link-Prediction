import time
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import negative_sampling, to_networkx
from torch_geometric.data import DataLoader, Data
import torch_sparse as spar
import networkx as nx
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from src.predictor import InnerProd
from src.utils import CosineDecayScheduler, get_commu_strength, average_precision, commu_repartition, visu_tsne

writer = SummaryWriter()


def valid_ap(model, data, split_edge, param):
    if isinstance(model, torch.nn.Module):
        model.eval()
    device = data.adj_t.device()
    adj_t = data.adj_t
    h = None if model is None else model(data.x, adj_t)
    predictor = InnerProd()
    def test_split(split):
        # pred positive edges and negatives edges for nodes in the split
        pos_test_edge = split_edge[split]['edge'].to(device)
        neg_test_edge = split_edge[split]['edge_neg'].to(device)
        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), param['batch_size']):
            edge = pos_test_edge[perm].t()
            out = predictor.predict(h, edge[0], edge[1])
            pos_test_preds += [out.squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)
        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), param['batch_size']):
            edge = neg_test_edge[perm].t()
            out = predictor.predict(h, edge[0], edge[1])
            neg_test_preds += [out.squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)
        return pos_test_pred, neg_test_pred
    
    pos_valid_pred, neg_valid_pred = test_split('valid')
    return average_precision(pos_valid_pred, neg_valid_pred)


### CONTRASTIVE FRAMEWORK ###

def pretrain(model_name, model, aug, param):
    switch = {"grace": pretrain_grace,
              "lgrace": pretrain_lgrace,
              "csgcl": pretrain_csgcl,
              "bgrl": pretrain_bgrl,
              "lbgrl": pretrain_lbgrl,
            #   "agrace": pretrain_agrace,
            #   "ândgrace":pretrain_and_grace,
            #   "âorgrace": pretrain_aor_grace,
            #   "extagrace": pretrain_extend_agrace,
            #   "a2grace": pretrain_a2grace,
            #   "abgrl": pretrain_abgrl,
            #   "âorbgrl": pretrain_or_bgrl,
            #   "extabgrl": pretrain_extend_abgrl,
            #   "a2bgrl": pretrain_a2bgrl,
              }
    return switch[model_name](model, aug, param)

def pretrain_grace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        # weight_decay=param['weight_decay']
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
        # with torch.no_grad():
        #     model.eval()
        #     aug.eval()
        #     x_1, edge_index_1, x_2, edge_index_2 = aug()
        #     z1 = model(x_1, edge_index_1)
        #     z2 = model(x_2, edge_index_2)
        #     h = model(aug.data.x, aug.data.edge_index)
        #     val_loss = model.loss(z1, z2)
        #     val_ap = valid_ap(model, aug.data, aug.split_edge, param)
        # writer.add_scalars("grace", {'tr_loss':loss, 'val_loss': val_loss, 'val_ap': val_ap}, epoch)

        # if epoch % 10 == 0:
        #     name = f'gif/grace/{epoch}.png'
        #     if not hasattr(aug.data, 'communities'):
        #         data = commu_repartition(aug.data, 'louvain')
        #     else:
        #         data = aug.data
        #     visu_tsne(h, partition=data.communities, name=name)
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
    return pre_time

def pretrain_lgrace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        # weight_decay=param['weight_decay']
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
        if epoch % 100 == 0:
            _loss = np.average([_.item() for _ in total_loss])
            loss_res.append(round(float(_loss), 2))
        
        # valid part
        # with torch.no_grad():
        #     model.eval()
        #     aug.eval()
        #     x_1, edge_index_1, x_2, edge_index_2 = aug()
        #     edge_index_1 = edge_index_1.T
        #     edge_index_2 = edge_index_2.T
        #     eq = torch.eq(edge_index_1[:, None], edge_index_2[None, :]).all(dim=2)
        #     intersection_idx = torch.nonzero(eq)
        #     and_edge_index = edge_index_1[intersection_idx[:, 0]].T
        #     or_edge_index = torch.unique(torch.cat((edge_index_1, edge_index_2)), dim=0).T
        #     edge_index_1 = edge_index_1.T
        #     edge_index_2 = edge_index_2.T
        #     if and_edge_index.size(1) == 0:
        #         nb_jump += 1
        #         continue
        #     neg_edge = negative_sampling(or_edge_index, num_neg_samples=and_edge_index.size(1))
        #     h1 = model(x_1, edge_index_1).to(aug.device)
        #     h2 = model(x_2, edge_index_2).to(aug.device)
        #     h = model(aug.data.x, aug.data.edge_index)
        #     val_loss = model.loss(h1, h2, and_edge_index, neg_edge)
        #     val_ap = valid_ap(model, aug.data, aug.split_edge, param)
        # writer.add_scalars("lgrace", {'tr_loss':loss, 'val_loss': val_loss, 'val_ap': val_ap}, epoch)

        # if epoch % 10 == 0:
        #     name = f'gif/lgrace/{epoch}.png'
        #     if not hasattr(aug.data, 'communities'):
        #         data = commu_repartition(aug.data, 'louvain')
        #     else:
        #         data = aug.data
        #     visu_tsne(h, partition=data.communities, name=name)
    print('real epochs: ', param['ct_epochs']-nb_jump)
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
    return pre_time

def pretrain_csgcl(model, aug, param):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=param['gnn_lr'],
                                #  weight_decay=param['weight_decay']
                                 )
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
        # with torch.no_grad():
        #     model.eval()
        #     aug.eval()
        #     x_1, edge_index_1, x_2, edge_index_2 = aug()
        #     z1, y2 = model.train_forward((x_1, edge_index_1), (x_2, edge_index_2))
        #     z2, y1 = model.train_forward((x_2, edge_index_2), (x_1, edge_index_1))

        #     h = model(aug.data.x, aug.data.edge_index)
        #     val_loss = model.loss(z1, z2, y1, y2)
        #     val_ap = valid_ap(model, aug.data, aug.split_edge, param)
        # writer.add_scalars("bgrl", {'tr_loss':loss, 'val_loss': val_loss, 'val_ap': val_ap}, epoch)

        # if epoch % 10 == 0:         
        #     name = f'gif/bgrl/{epoch}.png'
        #     if not hasattr(aug.data, 'communities'):
        #         data = commu_repartition(aug.data, 'louvain')
        #     else:
        #         data = aug.data
        #     visu_tsne(h, partition=data.communities, name=name)
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
    # G = to_networkx(aug.data, to_undirected=True)
    # G.remove_nodes_from(list(nx.isolates(G)))
    # nx.draw(G, node_size=20)
    # nx.spring_layout(G)
    # plt.savefig(f'train_split.png')
    # plt.close('all')
    # and_edge_index_total = None
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        aug.train()

        lr = lr_scheduler.get(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        mm = 1 - mm_scheduler.get(epoch)


        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        
        # data1 = Data(x=x_1, edge_index=edge_index_1)
        # print(data1)
        # G1 = to_networkx(data1, to_undirected=True)
        # G1.remove_nodes_from(list(nx.isolates(G1)))
        # nx.draw(G1, node_size=20)
        # nx.spring_layout(G1)
        # plt.savefig(f'{epoch}__aug1.png')
        # plt.close('all')
        # data2 = Data(x=x_2, edge_index=edge_index_2)
        # print(data2)
        # G2 = to_networkx(data2, to_undirected=True)
        # G2.remove_nodes_from(list(nx.isolates(G2)))
        # nx.draw(G2, node_size=20)
        # nx.spring_layout(G2)
        # plt.savefig(f'{epoch}__aug2.png')
        # plt.close('all')
        

        edge_index_1 = edge_index_1.T
        edge_index_2 = edge_index_2.T
        eq = torch.eq(edge_index_1[:, None], edge_index_2[None, :]).all(dim=2)
        intersection_idx = torch.nonzero(eq)
        and_edge_index = edge_index_1[intersection_idx[:, 0]].T
        # and_edge_index_total = torch.cat((and_edge_index_total, and_edge_index), dim=1) if and_edge_index_total is not None else and_edge_index
        # data3 = Data(x=x_2, edge_index=and_edge_index)
        # print(data3)
        # G3 = to_networkx(data3, to_undirected=True)
        # G3.remove_nodes_from(list(nx.isolates(G3)))
        # nx.draw(G3, node_size=20)
        # nx.spring_layout(G3)
        # plt.savefig(f'{epoch}__aug_inter.png')
        # plt.close('all')
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
        # with torch.no_grad():
        #     model.eval()
        #     aug.eval()
        #     x_1, edge_index_1, x_2, edge_index_2 = aug()
        #     edge_index_1 = edge_index_1.T
        #     edge_index_2 = edge_index_2.T
        #     eq = torch.eq(edge_index_1[:, None], edge_index_2[None, :]).all(dim=2)
        #     intersection_idx = torch.nonzero(eq)
        #     and_edge_index = edge_index_1[intersection_idx[:, 0]].T
        #     edge_index_1 = edge_index_1.T
        #     edge_index_2 = edge_index_2.T
        #     if and_edge_index.size(1) == 0:
        #         nb_jump += 1
        #         continue
        #     z1, y2 = model.train_forward((x_1, edge_index_1), (x_2, edge_index_2), and_edge_index)
        #     z2, y1 = model.train_forward((x_2, edge_index_2), (x_1, edge_index_1), and_edge_index)

        #     h = model(aug.data.x, aug.data.edge_index)
        #     val_loss = model.loss(z1, z2, y1, y2)
        #     val_ap = valid_ap(model, aug.data, aug.split_edge, param)
        # writer.add_scalars("lbgrl", {'tr_loss':loss, 'val_loss': val_loss, 'val_ap': val_ap}, epoch)

        # print('before visu')
        # if epoch % 10 == 0:
        #     print('visu')
        #     name = f'gif/lbgrl/{epoch}.png'
        #     if not hasattr(aug.data, 'communities'):
        #         data = commu_repartition(aug.data, 'louvain')
        #     else:
        #         data = aug.data
        #     visu_tsne(h, partition=data.communities, name=name)
    # data4 = Data(x=x_2, edge_index=and_edge_index_total)
    # print(data4)
    # G4 = to_networkx(data4, to_undirected=True)
    # G4.remove_nodes_from(list(nx.isolates(G4)))
    # nx.draw(G4, node_size=20)
    # nx.spring_layout(G4)
    # plt.savefig(f'final__aug_inter.png')
    # plt.close('all')
    print('real epochs: ', param['ct_epochs']-nb_jump)
    print('pretrain loss: ', loss_res, ' s')
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    writer.flush()
    return pre_time

""" ### Â loss
def pretrain_agrace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        # weight_decay=param['weight_decay']
    )
    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        aug.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        adj_t = spar.SparseTensor.from_edge_index(edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
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
    writer.flush()
    return pre_time

def pretrain_aor_grace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        # weight_decay=param['weight_decay']
    )
    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        aug.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        adj_1 = spar.SparseTensor.from_edge_index(edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes)).to_torch_sparse_coo_tensor().coalesce()
        adj_2 = spar.SparseTensor.from_edge_index(edge_index_2, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes)).to_torch_sparse_coo_tensor().coalesce()
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
    writer.flush()
    return pre_time

def pretrain_and_grace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        # weight_decay=param['weight_decay']
    )
    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        aug.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        adj_1 = spar.SparseTensor.from_edge_index(edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
        adj_1 = adj_1.to_symmetric().coalesce().to_torch_sparse_csr_tensor()
        adj_2 = spar.SparseTensor.from_edge_index(edge_index_2, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
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
    writer.flush()
    return pre_time

def pretrain_extend_agrace(model, aug, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['gnn_lr'],
        # weight_decay=param['weight_decay']
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
        # weight_decay=param['weight_decay']
    )
    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['ct_epochs'] + 1)):
        model.train()
        aug.train()
        optimizer.zero_grad()
        x_1, edge_index_1, x_2, edge_index_2 = aug()
        adj_t = spar.SparseTensor.from_edge_index(edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
        adj_t = adj_t.to_symmetric().coalesce()
        A_hat_1 = adj_t.to_dense()+torch.eye(aug.data.x.shape[0]).to(aug.device)

        adj_t = spar.SparseTensor.from_edge_index(edge_index_2, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
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
        adj_t = spar.SparseTensor.from_edge_index(edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
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
        adj_1 = spar.SparseTensor.from_edge_index(edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes)).to_torch_sparse_coo_tensor().coalesce()
        adj_2 = spar.SparseTensor.from_edge_index(edge_index_2, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes)).to_torch_sparse_coo_tensor().coalesce()
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
        adj_t = spar.SparseTensor.from_edge_index(edge_index_1, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
        adj_t = adj_t.to_symmetric().coalesce()
        A_hat_1 = adj_t.to_dense()+torch.eye(aug.data.x.shape[0]).to(aug.device)

        adj_t = spar.SparseTensor.from_edge_index(edge_index_2, sparse_sizes=(aug.data.num_nodes, aug.data.num_nodes))
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
    writer.flush()
    return pre_time
"""
