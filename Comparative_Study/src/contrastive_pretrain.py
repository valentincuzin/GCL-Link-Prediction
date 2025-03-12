import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import CosineDecayScheduler
from torch_geometric.utils import degree, to_undirected, to_networkx, dropout_adj
from contrastive_augmentation import drop_feature, drop_feature_weighted, drop_edge_weighted, feature_drop_weights, degree_drop_weights, pr_drop_weights, compute_pr, eigenvector_centrality, evc_drop_weights, cav, ced, community_detection, community_strength, transition, get_edge_weight

def pretrain_grace(model, data, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )
    t1 = time.time()
    for epoch in range(1, param['num_epochs'] + 1):
        model.train()
        optimizer.zero_grad()
        edge_index_1 = dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{1}'])[0]
        edge_index_2 = dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{2}'])[0]
        x_1 = drop_feature(data.x, param['drop_feature_rate_1'])
        x_2 = drop_feature(data.x, param['drop_feature_rate_2'])
        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        loss = model.loss(z1, z2)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
    print(f"pretrain time {time.time()-t1:.2f} s, loss {loss:.4f}", flush=True)
    
def pretrain_gca(model, data, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )
    device = data.x.device
    # compute drop_weights per centrality metrics
    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif param['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif param['drop_scheme'] == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    # # compute feature_weights per centrality metrics
    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif param['drop_scheme'] == 'pr':
        node_pr = compute_pr(data.edge_index)
        print(node_pr.shape, data.x.shape)
        feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif param['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)
    
    t1 = time.time()
    for epoch in range(1, param['num_epochs'] + 1):
        model.train()
        optimizer.zero_grad()
        edge_index_1 = drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{1}'], threshold=0.7)
        edge_index_2 = drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{2}'], threshold=0.7)
        x_1 = drop_feature_weighted(data.x, feature_weights, param['drop_feature_rate_1'])
        x_2 = drop_feature_weighted(data.x, feature_weights, param['drop_feature_rate_2'])

        z1 = model(x_1, edge_index_1)
        z2 = model(x_2, edge_index_2)

        loss = model.loss(z1, z2)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
    print(f"pretrain time {time.time()-t1:.2f} s, loss {loss:.4f}", flush=True)
    
def pretrain_bgrl(model, data, param):
      # optimizer
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])

    # scheduler
    lr_scheduler = CosineDecayScheduler(param['learning_rate'], 1000, param['num_epochs'])
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, param['num_epochs'])

    t1 = time.time()
    for epoch in range(1, param['num_epochs'] + 1):
        model.train()

        lr = lr_scheduler.get(epoch)
        mm = 1 - mm_scheduler.get(epoch)


        optimizer.zero_grad()
        data_c1 = data.clone()
        data_c2 = data.clone()
        data_c1.edge_index = dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{1}'])[0]
        data_c2.edge_index = dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{2}'])[0]
        
        data_c1.x = drop_feature(data.x, param['drop_feature_rate_1'])
        data_c2.x = drop_feature(data.x, param['drop_feature_rate_2'])

        z1, y2 = model.train_forward(data_c1, data_c2)
        z2, y1 = model.train_forward(data_c2, data_c1)

        loss = 2 - F.cosine_similarity(z1, y2.detach(), dim=-1).mean() - F.cosine_similarity(z2, y1.detach(), dim=-1).mean() # loss simple
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 100 == 0:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
    print(f"pretrain time {time.time()-t1:.2f} s, loss {loss:.4f}", flush=True)
    
def pretrain_bgrl_adaptative(model, data, param):
      # optimizer
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])

    # scheduler
    lr_scheduler = CosineDecayScheduler(param['learning_rate'], 1000, param['num_epochs'])
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, param['num_epochs'])
    device = data.x.device

    # compute drop_weights per centrality metrics
    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif param['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif param['drop_scheme'] == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    # # compute feature_weights per centrality metrics
    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif param['drop_scheme'] == 'pr':
        node_pr = compute_pr(data.edge_index)
        print(node_pr.shape, data.x.shape)
        feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif param['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)
    
    t1 = time.time()
    for epoch in range(1, param['num_epochs'] + 1):
        model.train()

        lr = lr_scheduler.get(epoch)
        mm = 1 - mm_scheduler.get(epoch)


        optimizer.zero_grad()
        data_c1 = data.clone()
        data_c2 = data.clone()
        """ legacy BGRL augmentation
        data_c1.edge_index = dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{1}'])[0]
        data_c2.edge_index = dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{2}'])[0]
        
        data_c1.x = drop_feature(data.x, param['drop_feature_rate_1'])
        data_c2.x = drop_feature(data.x, param['drop_feature_rate_2'])
        """
        data_c1.edge_index = drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{1}'], threshold=0.7)
        data_c2.edge_index = drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{2}'], threshold=0.7)
        data_c1.x = drop_feature_weighted(data.x, feature_weights, param['drop_feature_rate_1'])
        data_c2.x = drop_feature_weighted(data.x, feature_weights, param['drop_feature_rate_2'])
        z1, y2 = model.train_forward(data_c1, data_c2)
        z2, y1 = model.train_forward(data_c2, data_c1)

        loss = 2 - F.cosine_similarity(z1, y2.detach(), dim=-1).mean() - F.cosine_similarity(z2, y1.detach(), dim=-1).mean() # loss simple
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 100 == 0:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
    print(f"pretrain time {time.time()-t1:.2f} s, loss {loss:.4f}", flush=True)
    
def pretrain_csgcl(model, data, param):
    g = to_networkx(data, to_undirected=True)
    communities = community_detection('leiden')(g).communities
    com = transition(communities, g.number_of_nodes())
    com_cs, node_cs = community_strength(g, communities)
    edge_weight = get_edge_weight(data.edge_index, com, com_cs)
    com_size = [len(c) for c in communities]
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=param['learning_rate'],
                                 weight_decay=param['weight_decay'])
    t1 = time.time()
    for epoch in range(1, param['num_epochs'] + 1):
        model.train()
        optimizer.zero_grad()
    
        edge_index_1 = ced(data.edge_index, edge_weight, p=param['drop_edge_rate_1'])
        edge_index_2 = ced(data.edge_index, edge_weight, p=param['drop_edge_rate_2'])
        x1 = cav(data.x, node_cs, param["drop_feature_rate_1"])
        x2 = cav(data.x, node_cs, param['drop_feature_rate_2'])
        z1 = model(x1, edge_index_1)
        z2 = model(x2, edge_index_2)
        loss = model.team_up_loss(z1, z2,
                                  cs=node_cs,
                                  current_ep=epoch)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
    print(f"pretrain time {time.time()-t1:.2f} s, loss {loss:.4f}", flush=True)
    
def pretrain_bgrl_adaptative_cs(model, data, param):
      # optimizer
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])

    # scheduler
    lr_scheduler = CosineDecayScheduler(param['learning_rate'], 1000, param['num_epochs'])
    mm_scheduler = CosineDecayScheduler(1 - 0.99, 0, param['num_epochs'])

    g = to_networkx(data, to_undirected=True)
    communities = community_detection('leiden')(g).communities
    com = transition(communities, g.number_of_nodes())
    com_cs, node_cs = community_strength(g, communities)
    edge_weight = get_edge_weight(data.edge_index, com, com_cs)
    com_size = [len(c) for c in communities]
    
    t1 = time.time()
    for epoch in range(1, param['num_epochs'] + 1):
        model.train()

        lr = lr_scheduler.get(epoch)
        mm = 1 - mm_scheduler.get(epoch)


        optimizer.zero_grad()
        data_c1 = data.clone()
        data_c2 = data.clone()
        """ legacy BGRL augmentation
        data_c1.edge_index = dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{1}'])[0]
        data_c2.edge_index = dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{2}'])[0]
        
        data_c1.x = drop_feature(data.x, param['drop_feature_rate_1'])
        data_c2.x = drop_feature(data.x, param['drop_feature_rate_2'])
        """
        data_c1.edge_index = ced(data.edge_index, edge_weight, p=param['drop_edge_rate_1'])
        data_c2.edge_index = ced(data.edge_index, edge_weight, p=param['drop_edge_rate_2'])
        data_c1.x = cav(data.x, node_cs, param["drop_feature_rate_1"])
        data_c2.x = cav(data.x, node_cs, param['drop_feature_rate_2'])
        z1, y2 = model.train_forward(data_c1, data_c2)
        z2, y1 = model.train_forward(data_c2, data_c1)

        loss = 2 - F.cosine_similarity(z1, y2.detach(), dim=-1).mean() - F.cosine_similarity(z2, y1.detach(), dim=-1).mean() # loss simple
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)
        if epoch % 100 == 0:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')
    print(f"pretrain time {time.time()-t1:.2f} s, loss {loss:.4f}", flush=True)