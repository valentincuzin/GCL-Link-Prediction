import networkx as nx
import matplotlib.pyplot as plt
import torch
import time
import random
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from networkx.generators.community import stochastic_block_model
from torch_geometric.utils import from_networkx, dropout_adj, to_networkx
from ogb.linkproppred import Evaluator

import src.encoder as enc
import src.decoder as dec
import src.utils as ut
import src.train_utils as tr
import src.contrastive_pretrain as pretr
import src.contrastive_model as ctmod
from src.contrastive_augmentation import drop_feature, cav, ced, community_detection, community_strength, transition, get_edge_weight

hp = {
    'xdp': 0.7,
    'tdp': 0.3,
    'pt': 0.75,
    'gnnedp': 0.0,
    'preedp': 0.4,
    'predp': 0.05,
    'gnndp': 0.05,
    'res': False,
    'probscale': 4.3,
    'proboffset': 2.8,
    'alpha': 1.0,
    'gnnlr': 0.0043,
    'prelr': 0.0024,
    'batch_size': 1152,
    'ln': True,
    'lnnn': True,
    'epochs': 100,
    'model': 'puregcn',
    'runs': 10,
    'hiddim': 256,
    'mplayers': 1,
    'testbs': 8192,
    'maskinput': True,
    'jk': True,
    'use_xlin': True,
    'tailact': True,
    'use_valedges_as_input': False,
    'freeze': True,
    'inner': True,
    'ct_param': {
    	'learning_rate': 0.01,
    	'num_hidden': 256,
    	'num_proj_hidden': 32,
    	'activation': 'prelu',
    	'base_model': 'GCNConv',
    	'num_layers': 2,
    	'drop_edge_rate_1': 0.3,
    	'drop_edge_rate_2': 0.4,
    	'drop_feature_rate_1': 0.1,
    	'drop_feature_rate_2': 0.0,
    	'tau': 0.4,
    	'num_epochs': 500,
    	'weight_decay': 1e-5,
    	'drop_scheme': 'degree',
    }
}

# generator of graph
nb_nodes = 50
nb_block = 5
sizes = [nb_nodes for _ in range(nb_block)]
probs = [[0.3, 0.0001, 0.0005, 0.0002, 0.0004],
        [0.0001, 0.8, 0.0001, 0.0006, 0.0002],
        [0.0005, 0.0001, 0.65, 0.0001, 0.0003],
        [0.0002, 0.0006, 0.0001, 0.1, 0.0001],
        [0.0004, 0.0002, 0.0003, 0.0001, 0.4]]

def gen_sbm(sizes, probs, device=None, seed=random.randint(1, 10000)):
    G = stochastic_block_model(sizes, probs, seed=seed)
    #nx.draw(G)
    #plt.savefig(f'{random.randint(0, 10000)}sbm.png')
    data = from_networkx(G)
    data.num_nodes = sum(sizes)
    data.sizes = sizes
    data.probs = probs
    data.num_features = data.num_nodes
    data.x = F.one_hot(torch.arange(0, sum(sizes))).float()
    if device is not None:
        data = data.to(device)
    return data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = gen_sbm(sizes, probs, device, 0)
evaluator = Evaluator(name='ogbl-ppa')
data_split = ut.DataSplit([data], device, hp['runs'], hp['use_valedges_as_input'])

def pretrain_grace_commu(model, data, param):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )
    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['num_epochs'] + 1)):
        model.train()
        optimizer.zero_grad()
        data_sbm1 = gen_sbm(data.sizes, data.probs, data.x.device, epoch)
        data_sbm2 = gen_sbm(data.sizes, data.probs, data.x.device, epoch)
        # edge_index_1 = dropout_adj(data_sbm.edge_index, p=param[f'drop_edge_rate_{1}'])[0]
        # edge_index_2 = dropout_adj(data_sbm.edge_index, p=param[f'drop_edge_rate_{2}'])[0]
        # x_1 = drop_feature(data_sbm.x, param['drop_feature_rate_1'])
        # x_2 = drop_feature(data_sbm.x, param['drop_feature_rate_2'])
        z1 = model(data_sbm1.x, data_sbm1.edge_index)
        z2 = model(data_sbm2.x, data_sbm2.edge_index)
        loss = model.loss(z1, z2)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time

def pretrain_bgrl_commu(model, data, param):
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=param['learning_rate'], weight_decay=param['weight_decay'])

    lr_scheduler = ut.CosineDecayScheduler(param['learning_rate'], 1000, param['num_epochs'])
    mm_scheduler = ut.CosineDecayScheduler(1 - 0.99, 0, param['num_epochs'])

    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['num_epochs'] + 1)):
        model.train()

        lr = lr_scheduler.get(epoch)
        mm = 1 - mm_scheduler.get(epoch)

        optimizer.zero_grad()
        data_sbm = gen_sbm(data.sizes, data.probs, data.x.device)

        data_c1 = gen_sbm(data.sizes, data.probs, data.x.device, epoch)
        data_c2 = gen_sbm(data.sizes, data.probs, data.x.device, epoch)
        # data_c1.edge_index = dropout_adj(data_sbm.edge_index, p=param[f'drop_edge_rate_{1}'])[0]
        # data_c2.edge_index = dropout_adj(data_sbm.edge_index, p=param[f'drop_edge_rate_{2}'])[0]

        # data_c1.x = drop_feature(data_sbm.x, param['drop_feature_rate_1'])
        # data_c2.x = drop_feature(data_sbm.x, param['drop_feature_rate_2'])

        z1, y2 = model.train_forward(data_c1, data_c2)
        z2, y1 = model.train_forward(data_c2, data_c1)

        loss = 2 - F.cosine_similarity(z1, y2.detach(), dim=-1).mean() - F.cosine_similarity(z2, y1.detach(), dim=-1).mean() # loss simple
        loss.backward()
        optimizer.step()
        model.update_target_network(mm)

        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time

def pretrain_csgcl_commu(model, data, param):
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=param['learning_rate'],
                                 weight_decay=param['weight_decay'])
    t1 = time.time()
    loss_res = []
    for epoch in tqdm(range(1, param['num_epochs'] + 1)):
        model.train()
        optimizer.zero_grad()
        data_sbm1 = gen_sbm(data.sizes, data.probs, data.x.device)
        data_sbm2 = gen_sbm(data.sizes, data.probs, data.x.device)
        g = to_networkx(data_sbm1, to_undirected=True)
        communities = community_detection('leiden')(g).communities
        # com = transition(communities, g.number_of_nodes())
        com_cs, node_cs = community_strength(g, communities)
        # edge_weight = get_edge_weight(data_sbm.edge_index, com, com_cs)
        # edge_index_1 = ced(data_sbm.edge_index, edge_weight, p=param['drop_edge_rate_1'])
        # edge_index_2 = ced(data_sbm.edge_index, edge_weight, p=param['drop_edge_rate_2'])
        # x1 = cav(data_sbm.x, node_cs, param["drop_feature_rate_1"])
        # x2 = cav(data_sbm.x, node_cs, param['drop_feature_rate_2'])
        z1 = model(x1, edge_index_1)
        z2 = model(x2, edge_index_2)
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

########## RUNS ##########
full_res = []

def init_model(data, hp):
    _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
    encoder = ctmod.GRACE(_encoder, hp['ct_param']['num_hidden'], hp['ct_param']['num_proj_hidden']).to(device)
    predictor = dec.inner_prod().to(device)
    return encoder, predictor
full_res.append(tr.runs('SBM_GRACE+Inner', init_model, pretrain_grace_commu, data_split, evaluator, hp))


def init_model(data, hp):
    _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
    _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
    encoder = ctmod.BGRL(_encoder, _predictor).to(device)
    predictor = dec.inner_prod().to(device)
    return encoder, predictor
full_res.append(tr.runs('SBM_BGRL+Inner', init_model, pretrain_bgrl_commu, data_split, evaluator, hp))


def init_model(data, hp):
    _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
    encoder = ctmod.CSGCL(_encoder,
                      hp['ct_param']['num_hidden'],
                      hp['ct_param']['num_proj_hidden'],
                      hp['ct_param']['tau']).to(device)
    predictor = dec.inner_prod().to(device)
    return encoder, predictor
full_res.append(tr.runs('SBM_CSGCL+inner', init_model, pretrain_csgcl_commu, data_split, evaluator, hp))

def init_model(data, hp):
    encoder = enc.ENCODER_NCN(data.num_features, hp['hiddim'], hp['hiddim'], hp['mplayers'],
                    hp['gnndp'], hp['ln'], hp['res'], data.max_x,
                    hp['model'], edrop=hp['gnnedp'],  xdropout=hp['xdp'], taildropout=hp['tdp']).to(device)
    predictor = dec.inner_prod().to(device)
    return encoder, predictor
full_res.append(tr.runs('GCN+inner', init_model, None, data_split, evaluator, hp))


def init_model(data, hp):
    _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
    encoder = ctmod.GRACE(_encoder, hp['ct_param']['num_hidden'], hp['ct_param']['num_proj_hidden']).to(device)
    predictor = dec.inner_prod().to(device)
    return encoder, predictor
full_res.append(tr.runs('GRACE+inner', init_model, pretr.pretrain_grace, data_split, evaluator, hp))


hp['ct_param']['drop_scheme'] = 'degree'
def init_model(data, hp):
    _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
    encoder = ctmod.GRACE(_encoder, hp['ct_param']['num_hidden'], hp['ct_param']['num_proj_hidden']).to(device)
    predictor = dec.inner_prod().to(device)
    return encoder, predictor
full_res.append(tr.runs('GCA_deg+inner', init_model, pretr.pretrain_gca, data_split, evaluator, hp))


hp['ct_param']['drop_scheme'] = 'pr'
def init_model(data, hp):
    _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
    encoder = ctmod.GRACE(_encoder, hp['ct_param']['num_hidden'], hp['ct_param']['num_proj_hidden']).to(device)
    predictor = dec.inner_prod().to(device)
    return encoder, predictor
full_res.append(tr.runs('GCA_pr+inner', init_model, pretr.pretrain_gca, data_split, evaluator, hp))


hp['ct_param']['drop_scheme'] = 'evc'
def init_model(data, hp):
    _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
    encoder = ctmod.GRACE(_encoder, hp['ct_param']['num_hidden'], hp['ct_param']['num_proj_hidden']).to(device)
    predictor = dec.inner_prod().to(device)
    return encoder, predictor
full_res.append(tr.runs('GCA_evc+inner', init_model, pretr.pretrain_gca, data_split, evaluator, hp))


def init_model(data, hp):
    _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
    _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
    encoder = ctmod.BGRL(_encoder, _predictor).to(device)
    predictor = dec.inner_prod().to(device)
    return encoder, predictor
full_res.append(tr.runs('BGRL+inner', init_model, pretr.pretrain_bgrl, data_split, evaluator, hp))


hp['ct_param']['drop_scheme'] = 'degree'
def init_model(data, hp):
    _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
    _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
    encoder = ctmod.BGRL(_encoder, _predictor).to(device)
    predictor = dec.inner_prod().to(device)
    return encoder, predictor
full_res.append(tr.runs('BGRL_deg+inner', init_model, pretr.pretrain_bgrl_adaptative, data_split, evaluator, hp))


hp['ct_param']['drop_scheme'] = 'pr'
def init_model(data, hp):
    _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
    _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
    encoder = ctmod.BGRL(_encoder, _predictor).to(device)
    predictor = dec.inner_prod().to(device)
    return encoder, predictor
full_res.append(tr.runs('BGRL_pr+inner', init_model, pretr.pretrain_bgrl_adaptative, data_split, evaluator, hp))


hp['ct_param']['drop_scheme'] = 'evc'
def init_model(data, hp):
    _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
    _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
    encoder = ctmod.BGRL(_encoder, _predictor).to(device)
    predictor = dec.inner_prod().to(device)
    return encoder, predictor
full_res.append(tr.runs('BGRL_evc+inner', init_model, pretr.pretrain_bgrl_adaptative, data_split, evaluator, hp))


def init_model(data, hp):
    _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
    encoder = ctmod.CSGCL(_encoder,
                      hp['ct_param']['num_hidden'],
                      hp['ct_param']['num_proj_hidden'],
                      hp['ct_param']['tau']).to(device)
    predictor = dec.inner_prod().to(device)
    return encoder, predictor
full_res.append(tr.runs('CSGCL+inner', init_model, pretr.pretrain_csgcl, data_split, evaluator, hp))


def init_model(data, hp):
    _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
    _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
    encoder = ctmod.BGRL(_encoder, _predictor).to(device)
    predictor = dec.inner_prod().to(device)
    return encoder, predictor
full_res.append(tr.runs('BGRL_cs+inner', init_model, pretr.pretrain_bgrl_cs, data_split, evaluator, hp))


df, tex = ut.full_output(full_res)
df.to_csv(f'output/SBM_inner_250_res.csv', sep=';')