import networkx as nx
import torch
import time
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from networkx.generators.community import stochastic_block_model
from torch_geometric.utils import from_networkx
from torch_geometric import seed_everything
from ogb.linkproppred import Evaluator

import src.encoder as enc
import src.decoder as dec
import src.utils as ut
import src.train_utils as tr
import src.contrastive_model as ctmod

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
    	'num_epochs': 1500,
    	'weight_decay': 1e-5,
    	'drop_scheme': 'degree',
    }
}

# generator of graph
nb_nodes = 20
nb_block = 5
sizes = [nb_nodes for _ in range(nb_block)]
probs = [[0.2, 0.1, 0.05, 0.02, 0.04],
        [0.1, 0.6, 0.1, 0.06, 0.02],
        [0.05, 0.1, 0.75, 0.01, 0.03],
        [0.02, 0.06, 0.01, 0.1, 0.1],
        [0.04, 0.02, 0.03, 0.1, 0.4]]

def gen_sbm(sizes, probs):
    G = stochastic_block_model(sizes, probs)
    data = from_networkx(G)
    data.num_nodes = sum(sizes)
    data.sizes = sizes
    data.probs = probs
    data.x = F.one_hot(torch.arange(0, sum(sizes))).float()
    print(data)
    return data

data = gen_sbm(sizes, probs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
evaluator = Evaluator(name='ogbl-ppa')
data_split = ut.DataSplit([data], device, hp['runs'], hp['use_valedges_as_input'], 2)

def init_model(data, hp):
    _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
    encoder = ctmod.GRACE(_encoder, hp['ct_param']['num_hidden'], hp['ct_param']['num_proj_hidden']).to(device)
    predictor = dec.MlpProdDecoder(hp['hiddim'], hp['hiddim']).to(device)
    return encoder, predictor


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
        data1 = gen_sbm(data.sizes, data.probs)
        data2 = gen_sbm(data.sizes, data.probs)
        z1 = model(data1.x, data1.edge_index)
        z2 = model(data2.x, data2.edge_index)

        loss = model.loss(z1, z2)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            loss_res.append(round(float(loss), 2))
    print('pretrain loss: ', loss_res)
    pre_time = time.time()-t1
    print(f"pretrain time: {pre_time:.2f} s")
    return pre_time

tr.runs('Cora GCN+NCN', init_model, pretrain_grace_commu, data_split, evaluator, hp)