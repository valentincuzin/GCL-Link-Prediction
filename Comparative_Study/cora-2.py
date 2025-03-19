import torch
import torch.nn as nn
import pandas as pd
import src.utils as ut
import src.train_utils as tr
import src.contrastive_pretrain as pretr
import src.contrastive_model as ctmod
import src.encoder as enc
import src.decoder as dec
DATASET = 'Cora'
print(DATASET)
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

def run_all(dataset: str, hp: dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = ut.get_evaluator(DATASET)
    data_split = ut.DataSplit(DATASET, device, hp['runs'], hp['use_valedges_as_input'], 2)
    
    full_res = []
    
    def init_model(data, hp):
        encoder = enc.ENCODER_NCN(data.num_features, hp['hiddim'], hp['hiddim'], hp['mplayers'],
    					hp['gnndp'], hp['ln'], hp['res'], data.max_x,
    					hp['model'], edrop=hp['gnnedp'],  xdropout=hp['xdp'], taildropout=hp['tdp']).to(device)
        predictor = dec.CNLinkPredictor(hp['hiddim'], hp['hiddim'], 1, hp['mplayers'],
        						hp['predp'], hp['preedp'], hp['lnnn']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora GCN+NCN', init_model, None, data_split, evaluator, hp))
    
    def init_model(data, hp):
        encoder = enc.ENCODER_NCN(data.num_features, hp['hiddim'], hp['hiddim'], hp['mplayers'],
    					hp['gnndp'], hp['ln'], hp['res'], data.max_x,
    					hp['model'], edrop=hp['gnnedp'],  xdropout=hp['xdp'], taildropout=hp['tdp']).to(device)
        predictor = dec.MlpProdDecoder(hp['hiddim'], hp['hiddim']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora GCN+MLP', init_model, None, data_split, evaluator, hp))
    
    
    def init_model(data, hp):
        _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
        encoder = ctmod.GRACE(_encoder, hp['ct_param']['num_hidden'], hp['ct_param']['num_proj_hidden']).to(device)
        predictor = dec.CNLinkPredictor(hp['hiddim'], hp['hiddim'], 1, hp['mplayers'],
        						hp['predp'], hp['preedp'], hp['lnnn']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora GRACE+NCN', init_model, pretr.pretrain_grace, data_split, evaluator, hp))
    
    def init_model(data, hp):
        _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
        encoder = ctmod.GRACE(_encoder, hp['ct_param']['num_hidden'], hp['ct_param']['num_proj_hidden']).to(device)
        predictor = dec.MlpProdDecoder(hp['hiddim'], hp['hiddim']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora GRACE+MLP', init_model, pretr.pretrain_grace, data_split, evaluator, hp))
    
    
    hp['ct_param']['drop_scheme'] = 'degree'
    def init_model(data, hp):
        _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
        encoder = ctmod.GRACE(_encoder, hp['ct_param']['num_hidden'], hp['ct_param']['num_proj_hidden']).to(device)
        predictor = dec.CNLinkPredictor(hp['hiddim'], hp['hiddim'], 1, hp['mplayers'],
        						hp['predp'], hp['preedp'], hp['lnnn']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora GCA_deg+NCN', init_model, pretr.pretrain_gca, data_split, evaluator, hp))
    
    def init_model(data, hp):
        _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
        encoder = ctmod.GRACE(_encoder, hp['ct_param']['num_hidden'], hp['ct_param']['num_proj_hidden']).to(device)
        predictor = dec.MlpProdDecoder(hp['hiddim'], hp['hiddim']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora GCA_deg+MLP', init_model, pretr.pretrain_gca, data_split, evaluator, hp))
    
    
    hp['ct_param']['drop_scheme'] = 'pr'
    def init_model(data, hp):
        _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
        encoder = ctmod.GRACE(_encoder, hp['ct_param']['num_hidden'], hp['ct_param']['num_proj_hidden']).to(device)
        predictor = dec.CNLinkPredictor(hp['hiddim'], hp['hiddim'], 1, hp['mplayers'],
        						hp['predp'], hp['preedp'], hp['lnnn']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora GCA_pr+NCN', init_model, pretr.pretrain_gca, data_split, evaluator, hp))
    
    def init_model(data, hp):
        _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
        encoder = ctmod.GRACE(_encoder, hp['ct_param']['num_hidden'], hp['ct_param']['num_proj_hidden']).to(device)
        predictor = dec.MlpProdDecoder(hp['hiddim'], hp['hiddim']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora GCA_pr+MLP', init_model, pretr.pretrain_gca, data_split, evaluator, hp))
    
    
    hp['ct_param']['drop_scheme'] = 'evc'
    def init_model(data, hp):
        _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
        encoder = ctmod.GRACE(_encoder, hp['ct_param']['num_hidden'], hp['ct_param']['num_proj_hidden']).to(device)
        predictor = dec.CNLinkPredictor(hp['hiddim'], hp['hiddim'], 1, hp['mplayers'],
        						hp['predp'], hp['preedp'], hp['lnnn']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora GCA_evc+NCN', init_model, pretr.pretrain_gca, data_split, evaluator, hp))
    
    def init_model(data, hp):
        _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
        encoder = ctmod.GRACE(_encoder, hp['ct_param']['num_hidden'], hp['ct_param']['num_proj_hidden']).to(device)
        predictor = dec.MlpProdDecoder(hp['hiddim'], hp['hiddim']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora GCA_evc+MLP', init_model, pretr.pretrain_gca, data_split, evaluator, hp))
    
    
    def init_model(data, hp):
        _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
        _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
        encoder = ctmod.BGRL(_encoder, _predictor).to(device)
        predictor = dec.CNLinkPredictor(hp['hiddim'], hp['hiddim'], 1, hp['mplayers'],
        						hp['predp'], hp['preedp'], hp['lnnn']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora BGRL+NCN', init_model, pretr.pretrain_bgrl, data_split, evaluator, hp))
    
    def init_model(data, hp):
        _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
        _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
        encoder = ctmod.BGRL(_encoder, _predictor).to(device)
        predictor = dec.MlpProdDecoder(hp['hiddim'], hp['hiddim']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora BGRL+MLP', init_model, pretr.pretrain_bgrl, data_split, evaluator, hp))
    
    
    hp['ct_param']['drop_scheme'] = 'degree'
    def init_model(data, hp):
        _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
        _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
        encoder = ctmod.BGRL(_encoder, _predictor).to(device)
        predictor = dec.CNLinkPredictor(hp['hiddim'], hp['hiddim'], 1, hp['mplayers'],
        						hp['predp'], hp['preedp'], hp['lnnn']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora BGRL_deg+NCN', init_model, pretr.pretrain_bgrl_adaptative, data_split, evaluator, hp))
    
    def init_model(data, hp):
        _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
        _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
        encoder = ctmod.BGRL(_encoder, _predictor).to(device)
        predictor = dec.MlpProdDecoder(hp['hiddim'], hp['hiddim']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora BGRL_deg+MLP', init_model, pretr.pretrain_bgrl_adaptative, data_split, evaluator, hp))
    
    
    hp['ct_param']['drop_scheme'] = 'pr'
    def init_model(data, hp):
        _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
        _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
        encoder = ctmod.BGRL(_encoder, _predictor).to(device)
        predictor = dec.CNLinkPredictor(hp['hiddim'], hp['hiddim'], 1, hp['mplayers'],
        						hp['predp'], hp['preedp'], hp['lnnn']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora BGRL_pr+NCN', init_model, pretr.pretrain_bgrl_adaptative, data_split, evaluator, hp))
    
    def init_model(data, hp):
        _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
        _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
        encoder = ctmod.BGRL(_encoder, _predictor).to(device)
        predictor = dec.MlpProdDecoder(hp['hiddim'], hp['hiddim']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora BGRL_pr+MLP', init_model, pretr.pretrain_bgrl_adaptative, data_split, evaluator, hp))
    
    
    hp['ct_param']['drop_scheme'] = 'evc'
    def init_model(data, hp):
        _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
        _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
        encoder = ctmod.BGRL(_encoder, _predictor).to(device)
        predictor = dec.CNLinkPredictor(hp['hiddim'], hp['hiddim'], 1, hp['mplayers'],
        						hp['predp'], hp['preedp'], hp['lnnn']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora BGRL_evc+NCN', init_model, pretr.pretrain_bgrl_adaptative, data_split, evaluator, hp))
    
    def init_model(data, hp):
        _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
        _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
        encoder = ctmod.BGRL(_encoder, _predictor).to(device)
        predictor = dec.MlpProdDecoder(hp['hiddim'], hp['hiddim']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora BGRL_evc+MLP', init_model, pretr.pretrain_bgrl_adaptative, data_split, evaluator, hp))
    
    
    def init_model(data, hp):
        _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
        encoder = ctmod.CSGCL(_encoder,
                          hp['ct_param']['num_hidden'],
                          hp['ct_param']['num_proj_hidden'],
                          hp['ct_param']['tau']).to(device)
        predictor = dec.CNLinkPredictor(hp['hiddim'], hp['hiddim'], 1, hp['mplayers'],
        						hp['predp'], hp['preedp'], hp['lnnn']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora CSGCL+NCN', init_model, pretr.pretrain_csgcl, data_split, evaluator, hp))
    
    def init_model(data, hp):
        _encoder = enc.ENCODER_GRACE(data.num_features, hp['ct_param']['num_hidden'], nn.Identity()).to(device)
        encoder = ctmod.CSGCL(_encoder,
                          hp['ct_param']['num_hidden'],
                          hp['ct_param']['num_proj_hidden'],
                          hp['ct_param']['tau']).to(device)
        predictor = dec.MlpProdDecoder(hp['hiddim'], hp['hiddim']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora CSGCL+MLP', init_model, pretr.pretrain_csgcl, data_split, evaluator, hp))
    
    
    def init_model(data, hp):
        _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
        _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
        encoder = ctmod.BGRL(_encoder, _predictor).to(device)
        predictor = dec.CNLinkPredictor(hp['hiddim'], hp['hiddim'], 1, hp['mplayers'],
        						hp['predp'], hp['preedp'], hp['lnnn']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora BGRL_cs+NCN', init_model, pretr.pretrain_bgrl_cs, data_split, evaluator, hp))
    
    def init_model(data, hp):
        _encoder = enc.ENCODER_BGRL([data.num_features, hp['ct_param']['num_hidden']], batchnorm=True).to(device)
        _predictor = ut.MLP_Head_BGRL(hp['ct_param']['num_hidden'], hp['ct_param']['num_hidden']).to(device)
        encoder = ctmod.BGRL(_encoder, _predictor).to(device)
        predictor = dec.MlpProdDecoder(hp['hiddim'], hp['hiddim']).to(device)
        return encoder, predictor
    full_res.append(tr.runs('Cora BGRL_cs+MLP', init_model, pretr.pretrain_bgrl_cs, data_split, evaluator, hp))
    
    df, tex = ut.full_output(full_res)
    df.to_csv(f'{DATASET}-2_res.csv', sep=';')

if __name__ == '__main__':
    run_all(DATASET, hp)