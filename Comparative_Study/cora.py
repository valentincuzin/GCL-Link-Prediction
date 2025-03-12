from torch_geometric import seed_everything
import torch
import src.utils as ut
import src.train_utils as tr
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
    'probscale': 4.3,
    'proboffset': 2.8,
    'alpha': 1.0,
    'gnnlr': 0.0043,
    'prelr': 0.0024,
    'batch_size': 1152,
    'ln': True,
    'lnnn': True,
    'epochs': 100,
    'runs': 10,
    'hiddim': 256,
    'mplayers': 1,
    'testbs': 8192,
    'maskinput': True,
    'jk': True,
    'use_xlin': True,
    'tailact': True,
}
param = {
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
res_dict = {"Hits@20": [], "Hits@20_std": 0, "Hits@50": [], "Hits@50_std": 0, "Hits@100": [], "Hits@100_std": 0}
evaluator = ut.get_evaluator(DATASET)
data_split = ut.DataSplit(DATASET, device, hp['runs'], False)

print("### GCN+NCN ###")
for r in range(hp['runs']):
	seed_everything(r)
	data, split_edge = data_split.get(r)
	encoder = enc.GCN(data.num_features, hp['hiddim'], hp['hiddim'], hp['mplayers'],
					hp['gnndp'], hp['ln'], hp['res'], data.max_x,
					hp['model'], edrop=hp['gnnedp'],  xdropout=hp['xdp'], taildropout=hp['tdp']).to(device)
	predictor = dec.CNLinkPredictor(hp['hiddim'], hp['hiddim'], 1, hp['mplayers'],
						hp['predp'], hp['preedp'], hp['lnnn']).to(device)
	res_dict = tr.run(r, "GCN+NCN", encoder, None, predictor, data, split_edge, evaluator)
res_dict, res_latex = ut.compute_table(res_dict)
print("\n\n### GCN+NCN ###")
print(res_dict)
print(f'\n{res_latex}\n\n')
# for dataset in  ["collab", "ppa", "citation2"]:
#     hp = hps[dataset]
#     hp['runs'] = 5
#     evaluator = Evaluator(name=f'ogbl-{dataset}')
#     data, split_edge = loaddataset(dataset, hp['use_valedges_as_input']) # get a new split of dataset
#     data = data.to(device)
    
#     res_dict = {"Hits@20": [], "Hits@20_std": 0, "Hits@50": [], "Hits@50_std": 0, "Hits@100": [], "Hits@100_std": 0}
#     for r in range(hp['runs']):
#         set_seed(r)
#         model = GCN(data.num_features, hp['hiddim'], hp['hiddim'], hp['mplayers'],
#                      hp['gnndp'], hp['ln'], hp['res'], data.max_x,
#                      hp['model'], edrop=hp['gnnedp'],  xdropout=hp['xdp'], taildropout=hp['tdp']).to(device)
#         predictor = CNLinkPredictor(hp['hiddim'], hp['hiddim'], 1, hp['mplayers'],
#                            hp['predp'], hp['preedp'], hp['lnnn']).to(device)
#         res_dict = run(r, model, None, predictor, data, evaluator, hp, res_dict)
#     res_dict, res_latex = compute_table(res_dict)
#     print(f'######\t{dataset}\t NCN\t######')
#     print('\n\n', res_latex, '\n\n')

#     res_dict = {"Hits@20": [], "Hits@20_std": 0, "Hits@50": [], "Hits@50_std": 0, "Hits@100": [], "Hits@100_std": 0}
#     for r in range(hp['runs']):
#         set_seed(r)
#         model = GCN(data.num_features, hp['hiddim'], hp['hiddim'], hp['mplayers'],
#                      hp['gnndp'], hp['ln'], hp['res'], data.max_x,
#                      hp['model'], edrop=hp['gnnedp'],  xdropout=hp['xdp'], taildropout=hp['tdp']).to(device)
#         predictor = CNLinkPredictor(hp['hiddim'], hp['hiddim'], 1, hp['mplayers'],
#                            hp['predp'], hp['preedp'], hp['lnnn']).to(device)
#         res_dict = run(r, model, None, predictor, data, evaluator, hp, res_dict)
#     res_dict, res_latex = compute_table(res_dict)
#     print(f'######\t{dataset}\t MLP\t######')
#     print('\n\n', res_latex, '\n\n')