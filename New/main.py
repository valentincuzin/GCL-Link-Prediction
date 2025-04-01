import argparse
import json
import os
import torch
import torch.nn as nn
from torch_geometric import seed_everything

from src.augmentation import Aug
from src.model import get_model
from src.datasets import DataSplit, get_evaluator, full_eval
from src.predictor import get_predictor
from src.train import pretrain, pred_train, test, ncn_loss
from src.utils import store_res, compute_table, full_output

def arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='cora', choices=["cora", "citeseer", "pubmed", "collab"])
	parser.add_argument('--reduce_feature', type=int, default=None, help='0 for Identity matrix, >0 for PCA reduce')
	parser.add_argument('--only_feature', action='store_true', default=False, help='erase structure information')
	parser.add_argument('--model', type=str, default='grace', choices=["grace", "csgcl", "bgrl"])
	parser.add_argument('--augmentation', type=str, default='random', choices=["random", "deg", "pr", "evc", "coms", "sbm"])
	parser.add_argument('--predictor', type=str, default='inner', choices=["inner", "mlp"])
	parser.add_argument('--epochs', type=int, default=1500)
	parser.add_argument('--runs', type=int, default=10)
	parser.add_argument('--use_valedges_as_input', action='store_true', help="add validation edges to the input adjacency matrix of gnn")
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = arguments()
	hp_files = os.path.join('params', args.dataset+'.json')
	with open(hp_files) as json_file:
		hp = json.load(json_file)
		hp["epochs"] = args.epochs
		hp["use_valedges_as_input"] = args.use_valedges_as_input
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	data_split = DataSplit(args.dataset, device, args.runs, args.use_valedges_as_input, args.reduce_feature, args.only_feature)
	evaluator = get_evaluator(args.dataset)
	res_dict = {"Hits@10": [], "Hits@20": [], "Hits@50": [], "Hits@100": [], 'ROCAUC': [], 'pretrain_time': []}
	for r in range(args.runs):
		seed_everything(r)
		data, split_edge = data_split.get(r)
		aug = Aug(data, hp['augmentation'], args.augmentation)
		model = get_model(args.model, data, hp['model'])
		predictor = get_predictor(args.predictor, hp['model']).to(device)
		pre_time = pretrain(args.model, model, aug, hp['model'])
		res_dict['pretrain_time'].append(pre_time)
		if isinstance(predictor, nn.Module):
			pred_train(model, predictor, data, split_edge, ncn_loss, hp['model'])
		pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred = test(model, predictor, data, split_edge, hp['model'])
		val_res = full_eval(evaluator, pos_valid_pred, neg_valid_pred)
		test_res = full_eval(evaluator, pos_test_pred, neg_test_pred)
		for key, v_res, t_res in zip(val_res.items(), test_res.values()):
			print(f"{key}:  val: {100 * v_res:.2f}%, test: {100 * t_res:.2f}%")
		res_dict = store_res(test_res, res_dict)
	res_dict, res_latex = compute_table(res_dict, f"{args.model}_{args.augmentation}+{args.predictor}")
	# TODO save CSV des resultats
