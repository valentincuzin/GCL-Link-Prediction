import argparse
import torch

from src.datasets import DataSplit, get_evaluator, full_eval

def arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='cora', choices=["cora", "citeseer", "pubmed", "collab"])
	parser.add_argument('--reduce_feature', type=int, default=None, help='0 for Identity matrix, >0 for PCA reduce')
	parser.add_argument('--only_feature', action='store_true', default=False, help='erase structure information')
	parser.add_argument('--predictor', type=str, default='inner', choices=["inner", "mlp"])
	parser.add_argument('--epochs', type=int, default=1500)
	parser.add_argument('--runs', type=int, default=10)
	parser.add_argument('--use_valedges_as_input', action='store_true', help="add validation edges to the input adjacency matrix of gnn")
	# TODO hyper-paramètre à règler dans un json
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = arguments()
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	data_split = DataSplit(args.dataset, device, args.runs, args.use_valedges_as_input, args.reduce_feature, args.only_feature)
	evaluator = get_evaluator(args.dataset)
	# TODO charger les models et le prédicteur de lien
	# TODO le train et le test doivent être très modulaire
	# TODO pré-entrainement si besoin et train
	# TODO test et full_eval
	# TODO save CSV des resultats
