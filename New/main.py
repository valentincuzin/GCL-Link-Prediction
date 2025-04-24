import argparse
import json
import os
import torch
import torch.nn as nn
from torch_geometric import seed_everything

from src.augmentation import Aug
from src.model import get_model
from src.datasets import DataSplit, get_evaluator, full_eval
from src.predictor import get_predictor, ProbDecoder
from src.train import pretrain, pred_train, baseline_train, test, get_loss
from src.utils import store_res, compute_table, full_output

DATASETS = ["synthetic_1", "synthetic_2", "synthetic_3", "facebook_friends", "wiki_science", "crime", "cora", "citeseer", "pubmed", "collab"]
MODELS = ["baseline", "grace", "lgrace", "agrace", "ândgrace", "âorgrace", "extagrace", "a2grace", "csgcl", "bgrl", "âorbgrl", "extabgrl", "a2bgrl", "abgrl"]
AUGMENTATIONS = ["random", "deg", "pr", "evc", "scom", "sbm", "sbm2"]
LOSS = ["log_sig", "bce", "auc", "hinge_auc"]

def arguments():
    def multiparse(input: str, choices: list):
        if input == "all":
            return choices
        if ',' not in input and input in choices:
            return [input]
        inputs: list = [i.strip() for i in input.split(',')]
        return inputs

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--reduce_feature', type=int, default=None, help='0 for Identity matrix, >0 for PCA reduce')
    parser.add_argument('--only_feature', action='store_true', default=False, help='erase structure information')
    parser.add_argument('--model', type=str, default='baseline')
    parser.add_argument('--augmentation', type=str, default='random')
    parser.add_argument('--predictor', type=str, default='mlp', choices=["inner", "mlp", "prob"])
    parser.add_argument('--loss', type=str, default='log_sig')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--ct_epochs', type=int, default=500)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--use_valedges_as_input', action='store_true', default=True, help="add validation edges to the input adjacency matrix of gnn")
    args = parser.parse_args()
    args.dataset = multiparse(args.dataset, DATASETS)
    args.model = multiparse(args.model, MODELS)
    args.augmentation = multiparse(args.augmentation, AUGMENTATIONS)
    args.loss = multiparse(args.loss, LOSS)
    print(args)
    return args

def synthetic_pred(data_split, evaluator, hp):
    res_dict = {"Hits@10": [], "Hits@20": [], "Hits@50": [], "Hits@100": [], 'ROCAUC': [], 'AP': [], 'pretrain_time': []}
    for r in range(data_split.runs):
        seed_everything(r)
        data, split_edge = data_split.get(r)
        predictor = ProbDecoder(data.probs, data.block)
        _, _, pos_test_pred, neg_test_pred = test(None, predictor, data, split_edge, hp['model'])
        test_res = full_eval(evaluator, pos_test_pred, neg_test_pred)
        res_dict = store_res(test_res, res_dict)
    save_name = "synthetic_prob_pred"
    df_res, res_latex = compute_table(res_dict, save_name)
    print(df_res)
    return df_res
        
        
def hp_load(dataset: str, args):
    print(f"....{dataset}....")
    if "synthetic" in dataset:
        hp_files = os.path.join('params','synthetic.json')
    else:
        hp_files = os.path.join('params', dataset+'.json')
    with open(hp_files) as json_file:
        hp = json.load(json_file)
        hp["model"]["epochs"] = args.epochs
        hp["model"]["ct_epochs"] = args.ct_epochs
        hp["model"]["use_valedges_as_input"] = args.use_valedges_as_input
    return hp

if __name__ == "__main__":
    args = arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for dataset in args.dataset:
        hp = hp_load(dataset, args)
        data_split = DataSplit(dataset, device, args.runs, args.use_valedges_as_input, args.reduce_feature, args.only_feature)
        evaluator = get_evaluator(dataset)
        full_res = []
        if "synthetic" in dataset:
            full_res.append(synthetic_pred(data_split, evaluator, hp))
        for model_name in args.model:
            print(f"...{model_name}...")
            for augmentation in args.augmentation:
                res_dict = {"Hits@10": [], "Hits@20": [], "Hits@50": [], "Hits@100": [], 'ROCAUC': [], 'AP': [], 'pretrain_time': []}
                for loss_name in args.loss:
                    # print(f".{loss_name}.")
                    for r in range(args.runs):
                        seed_everything(r)
                        data, split_edge = data_split.get(r)
                        model = get_model(model_name, data, hp['model'])
                        predictor = get_predictor(args.predictor, hp['model'])
                        if model_name != "baseline":
                            print(f"..{augmentation}..")
                            aug = Aug(data, hp['augmentation'], augmentation)
                            pre_time = pretrain(model_name, model, aug, hp['model'])
                            res_dict['pretrain_time'].append(pre_time)
                            if isinstance(predictor, nn.Module):
                                predictor = predictor.to(device)
                                pred_train(model, predictor, data, split_edge, loss_name, hp['model'])
                        else:
                            baseline_train(model, predictor, data, split_edge, loss_name, hp['model'])
                        pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred = test(model, predictor, data, split_edge, hp['model'])
                        val_res = full_eval(evaluator, pos_valid_pred, neg_valid_pred)
                        test_res = full_eval(evaluator, pos_test_pred, neg_test_pred)
                        for (key, v_res), t_res in zip(val_res.items(), test_res.values()):
                            print(f"{key}:  val: {100 * v_res:.2f}%, test: {100 * t_res:.2f}%")
                        res_dict = store_res(test_res, res_dict)
                    save_name = f"{model_name}{'_'+loss_name if loss_name != "log_sig" else ""}{'_'+augmentation if model_name != "baseline" else ""}"
                    df_res, res_latex = compute_table(res_dict, save_name)
                    print(df_res)
                    full_res.append(df_res)
                if model_name == "baseline":
                    break
        df, tex = full_output(full_res)
        df.to_csv(f'output/{args.name}_{dataset}_res.csv', sep=';')
