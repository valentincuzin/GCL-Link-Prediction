import argparse
import json
import os
import optuna
from optuna.trial import TrialState
import numpy as np
import torch
import torch.nn as nn
from torch_geometric import seed_everything

from src.augmentation import Aug
from src.model import get_model, define_model
from src.datasets import DataSplit, get_evaluator, full_eval
from src.predictor import get_predictor, ProbDecoder
from src.train import pred_train, baseline_train, test
from src.ctrain import pretrain
from src.utils import store_res, compute_table, full_output, commu_repartition

SMALL_DATASETS = ["facebook_friends", "wiki_science", "crime", 
                  "power", "unicodelang", "euroroad"]
DATASETS = ["synthetic_1", "synthetic_2", "synthetic_3", 
            "cora", "citeseer", "pubmed", "collab"]+SMALL_DATASETS
MODELS = ["baseline",
          "grace", "lgrace", "agrace", "ândgrace", "âorgrace", "extagrace", "a2grace", "csgcl", 
          "bgrl", "lbgrl", "âorbgrl", "extabgrl", "a2bgrl", "abgrl"]
AUGMENTATIONS = ["random", "rjc", "rjc2", "raa", "rra", "deg", "pr", "evc", "scom", "sbm", "sbm2"]
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
    parser.add_argument('--use_valedges_as_input', type=int, default=True, choices=[0,1], help="add validation edges to the input adjacency matrix of gnn")
    parser.add_argument('--hp_search', type=int, default=0, help="enter the number of trials, for the search")
    args = parser.parse_args()
    args.dataset = multiparse(args.dataset, DATASETS)
    args.model = multiparse(args.model, MODELS)
    args.augmentation = multiparse(args.augmentation, AUGMENTATIONS)
    args.loss = multiparse(args.loss, LOSS)
    print(args)
    return args

def synthetic_pred(data_split, evaluator, hp):
    res_dict = {"Hits@10": [], "Hits@20": [], "Hits@50": [], "Hits@100": [], 
                "ROCAUC": [], "AP": [], "pretrain_time": []}
    for r in range(data_split.runs):
        seed_everything(r)
        data, split_edge = data_split.get(r)
        if not hasattr(data, "probs") and not hasattr(data, "sizes"):
            data = commu_repartition(data, 'louvain').to(data.x.device)
        # print('block', data.block)
        predictor = ProbDecoder(data.probs, data.block)
        _, _, pos_test_pred, neg_test_pred = test(None, predictor, data, split_edge, hp['model'])
        pos_test_pred += np.random.uniform(-0.0001, 0.0001, pos_test_pred.shape)
        neg_test_pred += np.random.uniform(-0.0001, 0.0001, neg_test_pred.shape)
        test_res = full_eval(evaluator, pos_test_pred, neg_test_pred)
        res_dict = store_res(test_res, res_dict)
    save_name = "louvain_prob_pred"
    df_res, res_latex = compute_table(res_dict, save_name)
    print(df_res)
    return df_res
        
        
def hp_load(dataset: str, args):
    print(f"....{dataset}....")
    if "synthetic" in dataset:
        hp_files = os.path.join('params','synthetic.json')
    elif dataset in SMALL_DATASETS:
        hp_files = os.path.join('params','small.json')
    else:
        hp_files = os.path.join('params', dataset+'.json')
    with open(hp_files) as json_file:
        hp = json.load(json_file)
        hp["model"]["hp_search"] = args.hp_search != 0
        hp["model"]["epochs"] = args.epochs
        hp["model"]["ct_epochs"] = args.ct_epochs
        hp["model"]["use_valedges_as_input"] = args.use_valedges_as_input
    return hp

def update_hp(study, hp):
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        hp[key] = value
    return hp

def train_test_run(model, predictor, data, split_edge, model_name, augmentation, loss_name, evaluator, args, hp, res_dict):
    if model_name != "baseline":
        print(f"..{augmentation}..")
        aug = Aug(data, split_edge, hp['augmentation'], augmentation)
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
    return store_res(test_res, res_dict)

if __name__ == "__main__":
    args = arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for dataset in args.dataset:
        hp = hp_load(dataset, args)
        data_split = DataSplit(dataset, device, args.runs, args.use_valedges_as_input, args.reduce_feature, args.only_feature)
        evaluator = get_evaluator(dataset)
        full_res = []
        if "sbm" in args.augmentation:
            full_res.append(synthetic_pred(data_split, evaluator, hp))
        for model_name in args.model:
            print(f"...{model_name}...")
            for augmentation in args.augmentation:
                for loss_name in args.loss:
                    # print(f".{loss_name}.")
                    # , "PHits@10": [], "PHits@20": [], "PHits@50": [], "PHits@100": [], similar
                    res_dict = {"Hits@10": [], "Hits@20": [], "Hits@50": [], "Hits@100": [], 
                            "ROCAUC": [], "AP": [], "pretrain_time": []}
                    # hyperparameter search then classic runs
                    def _objective(trial):
                        res_dict = {"Hits@10": [], "Hits@20": [], "Hits@50": [], "Hits@100": [], "ROCAUC": [], "AP": [], "pretrain_time": []}
                        r = 0
                        hp['model']['ct_epochs'] = trial.suggest_int('ct_epochs', 600, 2000, 200)
                        hp['model']['gnn_lr'] = trial.suggest_float('gnn_lr', 0.001, 0.1)
                        hp['model']['tau'] = trial.suggest_int('tau', 2, 5)/10
                        hp['model']['batch_size'] = trial.suggest_int('batch_size', 128, 2000, 256)
                        seed_everything(r)
                        data, split_edge = data_split.get(r)
                        model = define_model(trial, model_name, data, hp['model'])
                        predictor = get_predictor(args.predictor, hp['model'])
                        res_dict = train_test_run(model, predictor, data, split_edge, model_name, augmentation, loss_name, evaluator, args, hp, res_dict)
                        hit50 = res_dict.pop("Hits@50")[0]
                        trial.report(hit50, hp['model']['ct_epochs'])
                        # # Handle pruning based on the intermediate value.
                        # if trial.should_prune():
                        #     raise optuna.exceptions.TrialPruned()
                        return hit50
                    if args.hp_search != 0:
                        study = optuna.create_study(direction='maximize')
                        study.optimize(_objective, n_trials=args.hp_search)
                        hp['model'] = update_hp(study, hp['model'])
                    for r in range(args.runs):
                        seed_everything(r)
                        data, split_edge = data_split.get(r)
                        model = get_model(model_name, data, hp['model'])
                        predictor = get_predictor(args.predictor, hp['model'])
                        res_dict = train_test_run(model, predictor, data, split_edge, model_name,augmentation, loss_name, evaluator, args, hp, res_dict)
                    save_name = f"{model_name}{'_'+loss_name if loss_name != "log_sig" else ""}{'_'+augmentation if model_name != "baseline" else ""}"
                    df_res, res_latex = compute_table(res_dict, save_name)
                    print(df_res)
                    full_res.append(df_res)
                if model_name == "baseline":
                    break
        df, tex = full_output(full_res)
        df.to_csv(f'output/{args.name}_{dataset}_res.csv', sep=';')
