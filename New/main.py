import argparse
import optuna
import numpy as np
import torch
import torch.nn as nn
from torch_geometric import seed_everything

from src.augmentation import Aug
from src.model import get_model
from src.datasets import DataSplit, get_evaluator, full_eval
from src.predictor import get_predictor, ProbDecoder
from src.train import pred_train, baseline_train, test
from src.ctrain import pretrain
from src.utils import store_res, compute_table, full_output, commu_repartition
from src.hp import hp_load, update_hp, hp_augmentation, hp_train, hp_bgrl_gcn, hp_grace_gcn

SMALL_DATASETS = ["facebook_friends", "wiki_science", "crime", 
                  "power", "unicodelang", "euroroad", 
                  "escort", "tips", "pol_kato", "pol_robertson", "yeast", "netscience", 
                  'USAir', 'NS', 'PB', 'Yeast', 'Celegans', 'Power', 'Router', 'Ecoli']
DATASETS = ["synthetic_1", "synthetic_2", "synthetic_3", 
            "cora", "citeseer", "pubmed", 
            "cs", "physics", "computers", "photo",
            "collab", "ddi"]+SMALL_DATASETS
MODELS = ["baseline",
          "grace", "lgrace", "a2grace", "csgcl", 
          "bgrl", "lbgrl", "a2bgrl"]
AUGMENTATIONS = ["random", "rjc", "rjc2", "raa", "rra", "deg", "pr", "evc", "scom", "sbm", "sbm2", "sgf"]
LOSS = ["log_sig", "bce", "auc", "hinge_auc"]
ENCODER = ['grace', 'bgrl', 'ncn', 'mplp']
PREDICTOR = ['inner', 'mlp', 'ncn']

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
    parser.add_argument('--encoder', type=str, default='gcn_grace')
    parser.add_argument('--predictor', type=str, default='mlp')
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--hp_search', type=int, default=0, help="enter the number of trials, for the search")
    args = parser.parse_args()
    args.dataset = multiparse(args.dataset, DATASETS)
    args.model = multiparse(args.model, MODELS)
    args.augmentation = multiparse(args.augmentation, AUGMENTATIONS)
    args.encoder = multiparse(args.encoder, ENCODER)
    args.predictor = multiparse(args.predictor, PREDICTOR)
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
        _, _, pos_test_pred, neg_test_pred = test(None, predictor, data, split_edge, hp)
        pos_test_pred += np.random.uniform(-0.0001, 0.0001, pos_test_pred.shape)
        neg_test_pred += np.random.uniform(-0.0001, 0.0001, neg_test_pred.shape)
        test_res = full_eval(evaluator, pos_test_pred, neg_test_pred)
        res_dict = store_res(test_res, res_dict)
    save_name = "louvain_prob_pred"
    df_res, res_latex = compute_table(res_dict, save_name)
    print(df_res)
    return df_res

def train_test_run(model, predictor, data, split_edge, model_name, augmentation, evaluator, hp, res_dict, valid = False):
    if model_name != "baseline":
        print(f"..{augmentation}..")
        aug = Aug(data, split_edge, hp, augmentation)
        pre_time = pretrain(model_name, model, aug, hp)
        res_dict['pretrain_time'].append(pre_time)
        if isinstance(predictor, nn.Module):
            predictor = predictor.to(device)
            pred_train(model, predictor, data, split_edge, hp)
    else:
        baseline_train(model, predictor, data, split_edge, hp)
    pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred = test(model, predictor, data, split_edge, hp)
    val_res = full_eval(evaluator, pos_valid_pred, neg_valid_pred)
    test_res = full_eval(evaluator, pos_test_pred, neg_test_pred)
    for (key, v_res), t_res in zip(val_res.items(), test_res.values()):
        print(f"{key}:  val: {100 * v_res:.2f}%, test: {100 * t_res:.2f}%")
    res = store_res(val_res, res_dict) if valid else store_res(test_res, res_dict)
    return res

if __name__ == "__main__":
    args = arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for dataset in args.dataset:
        data_split = DataSplit(dataset, device, args.runs, args.reduce_feature, args.only_feature)
        evaluator = get_evaluator(dataset)
        full_res = []
        for model_name in args.model:
            for encoder_name in args.encoder:
                for predictor_name in args.predictor:
                    print(f"...{model_name}...")
                    for augmentation in args.augmentation:
                        
                        if args.hp_search == 0:
                            hp = hp_load(dataset, model_name, augmentation, encoder_name, predictor_name)
                        else:
                            hp = {}
                        res_dict = {"Hits@10": [], "Hits@20": [], "Hits@50": [], "Hits@100": [], 
                                "ROCAUC": [], "AP": [], "pretrain_time": []}
                        # hyperparameter search then classic runs
                        def _objective(trial):
                            res_dict = {"Hits@10": [], "Hits@20": [], "Hits@50": [], "Hits@100": [], "ROCAUC": [], "AP": [], "pretrain_time": []}
                            seed_everything(0)
                            global hp
                            if model_name != 'baseline':
                                hp = hp_augmentation(augmentation, trial, hp)
                            hp = hp_train(predictor_name, trial, hp)
                            switch = {
                                'gcn_bgrl': hp_bgrl_gcn,
                                'gcn_grace': hp_grace_gcn
                            }
                            hp = switch[encoder_name](trial, hp)
                            data, split_edge = data_split.get(0)
                            model = get_model(encoder_name, model_name, data, hp)
                            predictor = get_predictor(predictor_name, hp)
                            res_dict = train_test_run(model, predictor, data, split_edge, model_name, augmentation, evaluator, hp, res_dict, valid=True)
                            hit50 = res_dict.pop("Hits@50")[0]
                            trial.report(hit50, hp['ct_epochs'])
                            return hit50

                        if args.hp_search != 0:
                            study = optuna.create_study(direction='maximize')
                            study.optimize(_objective, n_trials=args.hp_search)
                            hp = update_hp(study, hp, f"params/{dataset}_{model_name}_enc:{encoder_name}_pred:{predictor_name}{'_'+augmentation if model_name != "baseline" else ""}")
                        if "sbm" in augmentation:
                            full_res.append(synthetic_pred(data_split, evaluator, hp))
                        for r in range(args.runs):
                            seed_everything(r)
                            data, split_edge = data_split.get(r)
                            model = get_model(encoder_name, model_name, data, hp)
                            predictor = get_predictor(predictor_name, hp)
                            res_dict = train_test_run(model, predictor, data, split_edge, model_name,augmentation, evaluator, hp, res_dict)
                        save_name = f"{model_name}_enc:{encoder_name}_pred:{predictor_name}{'_'+augmentation if model_name != "baseline" else ""}"
                        df_res, res_latex = compute_table(res_dict, save_name)
                        print(df_res)
                        full_res.append(df_res)
                        df, tex = full_output(full_res)
                        df.to_csv(f'output/{args.name}_{dataset}_res.csv', sep=';')
                        if model_name == "baseline":
                            break
        