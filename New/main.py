import argparse
import optuna
import numpy as np
import torch
import torch.nn as nn
from torch_geometric import seed_everything

from ogb.linkproppred import Evaluator
from torch_geometric.data import Data

from src.augmentation import Aug, gen_sbm_bank
from src.model import get_model
from src.datasets import DataSplit, get_evaluator, full_eval
from src.predictor import get_predictor, ProbDecoder, InnerProd
from src.train import pred_train, baseline_train, test
from src.ctrain import pretrain
from src.utils import store_res, compute_table, full_output, commu_distrib
import src.hp as hp

SMALL_DATASETS = [
    # from https://networks.skewed.de/
    "facebook_friends",
    "wiki_science",
    "crime",
    "power",
    "unicodelang",
    "euroroad",
    "escort",
    "tips",
    "pol_kato",
    "pol_robertson",
    "yeast",
    "netscience",
    # from https://github.com/Barcavin/efficient-node-labelling
    "USAir",
    "NS",
    "PB",
    "Yeast",
    "Celegans",
    "Power",
    "Router",
    "Ecoli",
]
DATASETS = [
    "synthetic_1",
    "synthetic_2",
    "synthetic_3",
    # from https://pytorch-geometric.readthedocs.io/en/latest/cheatsheet/data_cheatsheet.html
    "cora",
    "citeseer",
    "pubmed",
    "cs",
    "physics",
    "computers",
    "photo",
    # from https://ogb.stanford.edu/
    "collab",
    "ddi",
] + SMALL_DATASETS
MODELS = ["baseline", "grace", "lgrace", "agrace", "csgcl", "bgrl", "lbgrl", "abgrl"]
AUGMENTATIONS = [
    "random",
    "deg",
    "pr",
    "evc",
    "scom",
    "sbm",
]
ENCODER = ["grace", "bgrl", "ncn"]
PREDICTOR = ["inner", "mlp", "ncn"]


def arguments():
    def multiparse(input: str, choices: list):
        if input == "all":
            return choices
        if "," not in input and input in choices:
            return [input]
        inputs: list = [i.strip() for i in input.split(",")]
        return inputs

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument(
        "--reduce_feature",
        type=int,
        default=None,
        help="0 for Identity matrix, >0 for PCA reduce",
    )
    parser.add_argument(
        "--only_feature",
        action="store_true",
        default=False,
        help="erase structure information",
    )
    parser.add_argument("--model", type=str, default="baseline")
    parser.add_argument("--augmentation", type=str, default="random")
    parser.add_argument("--encoder", type=str, default="gcn_ncn")
    parser.add_argument("--predictor", type=str, default="mlp")
    parser.add_argument("--save", type=str, default="test/")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument(
        "--hp_search",
        type=int,
        default=0,
        help="enter the number of trials, for the search",
    )
    parser.add_argument(
        "--hp_metric",
        type=str,
        default="Hits@50",
        choices=["Hits@20", "Hits@50", "Hits@100", "AUC", "AP"],
    )
    args = parser.parse_args()
    args.dataset = multiparse(args.dataset, DATASETS)
    args.model = multiparse(args.model, MODELS)
    args.augmentation = multiparse(args.augmentation, AUGMENTATIONS)
    args.encoder = multiparse(args.encoder, ENCODER)
    args.predictor = multiparse(args.predictor, PREDICTOR)
    print(args)
    return args


def commu_prob_pred(data_split: DataSplit, evaluator: Evaluator, param: dict):
    """
    runs and test of the communities probs based predictor

    Args:
        data_split (DataSplit):
        evaluator (Evaluator):
        param (dict):

    Returns:
        pd.DataFrame: results
    """
    res_dict = {
        "Hits@10": [],
        "Hits@20": [],
        "Hits@50": [],
        "Hits@100": [],
        "ROCAUC": [],
        "AP": [],
        "pretrain_time": [],
    }
    for r in range(data_split.runs):
        seed_everything(r)
        data, split_edge = data_split.get(r)
        if not hasattr(data, "probs") and not hasattr(data, "sizes"):
            data = commu_distrib(data, param["commu_detect"]).to(data.x.device)
        predictor = ProbDecoder(data.probs, data.block)
        _, _, pos_test_pred, neg_test_pred = test(
            None, predictor, data, split_edge, param
        )
        pos_test_pred += np.random.uniform(-0.0001, 0.0001, pos_test_pred.shape)
        neg_test_pred += np.random.uniform(-0.0001, 0.0001, neg_test_pred.shape)
        test_res = full_eval(evaluator, pos_test_pred, neg_test_pred)
        res_dict = store_res(test_res, res_dict)
    save_name = f"{param['commu_detect']}_prob_pred"
    df_res, res_latex = compute_table(res_dict, save_name)
    print(df_res)
    return df_res


def train_test_run(
    model: nn.Module,
    predictor: nn.Module | InnerProd,
    data: Data,
    split_edge: dict,
    model_name: str,
    augmentation_name: str,
    evaluator: Evaluator,
    param: dict,
    res_dict: dict,
    run: int = 0,
    valid: bool = False,
):
    """
    function to run a train then a test on link_prediction

    Args:
        model (nn.Module): GCN encoder
        predictor (nn.Module | InnerProd): Link Predictor
        data (Data): input graph
        split_edge (dict):
        model_name (str): Instance Discrimination learning method name
        augmentation_name (str):
        evaluator (Evaluator):
        param (dict):
        res_dict (dict):
        run (int, optional): number of the current run. Defaults to 0.
        valid (bool, optional): return valid score if true. Defaults to False.

    Returns:
        pd.DataFrame: results
    """
    if model_name != "baseline":
        aug = Aug(data, split_edge, param, augmentation_name, run)
        pre_time = pretrain(model_name, model, aug, param)
        res_dict["pretrain_time"].append(pre_time)
        if isinstance(predictor, nn.Module):
            predictor = predictor.to(device)
            pred_train(model, predictor, data, split_edge, param)
    else:
        baseline_train(model, predictor, data, split_edge, param)
    pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred = test(
        model, predictor, data, split_edge, param
    )
    val_res = full_eval(evaluator, pos_valid_pred, neg_valid_pred)
    test_res = full_eval(evaluator, pos_test_pred, neg_test_pred)
    for (key, v_res), t_res in zip(val_res.items(), test_res.values()):
        print(f"{key}:  val: {100 * v_res:.2f}%, test: {100 * t_res:.2f}%")
    res = store_res(val_res, res_dict) if valid else store_res(test_res, res_dict)
    return res


if __name__ == "__main__":
    args = arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for dataset in args.dataset:
        data_split = DataSplit(
            dataset, device, args.runs, args.reduce_feature, args.only_feature
        )
        evaluator = get_evaluator(dataset)
        full_res = []
        if any(
            ("sbm" in element and "fast" not in element)
            for element in args.augmentation
        ):
            gen_sbm_bank(data_split, args.runs)
        for model_name in args.model:
            for encoder_name in args.encoder:
                for predictor_name in args.predictor:
                    for augmentation_name in args.augmentation:
                        save_name = f"{model_name},enc:{encoder_name},pred:{predictor_name}{',' + augmentation_name if model_name != 'baseline' else ''}"
                        print(f"...{dataset},{save_name}...")

                        # hyperparameter search then classic runs
                        if args.hp_search == 0:
                            param = hp.hp_load(dataset, save_name)
                        else:
                            param = {}
                        if args.hp_search != 0:

                            def _objective(trial):
                                res_dict = {
                                    "Hits@10": [],
                                    "Hits@20": [],
                                    "Hits@50": [],
                                    "Hits@100": [],
                                    "ROCAUC": [],
                                    "AP": [],
                                    "pretrain_time": [],
                                }
                                seed_everything(0)
                                global param
                                if model_name != "baseline":
                                    param = hp.hp_augmentation(
                                        augmentation_name, trial, param
                                    )
                                param = hp.hp_train(predictor_name, trial, param)
                                switch = {
                                    "gcn_bgrl": hp.hp_bgrl_gcn,
                                    "gcn_grace": hp.hp_grace_gcn,
                                    "gcn_ncn": hp.hp_ncn_gcn,
                                }
                                param = switch[encoder_name](trial, param)
                                if "grace" in model_name:
                                    param["tau"] = trial.suggest_float(
                                        "tau", 0.1, 0.9, step=0.1
                                    )
                                data, split_edge = data_split.get(0)
                                model = get_model(encoder_name, model_name, data, param)
                                predictor = get_predictor(predictor_name, param)
                                res_dict = train_test_run(
                                    model,
                                    predictor,
                                    data,
                                    split_edge,
                                    model_name,
                                    augmentation_name,
                                    evaluator,
                                    param,
                                    res_dict,
                                    run=0,
                                    valid=True,
                                )
                                score = res_dict.pop(args.hp_metric)[0]
                                return score

                            sampler = optuna.samplers.TPESampler(multivariate=True)
                            study = optuna.create_study(
                                sampler=sampler, direction="maximize"
                            )
                            study.optimize(_objective, n_trials=args.hp_search)
                            param = hp.update_hp(
                                study, param, f"params/{dataset}/{save_name}"
                            )

                        res_dict = {
                            "Hits@10": [],
                            "Hits@20": [],
                            "Hits@50": [],
                            "Hits@100": [],
                            "ROCAUC": [],
                            "AP": [],
                            "pretrain_time": [],
                        }
                        if "sbm" in augmentation_name and model_name != "baseline":
                            full_res.append(
                                commu_prob_pred(data_split, evaluator, param)
                            )
                        # classic runs
                        for r in range(args.runs):
                            seed_everything(r)
                            data, split_edge = data_split.get(r)
                            model = get_model(encoder_name, model_name, data, param)
                            predictor = get_predictor(predictor_name, param)
                            res_dict = train_test_run(
                                model,
                                predictor,
                                data,
                                split_edge,
                                model_name,
                                augmentation_name,
                                evaluator,
                                param,
                                res_dict,
                                run=r,
                            )
                        df_res, res_latex = compute_table(res_dict, save_name)
                        print(df_res)
                        full_res.append(df_res)
                        df, tex = full_output(full_res)
                        df.to_csv(f"output/{args.save}_{dataset}.csv", sep=";")
                        if model_name == "baseline":
                            break
