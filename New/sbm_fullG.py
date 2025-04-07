import json
import os
import copy


import torch
import torch.nn as nn
from torch_geometric import seed_everything
from torch_geometric.utils import to_undirected

from src.augmentation import Aug
from src.model import get_model
from src.datasets import DataSplit, get_evaluator, full_eval
from src.predictor import get_predictor
from src.train import pretrain, pred_train, baseline_train, test, ncn_loss
from src.utils import store_res, compute_table, full_output

DATASETS = ["cora"]
MODELS = ["baseline", "grace", "csgcl", "bgrl"]
AUGMENTATION = ["sbm", "random"]

def hp_load(dataset: str):
    print(f"....{dataset}....")
    hp_files = os.path.join('params', dataset+'.json')
    with open(hp_files) as json_file:
        hp = json.load(json_file)
        hp["model"]["epochs"] = 100
        hp["model"]["ct_epochs"] = 500
        hp["model"]["use_valedges_as_input"] = True
    return hp

if __name__ == "__main__":
    dataset = DATASETS[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hp = hp_load(dataset)
    data_split = DataSplit(dataset, device, 1, hp["model"]["use_valedges_as_input"])
    evaluator = get_evaluator(dataset)
    full_res = []
    for model_name in MODELS:
        print(f"...{model_name}...")
        for augmentation in AUGMENTATION:
            res_dict = {"Hits@10": [], "Hits@20": [], "Hits@50": [], "Hits@100": [], 'ROCAUC': [], 'AP': [], 'pretrain_time': []}
            for r in range(1):
                seed_everything(r)
                data, split_edge = data_split.get(r)
                model = get_model(model_name, data, hp['model'])
                predictor = get_predictor('mlp', hp['model'])
                if model_name != "baseline":
                    data_full = copy.deepcopy(data)
                    trn_edge_index = split_edge['train']['edge'].t()
                    val_edge_index = split_edge['valid']['edge'].t()
                    # test_edge_index = split_edge['test']['edge'].t()
                    data_full.edge_index = torch.cat([trn_edge_index, val_edge_index], dim=-1)
                    data_full.edge_index = to_undirected(data_full.edge_index)
                    aug = Aug(data_full, hp['augmentation'], 'sbm')
                    pre_time = pretrain(model_name, model, aug, hp['model'])
                    res_dict['pretrain_time'].append(pre_time)
                    if isinstance(predictor, nn.Module):
                        predictor = predictor.to(device)
                        pred_train(model, predictor, data, split_edge, ncn_loss, hp['model'])
                else:
                    baseline_train(model, predictor, data, split_edge, ncn_loss, hp['model'])
                pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred = test(model, predictor, data, split_edge, hp['model'])
                val_res = full_eval(evaluator, pos_valid_pred, neg_valid_pred)
                test_res = full_eval(evaluator, pos_test_pred, neg_test_pred)
                for (key, v_res), t_res in zip(val_res.items(), test_res.values()):
                    print(f"{key}:  val: {100 * v_res:.2f}%, test: {100 * t_res:.2f}%")
                res_dict = store_res(test_res, res_dict)
            save_name = f"{model_name}{'_'+'sbm' if model_name != "baseline" else ""}"
            df_res, res_latex = compute_table(res_dict, save_name)
            full_res.append(df_res)
df, tex = full_output(full_res)
df.to_csv(f'output/cora-test_sbm_res.csv', sep=';')
