import json
import os
from optuna.trial import TrialState


def hp_load(dataset: str, save_name: str):
    print(f"....{dataset}....")
    if "synthetic" in dataset:
        hp_files = "params/synthetic.json"
    elif os.path.exists(
        f"params/{dataset}/{save_name}.json"
    ):
        hp_files = f"params/{dataset}/{save_name}.json"
        print('hp loaded !')
    else:
        hp_files = "params/default.json"
        print("no hp file, default setting load...")
    with open(hp_files) as json_file:
        hp = json.load(json_file)

    print("HYPER-PARAM:", hp)
    return hp


def update_hp(study, hp, name):
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

    with open(f"{name}.json", "w", encoding="utf-8") as fichier:
        json.dump(hp, fichier, ensure_ascii=False, indent=4)
    return hp

def hp_augmentation(augmentation, trial, hp):
    if not('sbm' in augmentation and '+' not in augmentation):
        hp["drop_edge_rate_1"] = trial.suggest_float(
            "drop_edge_rate_1", 0.0, 0.9, step=0.1)
        hp["drop_edge_rate_2"] = trial.suggest_float(
            "drop_edge_rate_2", 0.0, 0.9, step=0.1)
    if hp['attr']:
        hp["drop_feature_rate_1"] = trial.suggest_float(
            "drop_feature_rate_1", 0.0, 0.9, step=0.1)
        hp["drop_feature_rate_2"] = trial.suggest_float(
            "drop_feature_rate_2", 0.0, 0.9, step=0.1)
    else:
        hp["drop_feature_rate_1"] = 0.0
        hp["drop_feature_rate_2"] = 0.0
    if "sbm" in augmentation:
        hp["commu_detect"] = trial.suggest_categorical("commu_detect", ["louvain", "leiden", "infomap"])

    # if any(x in augmentation for x in ["rjc", "raa", "rra"]):
    #     hp["reconstruction_rate"] = trial.suggest_float(
    #         "reconstruction_rate", 0.1, 0.90, step=0.1)
    return hp


def hp_train(predictor, trial, hp):
    hp["ct_epochs"] = trial.suggest_categorical("ct_epochs", [100, 500, 1500, 3000])
    hp["proj_hidden"] = trial.suggest_int("proj_hidden", 64, 512, 64)

    hp["gnn_lr"] = trial.suggest_float("gnn_lr", 0.0001, 0.01, log=True)
    hp["batch_size"] = trial.suggest_int("batch_size", 256, 6400, 64)
    hp["use_valedges_as_input"] = False

    hp["epochs"] = 100
    hp["loss_func"] = trial.suggest_categorical("loss_func", ["log_sig", "bce"])
    if predictor != "inner":
        hp["pre_lr"] = trial.suggest_float("pre_lr", 0.0001, 0.01, log=True)
    if predictor == "ncn":
        hp["predp"] = trial.suggest_float("predp", 0.00, 0.90, step=0.01)
        hp["preedp"] = trial.suggest_float("preedp", 0.00, 0.90, step=0.01)
        hp["lnnn"] = trial.suggest_categorical("lnnn", [True, False])
        hp["use_xlin"] = trial.suggest_categorical("use_xlin", [True, False])
        hp["tailact"] = trial.suggest_categorical("tailact", [True, False])
        hp["twolayerlin"] = trial.suggest_categorical("twolayerlin", [True, False])
    
    hp["mask_input"] = trial.suggest_categorical("mask_input", [True, False])
    hp["weight_decay"] = trial.suggest_float("weight_decay", 1e-6, 1e-4)
    return hp


# encoder
def hp_bgrl_gcn(trial, hp):
    hp["n_layers"] = trial.suggest_int("n_layers", 1, 4)
    layer_size = trial.suggest_int("layer_size",  64, 512, 64)
    hp["layer_sizes"] = []
    for _ in range(hp["n_layers"]):
        hp["layer_sizes"].append(layer_size)
    hp["layer_sizes"][-1] = int(hp["layer_sizes"][-1]/2)
    print(hp["layer_sizes"])
    hp["hidden"] = hp["layer_sizes"][-1]
    hp["batch_layer_norm"] = trial.suggest_categorical(
        "batch_layer_norm", [True, False]
    )
    if hp["batch_layer_norm"]:
        hp["batchnorm_mm"] = trial.suggest_float(
            "batchnorm_mm", 0.80, 1, step=0.01)
    else:
        hp["batchnorm_mm"] = None
    hp["weight_standardization"] = trial.suggest_categorical(
        "weight_standardization", [True, False]
    )
    return hp


def hp_grace_gcn(trial, hp):
    hp["n_layers"] = trial.suggest_int("n_layers", 2, 4)
    hp["hidden"] = trial.suggest_int("hidden", 64, 512, 64)
    hp["activation"] = trial.suggest_categorical(
        "activation", ["identity", "relu", "prelu"]
    )
    hp["skip"] = trial.suggest_categorical("skip", [True, False])

    return hp


def hp_ncn_gcn(trial, hp):
    hp["n_layers"] = trial.suggest_int("n_layers", 1, 4)
    hp["hidden"] = trial.suggest_int("hidden", 64, 512, 64)
    hp["gnn_dp"] = trial.suggest_float("gnn_dp", 0.00, 0.90, step=0.01)
    hp["layer_norm"] = trial.suggest_categorical("layer_norm", [True, False])
    hp["res"] = trial.suggest_categorical("res", [True, False])
    hp["conv_fn"] = trial.suggest_categorical("conv_fn", ["gcn", "puregcn"])
    hp["jk"] = trial.suggest_categorical("jk", [True, False])
    hp["edrop"] = trial.suggest_float("edrop", 0.00, 0.90, step=0.01)
    hp["xdropout"] = trial.suggest_float("xdropout", 0.00, 0.90, step=0.01)
    hp["taildropout"] = trial.suggest_float(
        "taildropout", 0.00, 0.90, step=0.01)
    return hp
