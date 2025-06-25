import json
import os
import numpy as np
from optuna.trial import TrialState

def hp_load(dataset: str, model: str, augmentation: str, encoder: str, predictor: str):
    print(f"....{dataset}....")
    if "synthetic" in dataset:
        hp_files = 'params/synthetic.json'
    elif os.path.exists(f'params/{dataset}_{model}_enc:{encoder}_pred:{predictor}_{augmentation}.json'):
        hp_files =  f'params/{dataset}_{model}_enc:{encoder}_pred:{predictor}_{augmentation}.json'
    else:
        hp_files = 'params/default.json'
        print("no hp file, default setting load...")
    with open(hp_files) as json_file:
        hp = json.load(json_file)
    # hp['epochs'] = args.epochs
    # hp['ct_epochs'] = args.ct_epochs
    # hp['use_valedges_as_input'] = args.use_valedges_as_input

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
        if key in ['drop_edge_rate_1', 'drop_edge_rate_2', 'drop_feature_rate_1', 'drop_feature_rate_2', 'commu_detect', 'reconstruction_rate']:
            hp[key] = value
        else:
            hp[key] = value
    
    with open(f'{name}.json', 'w', encoding='utf-8') as fichier:
        json.dump(trial.params, fichier, ensure_ascii=False, indent=4)
    return hp

def hp_augmentation(augmentation, trial, hp):
    hp['drop_edge_rate_1'] = trial.suggest_categorical('drop_edge_rate_1', np.arange(0.0, 0.91, 0.1))
    hp['drop_edge_rate_2'] = trial.suggest_categorical('drop_edge_rate_2', np.arange(0.0, 0.91, 0.1))
    hp['drop_feature_rate_1'] = trial.suggest_categorical('drop_feature_rate_1', np.arange(0.0, 0.91, 0.1))
    hp['drop_feature_rate_2'] = trial.suggest_categorical('drop_feature_rate_2', np.arange(0.0, 0.91, 0.1))

    if 'sbm' in augmentation:
        hp['commu_detect'] = trial.suggest_categorical('commu_detect', ['louvain', 'leiden', 'infomap'])

    if any(x in augmentation for x in ['rjc', 'raa', 'rra']):
        hp['reconstruction_rate'] = trial.suggest_categorical('reconstruction_rate', np.arange(0.1, 0.91, 0.1))
    return hp

def hp_train(predictor, trial, hp):
    hp['ct_epochs'] = 5000
    hp['gnn_lr'] = trial.suggest_float('gnn_lr', 0.001, 0.1)
    hp['batch_size'] = trial.suggest_int('batch_size', 64, 6400, 64)
    hp['use_valedges_as_input'] = trial.suggest_categorical('use_valedges_as_input', [True, False])

    hp['epochs'] = trial.suggest_int('epochs', 10, 190, 10)
    if predictor != 'inner':
        hp['pre_lr'] = trial.suggest_float('pre_lr', 0.001, 0.1)
        hp['loss_func'] = trial.suggest_categorical('loss_func', ['log_sig', 'bce'])
    hp['mask_input'] = trial.suggest_categorical('mask_input', [True, False])

    hp['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-4)
    return hp

# encoder
def hp_bgrl_gcn(trial, hp):
    hp['n_layers'] = trial.suggest_int('n_layers', 1, 4)
    hp['layer_sizes'] = []
    for i in range(hp['n_layers']):
        hp['layer_sizes'].append(trial.suggest_int(f'{i+1}_layer_size', 32, 512, 32))
    print(hp['layer_sizes'])
    hp['hidden'] = hp['layer_sizes'][-1]
    hp['batch_layer_norm'] = trial.suggest_categorical('batch_layer_norm', [True, False])
    hp['batchnorm_mm'] = trial.suggest_categorical('batchnorm_mm', np.arange(0.80, 1, 0.01))
    hp['weight_standardization'] = trial.suggest_categorical('weight_standardization', [True, False])
    return hp

def hp_grace_gcn(trial, hp):
    hp['n_layers'] = trial.suggest_int('n_layers', 2, 4)
    hp['hidden'] = trial.suggest_int('hidden', 32, 512, 32)
    hp['activation'] = trial.suggest_categorical('activation', ['identity', 'relu', 'prelu', 'rrelu'])
    hp['skip'] = trial.suggest_categorical('skip', [True, False])

    hp['proj_hidden'] = trial.suggest_int('proj_hidden', 32, 512, 32)
    hp['tau'] = trial.suggest_categorical('tau', np.arange(0.1, 0.91, 0.1))
    return hp
