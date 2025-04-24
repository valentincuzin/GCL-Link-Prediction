import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch_geometric import seed_everything
from ogb.linkproppred import Evaluator
from utils import compute_table

def get_evaluator(dataset: str = 'ogbl-ppa'):
    if dataset in ["Cora", "Citeseer", "Pubmed", 'ogbl-ppa']:
        evaluator = Evaluator(name='ogbl-ppa')
    else:
        evaluator = Evaluator(name=f'ogbl-{dataset}')
    return evaluator

@torch.no_grad()
def test(encoder: nn.Module, predictor: nn.Module, data, split_edge: dict, evaluator: Evaluator, batch_size: int,
         use_valedges_as_input, res_dict: dict):
    # adapted from MPLP code
    encoder.eval()
    predictor.eval()
    device = data.adj_t.device()
    adj_t = data.adj_t
    h = encoder(data.x, adj_t)

    def test_split(split):
        # pred positive edges and negatives edges for nodes in the split
        pos_test_edge = split_edge[split]['edge'].to(device)
        neg_test_edge = split_edge[split]['edge_neg'].to(device)
        pos_test_preds = []
        for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
            edge = pos_test_edge[perm].t()
            out = predictor(h, adj_t, edge)
            pos_test_preds += [out.squeeze().cpu()]
        pos_test_pred = torch.cat(pos_test_preds, dim=0)
        neg_test_preds = []
        for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
            edge = neg_test_edge[perm].t()
            out = predictor(h, adj_t, edge)
            neg_test_preds += [out.squeeze().cpu()]
        neg_test_pred = torch.cat(neg_test_preds, dim=0)

        return pos_test_pred, neg_test_pred

    pos_valid_pred, neg_valid_pred = test_split('valid')
    if use_valedges_as_input:
        adj_t = data.full_adj_t
        h = encoder(data.x, adj_t)
    pos_test_pred, neg_test_pred = test_split('test')
    
    results = {}
    evaluator.eval_metric = 'hits@k'
    for K in [10, 20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']
        results[f'Hits@{K}'] = (valid_hits, test_hits)
    pos_valid_pred = pos_valid_pred[:neg_valid_pred.shape[0]]
    pos_test_pred = pos_test_pred[:neg_test_pred.shape[0]]
    evaluator.eval_metric = 'rocauc'
    valid_auc = evaluator.eval({
        'y_pred_pos': pos_valid_pred,
        'y_pred_neg': neg_valid_pred,
    })['rocauc']
    test_auc = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['rocauc']
    results['ROCAUC'] = (valid_auc, test_auc)
    return results, pos_test_pred, neg_test_pred


def runs(name: str,
         model_init_function: callable,
         pretrain_function: callable,
         train_function: callable,
         data_split,
         evaluator,
         hp: dict):
    res_dict = {"Hits@10": [], "Hits@20": [], "Hits@50": [], "Hits@100": [], 'ROCAUC': [], 'pretrain_time': []}
    print(f"### {name} ###")
    for r in range(hp['runs']):
        seed_everything(r)
        data, split_edge = data_split.get(r)
        encoder, predictor = model_init_function(data, hp)
        
        # train and test the encoder and the predictor
        if pretrain_function is not None:
            pre_time = pretrain_function(encoder, data, hp['pre_param'])
            res_dict['pretrain_time'].append(pre_time)
        if train_function is not None:
            loss_res = []
            t1 = time.time()
            for epoch in tqdm(range(1, 1 + hp["epochs"])):
                loss = train_function(
                    encoder,
                    predictor,
                    data,
                    split_edge,
                    optimizer,
                    hp["batch_size"],
                    hp["maskinput"]
                )
                if epoch % 10 == 0:
                    loss_res.append(round(float(loss), 3))
            print('train loss: ', loss_res)
            print(f"train time: {time.time()-t1:.2f} s")

        print(f"Run: {r + 1}")
        t1 = time.time()
        results, pos_test_pred, neg_test_pred = test(
            encoder,
            predictor,
            data,
            split_edge,
            evaluator,
            hp["batch_size"],
            hp["use_valedges_as_input"],
            res_dict
        )
        print(f"test time {time.time() - t1:.2f} s")
        for key, result in results.items():
            valid_hits, test_hits = result
            res_dict[key].append(test_hits)
            print(
                f"{key}: "
                f"Valid: {100 * valid_hits:.2f}%, "
                f"Test: {100 * test_hits:.2f}%"
            )
        if run == 0:
            res_dict['test_pred'] = torch.cat((pos_test_pred, neg_test_pred)).tolist()
        print("---", flush=True)

    res_dict, res_latex = compute_table(res_dict, name)
    print(f"\n\n### {name} ###")
    print(f'\n{res_latex}\n\n')
    return res_dict
