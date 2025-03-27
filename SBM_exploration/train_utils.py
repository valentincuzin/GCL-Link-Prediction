import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from ogb.linkproppred import Evaluator

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

def test_output(
    run: int,
    encoder: nn.Module,
    predictor: nn.Module,
    data,
    split_edge: dict,
    evaluator: Evaluator,
    writer: SummaryWriter,
    hp: dict,
    res_dict: dict,
    epoch: int = 0,
):
    # make a test with the evaluator, then print and return results
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
    print(f"Run: {run + 1:02d}, "
            f"Epoch: {epoch:02d}, ")
    for key, result in results.items():
        writer.add_scalars(
            f"{key}_{run}",
            {"val": result[0], "tst": result[1]},
            epoch,
        )
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
    return res_dict

