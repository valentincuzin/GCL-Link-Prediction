import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch_sparse import SparseTensor
from tqdm import tqdm
from ogb.linkproppred import Evaluator


def train_mplp(encoder: nn.Module,
          predictor: nn.Module,
          data,
          split_edge: dict,
          optimizer,
          batch_size: int,
          maskinput: bool = True):
    encoder.train()
    predictor.train()
    device = data.adj_t.device()
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    pos_train_edge = split_edge['train']['edge'].to(device)
    
    optimizer.zero_grad()
    total_loss = total_examples = 0

    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)

    neg_edge_epoch = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
    for perm in tqdm(DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True),desc='Train'):
        edge = pos_train_edge[perm].t()
        if maskinput:
            # adj_t = data.adj_t
            # undirected_edges = torch.cat((edge, edge.flip(0)), dim=-1)
            # target_adj = SparseTensor.from_edge_index(undirected_edges, sparse_sizes=adj_t.sizes())
            # adj_t = spmdiff_(adj_t, target_adj, keep_val=True)
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask].t()
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj_t = data.adj_t


        h = encoder(data.x, adj_t)

        neg_edge = neg_edge_epoch[:,perm]
        train_edges = torch.cat((edge, neg_edge), dim=-1)
        train_label = torch.cat((torch.ones(edge.size()[1]), torch.zeros(neg_edge.size()[1])), dim=0).to(device)
        out = predictor.multidomainforward(h, adj_t, train_edges).squeeze()
        loss = criterion(out, train_label)

        loss.backward()

        if data.x is not None:
            nn.utils.clip_grad_norm_(data.x, 1.0)
        nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
        nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        total_examples += train_label.size(0)
        total_loss += loss.item() * train_label.size(0)
    
    return total_loss / total_examples

def train_ncn(encoder: nn.Module,
          predictor: nn.Module,
          data,
          split_edge: dict,
          optimizer,
          batch_size: int,
          maskinput: bool = True):
    encoder.train()
    predictor.train()
    device = data.x.device()

    pos_train_edge = split_edge['train']['edge'].to(device)
    pos_train_edge = pos_train_edge.t()

    total_loss = []
    adjmask = torch.ones_like(pos_train_edge[0], dtype=torch.bool)
    
    negedge = negative_sampling(data.edge_index.to(pos_train_edge.device), data.adj_t.sizes()[0])
    # for perm in PermIterator(
    #         adjmask.device, adjmask.shape[0], batch_size
    # ):
    for perm in DataLoader(range(adjmask.size(0)), batch_size, shuffle=True):
        optimizer.zero_grad()
        if maskinput:
            adjmask[perm] = 0
            tei = pos_train_edge[:, adjmask]
            adj = SparseTensor.from_edge_index(tei,
                               sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(
                                   pos_train_edge.device, non_blocking=True)
            adjmask[perm] = 1
            adj = adj.to_symmetric()
        else:
            adj = data.adj_t
        h = encoder(data.x, adj)
        edge = pos_train_edge[:, perm]
        pos_outs = predictor.multidomainforward(h,
                                                    adj,
                                                    edge)

        pos_losss = -F.logsigmoid(pos_outs).mean()
        edge = negedge[:, perm]
        neg_outs = predictor.multidomainforward(h, adj, edge)
        neg_losss = -F.logsigmoid(-neg_outs).mean()
        loss = neg_losss + pos_losss
        loss.backward()
        optimizer.step()

        total_loss.append(loss)
    total_loss = np.average([_.item() for _ in total_loss])
    return total_loss

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
    start_time = time.perf_counter()
    if use_valedges_as_input:
        adj_t = data.full_adj_t
        h = encoder(data.x, adj_t)
    pos_test_pred, neg_test_pred = test_split('test')
    end_time = time.perf_counter()
    total_time = end_time - start_time
    print(f'Inference for one epoch Took {total_time:.4f} seconds')
    
    results = {}
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

    return results

def test_output(
    run: int,
    epoch: int,
    encoder: nn.Module,
    predictor: nn.Module,
    data,
    split_edge: dict,
    evaluator: Evaluator,
    writer: SummaryWriter,
    hp: dict,
    res_dict: dict,
):
    # make a test with the evaluator, then print and return results
    t1 = time.time()
    results, h = test(
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
        writer.add_scalars(
            f"{key}_{run}",
            {"trn": result[0], "val": result[1], "tst": result[2]},
            epoch,
        )
        train_hits, valid_hits, test_hits = result
        res_dict[key].append(test_hits)
        print(key)
        print(
            f"Run: {run + 1:02d}, "
            f"Epoch: {epoch:02d}, "
            f"Train: {100 * train_hits:.2f}%, "
            f"Valid: {100 * valid_hits:.2f}%, "
            f"Test: {100 * test_hits:.2f}%"
        )
    print("---", flush=True)
    return res_dict


def run(
    r: int,
    name: str,
    encoder: nn.Module,
    pretrain_function: callable,
    predictor: nn.Module,
    data,
    split_edge: dict,
    evaluator: Evaluator,
    hp: dict,
):
    # train and test the encoder and the predictor
    writer = SummaryWriter(f"./rec/{name}")
    writer.add_text("hyperparams", str(hp))
    if pretrain_function is not None:
        pretrain_function(encoder, data)
    optimizer = torch.optim.Adam(
        [
            {"params": encoder.parameters(), "lr": hp["gnnlr"]},
            {"params": predictor.parameters(), "lr": hp["prelr"]},
        ]
    )

    for epoch in tqdm(range(1, 1 + hp["epochs"])):
        loss = train_ncn(
            encoder,
            predictor,
            data,
            split_edge,
            optimizer,
            hp["batch_size"],
            hp["maskinput"]
        )
        if epoch % 10 == 0:
            print(f"loss {loss:.4f}", flush=True)
    return test_output(
        r, epoch, encoder, predictor, data, split_edge, evaluator, writer, hp
    )
