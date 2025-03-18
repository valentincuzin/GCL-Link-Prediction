import time
import math
import torch
import torchhd
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch import Tensor
from tqdm import tqdm
from torch_sparse import SparseTensor, masked_select_nnz, matmul
from ogb.linkproppred import Evaluator
from src.ogbdataset import loaddataset
from torch_geometric import seed_everything
from torch_geometric.utils import degree, k_hop_subgraph, to_edge_index


###############################################
# Code from NCN

def elem2spm(element: Tensor, sizes: list[int]) -> SparseTensor:
    # Convert adjacency matrix to a 1-d vector
    col = torch.bitwise_and(element, 0xffffffff)
    row = torch.bitwise_right_shift(element, 32)
    return SparseTensor(row=row, col=col, sparse_sizes=sizes).to_device(
        element.device).fill_value_(1.0)


def spm2elem(spm: SparseTensor) -> Tensor:
    # Convert 1-d vector to an adjacency matrix
    sizes = spm.sizes()
    elem = torch.bitwise_left_shift(spm.storage.row(),
                                    32).add_(spm.storage.col())
    #elem = spm.storage.row()*sizes[-1] + spm.storage.col()
    #assert torch.all(torch.diff(elem) > 0)
    return elem

def spmoverlap_(adj1: SparseTensor, adj2: SparseTensor) -> SparseTensor:
    '''
    Compute the overlap of neighbors (rows in adj). The returned matrix is similar to the hadamard product of adj1 and adj2
    '''
    assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

    if element2.shape[0] > element1.shape[0]:
        element1, element2 = element2, element1

    idx = torch.searchsorted(element1[:-1], element2)
    mask = (element1[idx] == element2)
    retelem = element2[mask]
    '''
    nnz1 = adj1.nnz()
    element = torch.cat((adj1.storage.row(), adj2.storage.row()), dim=-1)
    element.bitwise_left_shift_(32)
    element[:nnz1] += adj1.storage.col()
    element[nnz1:] += adj2.storage.col()
    
    element = torch.sort(element, dim=-1)[0]
    mask = (element[1:] == element[:-1])
    retelem = element[:-1][mask]
    '''

    return elem2spm(retelem, adj1.sizes())


def spmoverlap_notoverlap_(
        adj1: SparseTensor,
        adj2: SparseTensor) -> tuple[SparseTensor, SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''
    # assert adj1.sizes() == adj2.sizes()
    element1 = spm2elem(adj1)
    element2 = spm2elem(adj2)

    if element1.shape[0] == 0:
        retoverlap = element1
        retelem1 = element1
        retelem2 = element2
    else:
        idx = torch.searchsorted(element1[:-1], element2)
        matchedmask = (element1[idx] == element2)

        maskelem1 = torch.ones_like(element1, dtype=torch.bool)
        maskelem1[idx[matchedmask]] = 0
        retelem1 = element1[maskelem1]

        retoverlap = element2[matchedmask]
        retelem2 = element2[torch.logical_not(matchedmask)]
    sizes = adj1.sizes()
    return elem2spm(retoverlap,
                    sizes), elem2spm(retelem1,
                                     sizes), elem2spm(retelem2, sizes)

def sparsesample_reweight(adj: SparseTensor, deg: int) -> SparseTensor:
    '''
    another implementation for sampling elements from a adjacency matrix. It will also scale the sampled elements.
    
    '''
    rowptr, col, _ = adj.csr()
    rowcount = adj.storage.rowcount()
    mask = rowcount > deg

    rowcount = rowcount[mask]
    rowptr = rowptr[:-1][mask]

    rand = torch.rand((rowcount.size(0), deg), device=col.device)
    rand.mul_(rowcount.to(rand.dtype).reshape(-1, 1))
    rand = rand.to(torch.long)
    rand.add_(rowptr.reshape(-1, 1))

    samplecol = col[rand].flatten()

    samplerow = torch.arange(adj.size(0), device=adj.device())[mask].reshape(
        -1, 1).expand(-1, deg).flatten()
    samplevalue = (rowcount * (1/deg)).reshape(-1, 1).expand(-1, deg).flatten()

    mask = torch.logical_not(mask)
    nosamplerow, nosamplecol = adj[mask].coo()[:2]
    nosamplerow = torch.arange(adj.size(0),
                               device=adj.device())[mask][nosamplerow]

    ret = SparseTensor(row=torch.cat((samplerow, nosamplerow)),
                       col=torch.cat((samplecol, nosamplecol)),
                       value=torch.cat((samplevalue,
                                        torch.ones_like(nosamplerow))),
                       sparse_sizes=adj.sparse_sizes()).to_device(
                           adj.device()).coalesce()  #.fill_value_(1)
    return ret

def adjoverlap(adj1: SparseTensor,
               adj2: SparseTensor,
               tarei: Tensor,
               filled1: bool = False,
               calresadj: bool = False,
               cnsampledeg: int = -1,
               ressampledeg: int = -1):
    # a wrapper for functions above.
    adj1 = adj1[tarei[0]]
    adj2 = adj2[tarei[1]]
    if calresadj:
        adjoverlap, adjres1, adjres2 = spmoverlap_notoverlap_(adj1, adj2)
        if cnsampledeg > 0:
            adjoverlap = sparsesample_reweight(adjoverlap, cnsampledeg)
        if ressampledeg > 0:
            adjres1 = sparsesample_reweight(adjres1, ressampledeg)
            adjres2 = sparsesample_reweight(adjres2, ressampledeg)
        return adjoverlap, adjres1, adjres2
    else:
        adjoverlap = spmoverlap_(adj1, adj2)
        if cnsampledeg > 0:
            adjoverlap = sparsesample_reweight(adjoverlap, cnsampledeg)
    return adjoverlap

# Edge dropout with adjacency matrix as input
class DropAdj(nn.Module):
    def __init__(self, dp: float = 0.0, doscale=True) -> None:
        super().__init__()
        self.dp = dp
        self.register_buffer("ratio", torch.tensor(1/(1-dp)))
        self.doscale = doscale

    def forward(self, adj: SparseTensor)->SparseTensor:
        if self.dp < 1e-6 or not self.training:
            return adj
        mask = torch.rand_like(adj.storage.col(), dtype=torch.float) > self.dp
        adj = masked_select_nnz(adj, mask, layout="coo")
        if self.doscale:
            if adj.storage.has_value():
                adj.storage.set_value_(adj.storage.value()*self.ratio, layout="coo")
            else:
                adj.fill_value_(1/(1-self.dp), dtype=torch.float)
        return adj
    
###############################################
# code from BGRL

class MLP_Head_BGRL(nn.Module):
    r"""MLP used for predictor in BGRL. The MLP has one hidden layer.

    Args:
        input_size (int): Size of input features.
        output_size (int): Size of output features.
        hidden_size (int, optional): Size of hidden layer. (default: :obj:`4096`).
    """
    def __init__(self, input_size, output_size, hidden_size=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.PReLU(1),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        self.reset_parameters()

    def forward(self, x):
        return self.net(x)

    def reset_parameters(self):
        # kaiming_uniform
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

class CosineDecayScheduler:
    def __init__(self, max_val, warmup_steps, total_steps):
        self.max_val = max_val
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get(self, step):
        if step < self.warmup_steps:
            return self.max_val * step / self.warmup_steps # augmentation de plus en plus grande
        elif self.warmup_steps <= step <= self.total_steps:
            return self.max_val * (1 + np.cos((step - self.warmup_steps) * np.pi /
                                              (self.total_steps - self.warmup_steps))) / 2 # décroit de façon lisse et progressive.
        else:
            raise ValueError('Step ({}) > total number of steps ({}).'.format(step, self.total_steps))

###############################################
# code from MPLP

def spmdiff_(adj1: SparseTensor,
                   adj2: SparseTensor, keep_val=False) -> tuple[SparseTensor, SparseTensor]:
    '''
    return elements in adj1 but not in adj2 and in adj2 but not adj1
    '''
    # assert adj1.sizes() == adj2.sizes()
    
    element1, val1 = spm2elem(adj1)
    element2, val2 = spm2elem(adj2)

    if element1.shape[0] == 0:
        retelem1 = element1
        retelem2 = element2
    else:
        idx = torch.searchsorted(element1[:-1], element2)
        matchedmask = (element1[idx] == element2)

        maskelem1 = torch.ones_like(element1, dtype=torch.bool)
        maskelem1[idx[matchedmask]] = 0
        retelem1 = element1[maskelem1]
    
    if keep_val and val1 is not None:
        retval1 = val1[maskelem1]
        return elem2spm(retelem1, adj1.sizes(), retval1)
    else:
        return elem2spm(retelem1, adj1.sizes())

def dot_product(tensor1, tensor2):
    return (tensor1 * tensor2).sum(dim=-1)

def subgraph(edges: Tensor, adj_t: SparseTensor, k: int=2):
    row,col = edges
    nodes = torch.cat((row,col),dim=-1)
    edge_index,_ = to_edge_index(adj_t)
    subset, new_edge_index, inv, edge_mask = k_hop_subgraph(nodes, k, edge_index=edge_index, 
                                                                num_nodes=adj_t.size(0), relabel_nodes=True)
    # subset[inv] = nodes. The new node id is based on `subset`'s order.
    # inv means the new idx (in subset) of the old nodes in `nodes`
    new_adj_t = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], 
                                sparse_sizes=(subset.size(0), subset.size(0)))
    new_edges = inv.view(2,-1)
    return new_adj_t, new_edges, subset

def get_two_hop_adj(adj_t):
    # adj_t = adj_t.fill_value_(1.0) # no need to fill value because of subgraph op
    one_and_two_hop_adj = adj_t @ adj_t
    adj_t_with_self_loop = adj_t.fill_diag(1)
    two_hop_adj = spmdiff_(one_and_two_hop_adj, adj_t_with_self_loop)
    return adj_t, two_hop_adj

class NodeLabel(torch.nn.Module):
    MINIMUM_SIGNATURE_DIM = 64
    def __init__(self, dim: int=1024, signature_sampling="torchhd", prop_type="prop_only",
                 minimum_degree_onehot: int=-1):
        super().__init__()
        self.dim = dim
        self.signature_sampling = signature_sampling
        self.prop_type = prop_type
        self.cached_two_hop_adj = None
        self.minimum_degree_onehot = minimum_degree_onehot

    def forward(self, edges: Tensor, adj_t: SparseTensor, node_weight: Tensor=None, cache_mode=None, adj2: SparseTensor=None):
        # if self.training and (cache_mode is not None):
        #     raise ValueError("Cannot use cache during training")
        if cache_mode is not None:
            if self.prop_type == "prop_only":
                return self.propagation_only_cache(edges, adj_t, node_weight, cache_mode)
            elif self.prop_type == "combine":
                return self.propagation_combine_cache(edges, adj_t, node_weight, cache_mode)
            elif self.prop_type == "precompute":
                return self.propagation_prop_only_cache(edges, adj_t, node_weight, cache_mode)
            else:
                raise NotImplementedError()
        else:
            if self.prop_type == "prop_only":
                return self.propagation_only(edges, adj_t, node_weight, adj2=adj2)
            elif self.prop_type == "exact":
                return self.propagation(edges, adj_t, node_weight)
            elif self.prop_type == "combine":
                return self.propagation_combine(edges, adj_t, node_weight)

    def get_random_node_vectors(self, adj_t: SparseTensor, node_weight) -> Tensor:
        num_nodes = adj_t.size(0)
        device = adj_t.device()
        if self.minimum_degree_onehot > 0:
            degree = adj_t.sum(dim=1)
            nodes_to_one_hot = degree >= self.minimum_degree_onehot
            one_hot_dim = nodes_to_one_hot.sum()
            # warnings.warn(f"number of nodes to one-hot: {one_hot_dim}", UserWarning)
            if one_hot_dim + self.MINIMUM_SIGNATURE_DIM > self.dim:
                raise ValueError(f"There are {int(one_hot_dim)} nodes with degree higher than {self.minimum_degree_onehot}, select a higher threshold to choose fewer nodes as hub")
            embedding = torch.zeros(num_nodes, self.dim, device=device)
            if one_hot_dim>0:
                one_hot_embedding = F.one_hot(torch.arange(0, one_hot_dim)).float().to(device)
                embedding[nodes_to_one_hot,:one_hot_dim] = one_hot_embedding
        else:
            embedding = torch.zeros(num_nodes, self.dim, device=device)
            nodes_to_one_hot = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            one_hot_dim = 0
        rand_dim = self.dim - one_hot_dim

        if self.signature_sampling == "torchhd":
            scale = math.sqrt(1 / rand_dim)
            node_vectors = torchhd.random(num_nodes - one_hot_dim, rand_dim, device=device)
            node_vectors.mul_(scale)  # make them unit vectors
        elif self.signature_sampling == "gaussian":
            node_vectors = F.normalize(torch.nn.init.normal_(torch.empty((num_nodes - one_hot_dim, rand_dim), dtype=torch.float32, device=device)))
        elif self.signature_sampling == "onehot":
            embedding = torch.zeros(num_nodes, num_nodes, device=device)
            node_vectors = F.one_hot(torch.arange(0, num_nodes)).float().to(device)

        embedding[~nodes_to_one_hot, one_hot_dim:] = node_vectors

        if node_weight is not None:
            node_weight = node_weight.unsqueeze(1) # Note: not sqrt here because it can cause problem for MLP when output is negative
                                                   # thus, it requires the MLP to approximate one more sqrt?
            embedding.mul_(node_weight)
        return embedding

    def propagation(self, edges: Tensor, adj_t: SparseTensor, node_weight=None):
        # get the 2-hop subgraph of the target edges
        adj_t, edges, subset = subgraph(edges, adj_t)
        node_weight = node_weight[subset] if node_weight is not None else None
        x = self.get_random_node_vectors(adj_t, node_weight=node_weight)

        one_hop_adj, two_hop_adj = get_two_hop_adj(adj_t)
        subset = edges.view(-1) # flatten the target nodes [row, col]

        # size: [(2 x num_target_edges(row,col)) , total_num_nodes_in_2_hop_subgraph ]
        one_hop_adj = one_hop_adj[subset]
        two_hop_adj = two_hop_adj[subset]

        ##### no cache for testing, because the adj_t may change when use_val_as_input is True #####
        # if self.training: # training always requires compute especially for target input mask
        #     one_hop_adj, two_hop_adj = self.get_two_hop_adj(adj_t)
        # else: # testing
        #     if self.cached_two_hop_adj is None:
        #         # caching
        #         one_hop_adj, two_hop_adj = self.get_two_hop_adj(adj_t)
        #         self.cached_two_hop_adj = two_hop_adj
        #     else:
        #         # load cache
        #         one_hop_adj = adj_t.fill_value_(1.0)
        #         two_hop_adj = self.cached_two_hop_adj
            

        degree_one_hop = one_hop_adj.sum(dim=1)
        degree_one_hop = degree_one_hop.view(2, edges.size(1))
        degree_two_hop = two_hop_adj.sum(dim=1)
        degree_two_hop = degree_two_hop.view(2, edges.size(1))

        one_hop_x = matmul(one_hop_adj, x) # size: [(2 x num_target_edges(row,col)) , dim]
        two_hop_x = matmul(two_hop_adj, x)

        one_hop_x = one_hop_x.view(2, edges.size(1), -1)
        two_hop_x = two_hop_x.view(2, edges.size(1), -1)

        count_1_1 = dot_product(one_hop_x[0,:,:], one_hop_x[1,:,:])
        count_1_2 = dot_product(one_hop_x[0,:,:] , two_hop_x[1,:,:]) + dot_product(two_hop_x[0,:,:] , one_hop_x[1,:,:])
        count_2_2 = dot_product(two_hop_x[0,:,:] , two_hop_x[1,:,:])

        count_1_inf = degree_one_hop[0,:] + degree_one_hop[1,:] - 2 * count_1_1 - count_1_2
        count_2_inf = degree_two_hop[0,:] + degree_two_hop[1,:] - 2 * count_2_2 - count_1_2

        # count_1_1 = dot_product(one_hop_x[edges[0]] , one_hop_x[edges[1]])
        # count_1_2 = dot_product(one_hop_x[edges[0]] , two_hop_x[edges[1]]) + dot_product(two_hop_x[edges[0]] , one_hop_x[edges[1]])
        # count_2_2 = dot_product(two_hop_x[edges[0]] , two_hop_x[edges[1]])

        # count_1_inf = degree_one_hop[edges[0]] + degree_one_hop[edges[1]] - 2 * count_1_1 - count_1_2
        # count_2_inf = degree_two_hop[edges[0]] + degree_two_hop[edges[1]] - 2 * count_2_2 - count_1_2
        return count_1_1, count_1_2, count_2_2, count_1_inf, count_2_inf
    
    def propagation_combine(self, edges: Tensor, adj_t: SparseTensor, node_weight=None):
        # get the 2-hop subgraph of the target edges
        adj_t, edges, subset = subgraph(edges, adj_t)
        node_weight = node_weight[subset] if node_weight is not None else None
        x = self.get_random_node_vectors(adj_t, node_weight=node_weight)

        one_hop_adj, two_hop_adj = get_two_hop_adj(adj_t)
        subset = edges.view(-1) # flatten the target nodes [row, col]

        # size: [(2 x num_target_edges(row,col)) , total_num_nodes_in_2_hop_subgraph ]
        # one_hop_adj = one_hop_adj[subset] # not subsetting adj because two-iter propagation needs all nodes embedding
        # two_hop_adj = two_hop_adj[subset] # comment out since we want to subset two_hop_adj by subset_unique

        degree_one_hop_subgraph_nodes = one_hop_adj.sum(dim=1)
        degree_one_hop = degree_one_hop_subgraph_nodes[subset]
        degree_one_hop = degree_one_hop.view(2, edges.size(1))
        degree_two_hop = two_hop_adj[subset].sum(dim=1)
        degree_two_hop = degree_two_hop.view(2, edges.size(1))

        degree_u = degree_one_hop[0,:]
        degree_v = degree_one_hop[1,:]
        degree_u_2 = degree_two_hop[0,:]
        degree_v_2 = degree_two_hop[1,:]


        subset_unique, inverse_indices = torch.unique(subset, return_inverse=True)
        one_hop_x_subgraph_nodes = matmul(one_hop_adj, x)
        two_iter_x = matmul(one_hop_adj[subset_unique], one_hop_x_subgraph_nodes)[inverse_indices]
        ## TODO: verify if matmul(one_hop_adj[subset], one_hop_x_subgraph_nodes)
        ##                 matmul(one_hop_adj, one_hop_x_subgraph_nodes) [subset] are the same
        one_hop_x = one_hop_x_subgraph_nodes[subset]
        # two_hop_x = matmul(two_hop_adj_subset, x) # size: [(2 x num_target_edges(row,col)) , dim]
        two_hop_x = matmul(two_hop_adj[subset_unique], x)[inverse_indices]

        one_hop_x = one_hop_x.view(2, edges.size(1), -1)
        two_hop_x = two_hop_x.view(2, edges.size(1), -1)
        two_iter_x = two_iter_x.view(2, edges.size(1), -1)

        count_1_1 = dot_product(one_hop_x[0,:,:], one_hop_x[1,:,:])
        count_1_2 = dot_product(one_hop_x[0,:,:] , two_hop_x[1,:,:])
        count_2_1 = dot_product(two_hop_x[0,:,:] , one_hop_x[1,:,:])
        count_2_2 = dot_product(two_hop_x[0,:,:] , two_hop_x[1,:,:])

        count_1_inf = degree_u + degree_v - 2 * count_1_1 - count_1_2 - count_2_1
        count_2_inf = degree_u_2 + degree_v_2 - 2 * count_2_2 - count_1_2 - count_2_1

        # combine part
        comb_count_1_2 = dot_product(one_hop_x[0,:,:] , two_iter_x[1,:,:])
        comb_count_2_1 = dot_product(two_iter_x[0,:,:] , one_hop_x[1,:,:])
        comb_count_2_2 = dot_product((two_iter_x[0,:,:] - degree_u.view(-1,1)*x[edges[0]]), # two-iter contains self return nodes, thus exclude them
                                     (two_iter_x[1,:,:] - degree_v.view(-1,1)*x[edges[1]]))
        # count those 1 step and 2 step away from the target nodes. thus they form triangles
        comb_count_self_1_2 = dot_product(one_hop_x[0,:,:] , two_iter_x[0,:,:])
        comb_count_self_2_1 = dot_product(one_hop_x[1,:,:] , two_iter_x[1,:,:])

        return count_1_1, count_1_2, count_2_1, count_2_2, count_1_inf, count_2_inf, \
                comb_count_1_2, comb_count_2_1, comb_count_2_2, comb_count_self_1_2, comb_count_self_2_1,\
                degree_u, degree_v, degree_u_2, degree_v_2
                

    def propagation_combine_cache(self, edges: Tensor, adj_t: SparseTensor, node_weight=None, cache_mode=None):
        if cache_mode == 'build':
            # get the 2-hop subgraph of the target edges
            x = self.get_random_node_vectors(adj_t, node_weight=node_weight)

            one_hop_adj, two_hop_adj = get_two_hop_adj(adj_t)

            degree_one_hop = one_hop_adj.sum(dim=1)
            degree_two_hop = two_hop_adj.sum(dim=1)

            one_hop_x = matmul(one_hop_adj, x)
            two_hop_x = matmul(two_hop_adj, x)
            two_iter_x = matmul(one_hop_adj, one_hop_x)

            # caching
            self.cached_x = x
            self.cached_degree_one_hop = degree_one_hop
            self.cached_degree_two_hop = degree_two_hop

            self.cached_one_hop_x = one_hop_x
            self.cached_two_hop_x = two_hop_x
            self.cached_two_iter_x = two_iter_x
            return
        if cache_mode == 'delete':
            del self.cached_x
            del self.cached_degree_one_hop
            del self.cached_degree_two_hop
            del self.cached_one_hop_x
            del self.cached_two_hop_x
            del self.cached_two_iter_x
            return
        if cache_mode == 'use':
            # loading
            x = self.cached_x
            degree_one_hop = self.cached_degree_one_hop
            degree_two_hop = self.cached_degree_two_hop

            one_hop_x = self.cached_one_hop_x
            two_hop_x = self.cached_two_hop_x
            two_iter_x = self.cached_two_iter_x

        count_1_1 = dot_product(one_hop_x[edges[0]] , one_hop_x[edges[1]])
        count_1_2 = dot_product(one_hop_x[edges[0]] , two_hop_x[edges[1]])
        count_2_1 = dot_product(two_hop_x[edges[0]] , one_hop_x[edges[1]])
        count_2_2 = dot_product(two_hop_x[edges[0]] , two_hop_x[edges[1]])

        count_1_inf = degree_one_hop[edges[0]] + degree_one_hop[edges[1]] - 2 * count_1_1 - count_1_2 - count_2_1
        count_2_inf = degree_two_hop[edges[0]] + degree_two_hop[edges[1]] - 2 * count_2_2 - count_1_2 - count_2_1


        comb_count_1_2 = dot_product(one_hop_x[edges[0]] , two_iter_x[edges[1]])
        comb_count_2_1 = dot_product(two_iter_x[edges[0]] , one_hop_x[edges[1]])
        comb_count_2_2 = dot_product((two_iter_x[edges[0]]-degree_one_hop[edges[0]].view(-1,1)*x[edges[0]]),\
                                     (two_iter_x[edges[1]]-degree_one_hop[edges[1]].view(-1,1)*x[edges[1]]))


        comb_count_self_1_2 = dot_product(one_hop_x[edges[0]] , two_iter_x[edges[0]])
        comb_count_self_2_1 = dot_product(one_hop_x[edges[1]] , two_iter_x[edges[1]])

        degree_u = degree_one_hop[edges[0]]
        degree_v = degree_one_hop[edges[1]]
        degree_u_2 = degree_two_hop[edges[0]]
        degree_v_2 = degree_two_hop[edges[1]]

        return count_1_1, count_1_2, count_2_1, count_2_2, count_1_inf, count_2_inf, \
                comb_count_1_2, comb_count_2_1, comb_count_2_2, comb_count_self_1_2, comb_count_self_2_1,\
                degree_u, degree_v, degree_u_2, degree_v_2

    def propagation_only(self, edges: Tensor, adj_t: SparseTensor, node_weight=None, adj2: SparseTensor=None):
        adj_t, new_edges, subset_nodes = subgraph(edges, adj_t, 2)
        node_weight = node_weight[subset_nodes] if node_weight is not None else None
        x = self.get_random_node_vectors(adj_t, node_weight=node_weight)
        subset = new_edges.view(-1) # flatten the target nodes [row, col]

        # remove values from adj_t
        # adj_t = adj_t.set_value(None)

        subset_unique, inverse_indices = torch.unique(subset, return_inverse=True)
        one_hop_x_subgraph_nodes = matmul(adj_t, x)
        one_hop_x = one_hop_x_subgraph_nodes[subset]
        two_hop_x = matmul(adj_t[subset_unique], one_hop_x_subgraph_nodes)[inverse_indices]
        degree_one_hop = adj_t.sum(dim=1)

        one_hop_x = one_hop_x.view(2, new_edges.size(1), -1)
        two_hop_x = two_hop_x.view(2, new_edges.size(1), -1)

        count_1_1 = dot_product(one_hop_x[0,:,:], one_hop_x[1,:,:])
        count_1_2 = dot_product(one_hop_x[0,:,:], two_hop_x[1,:,:])
        count_2_1 = dot_product(two_hop_x[0,:,:] , one_hop_x[1,:,:])
        count_2_2 = dot_product((two_hop_x[0,:,:]-degree_one_hop[new_edges[0]].view(-1,1)*x[new_edges[0]]) , (two_hop_x[1,:,:]-degree_one_hop[new_edges[1]].view(-1,1)*x[new_edges[1]]))
        

        count_self_1_2 = dot_product(one_hop_x[0,:,:] , two_hop_x[0,:,:])
        count_self_2_1 = dot_product(one_hop_x[1,:,:] , two_hop_x[1,:,:])
        degree_u = degree_one_hop[new_edges[0]]
        degree_v = degree_one_hop[new_edges[1]]
        if adj2 is None:
            return count_1_1, count_1_2, count_2_1, count_2_2, count_self_1_2, count_self_2_1, degree_u, degree_v
        else:
            raise NotImplementedError()
            # adj2 = adj2[subset_nodes, subset_nodes] # under the new node id
            # adj2 = adj2[subset_unique] # select those unique

            # combine above steps
            x = self.get_random_node_vectors(adj2, node_weight=None)
            adj2_new = adj2[subset_nodes[subset_unique], subset_nodes]

    def propagation_prop_only_cache(self, edges: Tensor, adj_t: SparseTensor, node_weight=None, cache_mode=None):
        if cache_mode == 'build':
            # get the 2-hop subgraph of the target edges
            x = self.get_random_node_vectors(adj_t, node_weight=node_weight)


            degree_one_hop = adj_t.sum(dim=1)

            one_hop_x = matmul(adj_t, x)
            two_iter_x = matmul(adj_t, one_hop_x)

            two_iter_x = two_iter_x - degree_one_hop.view(-1,1)*x

            # caching
            self.cached_degree_one_hop = degree_one_hop

            self.cached_one_hop_x = one_hop_x
            self.cached_two_iter_x = two_iter_x
            return
        if cache_mode == 'delete':
            del self.cached_degree_one_hop
            del self.cached_one_hop_x
            del self.cached_two_iter_x
            return
        if cache_mode == 'use':
            # loading
            degree_one_hop = self.cached_degree_one_hop

            one_hop_x = self.cached_one_hop_x
            two_iter_x = self.cached_two_iter_x
        count_1_1 = dot_product(one_hop_x[edges[0]] , one_hop_x[edges[1]])


        count_1_2_only = dot_product(one_hop_x[edges[0]] , two_iter_x[edges[1]])
        count_2_1_only = dot_product(two_iter_x[edges[0]] , one_hop_x[edges[1]])
        count_2_2_only = dot_product((two_iter_x[edges[0]]),\
                                     (two_iter_x[edges[1]]))


        count_self_1_2 = dot_product(one_hop_x[edges[0]] , two_iter_x[edges[0]])
        count_self_2_1 = dot_product(one_hop_x[edges[1]] , two_iter_x[edges[1]])

        degree_u = degree_one_hop[edges[0]]
        degree_v = degree_one_hop[edges[1]]
        return count_1_1, count_1_2_only, count_2_1_only, count_2_2_only, count_self_1_2, count_self_2_1, degree_u, degree_v

    def propagation_only_cache(self, edges: Tensor, adj_t: SparseTensor, node_weight=None, cache_mode=None):
        if cache_mode == 'build':
            # get the 2-hop subgraph of the target edges
            x = self.get_random_node_vectors(adj_t, node_weight=node_weight)


            degree_one_hop = adj_t.sum(dim=1)

            one_hop_x = matmul(adj_t, x)
            two_iter_x = matmul(adj_t, one_hop_x)

            # caching
            self.cached_x = x
            self.cached_degree_one_hop = degree_one_hop

            self.cached_one_hop_x = one_hop_x
            self.cached_two_iter_x = two_iter_x
            return
        if cache_mode == 'delete':
            del self.cached_x
            del self.cached_degree_one_hop
            del self.cached_one_hop_x
            del self.cached_two_iter_x
            return
        if cache_mode == 'use':
            # loading
            x = self.cached_x
            degree_one_hop = self.cached_degree_one_hop

            one_hop_x = self.cached_one_hop_x
            two_iter_x = self.cached_two_iter_x
        count_1_1 = dot_product(one_hop_x[edges[0]] , one_hop_x[edges[1]])
        count_1_2 = dot_product(one_hop_x[edges[0]] , two_iter_x[edges[1]])
        count_2_1 = dot_product(two_iter_x[edges[0]] , one_hop_x[edges[1]])
        count_2_2 = dot_product(two_iter_x[edges[0]]-degree_one_hop[edges[0]].view(-1,1)*x[edges[0]],\
                                two_iter_x[edges[1]]-degree_one_hop[edges[1]].view(-1,1)*x[edges[1]])

        count_self_1_2 = dot_product(one_hop_x[edges[0]] , two_iter_x[edges[0]])
        count_self_2_1 = dot_product(one_hop_x[edges[1]] , two_iter_x[edges[1]])

        degree_u = degree_one_hop[edges[0]]
        degree_v = degree_one_hop[edges[1]]
        return count_1_1, count_1_2, count_2_1, count_2_2, count_self_1_2, count_self_2_1, degree_u, degree_v


###############################################
# code from me

def compute_table(res_dict: dict[str, list | float], name: str):
    # Compute the mean and std from a dict return tab and latex table
    new_tab = []
    for key, result in res_dict.items():
        if isinstance(result, list):
            result = np.array(result)
            unit = 100 if key != 'pretrain_time' else 1
            mean = round(unit * np.mean(result), 2)
            std = round(unit * np.std(result), 2)
            new_tab.append({"metrics": key, name: fr"{mean}$\pm${std}"})
    df = pd.DataFrame(data=new_tab)
    df.set_index('metrics')
    res_latex = df.to_latex(
        index=False, formatters={"name": str.upper}, float_format="{:.1f}".format
    )
    return df, res_latex

def full_output(full_res: list):
    # concat full result to one dataframe, then print with latex
    full_res = pd.concat(full_res, axis=1)
    full_res = full_res.loc[:, ~full_res.columns.duplicated()]
    full_res.set_index('metrics', inplace=True)
    print(full_res)
    full_latex = full_res.to_latex(
        index=True, formatters={"name": str.upper}, float_format="{:.1f}".format
    )
    print(full_latex)
    return full_res, full_latex

def get_evaluator(dataset: str):
    if dataset in ["Cora", "Citeseer", "Pubmed"]:
        evaluator = Evaluator(name='ogbl-ppa')
    else:
        evaluator = Evaluator(name=f'ogbl-{dataset}')
    return evaluator

class DataSplit:
    def __init__(self, dataset: str, device: str, runs: int, use_valedges_as_input: bool = False, reduce_feature: int|None = None, only_feature: bool = False):
        print(f"{runs} split from the dataset {dataset}")
        self.device = device
        self.runs = runs
        self.data_runs: dict[int, tuple[any, dict]] = {}
        t1 = time.time()
        for r in tqdm(range(runs)):
            seed_everything(r)
            self.data_runs[r] = loaddataset(dataset, use_valedges_as_input, reduce_feature, only_feature)
        self.info_time = round(time.time()-t1, 2)
        self.info()

    def get(self, r):
        data, split_edge = self.data_runs[r]
        return data.to(self.device), split_edge

    def info(self):
        print("split time: ", self.info_time, " s")
        data, split_edge = self.data_runs[0]
        print("dataset split ")
        for key1 in split_edge:
            for key2  in split_edge[key1]:
                print(key1, key2, split_edge[key1][key2].shape[0])
