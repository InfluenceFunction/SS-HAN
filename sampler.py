import torch
import numpy as np

def sample(g, sample_nums):

    p_a, p_f = g.edges(etype='pa'), g.edges(etype='pf')
    p_a = torch.cat((p_a[0].reshape(1,-1), p_a[1].reshape(1,-1))).numpy()
    p_f = torch.cat((p_f[0].reshape(1,-1), p_f[1].reshape(1,-1))).numpy()

    papers = np.unique(p_a[0]) # several papers have not author
    sample_nums = papers.shape[0]
    split_rate = p_f.shape[1] / p_a.shape[1]
    p_a_nums = int(sample_nums * (1 - split_rate))
    p_f_nums = int(sample_nums * split_rate)

    papers = np.random.choice(papers, size=sample_nums, replace=False)
    pos_edge_index, neg_edge_index = np.empty(sample_nums), np.empty(sample_nums)

    for i, p in enumerate(papers):
        if i < p_a_nums:
            pos_edge_index[i] = sample_n_for_u(p_a, p, pos=True)
            neg_edge_index[i] = sample_n_for_u(p_a, p, neg=True)
        else:
            pos_edge_index[i] = sample_n_for_u(p_f, p, pos=True)
            neg_edge_index[i] = sample_n_for_u(p_f, p, neg=True)

    pos_edge_index[:p_a_nums] += g.num_nodes('paper')
    neg_edge_index[:p_a_nums] += g.num_nodes('paper')
    pos_edge_index[p_a_nums:] += g.num_nodes('paper') + g.num_nodes('author')
    neg_edge_index[p_a_nums:] += g.num_nodes('paper') + g.num_nodes('author')

    pos_edge_index = np.vstack((papers, pos_edge_index))
    neg_edge_index = np.vstack((papers, neg_edge_index))
    return torch.LongTensor(pos_edge_index), torch.LongTensor(neg_edge_index)

def sample_n_for_u(p_a_or_f, p, pos=False, neg=False):
    mask = p_a_or_f[0] == p
    if pos:
        node = np.random.choice(p_a_or_f[1][mask], 1, replace=False)
    else:
        node = np.random.choice(p_a_or_f[1][~mask], 1, replace=False)

    return node