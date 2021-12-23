import torch
import numpy as np

def find_group_idx(group_split_dict, index_list, train_rev, test_rev, val_rev):
    group0_idx = []
    group1_idx = []
    for rid, gid in group_split_dict.items():
        new_rid = ('u' + rid[0], 'p' + rid[1])
        if gid == 1:
            group1_idx.append(index_list[new_rid])
        elif gid == 0:
            group0_idx.append(index_list[new_rid])
    assert len(group1_idx) + len(group0_idx) == len(train_rev) + len(test_rev) + len(val_rev)
    return torch.tensor(group0_idx), torch.tensor(group1_idx)

def separate_group_idx(group_idx, idx_train, idx_val, idx_test):
    group_train = []
    group_val = []
    group_test = []
    
    for idx in group_idx:
        if idx in idx_train:
            group_train.append(idx)
        elif idx in idx_val:
            group_val.append(idx)
        else:
            group_test.append(idx)
    assert len(group_train) + len(group_val) + len(group_test) == len(group_idx)
    return torch.tensor(group_train), torch.tensor(group_val), torch.tensor(group_test)

def di_loss(group0_index, group1_index, output, labels): #added labels
    output_prob = torch.exp(output[:,1])

    group0_output = torch.gather(output_prob, dim=0, index=group0_index)
    group1_output = torch.gather(output_prob, dim=0, index=group1_index)

    group0_labels = torch.gather(labels, dim=0, index=group0_index)
    group1_labels = torch.gather(labels, dim=0, index=group1_index)

    group0_tpr = torch.masked_select(group0_output, group0_labels.gt(0))
    group0_fpr = torch.masked_select(group0_output, group0_labels.lt(1))

    group1_tpr = torch.masked_select(group1_output, group1_labels.gt(0))
    group1_fpr = torch.masked_select(group1_output, group1_labels.lt(1))

    tcr = torch.abs(torch.mean(group0_tpr) - torch.mean(group1_tpr))
    fcr = torch.abs(torch.mean(group0_fpr) - torch.mean(group1_fpr))

    loss = tcr + fcr
    return loss