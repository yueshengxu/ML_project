import numpy as np
import scipy.sparse as sp
import torch
from sklearn import metrics
import pickle
import os
import copy
from cvxopt import matrix, solvers

def seperate_r_u(features, idx_list, l_idx, l_fea, l_nums, temp):
    r_idx = []
    r_fea = []
    u_idx = []
    u_fea = []
    for idx in idx_list:
        if isinstance(idx, tuple):
            r_idx.append(idx)
            r_fea.append(features[idx])
        elif idx[0] == 'u':
            u_idx.append(idx)
            u_fea.append(features[idx])
    l_idx += (r_idx + u_idx)
    l_fea += (r_fea + u_fea)
    l_nums.append([len(r_idx) + temp, 
                   len(r_idx) + len(u_idx) + temp])
    temp += len(r_idx) + len(u_idx)
    return l_idx, l_fea, l_nums, temp
def feature_matrix(features, p_train, p_val, p_test):

    l_idx = []
    l_fea = []
    l_nums = []

    
    temp = 0
    l_idx, l_fea, l_nums, temp = seperate_r_u(features, p_train, l_idx, l_fea, l_nums, temp)
    l_idx, l_fea, l_nums, temp = seperate_r_u(features, p_val, l_idx, l_fea, l_nums, temp)    
    l_idx, l_fea, l_nums, temp = seperate_r_u(features, p_test, l_idx, l_fea, l_nums, temp)

    prod_idx = list( set(list(features.keys())) - set(p_train) - set(p_val) - set(p_test) )
    prod_fea = []
    for idx in prod_idx:
        prod_fea.append(features[idx])
    
    l_idx += prod_idx
    l_fea += prod_fea
    l_nums.append([len(p_train), 
                   len(p_train) + len(p_val), 
                   len(p_train) + len(p_val) + len(p_test), 
                   len(p_train) + len(p_val) + len(p_test) + len(prod_idx)])
    return l_idx, l_fea, l_nums


def read_data(tvt, urp, city_name, train_ratio, val_ratio):
#    tvt = 'train' / 'val' / 'test'
#    urp = 'user' / 'review' / 'prod'
    test_ratio = str(round(100*(1-train_ratio-val_ratio)))
    train_ratio = str(int(100*train_ratio))
    val_ratio = str(int(100*val_ratio))
    with open('../data/' + tvt + '_' + urp + '_' + city_name + '_' + train_ratio + val_ratio + test_ratio, 'rb') as f:
#          str(int(100*train_ratio)) + 
#          str(int(100*val_ratio)) + 
#          str(round(100*test_ratio)), 'rb') as f:
        nodelist = pickle.load(f)
    return nodelist

def read_user_prod(review_list):
    user_list = list(set([x[0] for x in review_list]))
    prod_list = list(set([x[1] for x in review_list]))
    return user_list, prod_list

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    # print(torch.where(preds==1))
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def construct_adj_matrix(ground_truth, idx_map, labels):
    edges = []
    
    for it, r_id in enumerate(ground_truth.keys()):
        edges.append((idx_map[r_id], idx_map[r_id[0]]))
        edges.append((idx_map[r_id], idx_map[r_id[1]]))
        
        edges.append((idx_map[r_id[0]], idx_map[r_id]))
        edges.append((idx_map[r_id[1]], idx_map[r_id]))
        
    edges = np.array(edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])),
                        shape = (labels.shape[0], labels.shape[0]),
                        dtype = np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)    
    return adj

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def onehot_label(ground_truth, list_idx):
    labels = np.zeros((len(list_idx),2))

    gt = {}
    user_gt = {}
    for k,v in ground_truth.items():
        u = k[0]
        p = k[1]
        if u not in gt.keys():
            gt[u] = v
            user_gt[u] = v
        else:
            gt[u] |= v
            user_gt[u] |= v
        if p not in gt.keys():
            gt[p] = v
        else:
            gt[p] |= v
    ground_truth = {**ground_truth, **gt}
    
    for it, k in enumerate(list_idx):
        labels[it][ground_truth[k]] = 1
    return labels, user_gt

def auc_score(output, ground_truth, list_idx, idx_range, u_or_r):
    prob = torch.exp(output[:,1]).cpu().detach().numpy()
    prob_dic = {}
    for it, idx in enumerate(list_idx):
        prob_dic[idx] = prob[it]
    sub_list = [list_idx[x] for x in idx_range]
    sub_true = []
    sub_prob = []
    if u_or_r == 'r':
        for x in sub_list:
            if isinstance(x, tuple):
                sub_true.append(ground_truth[x])
                sub_prob.append(prob_dic[x])
    elif u_or_r == 'u':
        for x in sub_list:
            if isinstance(x, str) and x[0]=='u':
                sub_true.append(ground_truth[x])
                sub_prob.append(prob_dic[x])
    fpr, tpr, thre = metrics.roc_curve(sub_true, sub_prob)
    return metrics.auc(fpr, tpr)

def save_output(GCN_output, list_index, type, scalar):
    GCN_prob = torch.exp(GCN_output[:, 1]).detach().numpy()
    product_prob = {}
    review_prob = {}
    user_prob = {}

    for index, id in enumerate(list_index):
        p = GCN_prob[index]
        if isinstance(id, tuple):
            review_prob[id] = p
        else:
            if id[0] == 'u':
                user_prob[id] = p
            elif id[0] == 'p':
                product_prob[id] = p


    user_num = len(list(user_prob.keys()))
    product_num = len(list(product_prob.keys()))
    review_num = len(list(review_prob.keys()))

    assert GCN_prob.shape[0] == user_num + product_num + review_num
    if type == 'eval':
        save_output_prob(product_prob, '/Users/kenny/Desktop/GCN_Yelp/output_prob/product_prob'+str(scalar)+'.pickle')
        save_output_prob(review_prob, '/Users/kenny/Desktop/GCN_Yelp/output_prob/review_prob'+str(scalar)+'.pickle')
        save_output_prob(user_prob, '/Users/kenny/Desktop/GCN_Yelp/output_prob/user_prob'+str(scalar)+'.pickle')
    elif type == 'train':
        save_output_prob(product_prob, '/Users/kenny/Desktop/GCN_Yelp/output_prob/train_product_prob.pickle')
        save_output_prob(review_prob, '/Users/kenny/Desktop/GCN_Yelp/output_prob/train_review_prob.pickle')
        save_output_prob(user_prob, '/Users/kenny/Desktop/GCN_Yelp/output_prob/train_user_prob.pickle')

def save_output_prob(object, path):
    with open(path, 'wb') as f:
        pickle.dump(object, f)

    print('save GCN output')



def auc(epoch,comparison_type,logit,labels,sub_set_index):
    # model.eval()
    # with torch.no_grad():
    #     output = model(features, adj, nums)
    # print(output[:,1])
    subset_prob = logit[sub_set_index][:,1].cpu().detach().numpy()
    subset_truth = labels[sub_set_index].cpu().detach().numpy()
    fpr, tpr, thre = metrics.roc_curve(subset_truth, subset_prob)
    auc_score = metrics.auc(fpr, tpr)
    print(epoch,"    ",auc_score)
    
    #review_auc = utils.auc_score(output[1], review_ground_truth, list_idx, idx_test, 'r')
    with open("auc.txt", 'a+') as f:
        f.write("                                                                {0} ----> Epoch: {1} Testing Review AUC: {2} \n".format(comparison_type,epoch,auc_score))
    #print("review AUC: ",review_auc)

def split_reviewer_nodes(node_idx, label):
    pos_idx = []
    neg_idx = []

    for idx in node_idx:
        if label[idx] == 1:
            pos_idx.append(idx)
        else:
            neg_idx.append(idx)
    return pos_idx, neg_idx


def read_pickle(path):
    with open(path, 'rb') as pf:
        return pickle.load(pf)


def concatenate_grad(model):
    grad_list = []
    grad_dict = dict()
    for n, _ in model.named_children():
        each_grad = getattr(model, n).weight.grad.data
        grad_dict[n] = copy.deepcopy(each_grad)
        each_grad = each_grad.cpu().detach().numpy().flatten()
        grad_list.append(each_grad)
    final_grad = np.concatenate(grad_list, axis=0)
    return final_grad, grad_dict

def construct_Pmatrix(grad_dict, loss_type):
    P = np.zeros((len(loss_type), len(loss_type)))
    for i, name_i in enumerate(loss_type):
        x_i = grad_dict[name_i]['concatenate_grad']
        for j, name_j in enumerate(loss_type):
            x_j = grad_dict[name_j]['concatenate_grad']
            x_ij = np.dot(x_i, x_j)
            P[i][j] = x_ij
    return 2*P

def solve_lambda(P, num_metrics, lambda_opt=None):

    q = matrix(0.0, (num_metrics, 1))
    G = matrix(0.0, (num_metrics, num_metrics))
    G[::num_metrics + 1] = -1.0
    h = matrix(0.0, (num_metrics, 1))
    A = matrix(1.0, (1, num_metrics))
    b = matrix(1.0)

    lambda_opt = np.array(solvers.qp(P, q, G, h, A, b)['x'])
    return lambda_opt


def weighted_grad(lambda_opt, loss_type, grad_dict, layer_name):
    new_grad = 0
    for i, l_type in enumerate(loss_type):
        new_grad += lambda_opt[i][0] * grad_dict[l_type]['grad'][layer_name]
    return new_grad