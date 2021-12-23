import pandas as pd
import torch
import numpy as np
import random as rand
#param:
#s  : logit of GCN  
#g  : ground truth (1 for spam, 0 for non-spam)

def DCG_score(logit, labels):
    #Sorting based on the value of "scores"
    d = {"scores":logit.cpu().detach().numpy(),"labels":labels.cpu().detach().numpy()}
    df = pd.DataFrame(d)
    df = df.sort_values("scores", ascending=False)    
    #reset index so that index is now reflecting the rank.
    df.reset_index(inplace=True,drop=True)
    r = df[df['labels']==1].index
    return sum(1/np.log2(2+r))
    
def max_DCG_score(num_spam):
    # rank all spams at the top
    r = list(range(2,2+num_spam))     #list of (spammers'ranks+2) 
    z = sum(1/np.log2(r))             
    return z

def NDCG_score(maxDCG, pos_group_idx, neg_group_idx, logit):
    num_total_sample = len(pos_group_idx) * int(0.1*len(neg_group_idx))
    subsample_idx = np.random.choice(neg_group_idx, num_total_sample)
    neg_subsample_logit = logit[subsample_idx].reshape((len(pos_group_idx), int(0.1*len(neg_group_idx))))
    pos_logit = logit[pos_group_idx].reshape((-1, 1))
    gap_logit = 1/(1+torch.exp(pos_logit - neg_subsample_logit))
    total_r = torch.sum(gap_logit, 1)
    total_dcg_score = (1/torch.log2(2+total_r))/(1/np.log2(2))
    dcg_score = torch.sum(total_dcg_score)/len(pos_group_idx)
    return dcg_score

def construct_comparison_matrix(g,k):
    # vectorization
    # for each positive node, sample k negative nodes
    positive_indices = [i for i, x in enumerate(g) if x == 1]    
    negative_indices = [i for i, x in enumerate(g) if x == 0] 
    comparison_matrix = []
    for i in positive_indices:
        arr = rand.sample(negative_indices, k)
        arr.insert(0,i)
        comparison_matrix.append(arr)
    return np.asarray(comparison_matrix).astype(int)
    #np.savetxt('test.out', m, delimiter=',')   # X is an array

def construct_group_comparison_matrix(comparison_type, group0_index, group1_index,g,k):

    if   (comparison_type=="4vs2"):
        positive_indices = [i for i, x in enumerate(g[group1_index]) if x == 1] 
        negative_indices = [i for i, x in enumerate(g[group1_index]) if x == 0] 
    elif (comparison_type=="3vs1"):
        positive_indices = [i for i, x in enumerate(g[group0_index]) if x == 1] 
        negative_indices = [i for i, x in enumerate(g[group0_index]) if x == 0] 
    elif (comparison_type=="3vs2"):
        positive_indices = [i for i, x in enumerate(g[group0_index]) if x == 1] 
        negative_indices = [i for i, x in enumerate(g[group1_index]) if x == 0] 
    elif (comparison_type=="4vs1"):    
        positive_indices = [i for i, x in enumerate(g[group1_index]) if x == 1] 
        negative_indices = [i for i, x in enumerate(g[group0_index]) if x == 0] 
    comparison_matrix = []
    for i in positive_indices:
        arr = rand.sample(negative_indices, k)
        arr.insert(0,i)
        comparison_matrix.append(arr)
    return np.asarray(comparison_matrix).astype(int)

def construct_difference_matrix(k):
    d = torch.diag(-1*torch.ones(k))
    first_col = torch.ones(k,1)
    difference_matrix = torch.cat((first_col,d),1)
    return difference_matrix

def avg_ranking_loss(comparison_matrix,difference_matrix,s):
    S = s[comparison_matrix].t()
    k, n = S.shape
    k = k -1  # exclude the first column
    return torch.sum(torch.log(1+torch.exp((torch.mm(-1*difference_matrix,S)))))/(k*n)

def vectorized_ndcg_score(comparison_matrix,difference_matrix,s,z):
    S = s[comparison_matrix].t()
    return torch.sum(1/torch.log2(2+torch.sum(1/(1+torch.exp(torch.mm(difference_matrix,S))),dim=0)))/z

def fairness_loss1(logit, maxDCG_g1, maxDCG_g0, pos_group1_idx, pos_group0_idx, neg_group1_idx, neg_group0_idx):
    xNDCG_group0 = NDCG_score(maxDCG_g0, pos_group0_idx, neg_group0_idx, logit)
    xNDCG_group1 = NDCG_score(maxDCG_g1, pos_group1_idx, neg_group1_idx, logit)

    return xNDCG_group0, xNDCG_group1

#def fairness_loss2(logit, pos_group1_idx, pos_group0_idx, neg_group1_idx, neg_group0_idx):
