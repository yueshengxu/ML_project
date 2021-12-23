# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 03:07:28 2021

@author: kaibu
"""
import torch
from sklearn.metrics import confusion_matrix
import numpy as np

def FNR_matrix(ground_truth, train_index, g1_index, g2_index):
    matrix = []
    g1_ratio = len(g1_index) / (len(g1_index) + len(g2_index))

    for index in train_index:
        if index in g1_index:
            a = 1
        elif index in g2_index:
            a = 0
        one_or_zero = ground_truth[index] # 1 if label=1, 0 if label=0
        matrix.append((a - g1_ratio) * one_or_zero)
    matrix_tensor =  torch.tensor(matrix)
    return torch.unsqueeze(matrix_tensor, 0)

def FPR_matrix(ground_truth, train_index, g1_index, g2_index):
    matrix = []
    g1_ratio = len(g1_index) / (len(g1_index) + len(g2_index))

    for index in train_index:
        if index in g1_index:
            a = 1
        elif index in g2_index:
            a = 0
        one_or_zero = (ground_truth[index] -1) * -1 # 1 if the label=0, and 0 if label=1
        matrix.append((a - g1_ratio) * one_or_zero)
    matrix_tensor =  torch.tensor(matrix)
    return torch.unsqueeze(matrix_tensor, 0)

def equalized_FNR_fairnessloss(fnr_matrix, logits):
    sum_all = torch.mm(fnr_matrix.cuda(), torch.unsqueeze(logits.cuda(), 1))
    return torch.abs(sum_all / len(logits))

def equalized_FPR_fairnessloss(fpr_matrix, logits):
    sum_all = torch.mm(fpr_matrix.cuda(), torch.unsqueeze(logits.cuda(), 1))
    return torch.abs(sum_all / len(logits))

def ture_equalized_FNR_FPR_fairnessloss(g1_index, g2_index, output, ground_truth, threshold):

    g1_index = g1_index
    g2_index = g2_index
    ground_truth = ground_truth.cpu()

    output_prob = torch.exp(output[:,1])
    prediction = (output_prob > threshold)
    output_prob = output_prob.cpu().detach().numpy()
    #prediction = get_top_rank(output_prob, 1 - threshold)
    prediction = prediction.cpu().detach().numpy()

    print("g1", len(g1_index))
    print("g2", len(g2_index))
    print("output prob", output_prob.shape)
    print("labels", ground_truth.size())
    print("prediction", prediction.shape)

    g1_FNR, g1_FPR = FNR_FPR(ground_truth[g1_index], prediction[g1_index], output_prob[g1_index])
    g2_FNR, g2_FPR = FNR_FPR(ground_truth[g2_index], prediction[g2_index], output_prob[g2_index])

    return abs(g1_FNR - g2_FNR), abs(g1_FPR - g2_FPR)

def get_threshold(index, ground_truth):
    index = index.cpu().numpy()
    ground_truth = ground_truth.cpu().numpy()
    true_vals = ground_truth[index]
    num_ones = np.count_nonzero(true_vals)
    threshold = 1 - (num_ones / len(true_vals))
    return threshold

def FNR_FPR(ground_truth, prediction, output_prob):
    tn, fp, fn, tp = confusion_matrix(ground_truth, prediction).ravel()
    print()
    print("tp:", tp, "tn", tn, "fp", fp, "fn", fn)
    print("pred non_zeros", np.count_nonzero(prediction), "pred zeros", len(prediction) - np.count_nonzero(prediction))
    print("true non_zeros", np.count_nonzero(ground_truth), "true zeros", len(ground_truth) - np.count_nonzero(ground_truth))
    print("prob range, min:", np.amin(output_prob), "max:", np.nanmax(output_prob))

    return (fn / (fn + tp)), (fp / (fp + tn)) #fn/(fn+tn) , fp/(fp+tp)

def get_top_rank(output_prob, n):
    array = np.array(output_prob)
    temp = array.argsort()[::-1]
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))+1 #descending ranking for each prob, Ex [0.4, 0.2, 0.7, 0.1] --> [2,3,1,4]
    pred = ranks <= len(array) * n
    pred = pred.astype(int)
    return pred



def FNR_manual(ground_truth, prediction):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for i in range(len(ground_truth)):
        if ground_truth[i] == prediction[i] == 1:
           tp += 1
        if prediction[i] == 1 and ground_truth[i] != prediction[i]:
           fp += 1
        if ground_truth[i] == prediction[i] == 0:
           tn += 1
        if prediction[i] == 0 and ground_truth[i] != prediction[i]:
           fn += 1

    return fn / (tp + fn)



