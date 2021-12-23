import time
import argparse
import numpy as np
import pickle
import scipy.sparse as sp
import random
import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import csv
import utils as U
from GCN_models import GCN
from cvxopt import matrix, solvers
from tensorboardX import SummaryWriter
from sklearn import metrics
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from ndcg_score import *
from di import *
import csv_util
from equal_odds import *
#torch.cuda.set_device(0)
device = torch.device('cuda:0')
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50,)
parser.add_argument('--data', nargs='*', default=['Chi'])
parser.add_argument('--optimize', nargs='*', default=['ndcg'])

args = parser.parse_args()

class Parameters:
    def __init__(self):
        self.seed = 42
        self.lr = 0.005
        self.wd = 1e-3
        self.hidden = 16
        self.dropout = 0.5

        self.train_ratio = 0.3
        self.val_ratio = 0.2
        self.test_ratio = 0.5
        self.data_prefix = '../data/'
        self.model_prefix = '../model/'
        self.tensorboard = True
        self.setting = 'fairness_NDCG'


class GCNModel:
    def __init__(self, parameters):
        self._param = parameters
        self.train_domain = self._param.train_domain

        self.list_idx = None
        self.features = None
        self.labels = None
        self.user_ground_truth = None
        self.nums = None
        self.rev_nums = None
        self.adj = None

        self.idx_train = None
        self.idx_train_rev = None
        self.idx_val = None
        self.idx_test = None
        self.idx_whole = None

        self.group0_train_idx = None
        self.group0_test_idx = None
        self.group1_train_idx = None
        self.group1_test_idx = None

        self.group0_train_idx_tensor = None
        self.group1_train_idx_tensor = None
        self.group0_test_idx_tensor = None
        self.group1_test_idx_tensor = None

        self.fpr_matrix_train =  None
        self.fnr_matrix_train = None
        self.fpr_matrix_test =  None
        self.fnr_matrix_test = None

        self.setup_seed()
        self.load_data()
        self.GCN, self.optimizer = self.build_GCN_model()
        if self._param.tensorboard:
            self.writer = SummaryWriter(comment='fairGCN/{0}_{1}{2}{3}_{4}_lr_{5}'.format(self._param.train_domain,
                                                                                          self._param.train_ratio,
                                                                                          self._param.val_ratio,
                                                                                          self._param.test_ratio,
                                                                                          self._param.setting,
                                                                                          self._param.lr))


    def setup_seed(self):
        torch.manual_seed(self._param.seed)
        torch.cuda.manual_seed_all(self._param.seed)
        np.random.seed(self._param.seed)
        random.seed(self._param.seed)
        torch.backends.cudnn.deterministic = True


    def load_data(self):
        raw_features = U.read_pickle(self._param.data_prefix+self.train_domain+'_features.pickle')
        review_ground_truth = U.read_pickle(self._param.data_prefix+'ground_truth_'+self.train_domain)
        messages = U.read_pickle(self._param.data_prefix+'messages_'+self.train_domain)
        review_group = U.read_pickle(self._param.data_prefix+'review_groups_by_user_'+self.train_domain+'.pkl')

        print('read data...')
        train_rev = U.read_data('train', 'review', self.train_domain, self._param.train_ratio, self._param.val_ratio)
        val_rev = U.read_data('val', 'review', self.train_domain, self._param.train_ratio, self._param.val_ratio)
        test_rev = U.read_data('test', 'review', self.train_domain, self._param.train_ratio, self._param.val_ratio)
        self.rev_nums = len(train_rev) + len(val_rev) + len(test_rev)

        print('read user product')
        train_user, train_prod = U.read_user_prod(train_rev)
        val_user, val_prod = U.read_user_prod(val_rev)
        test_user, test_prod = U.read_user_prod(test_rev)

        portion_train = train_rev + train_user
        portion_val = val_rev + val_user
        portion_test = test_rev + test_user

        print('building feature matrix')
        self.list_idx, self.features, self.nums = U.feature_matrix(raw_features, portion_train, portion_val, portion_test)

        print('building label list')
        self.labels, self.user_ground_truth = U.onehot_label(review_ground_truth, self.list_idx)

        print('building adj matrix')
        idx_map = {j: i for i, j in enumerate(self.list_idx)}
        self.adj = U.construct_adj_matrix(review_ground_truth, idx_map, self.labels)
        self.adj = U.normalize(self.adj + sp.eye(self.adj.shape[0]))
        self.adj = U.sparse_mx_to_torch_sparse_tensor(self.adj).cuda()

        self.labels = torch.LongTensor(np.where(self.labels)[1]).cuda()

        self.idx_train = torch.LongTensor(range(self.nums[-1][0])).cuda()
        self.idx_train_rev = torch.LongTensor(range(self.nums[0][0])).cuda()
        self.idx_val = torch.LongTensor(range(self.nums[-1][0], self.nums[-1][1])).cuda()
        self.idx_test = torch.LongTensor(range(self.nums[-1][1], self.nums[-1][2])).cuda()
        self.idx_test_rev = torch.LongTensor(range(self.nums[1][1], self.nums[2][0])).cuda()
        self.idx_whole = torch.LongTensor(range(self.nums[-1][2])).cuda()

        print('review group size: ', len(review_group))
        self.find_group_idx(review_group, idx_map)



    def find_group_idx(self, group_split_dict, idx_list):
        group0_idx = []
        group1_idx = []
        for rid, gid in group_split_dict.items():
            new_rid = ('u' + rid[0], 'p' + rid[1])
            if gid == 1:
                group1_idx.append(idx_list[new_rid])
            elif gid == 0:
                group0_idx.append(idx_list[new_rid])
        assert len(group1_idx) + len(group0_idx) == self.rev_nums

        # split group0 group1 as train and test set group0:
        self.group0_train_idx = list(set(self.idx_train.tolist()).intersection(set(group0_idx)))
        self.group0_test_idx = list(set(self.idx_test.tolist()).intersection(set(group0_idx)))
        self.group1_train_idx = list(set(self.idx_train.tolist()).intersection(set(group1_idx)))
        self.group1_test_idx = list(set(self.idx_test.tolist()).intersection(set(group1_idx)))

        self.group0_train_idx_tensor = torch.LongTensor(self.group0_train_idx).cuda()
        self.group1_train_idx_tensor = torch.LongTensor(self.group1_train_idx).cuda()
        self.group0_test_idx_tensor = torch.LongTensor(self.group0_test_idx).cuda()
        self.group1_test_idx_tensor = torch.LongTensor(self.group1_test_idx).cuda()

        self.fpr_matrix_train =  FPR_matrix(self.labels, self.idx_train_rev, self.group0_train_idx_tensor, self.group1_train_idx_tensor)
        self.fnr_matrix_train =  FNR_matrix(self.labels, self.idx_train_rev, self.group0_train_idx_tensor, self.group1_train_idx_tensor)
        self.fpr_matrix_test =  FPR_matrix(self.labels, self.idx_test_rev, self.group0_test_idx_tensor, self.group1_test_idx_tensor)
        self.fnr_matrix_test = FNR_matrix(self.labels, self.idx_test_rev, self.group0_test_idx_tensor, self.group1_test_idx_tensor)
        
        
        if os.path.exists("./group_gt/"+str(self.train_domain)+"_group_ground_truth.pickle"):
            print("File already exists")
        self.construct_group_truth_dict(self.train_domain)
        if self.train_domain == 'Zip':
            self.group1_train_idx = np.random.choice(self.group1_train_idx, int(0.5 * len(self.group1_train_idx)), replace=False)
            self.sampled_group1_test_idx = np.random.choice(self.group1_test_idx, int(0.5 * len(self.group1_test_idx)), replace=False)

        print('group 0 : {0}, group 1: {1}'.format(len(group0_idx), len(group1_idx)))
        # return torch.tensor(group0_idx), torch.tensor(group1_idx)

    def split_review(self):
        # split reviews as group1 and group0 pos and neg
        self.pos_g0_train, self.neg_g0_train = U.split_reviewer_nodes(self.group0_train_idx, self.labels)
        self.pos_g0_test, self.neg_g0_test = U.split_reviewer_nodes(self.group0_test_idx, self.labels)
        self.pos_g1_train, self.neg_g1_train = U.split_reviewer_nodes(self.group1_train_idx, self.labels)
        self.pos_g1_test, self.neg_g1_test = U.split_reviewer_nodes(self.group1_test_idx, self.labels)
        if self.train_domain == 'Zip':
            self.sampled_pos_g1_test, self.sampled_neg_g1_test = U.split_reviewer_nodes(self.sampled_group1_test_idx, self.labels)

        pos_train_rev, _ = U.split_reviewer_nodes(self.idx_train_rev, self.labels)
        pos_test_rev, _ = U.split_reviewer_nodes(self.idx_test_rev, self.labels)

        # max DCG score on train and test g1 and g0
        self.maxDCG_g1_train = max_DCG_score(len(self.pos_g1_train))
        self.maxDCG_g0_train = max_DCG_score(len(self.pos_g0_train))
        self.maxDCG_g1_test = max_DCG_score(len(self.pos_g1_test))
        self.maxDCG_g0_test = max_DCG_score(len(self.pos_g0_test))

        self.maxDCG_train = max_DCG_score(len(pos_train_rev))
        self.maxDCG_test = max_DCG_score(len(pos_test_rev))

        print('maxDCG on group1 train: {0}, test: {1}'.format(self.maxDCG_g1_train, self.maxDCG_g1_test))
        print('maxDCG on group0 train: {0}, test: {1}'.format(self.maxDCG_g0_train, self.maxDCG_g0_test))
        print('maxDCG on train review: {0}, test review: {1}'.format(self.maxDCG_train, self.maxDCG_test))

    def print_data(self):
        print("index_train", self.idx_train.size())
        print("index_train_rev", self.idx_train_rev.size())
        print("group0_train_idx", len(self.group0_train_idx))
        print("group1_train_idx", len(self.group1_train_idx))

    def build_GCN_model(self):
        model = GCN(nfeat=32,nhid=self._param.hidden, nclass=self.labels.max().item()+1, dropout=self._param.dropout)
        optimizer = optim.Adam(model.parameters(), lr=self._param.lr, weight_decay=self._param.wd)
        return model.cuda(), optimizer

    def write_group_truth_to_file(self,train_domain):
        idx_review = torch.cat((self.idx_train_rev,self.idx_test_rev))
        g0 = self.group0_train_idx + self.group0_test_idx
        g1 = self.group1_train_idx + self.group1_test_idx
        idx_review = idx_review.cpu().detach().numpy()

        z = []
        for i in idx_review:
            if (i in g0):
                z.append(0)
            elif (i in g1):
                z.append(1)
        with open(str(train_domain)+'_group_ground_truth.pickle', 'wb') as f:
            pickle.dump(z,f)
    def construct_group_truth_dict(self,train_domain):
        with open('./group_gt/'+str(train_domain)+'_group_ground_truth.pickle', 'rb') as f:
            z = pickle.load(f)
        self.z = torch.Tensor(z).cuda()

    def train2(self, epoch, K, difference_matrix, loss_type):
        self.GCN.train()
        self.optimizer.zero_grad()
        criterion = torch.nn.CrossEntropyLoss()

        logit, log_softmax, y_pred, group_pred = self.GCN(self.features, self.adj, self.nums)
        loss1 = criterion(y_pred[self.idx_train_rev],self.labels[self.idx_train_rev])/len(self.idx_train_rev) 


        idx_review = torch.cat((self.idx_train_rev,self.idx_test_rev))
        loss2 = criterion(group_pred[idx_review], self.z.long())/len(idx_review)
        loss = loss1 - loss2

		# prob = torch.exp(log_softmax[self.idx_train_rev])
		# loss1 = criterion(prob ,self.labels[self.idx_train_rev])/len(self.idx_train_rev) 
		# idx_review = torch.cat((self.idx_train_rev,self.idx_test_rev))
		# loss2 = criterion(torch.sigmoid(logit[idx_review]), self.z.long())/len(idx_review)
		# loss = loss1 - loss2

        loss.backward()
		# update the GCN parameter
        self.optimizer.step()
        with torch.no_grad():
            comparison_matrix = construct_comparison_matrix(self.labels[self.idx_train_rev], K)
            ndcg_loss = avg_ranking_loss(comparison_matrix, difference_matrix, logit[self.idx_train_rev][:, 1])
            print(loss)
            print(ndcg_loss)




    def train(self, epoch, K, difference_matrix, loss_type):
        self.GCN.train()
        self.optimizer.zero_grad()
        self.GCN.zero_grad()
        logit, log_softmax , y_pred, group_pred = self.GCN(self.features, self.adj, self.nums)
        comparison_matrix = construct_comparison_matrix(self.labels[self.idx_train_rev], K)
        ndcg_loss = avg_ranking_loss(comparison_matrix, difference_matrix, logit[self.idx_train_rev][:, 1])


        out_arr = []
        xNDCG_group0, xNDCG_group1 = fairness_loss1(logit[:, 1], self.maxDCG_g1_train, self.maxDCG_g0_train, self.pos_g1_train, self.pos_g0_train, self.neg_g1_train, self.neg_g0_train)
        #fairness_loss = torch.abs(xNDCG_group1 - xNDCG_group0)
        fairness_loss = torch.maximum(xNDCG_group1/xNDCG_group0,xNDCG_group0/xNDCG_group1)
        # prob = np.exp(log_softmax.cpu().detach().numpy()[:,1])
        # th = np.mean(prob)
        # print(th)
        # pred_g0 = prob[self.group0_train_idx] > th
        # pred_g1 = prob[self.group1_train_idx] > th
        # fairness_loss = np.minimum(np.sum(pred_g0)/len(pred_g0),np.sum(pred_g1)/len(pred_g1))
        print(ndcg_loss)
        print(fairness_loss)
        print("------------------------------------------")
        out_arr.append(ndcg_loss.cpu().detach())
        out_arr.append(fairness_loss.cpu().detach())
        
        loss = ndcg_loss + fairness_loss
        loss.backward()

        self.optimizer.step()
        with torch.no_grad():
            self.GCN.zero_grad()
            logit, log_softmax , y_pred, group_pred = self.GCN(self.features, self.adj, self.nums)
            true_NDCG = DCG_score(logit[self.idx_train_rev][:,1], self.labels[self.idx_train_rev])/self.maxDCG_train
            true_NDCG_group1 = DCG_score(logit[self.group1_train_idx][:, 1], self.labels[self.group1_train_idx])/self.maxDCG_g1_train
            true_NDCG_group0 = DCG_score(logit[self.group0_train_idx][:, 1], self.labels[self.group0_train_idx])/self.maxDCG_g0_train
            out_arr.append(true_NDCG)
            out_arr.append(true_NDCG_group1)
            out_arr.append(true_NDCG_group0)
            print(true_NDCG,true_NDCG_group1,true_NDCG_group0)
        return out_arr
        


    def test(self, epoch, loss_type, difference_matrix):
        self.GCN.eval()
        loss_dict = dict()
        loss = 0

        with torch.no_grad():
            logit, output, y_pred, group_pred = self.GCN(self.features, self.adj, self.nums)
        true_NDCG = DCG_score(logit[self.idx_test_rev][:, 1], self.labels[self.idx_test_rev])/self.maxDCG_test
        true_NDCG_group0 = DCG_score(logit[self.group0_test_idx][:, 1], self.labels[self.group0_test_idx])/self.maxDCG_g0_test
        true_NDCG_group1 = DCG_score(logit[self.group1_test_idx][:, 1], self.labels[self.group1_test_idx])/self.maxDCG_g1_test



        
        out_arr = []
        out_arr.append(true_NDCG)
        out_arr.append(true_NDCG_group0)
        out_arr.append(true_NDCG_group1)
        print("TEST:          ", true_NDCG,true_NDCG_group0,true_NDCG_group1)
        return out_arr
        



    def main(self, loss_type):
        output_list = []


        difference_matrix = construct_difference_matrix(300).cuda()
        self.split_review()     # get different node group high-low, pos-neg
        
        print("Finish")
        out_tr = []
        out_ts = []
        for epoch in tqdm(range(args.epochs)):
            
            row = self.train(epoch, 300, difference_matrix, loss_type)
            out_tr.append(row)
            
            if (epoch)%4 == 0:
                row_ts = self.test(epoch, loss_type, difference_matrix)
                out_ts.append(row_ts)
        np.array(out_tr)
        np.array(out_ts)
        import pickle
        #pickle.dump( out_tr, open( "result_ratio5.pkl", "wb" ) )
        pickle.dump( out_ts, open( "test_ratio.pkl", "wb" ) )
        
        # model_path = '../model/{0}/{1}{2}{3}/{4}/'.format(self.train_domain,
        #                                                   self._param.train_ratio,
        #                                                   self._param.val_ratio,
        #                                                   self._param.test_ratio,
        #                                                   self._param.setting)
        # if not os.path.exists(model_path):
        #     os.makedirs(model_path)
        # torch.save(self.GCN.state_dict(), model_path+'my_model.pth')



if __name__ == '__main__':
    loss_type = args.optimize
    domain = args.data

    for d in domain:
        myParam = Parameters()
        myParam.train_domain = d
        myParam.tensorboard = True
        print('training domain: {0}'.format(myParam.train_domain))
        print('optimize:', loss_type)
        myFairGCN = GCNModel(myParam)
        myFairGCN.main(loss_type=loss_type)