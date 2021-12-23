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
import utils as U
from tensorboardX import SummaryWriter
from sklearn import metrics
from os import listdir
from os.path import isfile, join
from tqdm import tqdm
from ndcg_score import *
import matplotlib.pyplot as plt
import collections
torch.cuda.set_device(0)
device = torch.device('cuda:0')
torch.cuda.empty_cache()
class Parameters:
	def __init__(self):
		self.seed = 42
		self.epochs = 70
		self.lr = 0.005
		self.wd = 1e-3
		self.hidden = 16
		self.dropout = 0.5
		self.train_ratio = 0.3
		self.val_ratio = 0.2
		self.test_ratio = 0.5
		self.train_domain = 'Chi'
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
		self.z = None
		self.setup_seed()
		self.load_data()

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
		self.train_rev = U.read_data('train', 'review', self.train_domain, self._param.train_ratio, self._param.val_ratio)
		self.val_rev = U.read_data('val', 'review', self.train_domain, self._param.train_ratio, self._param.val_ratio)
		self.test_rev = U.read_data('test', 'review', self.train_domain, self._param.train_ratio, self._param.val_ratio) 
		self.rev_nums = len(self.train_rev) + len(self.val_rev) + len(self.test_rev)

		print('read user product')
		self.train_user, self.train_prod = U.read_user_prod(self.train_rev)
		self.val_user, self.val_prod = U.read_user_prod(self.val_rev)
		self.test_user, self.test_prod = U.read_user_prod(self.test_rev)
        
	def write_user_degree_list_to_file2(self):
		print(" ")
		print(" ")
		print("==================================")
		print(" ")
		print(" ")
		print("Reivew example:", self.train_rev[0])
		print("User example:", self.train_user[0])
		reviews = self.train_rev + self.val_rev + self.test_rev
		users = self.train_user + self.val_user + self.test_user
		
		print("Total number of Reviews: ", len(reviews))
		print("Total number of Users:   ", len(users))

		user_degree_dict = {}
		list_out = []
		for u in users:
			user_degree_dict[u] = 0
		for r in reviews:
			userId = r[0]
			user_degree_dict[userId] += 1
		for key, value in user_degree_dict.items():
			row = {}
			row["degree"] = value
			list_out.append(row)
		with open(str(self.train_domain)+'_user_degree_list.pickle', 'wb') as f:
			pickle.dump(list_out,f)

	def count(self):
		print(" ")
		print(" ")
		print("==================================")
		print(" ")
		print(" ")
		print("Reivew example:", self.train_rev[0])
		print("User   example:", self.train_user[0])
		with open(str(self.train_domain)+'_user_degree_list.pickle', 'rb') as f:
			user_degree_list = pickle.load(f)	
		unique_counts = collections.Counter(d['degree'] for d in user_degree_list)

		cutoff_point1 = 2
		cutoff_point2 = 5
		res1,res2,res3 = 0,0,0
		for key,value in unique_counts.items():
			if (key < cutoff_point1):
				res1 += value
			elif (key >= cutoff_point1 and key < cutoff_point2):
				res2 += value
			elif (key >= cutoff_point2):
				res3 += value
			else: 
				print("Error")
		total = res1 + res2 + res3 
		x = ['Less than 2', 'Between 2 and 5', 'greater than or equal to 5']
		y = [res1/total*100,res2/total*100,res3/total*100]
		x_pos = [i for i, _ in enumerate(x)]
		plt.xlabel("Degree")
		plt.ylabel("Percentage of Accounts")
		plt.bar(x_pos, y, color='green')
		plt.xticks(x_pos, x)
		save_results_to = './plots/'
		plt.savefig(save_results_to + str(self.train_domain)+'.png', dpi = 400)

	def main(self, loss_type):
		self.write_user_degree_list_to_file2()
		self.count()
		print('Finish!')
        

if __name__ == '__main__':
	loss_type = ['fairness', 'ndcg']
	domain = ['Chi','Zip','NYC']

	for d in domain:
		myParam = Parameters()
		myParam.train_domain = d
		myParam.tensorboard = False
		print('training domain: {0}'.format(myParam.train_domain))
		myFairGCN = GCNModel(myParam)
		myFairGCN.main(loss_type=loss_type)