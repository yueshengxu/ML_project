import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.nfeat = nfeat
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.linear_r = nn.Linear(5, nfeat)
        self.linear_p = nn.Linear(6, nfeat)
        self.linear_u = nn.Linear(7, nfeat)
        self.linear_y = nn.Linear(2, 2)
        self.linear_a = nn.Linear(2, 2)
        
    def forward(self, inputx, adj, nums):
        x = torch.zeros(len(inputx),self.nfeat)
        x = x.cuda()
        # review
        x[:nums[0][0]] = self.linear_r(torch.cuda.FloatTensor(inputx[:nums[0][0]]))
        # user
        x[nums[0][0]:nums[0][1]] = self.linear_u(torch.cuda.FloatTensor(inputx[nums[0][0]:nums[0][1]]))

        if nums[1][0] != nums[1][1]:
            x[nums[0][1]:nums[1][0]] = self.linear_r(torch.cuda.FloatTensor(inputx[nums[0][1]:nums[1][0]]))
            x[nums[1][0]:nums[1][1]] = self.linear_u(torch.cuda.FloatTensor(inputx[nums[1][0]:nums[1][1]]))
        if nums[2][0] != nums[2][1]:
            x[nums[1][1]:nums[2][0]] = self.linear_r(torch.cuda.FloatTensor(inputx[nums[1][1]:nums[2][0]]))
            x[nums[2][0]:nums[2][1]] = self.linear_u(torch.cuda.FloatTensor(inputx[nums[2][0]:nums[2][1]]))
        # product
        x[nums[2][1]:] = self.linear_p(torch.cuda.FloatTensor(inputx[nums[2][1]:]))
        x = F.relu(self.gc1(x, adj))
        # x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        out = self.linear_y(x)
        a = self.linear_a(x)
        return x,F.log_softmax(x, dim=1),torch.sigmoid(out),torch.sigmoid(a)

        #return x,F.log_softmax(x, dim=1)
        #return F.log_softmax(x, dim=1)
