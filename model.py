import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class GCNLayer(nn.Module):

    def __init__(self, in_features, out_features, gcnii_varient = False):

        super(GCNLayer, self).__init__() 
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(init.kaiming_uniform_(torch.empty(self.in_features, self.out_features), mode='fan_in', nonlinearity='relu'))

    def forward(self, input, adj , h_0 , lamda, alpha, l):

        h_l = torch.spmm(adj, input)
        features = (1 - alpha) * h_l + alpha * h_0
        n = self.weight.shape[0]
        I_n = torch.eye(n) 
        beta = np.log((lamda / l) + 1)
        term1 = (1 - beta) * I_n
        term2 = beta * self.weight
        weights = term1 + term2
        output = torch.mm(features, weights)
        return output

if __name__ == '__main__':
    pass






