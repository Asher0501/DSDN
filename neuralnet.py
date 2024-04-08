
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from util_func import cluster_k, get_eigen_vec

def get_model():
    model = nn.Sequential(
        nn.Linear(64, 16),
        torch.nn.BatchNorm1d(16, eps=1e-05, momentum=0.1),
        torch.nn.LeakyReLU(negative_slope=0.01, inplace=False),
        nn.Linear(16, 2),
        torch.nn.BatchNorm1d(2, eps=1e-05, momentum=0.1),
        torch.nn.LeakyReLU(negative_slope=0.01, inplace=False),
        nn.Linear(2, 1),
        torch.nn.Sigmoid(),
    )
    return model

class Net(torch.nn.Module):
    def __init__(self, centers, num_seed):
        super(Net,self).__init__()
        self.seeds = torch.nn.Parameter(centers)         
        self.h = torch.nn.Parameter(torch.ones(1))         
        self.h = torch.nn.init.constant_(self.h, 0.025)
        self.hidden_2 = torch.nn.Linear(num_seed, 64)
        self.hidden_3 = torch.nn.Linear(64, 16)
        self.hidden_4 = torch.nn.Linear(16, 2)
        self.hidden_5 = torch.nn.Linear(2, 1)
        
        self.bn_3 = torch.nn.BatchNorm1d(64, eps = 1e-03, momentum = 0.1)
        self.bn_1 = torch.nn.BatchNorm1d(16, eps = 1e-03, momentum = 0.1)
        self.LLU = torch.nn.LeakyReLU(negative_slope = 0.01, inplace = False)
        self.bn_2 = torch.nn.BatchNorm1d(2, eps = 1e-03, momentum = 0.1)
        
    def init_weights(layer):
        if type(layer) == nn.Linear:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out')
            nn.init.constant_(layer.bias, 0.1)
 
    def forward(self, cords, ind):
        W, _ = cluster_k(cords, self.seeds, self.h)    
        x = get_eigen_vec(W, ind)
        #print(x.device)
        x = self.LLU(self.hidden_2(x))
        x = self.bn_3(x)
        x = self.LLU(self.hidden_3(x))
        x = self.bn_1(x)
        x = self.LLU(self.hidden_4(x))
        x = self.bn_2(x)
        x = torch.sigmoid(self.hidden_5(x))
        #x = F.log_softmax(self.hidden_5(x),dim=1)
        return x