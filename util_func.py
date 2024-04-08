import scipy.sparse as ssp
import numpy as np
import argparse
import networkx as nx
import scipy.io as sio
import os.path
import sys, copy, math, time, pdb
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm, trange
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score, auc, roc_curve, roc_auc_score,confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
import multiprocessing as mp

class Net(torch.nn.Module):
    def __init__(self, centers, num_seed, h_init):
        super(Net,self).__init__()
        self.seeds = torch.nn.Parameter(centers)         
        self.h = torch.nn.Parameter(torch.ones(1))         
        self.h = torch.nn.init.constant_(self.h, h_init)
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

if(torch.cuda.is_available()):
    device = 'cuda'
else:
    device = 'cpu'
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        return argparse.ArgumentTypeError('Boolean value expected')

def set_seed(seed):
    torch.manual_seed(seed)  # 
    torch.cuda.manual_seed(seed)  # 
    torch.backends.cudnn.deterministic = True  # cudnn
    np.random.seed(seed)  # numpy
    random.seed(seed)  # random and transforms

def get_net(data_name):
    data_name = data_name
    data_dir = os.path.join('data/{}'.format(data_name))
    data = sio.loadmat(data_dir)
    net = data['net']
    
    if 'group' in data:
        attributes = data['group'].toarray().astype('float32')
    else:
        attributes = None
    
    if True:
        net_ = net.toarray()
        assert(np.allclose(net_, net_.T, atol = 1e-8))
    return net, attributes

def load_data(dataset, index, use_splitted, net, test_ratio):
    if(use_splitted == False):
        train_pos, train_neg, test_pos, test_neg = sample_neg(net, test_ratio = test_ratio)
        """
        torch.save(train_pos, 'data/samples/' + dataset + '/' + dataset + 'train_pos_' + str(index) +'.pt')
        torch.save(train_neg, 'data/samples/' + dataset + '/' + dataset + 'train_neg_' + str(index) +'.pt')
        torch.save(test_pos, 'data/samples/' + dataset + '/' + dataset + 'test_pos_' + str(index) +'.pt')
        torch.save(test_neg, 'data/samples/' + dataset + '/' + dataset + 'test_neg_' + str(index) +'.pt')
        """
    else:
        train_pos = torch.load('data/samples/' + dataset + '/' + dataset + 'train_pos_' + str(index) +'.pt')
        train_neg = torch.load('data/samples/' + dataset + '/' + dataset + 'train_neg_' + str(index) +'.pt')
        test_pos = torch.load('data/samples/' + dataset + '/' + dataset + 'test_pos_' + str(index) +'.pt')
        test_neg = torch.load('data/samples/' + dataset + '/' + dataset + 'test_neg_' + str(index) +'.pt')
    return train_pos, train_neg, test_pos, test_neg

def sample_neg(net, train_pos = None, test_pos = None, max_train_num = None, all_unknown_as_negative = False, test_ratio = 0.1):
    net_triu = ssp.triu(net, k = 1)
    row, col , _ = ssp.find(net_triu)
    if train_pos is None and test_pos is None:
        # do sampling
        perm = np.random.choice(range(len(row)), len(row), replace = False)
        row, col = row[perm], col[perm]
        ## split train and test data
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])
    if max_train_num is not None and train_pos is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (train_pos[0][perm], train_pos[1][perm])
    # sample negative link
    train_num = len(train_pos[0]) if train_pos else 0
    test_num = len(test_pos[0]) if test_pos else 0
    neg = ([], [])
    n = net.shape[0]
    print('sampling negative links for train and test')
    if not all_unknown_as_negative:
        ## #negative link == #positive link
        while len(neg[0]) < train_num + test_num:
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            if i < j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg = (neg[0][:train_num], neg[1][:train_num])
        test_neg = (neg[0][train_num:], neg[1][train_num:])
    else:
        while len(neg[0]) < train_num:
            i, j = random.randint(0, n-1), random.randint(0, n - 1)
            if i< j and net[i, j] == 0:
                neg[0].append(i)
                neg[1].append(j)
            else:
                continue
        train_neg = (neg[0], neg[1])
        test_neg_i, test_neg_j, _ = ssp.find(ssp.triu(net == 0, k = 1))
        test_neg = (test_neg_i.tolist(), test_neg_j.tolist())
    
    train_pos = np.array(train_pos)
    train_pos = torch.tensor(train_pos)
    train_neg = np.array(train_neg)
    train_neg = torch.tensor(train_neg)
    test_pos = np.array(test_pos)
    test_pos = torch.tensor(test_pos)
    test_neg = np.array(test_neg)
    test_neg = torch.tensor(test_neg)
    return train_pos, train_neg, test_pos, test_neg

# return all 1-hop neighbors in fringe           
def neighbors(fringe, A):
    res = set()
    for node in fringe:
        nei, _, _ = ssp.find(A[:, node])
        nei = set(nei)
        res = res.union(nei)
    return res

def subgraph_extraction_labeling(ind, A, hop, max_nodes_per_hop,
                                node_info = None
                                ):
    dist = 0
    nodes = set([ind[0], ind[1]])
    visited = set([ind[0], ind[1]])
    fringe = set([ind[0], ind[1]])
    nodes_dist = [0, 0]
    for dist in range(1, hop+1):
        fringe = neighbors(fringe, A)
        fringe = fringe - visited 
        visited = visited.union(fringe)
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(fringe):
                fringe = np.random.choice(list(fringe), max_nodes_per_hop)
        if len(fringe) == 0:
            break
        nodes = nodes.union(fringe)
        nodes_dist += [dist] * len(fringe)
        
    ## move the focal node pair to the top
    nodes.remove(ind[0])
    nodes.remove(ind[1])
    nodes = [ind[0], ind[1]] + list(nodes)
    subgraph = A[nodes, :][:, nodes]
    
    g = nx.from_scipy_sparse_matrix(subgraph)
    ## remove focal link
    if g.has_edge(0, 1):
        g.remove_edge(0, 1)
    
    return g, nodes

def get_cord(net, num_subgraph, ppr):
    cord_x = []
    cord_y = []
    indic = []
    memo = 0
    
    for i in tqdm(range(num_subgraph), desc = 'getting subgraph cords'):
        g = nx.adjacency_matrix(net[i]).todense()
        
        
        g = torch.tensor(g).to(torch.float32)                    #### 
        
        
        g = g/(g.sum(dim = 0) + 1e-7)
        e_1 = torch.zeros(g.shape[0], 1)
        e_2 = torch.zeros(g.shape[0], 1)
        e_1[0, 0] = 1
        e_2[1, 0] = 1
        p_1 = torch.mm(torch.inverse(torch.eye(g.shape[0]) - ppr*g), e_1).squeeze()
        p_2 = torch.mm(torch.inverse(torch.eye(g.shape[0]) - ppr*g), e_2).squeeze()
        
     
        #cord_x.extend(p_1.numpy().tolist())
        #cord_y.extend(p_2.numpy().tolist())
        
        cord_x.extend((p_1/torch.max(p_1)).numpy().tolist())      #### 
        cord_y.extend((p_2/torch.max(p_2)).numpy().tolist())
        
        indic.append(memo + g.shape[0] - 1)
        memo = memo + g.shape[0]
        
    cord_x = torch.tensor(np.array(cord_x), dtype=torch.float32).unsqueeze(dim = 0)
    cord_y = torch.tensor(np.array(cord_y), dtype=torch.float32).unsqueeze(dim = 0)
    cords = torch.cat([cord_x, cord_y], dim = 0)
    indic = np.array(indic)
    return cords.T, indic

def parallel(pos_or_neg, net, hop, max_nodes_per_hop):
    start = time.time()
    #pool = mp.Pool(1)
    pool = mp.Pool(mp.cpu_count())
    results = pool.map_async(parallel_worker, 
                                [((i,j), net, hop, max_nodes_per_hop) for i,j in zip(pos_or_neg[0], pos_or_neg[1])])
    remaining = results._number_left
    pbar = tqdm(total=remaining)
    while True:
        pbar.update(remaining - results._number_left)
        if results.ready(): break
        remaining = results._number_left
        time.sleep(1)

    results = results.get()
    pool.close()
    pbar.close()
    g_list = [g for g, _ in results]
    return g_list

def get_subgraph_set(pos_or_neg, net, hop, max_nodes_per_hop = 100):
    net_set = []
    nodes_set = []
    
    g_list = parallel(pos_or_neg, net, hop, max_nodes_per_hop)
    return g_list

def parallel_worker(x):
    return subgraph_extraction_labeling(*x)

def centers_initial(num_centers, cords):
    inter = 1/math.sqrt(num_centers)
    start_index = inter / 2
    length = int(math.sqrt(num_centers))
    centers_x = torch.arange(start_index, 1 + start_index, inter)
    for i in range(length - 1):
        centers_x = torch.cat((centers_x, torch.arange(start_index, 1 + start_index, inter)), 0)
    centers_x = centers_x.unsqueeze(dim = 1)
    centers_y = torch.zeros(length, 1)
    centers_y = centers_y + start_index
    for i in range(1, length):
        temp = torch.zeros(length, 1)
        temp = temp + start_index + i * inter
        centers_y = torch.cat((centers_y, temp), 0)
    centers = torch.cat((centers_x, centers_y), 1)
    index_dict = {}
    for i in range(num_centers):
        index_dict[i] = 0
    for i in range(cords.shape[0]):
        if(math.floor(cords[i,0]) == 1):
            x_index = 19
        else:
            x_index = math.floor(cords[i,0]*math.sqrt(num_centers))
        if(math.floor(cords[i,1]) == 1):
            y_index = 19
        else:
            y_index = math.floor(cords[i,1]*math.sqrt(num_centers))
        index_dict[x_index + y_index * length] += 1
    i = 0
    for key, value in index_dict.items():
        if value > 0:
            if(i == 0):
                c = centers[key, :].unsqueeze(dim = 0)
                i += 1
            else:
                c = torch.cat((c, centers[key, :].unsqueeze(dim = 0)), dim = 0)
    '''index_dict = sorted(index_dict.items(), key=lambda item:item[1], reverse = True)
    for i in range(len(index_dict)):
        if(i == 0):
            c = centers[index_dict[i][0], :].unsqueeze(dim = 0)
        else:
            c = torch.cat((c, centers[index_dict[i][0], :].unsqueeze(dim = 0)), dim = 0)
        if(i == seed_num):
            break'''
    return c

from sklearn.cluster import KMeans

def cluster(centers_num, centers, cords):
    # ini_c = centers.numpy()
    data = cords.numpy()
    # estimator = KMeans(n_clusters = centers_num, init = ini_c)
    estimator = KMeans(n_clusters = centers_num, n_jobs = -1)
    estimator.fit(data)
    centers = torch.from_numpy(estimator.cluster_centers_)
    """
    plt.text(0.962, 0.875, dataset, zorder = 11, size = 20, backgroundcolor = 'mediumpurple', color = 'white', fontweight = 'bold', horizontalalignment = 'right')
            
    plt.grid(axis = 'both', linestyle = 'dashed', linewidth = 1.5, zorder = 0)
    plt.scatter(data[:, 0], data[:, 1], s = 100, color = 'aqua', alpha = 0.1)
    plt.scatter(centers[:, 0], centers[:, 1], color = 'pink', linewidth = 5, s = 800, alpha = 0.2)
    plt.show()
    """
    return centers

def cal_dist(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist

def normalize(Z):
    d = torch.sum(Z, dim=1)
    d = d.reshape([d.shape[0], 1])
    DK = Z / (d + 1e-8)
    return DK

def cluster_k(dataset, centers, h):
    epsilon = 10e-7
    D = cal_dist(dataset, centers)
    W = torch.exp((-D)/(2*h*h + epsilon))
    W = normalize(W)
    loss = torch.norm(W*D)
    return W, loss

def print_result(AUC, AP, dataset, filename, if_save = False, ppr = 0.5, hop = 1, centers_num = 64, bw = 0.025):
    # print("$$$$$$$$$$$$$$$$$$$$$$$$  " + dataset + "   AUC  AP   $$$$$$$$$$$$$$$$$$$$$$$$$$")
    meanAUC = AUC
    meanAP = AP
    if(if_save == True):
        with open(filename, 'a') as f:
            f.write('Dataset:   ' + dataset + '    || restart probability: ' + str(ppr) + '  || hop: ' + str(hop) + ' ||   ' + '# Gauss sensors: ' + str(centers_num) + ' || bandwidth: ' + str(bw) + 
                    '\n' + 'AUC: ' + str(meanAUC)  + 'AP: ' + str(meanAP) + '\n')

# 
def indic_matrix(indic):
    #Ind = torch.zeros(len(indic), indic[-1]+1)
    row = []
    col = []
    memo = 0
    for i in range(len(indic)):
        index = indic[i] + 1
        row += (index-memo)*[i]
        col += range(memo, index)
        memo = index
    indices = [row, col]
    values = []
    values += len(indices[0])*[1.0]
    Ind = torch.sparse_coo_tensor(indices=indices, values=values, 
                                  size=[len(indic), indic[-1]+1])
    return Ind

def get_eigen_vec(W, ind):
    return torch.mm(ind, W)

def get_label(train_pos, train_neg):
    y_pos = torch.ones(train_pos.shape[1], 1)
    y_neg = torch.zeros(train_neg.shape[1], 1)
    y = torch.cat([y_pos, y_neg], dim = 0)
    return y

def loop(model, graph_list, c, num_seeds=900, h_init=0.025, if_train=False):
    if(torch.cuda.is_available()):
        device = 'cuda'
    else:
        device = 'cpu'

    y_pos = torch.ones(int(len(graph_list)/2), 1)
    y_neg = torch.zeros(int(len(graph_list)/2), 1)
    y = torch.cat([y_pos, y_neg], dim = 0)
    
    cords, indic = get_cord(graph_list, len(graph_list), c)
    ind = indic_matrix(indic)
    ind = indic_matrix(indic)
    ind = ind.to(device)
    y = y.to(device)

    if(if_train):
        centers = centers_initial(num_seeds, cords)
        cords = cords.to(device)
        set_seed(114514)
        model =Net(centers, centers.shape[0], h_init)
        model.init_weights()           # 
        model = model.to(device)
        model.train()
        loss = 1
        epoch = 0
        epsilon = 10e-7
        optimizer = optim.Adam([
            {'params': model.hidden_2.parameters(), 'lr': 1e-1, 'weight_decay': 0.005},
            {'params': model.hidden_3.parameters(), 'lr': 1e-1, 'weight_decay': 0.005},
            {'params': model.hidden_4.parameters(), 'lr': 1e-1, 'weight_decay': 0.005},
            {'params': model.hidden_5.parameters(), 'lr': 1e-1, 'weight_decay': 0.005},
            {'params': model.seeds, 'lr': 1e-3},
            {'params': model.h, 'lr': 1e-4}
        ],
        )
        criterion = nn.BCELoss()
        ACC = []
        EPOCH = []
        LOSS = []
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.988)#gamma=0.988
        while(epoch <= 700):
            prediction = model(cords, ind)
            regularization_loss = 0
            i=0
            for param in model.parameters():
                if(i==2 or i == 4 or i==6 or i==8):
                    regularization_loss += torch.sum(abs(param))
                i += 1
            loss = criterion(prediction, y) + regularization_loss * 0.01
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            EPOCH.append(epoch)
            if(epoch % 40 == 0):
                print("epoch:" + str(epoch) + "    ||Loss: " + str(loss))
            epoch = epoch + 1
        return model, roc_auc_score(y.detach().cpu(), prediction.detach().cpu())
    else:
        cords = cords.to(device)
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            prediction = model(cords, ind)
        pred = prediction.detach()
        pred = pred.squeeze()
        pred = pred.to('cpu')
        pred = np.array(pred)
    return roc_auc_score(y.cpu(), pred)


