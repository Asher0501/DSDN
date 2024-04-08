from util_func import *
from neuralnet import *
from torch.autograd import Variable
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser(description='Link Prediction with DSDN')
# general settings
parser.add_argument('--data-name', default='NS', help='network name')
parser.add_argument('--batch-size', type=int, default=40000)
parser.add_argument('--use-splitted', type=str2bool, default=True, 
                    help='use splitted data or not, test ratio can not be changed if it is True')
parser.add_argument('--split-index', type=int, default=1,
                    help='choose which split data for training and testing')
parser.add_argument('--test-ratio', type=float, default=0.1,
                    help='ratio of test links')
parser.add_argument('--save-path', type=str, default='result/result.txt',
                    help='the saving path of link prediction results')
parser.add_argument('--if-save', type=str2bool, default=True,
                    help='save link prediction results or not')
parser.add_argument('--use-attribute', type=str2bool, default=False,
                    help='use attribute to predict or not')
# model settings
parser.add_argument('--hop', default=None, type=int, 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--sensors-num', type=int, default=64, 
                    help='the number of Gauss sensors (default: 64)')
parser.add_argument('--bandwidth', type=float, default=0.025,
                    help='the bandwidth of Gauss sensors')
parser.add_argument('--restart-value', type=float, default=None,
                    help='the restart probability of Personalized Pagerank')
parser.add_argument('--max-nodes-per-hop', default=90, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')

args = parser.parse_args()

def LP(train_pos, train_neg, test_pos, test_neg):
    global model
    train_auc, valid_auc, test_auc = [], [], []
    count = 0
    #global hop, c
    valid_num = int(math.ceil(train_pos.shape[1]/0.9*0.05))
    valid_split = np.random.choice(train_pos.shape[1], valid_num, replace=False)
    valid_pos = train_pos[:, valid_split]
    valid_neg = train_neg[:, valid_split]
    train_split = list((set(list(range(train_pos.shape[1]))) - set(list(valid_split))))
    train_pos_ = train_pos[:, train_split]
    train_neg_ = train_neg[:, train_split]
    train = torch.cat([train_pos_, train_neg_], dim = 1).tolist()

    valid = torch.cat([valid_pos, valid_neg], dim = 1).tolist()
    tes = torch.cat([test_pos, test_neg], dim = 1).tolist()

    #if(hop == None):
    c = 0.7
    for hop in [1,2,3]:
        if(count == 0):
            train_graph_list = get_subgraph_set(train, net, hop, int(args.max_nodes_per_hop))
        model, auc = loop(None, train_graph_list, c, if_train=True)
        train_auc.append(auc)
        print('train auc: ', auc)

        if(count == 0):
            valid_graph_list = get_subgraph_set(valid, net, hop, int(args.max_nodes_per_hop))
        auc = loop(model, valid_graph_list, c, if_train=False)
        valid_auc.append(auc)
        print('valid auc: ', auc)
    
    with open(args.save_path, 'a') as f:
        f.write(args.data_name + ' hop\n' + str(valid_auc) + '\n')
    hop = np.argmax(valid_auc) + 1
    print(hop)
    train_auc = []
    valid_auc = []
    for num_seed, h_init in [(400, 0.05), (900, 0.025)]:
        train_graph_list = get_subgraph_set(train, net, hop, int(args.max_nodes_per_hop))
        model, auc = loop(None, train_graph_list, c, num_seed, h_init, if_train = True)
        train_auc.append(auc)
        valid_graph_list = get_subgraph_set(valid, net, hop, int(args.max_nodes_per_hop))
        auc = loop(model, valid_graph_list, c, if_train=False)
        valid_auc.append(auc)
    
    with open(args.save_path, 'a') as f:
        f.write(args.data_name + ' numseed, h_init\n' + str(valid_auc) + '\n')
    
    if(np.argmax(valid_auc)==0):
        num_seed = 400
        h_init = 0.05
    else:
        num_seed = 900
        h_init = 0.025

        
    train_auc = []
    valid_auc = []
    best_auc = 0

    count = 0
    for c in [0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8]:
        if(count == 0):
            train_graph_list = get_subgraph_set(train, net, hop, int(args.max_nodes_per_hop))
        model, auc = loop(None, train_graph_list, c, num_seed, h_init, if_train=True)
        train_auc.append(auc)
        print('train auc: ', auc)

        if(count == 0):
            valid_graph_list = get_subgraph_set(valid, net, hop, int(args.max_nodes_per_hop))
        auc = loop(model, valid_graph_list, c, num_seed, h_init, if_train=False)
        valid_auc.append(auc)
        print('valid auc: ', auc)

        if(best_auc < auc):
            best_auc = auc
            best_c = c

        count += 1
    
    train = torch.cat([train_pos, train_neg], dim = 1).tolist()
    train_graph_list = get_subgraph_set(train, net, hop, int(args.max_nodes_per_hop))
    model, auc = loop(None, train_graph_list, best_c, num_seed, h_init, if_train=True)
        
    test_graph_list = get_subgraph_set(tes, net, hop, int(args.max_nodes_per_hop))
    auc = loop(model, test_graph_list, best_c, num_seed, h_init, if_train=False)
    test_auc.append(auc)
    print('test auc: ', auc)

    with open(args.save_path, 'a') as f:
        f.write(args.data_name + '  C \n' + 'train_auc: '+str(train_auc)+'\n'+
        'valid_auc: '+ str(valid_auc) + '\n' + 
        'test_auc: '+ str(test_auc) + '\n\n')
    return auc

# datasets: 'USAir', 'NS', 'PB', 'Yeast', 'Router', 'Celegans', 'pubmed', 'PPI_subgraph', 'citeseer', 'cora', 'Power'
dataset = args.data_name
num_sensors = args.sensors_num
bandwidth = args.bandwidth
global c 
c = args.restart_value
global hop 
hop = args.hop

use_splitted = args.use_splitted
if_save = args.if_save
filename = args.save_path # save path of results
j = args.split_index
batch_size = args.batch_size # for link prediction with attributes

if(torch.cuda.is_available()):
    device = 'cuda'
else:
    device = 'cpu'

net, attributes = get_net(dataset)

T1 = time.time()
print("Link prediction on " + dataset + " # " + str(j))
train_pos, train_neg, test_pos, test_neg = load_data(dataset, j, use_splitted, net, args.test_ratio)

net = net.copy()
net[test_pos.tolist()[0], test_pos.tolist()[1]] = 0  # mask test links
net[test_pos.tolist()[1], test_pos.tolist()[0]] = 0  # mask test links
net.eliminate_zeros()

auc = LP(train_pos, train_neg, test_pos, test_neg)

T2 = time.time()