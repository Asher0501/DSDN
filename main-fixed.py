from util_func import *
from neuralnet import *
from torch.autograd import Variable
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

parser = argparse.ArgumentParser(description='Link Prediction with DSDN')
# general settings
parser.add_argument('--data-name', default='NS', help='network name')
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
# model settings
parser.add_argument('--hop', default=2, type=int, 
                    help='enclosing subgraph hop number, \
                    options: 1, 2,..., "auto"')
parser.add_argument('--bins-num', type=int, default=900, 
                    help='the number of bins distributed evenly (default: 900)')
parser.add_argument('--h-init', type=float, default=0.025,
                    help='the intialization of bandwidth')
parser.add_argument('--c', type=float, default=0.7,
                    help='the restart probability of Personalized Pagerank')
parser.add_argument('--max-nodes-per-hop', default=100, 
                    help='if > 0, upper bound the # nodes per hop by subsampling')

args = parser.parse_args()

def LP(train_pos, train_neg, test_pos, test_neg):
    global model
    train_auc, valid_auc, test_auc = [], [], []
    count = 0

    train = torch.cat([train_pos, train_neg], dim = 1).tolist()
    tes = torch.cat([test_pos, test_neg], dim = 1).tolist()

    c = args.c
    hop = args.hop
    num_seed = args.bins_num
    h_init = args.h_init

    train_graph_list = get_subgraph_set(train, net, hop, int(args.max_nodes_per_hop))
    model, auc = loop(None, train_graph_list, c, num_seed, h_init, if_train=True)
    train_auc.append(auc)
    print('train auc: ', auc)

    test_graph_list = get_subgraph_set(tes, net, hop, int(args.max_nodes_per_hop))
    auc = loop(model, test_graph_list, c, num_seed, h_init, if_train=False)
    test_auc.append(auc)
    print('test auc: ', auc)
    
    """
    with open(args.save_path, 'a') as f:
        f.write(args.data_name + '  C \n' + 'train_auc: '+str(train_auc)+'\n'+
        'valid_auc: '+ str(valid_auc) + '\n' + 
        'test_auc: '+ str(test_auc) + '\n\n')"""
    
    return auc

# datasets: 'USAir', 'NS', 'PB', 'Yeast', 'Router', 'Celegans', 'pubmed', 'PPI_subgraph', 'citeseer', 'cora', 'Power'
use_splitted = args.use_splitted
if_save = args.if_save
filename = args.save_path # save path of results
j = args.split_index


if(torch.cuda.is_available()):
    device = 'cuda'
else:
    device = 'cpu'
dataset = args.data_name
net, _ = get_net(args.data_name)

T1 = time.time()
print("Link prediction on " + dataset + " # " + str(j))
train_pos, train_neg, test_pos, test_neg = load_data(dataset, j, use_splitted, net, args.test_ratio)

net = net.copy()
net[test_pos.tolist()[0], test_pos.tolist()[1]] = 0  # mask test links
net[test_pos.tolist()[1], test_pos.tolist()[0]] = 0  # mask test links
net.eliminate_zeros()

auc = LP(train_pos, train_neg, test_pos, test_neg)

T2 = time.time()