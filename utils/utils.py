import torch.optim as optim
import numpy as np
import os.path as osp
import torch
import subprocess

def np_random(seed=None):
    rng = np.random.RandomState()
    rng.seed(seed)
    return rng

def get_activation(activation):
    if activation == 'relu':
        return torch.nn.ReLU()
    elif activation == 'prelu':
        return torch.nn.PReLU()
    elif activation == 'tanh':
        return torch.nn.Tanh()
    elif (activation is None) or (activation == 'none'):
        return torch.nn.Identity()
    else:
        raise NotImplementedError

def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p : p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adamW':
        optimizer = optim.AdamW(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    return scheduler, optimizer

class objectview(object):
    def __init__(self, d):
        self.__dict__ = d

def save_mask(length,true_rate,log_dir,seed):
    np.random.seed(seed)
    mask = np.random.rand(length) < true_rate
    np.save(osp.join(log_dir,'len'+str(length)+'rate'+str(true_rate)+'seed'+str(seed)),mask)
    return mask

def get_known_mask(known_prob, edge_num):
    known_mask = (torch.FloatTensor(edge_num, 1).uniform_() < known_prob).view(-1)

    return known_mask

def get_test_mask(known_prob, test_edge_mask, edge_num):

    num = test_edge_mask.sum()

    known_mask = (torch.FloatTensor(num, 1).uniform_() < known_prob).view(-1)
    test_arr = np.zeros((edge_num, 1), dtype=int)

    j = 0
    for i in range(edge_num):
        if test_edge_mask[i] == True:
            if known_mask[j] == True:
                test_arr[i] = True
            j += 1
            
    test_mask = (torch.Tensor(test_arr) > 0).view(-1)

    return test_mask

def get_nonnan_known_mask(test_num, values, val_num):
    nan_mask = (torch.isnan(values)).view(-1)
    
    nonnan_mask = ~nan_mask
   
    train_arr = np.zeros((len(values), 1), dtype=int)
    val_arr = np.zeros((len(values), 1), dtype=int)
    test_arr = np.zeros((len(values), 1), dtype=int)
        
    for i in range(len(values)):
        if nonnan_mask[i] == True:
            if i < len(values)-val_num-test_num:
                train_arr[i] = True
            elif i < len(values)-test_num:
                val_arr[i] = True
            else:
                test_arr[i] = True    
            
    train_mask = (torch.Tensor(train_arr) > 0).view(-1)
    val_mask = (torch.Tensor(val_arr) > 0).view(-1)
    test_mask = (torch.Tensor(test_arr) > 0).view(-1)

    return train_mask, test_mask, val_mask

def get_y_mask(test_num, df_y, valid_num):

    train_arr = np.zeros(df_y.shape, dtype=int)
    test_arr = np.zeros(df_y.shape, dtype=int)
    valid_arr = np.zeros(df_y.shape, dtype=int)

    for i in range(df_y.shape[0]):
        if i < df_y.shape[0]-test_num-valid_num:
            train_arr[i] = True
        elif i < df_y.shape[0]-test_num:
            valid_arr[i] = True
        else:
            test_arr[i] = True
    
    train_mask = (torch.Tensor(train_arr) > 0).view(-1)
    test_mask = (torch.Tensor(test_arr) > 0).view(-1)
    valid_mask = (torch.Tensor(valid_arr) > 0).view(-1)

    return train_mask, test_mask, valid_mask


def mask_edge(edge_index, edge_attr, mask, remove_edge, ncol, noSplitY=False):   
    edge_index = edge_index.clone().detach()
    edge_attr = edge_attr.clone().detach()    
    mask = mask.clone().detach().to(torch.device('cpu'))

    if noSplitY:
        if remove_edge:
            edge_index = edge_index[:,mask]
            edge_attr = edge_attr[mask]
        else:
            edge_attr[~mask] = 0.
        return edge_index, edge_attr
    else:
        tmp_arr = np.zeros(mask.shape, dtype=int)
        tmp_arr = torch.tensor(tmp_arr, dtype=torch.int)
        for i in range(int(mask.shape[0])):
            if (i+1) % ncol == 0:
                tmp_arr[i] = 1

        tmp_arr = tmp_arr.clone().detach()

        hasYmsk = (torch.Tensor(tmp_arr) > 0).view(-1)
        hasYmsk = torch.logical_and(hasYmsk, mask)
        hasnotYMsk = torch.logical_xor(hasYmsk, mask)

        if remove_edge:
            hasnotY_index = edge_index[:,hasnotYMsk]
            hasnotY_attr  = edge_attr[hasnotYMsk]

            hasY_index = edge_index[:, hasYmsk]
            hasY_attr  = edge_attr[hasYmsk]
            
        else:
            raise ValueError("cannot support this mode, must remove in-connected edge")
            
        return hasnotY_index, hasnotY_attr, hasY_index, hasY_attr, hasYmsk

# get gpu usage
def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory

def auto_select_gpu(memory_threshold = 7000, smooth_ratio=200, strategy='greedy'):
    gpu_memory_raw = get_gpu_memory_map() + 10
    if strategy=='random':
        gpu_memory = gpu_memory_raw/smooth_ratio
        gpu_memory = gpu_memory.sum() / (gpu_memory+10)
        gpu_memory[gpu_memory_raw>memory_threshold] = 0
        gpu_prob = gpu_memory / gpu_memory.sum()
        cuda = str(np.random.choice(len(gpu_prob), p=gpu_prob))
        print('GPU select prob: {}, Select GPU {}'.format(gpu_prob, cuda))
    elif strategy == 'greedy':
        cuda = np.argmin(gpu_memory_raw)
        print('GPU mem: {}, Select GPU {}'.format(gpu_memory_raw[cuda], cuda))
    return cuda