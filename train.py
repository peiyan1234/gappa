import argparse
import os
import os.path as osp

import numpy as np
import torch

from pa_gnn import train_gnn
from pa_subparser import add_pa_subparser
from pa_data import build_graph
from utils.utils import auto_select_gpu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_types', type=str, default='GAPPA_GAPPA_GAPPA')
    parser.add_argument('--post_hiddens', type=str, default=None,) 
    parser.add_argument('--concat_states', action='store_true', default=False)
    parser.add_argument('--norm_embs', type=str, default=None,) 
    parser.add_argument('--aggr', type=str, default='mean',)
    parser.add_argument('--node_dim', type=int, default=64) 
    parser.add_argument('--edge_dim', type=int, default=64) 
    parser.add_argument('--edge_mode', type=int, default=1) 
    parser.add_argument('--gnn_activation', type=str, default='relu')
    parser.add_argument('--pred_prob_hiddens', type=str, default='64') 
    parser.add_argument('--pred_prob_activation', type=str, default='relu')    
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--opt_scheduler', type=str, default='step')
    parser.add_argument('--opt_restart', type=int, default=0)
    parser.add_argument('--opt_decay_step', type=int, default=1000)
    parser.add_argument('--opt_decay_rate', type=float, default=0.9)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--known', type=float, default=0.7) # 1 - edge dropout rate
    parser.add_argument('--auto_known', action='store_true', default=False)
    parser.add_argument('--loss_mode', type=int, default = 0)     
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--save_model', action='store_true', default=True)
    parser.add_argument('--save_prediction', action='store_true', default=True)
    parser.add_argument('--transfer_dir', type=str, default=None)
    parser.add_argument('--transfer_extra', type=str, default='')
    parser.add_argument('--mode', type=str, default='train') # debug
    parser.add_argument('--Lagrange_multiplier', type=float, default=0.5) # Lagrange multiplier
    subparsers = parser.add_subparsers()
    add_pa_subparser(subparsers)
    args = parser.parse_args()

    # select device
    if torch.cuda.is_available():
        cuda = auto_select_gpu()
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda)
        print('Using GPU {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
        device = torch.device('cuda:{}'.format(cuda))
    else:
        print('Using CPU')
        device = torch.device('cpu')

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    source = "sfs_selected_const"    
    dataFolder = f"../../data_cleaning/{source}"

    for k in range(25):
                
        cv_data_npz = osp.join(dataFolder, f"CV_{k}_data.npz")
        cv_data = np.load(cv_data_npz)  

        data = build_graph( cv_data['Train_X'].astype("float32"),
                            cv_data['Train_y'].astype("int32"),
                            cv_data['Val_X'].astype("float32"),
                            cv_data['Val_y'].astype("int32"),
                            cv_data['Test_X'].astype("float32"),
                            cv_data['Test_y'].astype("int32"),
                            args.node_mode, 
                            args.seed,                            
                            cv_data['cMsk'] ) 
        
        log_path = f'../gnn_{source}/CV_{k}/{args.log_dir}/'
        if os.path.isdir(log_path) == False:
            os.makedirs(log_path)
        train_gnn(data, args, source, k, log_path, device)

if __name__ == '__main__':
    main()
