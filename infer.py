import argparse
import os.path as osp

import numpy as np
import torch
import pandas as pd

from pa_gnn import *
from pa_subparser import add_pa_subparser
from pa_data import build_graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    subparsers = parser.add_subparsers()
    add_pa_subparser(subparsers)
    args = parser.parse_args()    

    print('Using CPU')
    device = torch.device('cpu')

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    source = "sfs_selected_const"   
    dataFolder = f"../../data_cleaning/{source}"    

    binaryF1 = []
    Accuracy = []
    AUCs = []
    Sensitivitys = []
    Specificitys = []

    mean_fpr_root = np.linspace(0, 1, 100)
    TPRs = [] 
    
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
        
        f1_score, acc, roc_info, auroc, specificity, sensitivity = infer_gnn(data, args, source, k, device)        
        fpr, tpr, thresholds = roc_info
        
        interp_tpr = np.interp(mean_fpr_root, fpr, tpr)
        interp_tpr[0] = 0.0
        TPRs.append(interp_tpr)
        
        binaryF1.append(f1_score)
        Accuracy.append(acc)
        AUCs.append(auroc)
        Sensitivitys.append(sensitivity)
        Specificitys.append(specificity)            
        
    outputfile = f'perform_results.npz'
    np.savez(outputfile, AUC = AUCs,
                         F1 = binaryF1,
                         Accuracy = Accuracy,
                         Sensitivitys = Sensitivitys,
                         Specificitys = Specificitys,
                         TPRs = TPRs)

if __name__ == '__main__':
    main()
