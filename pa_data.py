import pandas as pd
from torch_geometric.data import Data
import torch
import numpy as np

from utils.utils import get_nonnan_known_mask, mask_edge, get_y_mask

def create_node(df, mode):
    if mode == 0: # onehot feature node, all 1 sample node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol))
        feature_node[np.arange(ncol), feature_ind] = 1
        sample_node = [[1]*ncol for i in range(nrow)]
        node = sample_node + feature_node.tolist()
    elif mode == 1: # onehot sample and feature node
        nrow, ncol = df.shape
        feature_ind = np.array(range(ncol))
        feature_node = np.zeros((ncol,ncol+1))
        feature_node[np.arange(ncol), feature_ind+1] = 1
        sample_node = np.zeros((nrow,ncol+1))
        sample_node[:,0] = 1
        node = sample_node.tolist() + feature_node.tolist()
    return node

def create_edge(df):
    n_row, n_col = df.shape
    edge_start = []
    edge_end = []
    for x in range(n_row):
        edge_start = edge_start + [x] * n_col # obj
        edge_end = edge_end + list(n_row+np.arange(n_col)) # att    

    edge_start_new = edge_start + edge_end
    edge_end_new = edge_end + edge_start

    return (edge_start_new, edge_end_new)

def create_edge_attr(df, cMsk):
    nrow, ncol = df.shape
    edge_attr = []
    values = []
    edge_cMsk = []
    if np.sum(cMsk) > 0:
        cMsk = cMsk.tolist()
        for i in range(nrow):            
            for j in range(ncol):
                edge_attr.append([float(df.iloc[i,j])])
                values.append(float(df.iloc[i,j]))   
                if j == ncol-1:
                    edge_cMsk.append(1)
                else:
                    if cMsk[j] == True:
                        edge_cMsk.append(1)
                    else:
                        edge_cMsk.append(0)

        edge_cMsk += edge_cMsk
        edge_cMsk = (torch.Tensor(edge_cMsk) > 0).view(-1)             
    else:
        for i in range(nrow):
            for j in range(ncol):
                edge_attr.append([float(df.iloc[i,j])])
                values.append(float(df.iloc[i,j]))
    
    edge_attr = edge_attr + edge_attr
    return edge_attr, values, edge_cMsk

def build_graph( Train_X,
                Train_y,
                Val_X,
                Val_y,
                Test_X,
                Test_y,
                node_mode,
                seed=0,                
                cMsk=None ):

    train_num, _ = Train_y.shape
    test_num, _ = Test_y.shape
    val_num, _ = Val_y.shape
    
    # create Data Frame to prepare the generation of graph for train / val / test
    X = np.concatenate((Train_X, Val_X, Test_X), axis=0)
    df_X = pd.DataFrame(X)
    df_y = np.concatenate((Train_y, Val_y, Test_y), axis=0)

    _, ncol = df_X.shape
    
    edge_start, edge_end = create_edge(df_X)
    edge_index = torch.tensor([edge_start, edge_end], dtype=int)
    
    edge_attr, values, edge_cMsk = create_edge_attr(df_X, cMsk)
    edge_attr  = torch.tensor(edge_attr, dtype=torch.float)
    values = torch.tensor(values, dtype=torch.float)
    node_init  = create_node(df_X, node_mode) 

    x = torch.tensor(node_init, dtype=torch.float)
    y = torch.tensor(df_y, dtype=torch.float)

    torch.manual_seed(seed)
        
    train_edge_mask, test_edge_mask, valid_edge_mask = get_nonnan_known_mask(test_num*ncol, values, val_num*ncol)

    # get train graph
    double_train_edge_mask = torch.cat((train_edge_mask, train_edge_mask), dim=0)
    train_edge_index, train_edge_attr, train_edge_y_index, train_edge_y_attr, train_edge_y_mask = mask_edge(edge_index, edge_attr,
                                                                     double_train_edge_mask, True, ncol)
    train_labels = torch.tensor(Train_y, dtype=torch.float)

    # get valid graph
    valid_edge_mask = torch.logical_or(train_edge_mask, valid_edge_mask)
    double_valid_edge_mask = torch.cat((valid_edge_mask, valid_edge_mask), dim=0)
    double_valid_edge_mask = torch.logical_xor(double_valid_edge_mask, train_edge_y_mask)
    valid_edge_index, valid_edge_attr, valid_edge_y_index, valid_edge_y_attr, valid_edge_y_mask = mask_edge(edge_index, edge_attr,
                                                                     double_valid_edge_mask, True, ncol)
    valid_labels = torch.tensor(Val_y, dtype=torch.float)

    # get test graph
    test_edge_mask = torch.logical_or(train_edge_mask, test_edge_mask)
    test_edge_mask = torch.logical_or(valid_edge_mask, test_edge_mask)
    double_test_edge_mask = torch.cat((test_edge_mask, test_edge_mask), dim=0)
    double_test_edge_mask = torch.logical_xor(double_test_edge_mask, train_edge_y_mask)
    double_test_edge_mask = torch.logical_xor(double_test_edge_mask, valid_edge_y_mask)
    test_edge_index, test_edge_attr, test_edge_y_index, test_edge_y_attr, test_edge_y_mask = mask_edge(edge_index, edge_attr,
                                                                  double_test_edge_mask, True, ncol)

    test_labels = torch.tensor(Test_y, dtype=torch.float) 
    train_y_mask, test_y_mask, valid_y_mask = get_y_mask(test_num, df_y, val_num)    

    return Data( x=x,
                 edge_index=edge_index, 
                 edge_attr=edge_attr,
                 edge_cMsk=edge_cMsk,
                 cMsk = cMsk,                 

                 train_y_mask = train_y_mask, 
                 test_y_mask  = test_y_mask,
                 valid_y_mask = valid_y_mask,

                 train_edge_y_mask = train_edge_y_mask,
                 valid_edge_y_mask = valid_edge_y_mask,
                 test_edge_y_mask  = test_edge_y_mask,

                 train_edge_y_index = train_edge_y_index,
                 valid_edge_y_index = valid_edge_y_index,
                 test_edge_y_index = test_edge_y_index,

                 train_edge_y_attr = train_edge_y_attr,
                 valid_edge_y_attr = valid_edge_y_attr,
                 test_edge_y_attr = test_edge_y_attr,

                 train_edge_index = train_edge_index,
                 valid_edge_index = valid_edge_index,
                 test_edge_index  = test_edge_index,

                 train_edge_attr = train_edge_attr,
                 valid_edge_attr = valid_edge_attr,
                 test_edge_attr  = test_edge_attr,

                 train_edge_mask = train_edge_mask,
                 valid_edge_mask = valid_edge_mask,
                 test_edge_mask  = test_edge_mask,

                 train_labels = train_labels,
                 valid_labels = valid_labels,                 
                 test_labels  = test_labels,
                 
                 train_num = train_num,   
                 valid_num = val_num,
                 test_num  = test_num,

                 df_X=df_X,                 
                 edge_attr_dim=train_edge_attr.shape[-1],
                 user_num=df_X.shape[0]                 
                )

