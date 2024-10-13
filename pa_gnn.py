import numpy as np
import torch
import torch.nn.functional as F

from gnn_framework.gnn_model import get_gnn
from gnn_framework.prediction_model import MLPNet
from utils.utils import build_optimizer, get_known_mask, mask_edge

from torchmetrics.classification import BinaryF1Score, BinaryAccuracy, BinaryROC
from torchmetrics.classification import BinarySpecificity, BinaryAUROC, BinaryRecall

def infer_gnn(data, args, source, cv_iter, device=torch.device('cpu')):   
    
    x = data.x.clone().detach().to(device)
        
    test_edge_y_index  = data.test_edge_y_index.clone().detach().to(device)   
    test_edge_y_attr  = data.test_edge_y_attr.clone().detach().to(device)
    
    test_edge_index  = data.test_edge_index.clone().detach().to(device)    
    test_edge_attr  = data.test_edge_attr.clone().detach().to(device)

    test_labels  = data.test_labels.clone().detach().to(device)    
    
    load_path = f'../gnn_{source}/CV_{cv_iter}/log/'    
    
    model = torch.load(load_path+"model_best_valid.pt", map_location=torch.device("cpu"))
    pred_prob_model = torch.load(load_path+'pred_prob_model_best_valid.pt',map_location=torch.device('cpu'))
    
    model.eval()
    pred_prob_model.eval()

    classifier = torch.nn.Sigmoid()
    ROCmetric = BinaryROC(thresholds=None)
    AUCmetric = BinaryAUROC(thresholds=None)
    Specmetric = BinarySpecificity(threshold=0.5)
    Sensmetric = BinaryRecall(threshold=0.5)
    f1_metric = BinaryF1Score()
    acc_metric = BinaryAccuracy()

    with torch.no_grad():
        x_embd = model(x, test_edge_attr, test_edge_index)        
        x_pred = pred_prob_model([x_embd[test_edge_y_index[0], :], x_embd[test_edge_y_index[1], :]])        
        pred = classifier( x_pred[:int(test_edge_y_attr.shape[0] / 2)] )        
        label_test = test_labels.reshape((-1, 1))    
    return f1_metric(pred, label_test), acc_metric(pred, label_test), ROCmetric(pred, label_test.long()), AUCmetric(pred, label_test), Specmetric(pred, label_test), Sensmetric(pred, label_test)

def train_gnn(data, args, source, cv_iter, log_path, device=torch.device('cpu')):
    
    model = get_gnn(data, args).to(device)
    
    if args.pred_prob_hiddens == '':
        pred_prob_hiddens = []
    else:
        pred_prob_hiddens = list(map(int,args.pred_prob_hiddens.split('_')))
        
    input_dim = args.node_dim * 2    
    output_dim = 1

    pred_prob_model = MLPNet(  input_dim, output_dim,
                            hidden_layer_sizes=pred_prob_hiddens,
                            hidden_activation=args.pred_prob_activation,
                            dropout=args.dropout).to(device)        
    
    if args.transfer_dir: # this ensures the valid mask is consistant
        load_path = f'./gnn_{source}/CV_{cv_iter}/{args.transfer_dir}/'
        print("loading fron {} with {}".format(load_path,args.transfer_extra))
        model = torch.load(load_path+'model'+args.transfer_extra+'.pt',map_location=device)
        pred_prob_model = torch.load(load_path+'pred_prob_model'+args.transfer_extra+'.pt',map_location=device)

    trainable_parameters = list(model.parameters()) \
                           + list(pred_prob_model.parameters())
    print("total trainable_parameters: ",len(trainable_parameters))
    # build optimizer
    
    scheduler, opt = build_optimizer(args, trainable_parameters)
    
    x = data.x.clone().detach().to(device)
    
    _, n_col = data.df_X.shape

    edge_cMsk = data.edge_cMsk.clone().detach().to(device)

    edge_index = data.edge_index.clone().detach().to(device)
    edge_attr = data.edge_attr.clone().detach().to(device)
    
    train_edge_mask = data.train_edge_mask.clone().detach().to(device)
    double_train_edge_mask = torch.cat((train_edge_mask, train_edge_mask), dim=0)
    all_train_edge_index, all_train_edge_attr = mask_edge(edge_index, edge_attr, double_train_edge_mask, True, n_col, noSplitY=True)
    
    valid_edge_mask = data.valid_edge_mask.clone().detach().to(device)
    double_valid_edge_mask = torch.cat((valid_edge_mask, valid_edge_mask), dim=0)  

    train_edge_y_index = data.train_edge_y_index.clone().detach().to(device)
    valid_edge_y_index = data.valid_edge_y_index.clone().detach().to(device)

    train_edge_y_attr = data.train_edge_y_attr.clone().detach().to(device)
    valid_edge_y_attr = data.valid_edge_y_attr.clone().detach().to(device)    

    train_edge_index = data.train_edge_index.clone().detach().to(device)
    valid_edge_index = data.valid_edge_index.clone().detach().to(device)    
    
    train_edge_attr = data.train_edge_attr.clone().detach().to(device)
    valid_edge_attr = data.valid_edge_attr.clone().detach().to(device)    

    train_labels = data.train_labels.clone().detach().to(device)
    valid_labels = data.valid_labels.clone().detach().to(device)    

    best_valid_L = np.inf
    best_valid_L_epoch = 0
            
    Lr = []

    # train_ceLoss = torch.nn.BCEWithLogitsLoss()
    # valid_ceLoss = torch.nn.BCEWithLogitsLoss()

    classifier = torch.nn.Sigmoid()
    
    _lambda_ = args.Lagrange_multiplier     
    for epoch in range(args.epochs):
        model.train()
        pred_prob_model.train()

        known_mask = get_known_mask(args.known, int(all_train_edge_attr.shape[0] / 2)).to(device)
        double_known_mask = torch.cat((known_mask, known_mask), dim=0)
        known_edge_index, known_edge_attr = mask_edge(all_train_edge_index, all_train_edge_attr, double_known_mask, True, n_col, noSplitY=True)

        edge_cMsk_known = edge_cMsk[double_train_edge_mask]       
        edge_cMsk_known = edge_cMsk_known[:int(train_edge_attr.shape[0] / 2)]

        opt.zero_grad()
        x_embd = model(x, known_edge_attr, known_edge_index)
        
        pred_noY = pred_prob_model([x_embd[train_edge_index[0]], x_embd[train_edge_index[1]]])     
        pred_Y   = pred_prob_model([x_embd[train_edge_y_index[0]], x_embd[train_edge_y_index[1]]])       
        
        pred_train_noY = pred_noY[:int(train_edge_attr.shape[0] / 2),0]
        pred_train_Y   = classifier(pred_Y[:int(train_edge_y_attr.shape[0] / 2),0])

        label_train_noY = train_edge_attr[:int(train_edge_attr.shape[0] / 2), 0]
        label_train_Y   = train_labels[:, 0]  

        pred_train_noY_class = classifier(pred_train_noY[edge_cMsk_known])
        pred_train_noY_conti = pred_train_noY[~edge_cMsk_known]  
        
        label_train_noY_class = label_train_noY[edge_cMsk_known]  
        label_train_noY_conti = label_train_noY[~edge_cMsk_known]  

        loss = ( _lambda_ * ( F.mse_loss( pred_train_noY_conti, label_train_noY_conti ) + 
                              F.mse_loss( pred_train_noY_class, label_train_noY_class ) )
                 + F.mse_loss( pred_train_Y, label_train_Y ) )

        loss.backward()
        opt.step()

        train_loss = loss.item()

        if scheduler is not None:
            scheduler.step(epoch)
        
        for param_group in opt.param_groups:
            Lr.append(param_group['lr'])

        model.eval()
        pred_prob_model.eval()

        with torch.no_grad():
            x_embd = model(x, valid_edge_attr, valid_edge_index)

            pred_noY = pred_prob_model([x_embd[valid_edge_index[0]], x_embd[valid_edge_index[1]]])     
            pred_Y   = pred_prob_model([x_embd[valid_edge_y_index[0]], x_embd[valid_edge_y_index[1]]])   
            
            pred_valid_noY = pred_noY[:int(valid_edge_attr.shape[0] / 2),0]
            pred_valid_Y   = classifier(pred_Y[:int(valid_edge_y_attr.shape[0] / 2),0])
            
            label_valid_noY = valid_edge_attr[:int(valid_edge_attr.shape[0] / 2), 0]
            label_valid_Y = valid_labels[:, 0]

            edge_cMsk_known = edge_cMsk[double_valid_edge_mask]       
            edge_cMsk_known = edge_cMsk_known[:int(valid_edge_attr.shape[0] / 2)]

            pred_valid_noY_class = classifier(pred_valid_noY[edge_cMsk_known])
            pred_valid_noY_conti = pred_valid_noY[~edge_cMsk_known]  
            
            label_valid_noY_class = label_valid_noY[edge_cMsk_known]  
            label_valid_noY_conti = label_valid_noY[~edge_cMsk_known]  

            vloss = ( _lambda_ * ( F.mse_loss( pred_valid_noY_conti, label_valid_noY_conti ) + 
                                   F.mse_loss( pred_valid_noY_class, label_valid_noY_class ) )
                        + F.mse_loss( pred_valid_Y, label_valid_Y ) )
            
            vLoss = vloss.item()

            if vLoss < best_valid_L:
                best_valid_L = vLoss
                best_valid_L_epoch = epoch
                
                if args.save_model:
                    torch.save(model, log_path + 'model_best_valid.pt')
                    torch.save(pred_prob_model, log_path + 'pred_prob_model_best_valid.pt')                              
            
            if args.mode == 'debug':
                torch.save(model, log_path + 'model_{}.pt'.format(epoch))
                torch.save(pred_prob_model, log_path + 'pred_prob_model_{}.pt'.format(epoch))                        
            
            print('CV-iter: ', cv_iter)
            print(' epoch: ', epoch)
            print(' train loss: ', train_loss)
            print(' valid loss', vLoss)            

    if args.save_model:
        torch.save(model, log_path + 'model.pt')
        torch.save(pred_prob_model, log_path + 'pred_prob_model.pt')
        
    print("best valid loss is {:.3g} at epoch {}".format(best_valid_L, best_valid_L_epoch))
