import time
import sys
import torch
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
from tqdm import tqdm
from evaluator import evaluator as ev
from util import  bipartite_dataset, deg_dist,gen_top_k
from data_loader import Data_loader
from siren import SiReN
from sbgnn import load_edgelists, MeanAggregator, AttentionAggregator, SBGNN
from sklearn.metrics import roc_auc_score
import argparse


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_class=Data_loader(args.dataset,args.version)
    print('data loading...');st=time.time()
    train,test = data_class.data_load();
    train = train.astype({'userId':'int64', 'movieId':'int64'})
    # see 22-GraphFM: Graph Factorization Machines for Feature Interaction Modeling
    print('before remove rating 3, train: {}, test: {}'.format(len(train['rating']), len(test['rating'])))
    train.drop(train.loc[train['rating'] == 3].index, inplace=True)
    test.drop(test.loc[test['rating'] == 3].index, inplace=True)
    print('after remove rating 3, train: {}, test: {}'.format(len(train['rating']), len(test['rating'])))
    test_user = torch.tensor(test['userId'].to_numpy() - 1).to(device)
    test_item = torch.tensor(test['movieId'].to_numpy() - 1).to(device)
    test_rating = np.maximum(0, np.sign(test['rating'].to_numpy() - 3.5))
    data_class.train = train; data_class.test = test
    print('loading complete! time :: %s'%(time.time()-st))
    print('generate negative candidates...'); st=time.time()
    neg_dist = deg_dist(train,data_class.num_v)
    print('complete ! time : %s'%(time.time()-st))

    train_edgelist = zip(train['userId'] - 1, train['movieId'] - 1, np.sign(train['rating'] - 3.5))
    edgelists = load_edgelists(train_edgelist)
    if args.agg == 'MeanAggregator':
        agg = MeanAggregator
    else:
        agg = AttentionAggregator
    model = SBGNN(edgelists, data_class.num_u,data_class.num_v, args.reg, dropout=args.dropout,
                  layer_num=args.num_layers, aggregator=agg)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = args.lr)
    
    print("\nTraining on {}...\n".format(device))
    model.train()
    training_dataset=bipartite_dataset(train,neg_dist,args.offset,data_class.num_u,data_class.num_v,args.K);
    batch_size = len(train['rating']) // 80
    print('batch_size:', batch_size)
    best_auc = 0.
    
    for EPOCH in range(1,args.epoch+1):
        if EPOCH % 20 - 1 == 0:
            training_dataset.negs_gen_EP(20)
        LOSS=0
        training_dataset.edge_4 = training_dataset.edge_4_tot[:,:,EPOCH%20-1]
        ds = DataLoader(training_dataset,batch_size=batch_size,shuffle=True)
        q=0
        pbar = tqdm(desc = 'Version : {} Epoch {}/{}'.format(args.version,EPOCH,args.epoch),total=len(ds),position=0)
        
        for u,v,w,negs in ds:   
            u, v, w, negs = map(lambda x: x.to(device), [u, v, w, negs])
            q+=len(u)
            st=time.time()
            optimizer.zero_grad()
            loss = model(u,v,w,negs) # original
            loss.backward()                
            optimizer.step()
            LOSS+=loss.item()
            pbar.update(1);
            pbar.set_postfix({'loss': LOSS / q})

        pbar.close()

        if EPOCH%2 ==0:
            model.eval()
            emb = model.aggregate();
            emb_u, emb_v = torch.split(emb,[data_class.num_u,data_class.num_v])
            pred = torch.sigmoid(torch.einsum(
                'ij,ij->i', (emb_u[test_user], emb_v[test_item]))).cpu().detach()
            auc = roc_auc_score(test_rating, pred)
            best_auc = max(best_auc, auc)
            print('Epoch ', EPOCH, 'test auc:', auc, ', best_auc:', best_auc, file=sys.stderr)
            model.train()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        type = str,
                        default = 'ML-1M',
                        help = "Dataset"
                        )
    parser.add_argument('--version',
                        type = int,
                        default =1,
                        help = "Dataset version"
                        )
    parser.add_argument('--batch_size',
                        type = int,
                        default = 1024,
                        help = "Batch size"
                        )

    parser.add_argument('--dim',
                        type = int,
                        default = 64,
                        help = "Dimension"
                        )
    parser.add_argument('--lr',
                        type = float,
                        default = 1e-3,
                        help = "Learning rate"
                        )
    parser.add_argument('--offset',
                        type = float,
                        default = 3.5,
                        help = "Criterion of likes/dislikes"
                        )
    parser.add_argument('--K',
                        type = int,
                        default = 50,
                        help = "The number of negative samples"
                        )
    parser.add_argument('--num_layers',
                        type = int,
                        default = 3,
                        help = "The number of layers of a GNN model for the graph with positive edges"
                        )
    parser.add_argument('--MLP_layers',
                        type = int,
                        default = 3,
                        help = "The number of layers of MLP for the graph with negative edges"
                        )
    parser.add_argument('--epoch',
                        type = int,
                        default = 100,
                        help = "The number of epochs"
                        )
    parser.add_argument('--reg',
                        type = float,
                        default = 0.05,
                        help = "Regularization coefficient"
                        )
    parser.add_argument('--agg', type=str, default='AttentionAggregator',
                        choices=['AttentionAggregator', 'MeanAggregator'], help='Aggregator')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout')
    args = parser.parse_args()
    main(args)
