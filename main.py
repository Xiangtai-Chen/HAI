import torch
from sklearn.metrics import f1_score
import torch.nn as nn
import numpy as np
import json
import csv
import pickle
from utils import load_data, EarlyStopping
from valid import valid
import sys
import nni

'''@nni.get_next_parameter()'''

def main(args):
    # If args['hetero'] is True, g would be a heterogeneous graph.
    # Otherwise, it will be a list of homogeneous graphs.
    print(args['dataset'])
    g, features, num_classes = load_data(args['dataset'])
    features = features.to(args['device'])
    from model import HAN
    # A_ap,A_pa,A_ao,A_oa
    model = HAN(meta_paths=[['ap', 'pa'], ['ao', 'oa']],
                in_size=features.shape[1],
                hidden_size=args['hidden_units'],
                out_size=num_classes,
                num_heads=args['num_heads'],
                dropout=args['dropout']).to(args['device'])
    g = g.to(args['device'])


    b_xent = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    best_AUC = 0
    for epoch in range(args['num_epochs']):
        idx = np.random.permutation(features.shape[0])
        shuf_fts = features[idx, :]
        lbl_1 = torch.ones( features.shape[0])
        lbl_2 = torch.zeros( features.shape[0])
        lbl = torch.cat((lbl_1, lbl_2), 0)
        model.train()
        logits = model(g, features, shuf_fts)
        loss = b_xent(logits, lbl)
        # print('Epoch:', epoch, 'Loss:', loss)
        embedings = model.embed(g, features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.detach().cpu().numpy()
        loss = float(loss)
        nni.report_intermediate_result(loss)
        """@nni.report_intermediate_result(loss)"""
        embedings = embedings.detach().numpy()
        # valid
        validation = valid(embedings)
        AUC = validation.forward()
        print(f'AUC:{AUC}')
        if AUC[0] > best_AUC:
            best_AUC = AUC[0]
            np.savetxt('embedings.csv', embedings, delimiter=',')
    nni.report_final_result(best_AUC)
    """@nni.report_final_result(best_AUC)"""

if __name__ == '__main__':
    import argparse
    from utils import setup
    parser = argparse.ArgumentParser('HAN')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    parser.add_argument('--hetero', action='store_true',
                        help='Use metapath coalescing with DGL\'s own dataset'),
    parser.add_argument('--dataset', type=str, default='dblp',
                        help='Dataset')
    args = parser.parse_args().__dict__
    args = setup(args)
    main(args)