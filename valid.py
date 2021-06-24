import json
from sklearn import metrics
import numpy as np
import sys
class valid():
    def __init__(self, embeds):
        self.embedings = embeds
        self.topk = [0.05,0.06,0.07,0.08,0.09,0.1]

    def forward(self):
        with open('Dataprocessing/dblp.v12/train.json','r',encoding='utf8') as f:
            train_author = dict(json.load(f))
        author_train = dict((v[0],k) for k,v in train_author.items())
        with open('Dataprocessing/dblp.v12/valid.json','r',encoding='utf8') as f:
            valid_author = json.load(f)
        with open('Dataprocessing/dblp.v12/abstract/node_list.json','r',encoding='utf8') as f:
            node_list = json.load(f)
        author_list = node_list[0]

        embedings = self.embedings

        matching_score = metrics.pairwise.cosine_similarity(embedings, embedings)

        # AUC
        AUC = list()
        for i in range(matching_score.shape[0]):
            y = np.zeros(matching_score.shape[1])
            try:
                valid_org = valid_author[str(author_list[i])][0]
                author_inorg = int(author_train[valid_org])
                y[author_list.index(author_inorg)] = 1
            except Exception as e:
                continue
            y_score = matching_score[i]
            auc = metrics.roc_auc_score(y, y_score)
            AUC.append(auc)
        AUC = np.array(AUC).mean()
        AUC = float(AUC)
        
        # TOP-k HR
        all_k = list()
        for k in self.topk:
            TOPK = list()
            for i in range(matching_score.shape[0]):
                try:
                    valid_org = valid_author[str(author_list[i])][0]
                    author_inorg = int(author_train[valid_org])
                    simi = matching_score[i][author_list.index(author_inorg)]
                except Exception as e:
                    continue
                ordered_score = sorted(matching_score[i].tolist(),reverse=True)
                if ordered_score.index(simi) > k*matching_score.shape[1]:
                    TOPK.append(0)
                else:
                    TOPK.append(1)
            topk_ = 0
            for i in TOPK:
                topk_ += i/len(TOPK)
            all_k.append(topk_)
        return AUC, all_k