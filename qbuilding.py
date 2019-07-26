import pickle
import numpy as np
from time import time
corpus = {}
id = 0
def build_from_cor(cor_ogm_train_data, ogm_train_data):
    counter = 1
    for index_i, wordbag in enumerate(cor_ogm_train_data):
        if counter % 500 == 0:
            print('processing:',counter,'/',len(cor_ogm_train_data))
        for index_j, words in enumerate(wordbag):
            temp = -1
            temp_value = -1
            flag_t = 0
            if Jaccard(frozenset(ogm_train_data[index_i][index_j]), frozenset(words)) > J_threshold:
                corpus[frozenset(words)] = corpus[frozenset(ogm_train_data[index_i][index_j])]
            else:
                for key, value in corpus.items():
                    J = Jaccard(frozenset(words),key)
                    if temp < J:
                        temp = J
                        temp_value = value
                    if temp > J_threshold:
                        corpus[frozenset(words)] = temp_value
                        flag_t = 1
                        break
                if flag_t == 0:
                    corpus[frozenset(words)] = id
                    id = id + 1
        counter = counter + 1

def Jaccard(A,B):
    jtemp = len(A&B)
    return round(jtemp/(len(A)+len(B)-jtemp),2)

if __name__ == '__main__':
    path1=r'SoccerData\\'
    J_threshold = 0.3
    start = time()
    ogm_train_data = pickle.load(open(path1+'ogm_train_data', 'rb'), encoding='bytes')
    #cor_ogm_train_data = pickle.load(open(path1+'drop_ogm_train_data', 'rb'), encoding='bytes')
    cor_ogm_train_data = pickle.load(open(path1+'noise_ogm_train_data', 'rb'), encoding='bytes')
    senbag = ogm_train_data
    counter = 1
    for wordbag in senbag:
        if counter % 50 == 0:
            print('processing:',counter,'/',len(senbag))
        for words in wordbag:
            temp = -1
            temp_value = -1
            flag_t = 0
            if id == 0:
                corpus[frozenset(words)] = id
                id = id + 1
                continue
            for key, value in corpus.items():
                J = Jaccard(frozenset(words),key)
                if temp < J:
                    temp = J
                    temp_value = value
                if temp > J_threshold:
                    corpus[frozenset(words)] = temp_value
                    flag_t = 1
                    break
            if flag_t == 0:
                corpus[frozenset(words)] = id
                id = id + 1
        counter = counter + 1
    stop = time()
    print('time cost:',str(stop - start)+' seconds')
    build_from_cor(cor_ogm_train_data, ogm_train_data) #if you want to consider the trajectory unicertainty
    pickle.dump(corpus, open(path1+'corpus', 'wb'), protocol=2)
    stop = time()
    print('time cost:',str(stop - start)+' seconds')