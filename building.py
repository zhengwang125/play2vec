import pickle
import numpy as np

def Jaccard(A,B):
    jtemp = len(A&B)
    return round(jtemp/(len(A)+len(B)-jtemp),2)

if __name__ == '__main__':
    path1=r'SoccerData/'
    J_threshold = 0.3
    ogm_train_data = pickle.load(open(path1+'ogm_train_data', 'rb'), encoding='bytes')
    #cor_ogm_train_data = pickle.load(open(path1+'drop_ogm_train_data', 'rb'), encoding='bytes')
    cor_ogm_train_data = pickle.load(open(path1+'noise_ogm_train_data', 'rb'), encoding='bytes')
    senbag = ogm_train_data + cor_ogm_train_data
    corpus = {}
    id = 0
    counter = 1
    for wordbag in senbag:
        if counter % 500 == 0:
            print('processing:',counter,'/',len(senbag))
        for words in wordbag:
            temp = -1
            temp_value = -1
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
            else:
                corpus[frozenset(words)] = id
                id = id + 1
        counter = counter + 1
    pickle.dump(corpus, open(path1+'corpus', 'wb'), protocol=2)
    