# -*- coding: utf-8 -*-
import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
from sklearn.manifold import TSNE
import random
import time
from ogm import viz_ogm, grid_index_to_array
device_name = "/gpu:0"
#def cmp(a,b):
#    '''
#    a: listA
#    b: listB
#    '''
#    return list(set(a).difference(set(b))) == []
#
#def in_unique_words(item, UNIQUE_WORDS):
#    for uw in UNIQUE_WORDS:
#        if set(uw) == set(item):
#            return True
#    return False


#def filter_sample(int_words):
#    t = 1e-5
#    threshold = 0.8
#    int_word_counts = Counter(int_words)
#    total_count = len(int_words)
#    word_freqs = {w: c/total_count for w, c in int_word_counts.items()}
#    prob_drop = {w: 1 - np.sqrt(t / word_freqs[w]) for w in int_word_counts}
#    train_words = [w for w in int_words if prob_drop[w] < threshold]
#    print(len(train_words))
#    return train_words

def get_targets(words, idx, window_size=5):
    '''
    get input word context
    
    para
    ---
    words: word list
    idx: input word index
    window_size: the size of window
    '''
    target_window = np.random.randint(1, window_size+1)
    start_point = idx - target_window if (idx - target_window) > 0 else 0
    end_point = idx + target_window
    targets = set(words[start_point: idx] + words[idx+1: end_point+1])
    return list(targets)

def get_batches(words, batch_size, window_size=5):
    '''
    batch generator
    '''
    n_batches = len(words) // batch_size
    words = words[:n_batches*batch_size]
    
    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx: idx+batch_size]
        for i in range(len(batch)):
            batch_x = batch[i]
            batch_y = get_targets(batch, i, window_size)
            x.extend([batch_x]*len(batch_y))
            y.extend(batch_y)
        yield x, y
    
    
if __name__ == '__main__':
    print('ogm embedding')
    path1=r'SoccerData/'
    UNIQUE_WORDS = pickle.load(open(path1+'corpus', 'rb'), encoding='bytes')
    vocab_to_int = {k: v for k, v in UNIQUE_WORDS.items()}
    int_to_vocab = {v: k for k, v in UNIQUE_WORDS.items()}
    int_words = list(UNIQUE_WORDS.values())
    #pickle.dump(int_words, open(path1+'int_words', 'wb'), protocol=2)
    #train_words = filter_sample(int_words) #option (if Subsampling)
    train_words = int_words

    with tf.device(device_name):
        train_graph = tf.Graph()
        
        with train_graph.as_default():
            inputs = tf.placeholder(tf.int32, shape=[None], name='inputs')
            labels = tf.placeholder(tf.int32, shape=[None, None], name='labels')
            
        vocab_size = len(int_to_vocab)
        embedding_size = 20
        with train_graph.as_default():
            embedding = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
            embed = tf.nn.embedding_lookup(embedding, inputs)
        
        n_sampled = 100
        
        with train_graph.as_default():
            softmax_w = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))
            softmax_b = tf.Variable(tf.zeros(vocab_size))
            
            #negative sampling loss
            loss = tf.nn.sampled_softmax_loss(softmax_w, softmax_b, labels, embed, n_sampled, vocab_size)
            
            cost = tf.reduce_mean(loss)
            optimizer = tf.train.AdamOptimizer().minimize(cost)
            
        with train_graph.as_default():
            valid_size = 16 
            valid_window = 50
            valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
            valid_examples = np.append(valid_examples, 
                                       random.sample(range(50,50+valid_window), valid_size//2))
            
            valid_size = len(valid_examples)
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
            norm = tf.sqrt(tf.reduce_sum(tf.square(embedding), 1, keep_dims=True))
            normalized_embedding = embedding / norm
            valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
            similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))
    
    
    epochs = 20
    batch_size = 10
    window_size = 5

    with train_graph.as_default():
        saver = tf.train.Saver()
    
    xlim = [-52.5,52.5]
    ylim = [-34,34]
    delta = 3.0
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    with tf.Session(graph=train_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        iteration = 1
        loss = 0
        sess.run(tf.global_variables_initializer())
    
        for e in range(1, epochs+1):
            batches = get_batches(train_words, batch_size, window_size)
            start = time.time()
            for x, y in batches:
                feed = {inputs: x,
                        labels: np.array(y)[:, None]}
                train_loss, _ = sess.run([cost, optimizer], feed_dict=feed)
                
                loss += train_loss
                
                if iteration % 100 == 0: 
                    end = time.time()
                    print("Epoch {}/{}".format(e, epochs),
                          "Iteration: {}".format(iteration),
                          "Avg. Training loss: {:.4f}".format(loss/100),
                          "{:.4f} sec/batch".format((end-start)/100))
                    loss = 0
                    start = time.time()
                if iteration % 1000 == 0:
                    sim = similarity.eval()
                    for i in range(valid_size):
                        valid_word = int_to_vocab[valid_examples[i]]
                        top_k = 1
                        nearest = (-sim[i, :]).argsort()[1:top_k+1]
                        log = 'Nearest to [%s]:' % valid_word
                        print('query')
                        viz_ogm(grid_index_to_array(set(valid_word), xlim, ylim, delta))
                        print('topk:',top_k)
                        for k in range(top_k):
                            close_word = int_to_vocab[nearest[k]]
                            log = '%s %s,' % (log, close_word)
                            viz_ogm(grid_index_to_array(set(close_word), xlim, ylim, delta))
                        #print(log)
                iteration += 1
        save_path = saver.save(sess, path1+"checkpoints/ogm.ckpt")
        embed_mat = sess.run(normalized_embedding)
    
    pickle.dump(embed_mat, open(path1+'embed_mat', 'wb'), protocol=2)
    viz_words = 100
    tsne = TSNE()
    embed_tsne = tsne.fit_transform(embed_mat[:viz_words, :])
    fig, ax = plt.subplots(figsize=(14, 14))
    for idx in range(viz_words):
        plt.scatter(*embed_tsne[idx, :], color='steelblue')
        plt.annotate(idx, (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)