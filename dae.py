import numpy as np
import pickle
import matplotlib.pyplot as plt
import random
from encoder import get_encoder_layer
from decoder import decoding_layer,process_decoder_input
import tensorflow as tf
import os
device_name = "/gpu:0"
#def corrupt_noise(traj, rate_noise, factor):
#    new_traj={}
#    for count, key in enumerate(traj):
#        if count%500==0:
#            print('count:',count)
#        new_traj[key] = traj[key]
#        for i in range(len(traj[key])):
#            seed = random.random()
#            if seed < rate_noise:
#                #adding gauss noise
#                for col in range(46):
#                    new_traj[key][i][col] = traj[key][i][col] + factor * random.gauss(0,1)
#    return new_traj
#
#def corrupt_drop(traj, rate_drop):
#    new_traj={}
#    for count, key in enumerate(traj):
#        if count%500==0:
#            print('count:',count)
#        new_traj[key] = traj[key]
#        droprow = []
#        for i in range(len(traj[key])):
#            seed = random.random()
#            if seed < rate_drop:
#                #dropping
#                droprow.append(i)
#        new_traj[key] = np.delete(new_traj[key], droprow, axis = 0)
#                
#    return new_traj

def get_inputs():
    '''
    model inputs tensor
    '''
    embed_seq = tf.placeholder(tf.float32, [None, None, 20], name='embed_seq')
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    # define target sequence length （target_sequence_length and source_sequence_length are used to paprameters of feed_dict）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')
    
    return embed_seq, inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length

def seq2seq_model(embed_seq,
                  input_data, 
                  targets, 
                  lr, 
                  target_sequence_length, 
                  max_target_sequence_length, 
                  source_sequence_length, 
                  source_vocab_size, 
                  target_vocab_size, 
                  encoder_embedding_size, 
                  decoder_embedding_size, 
                  rnn_size, 
                  num_layers):
    # get encoder output
    _, encoder_state = get_encoder_layer(embed_seq,
                                         input_data, 
                                         rnn_size, 
                                         num_layers, 
                                         source_sequence_length, 
                                         source_vocab_size, 
                                         encoding_embedding_size)
    
    
    #tf.add_to_collection("encoder_state",encoder_state)
    print(encoder_state)
    print('Done encoder state')
    # decoder input after preprocess
    decoder_input = process_decoder_input(targets, target_letter_to_int, batch_size)
    
    # state vector and input to decoder
    training_decoder_output, predicting_decoder_output = decoding_layer(target_letter_to_int, 
                                                                       decoding_embedding_size, 
                                                                       num_layers, 
                                                                       rnn_size,
                                                                       target_sequence_length,
                                                                       max_target_sequence_length,
                                                                       encoder_state, 
                                                                       decoder_input,
                                                                       batch_size) 
    
    return training_decoder_output, predicting_decoder_output

def pad_sentence_batch(sentence_batch, pad_int):
    '''
    batch completion，guarantee the same sequence_length
    
    parameters：
    - sentence batch
    - pad_int: <PAD> respond to index
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int, embed_mat):
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # complete sequence
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))
        
        # record each length
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))
        
        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))
        
        embed_batch = np.array([[embed_mat[i] for i in psb] for psb in pad_sources_batch])
        
        
        yield embed_batch, pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths

def extract_character_vocab(data, UNIQUE_WORDS):
    '''
    build data mapping
    '''
    vocab_to_int = {}
    int_to_vocab = {}
    special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']
    for plays in data:
        for segs in plays:
            if frozenset(segs) in UNIQUE_WORDS == False:
                print('No this segment! please build it.')
            else:
                vocab_to_int[frozenset(segs)] = UNIQUE_WORDS[frozenset(segs)]
                int_to_vocab[UNIQUE_WORDS[frozenset(segs)]] = frozenset(segs)
    
    vocab_to_int['<PAD>'] = max(UNIQUE_WORDS.values()) + 1
    vocab_to_int['<UNK>'] = max(UNIQUE_WORDS.values()) + 2
    vocab_to_int['<GO>']  = max(UNIQUE_WORDS.values()) + 3
    vocab_to_int['<EOS>'] = max(UNIQUE_WORDS.values()) + 4
    
    int_to_vocab[max(UNIQUE_WORDS.values()) + 1] = '<PAD>'
    int_to_vocab[max(UNIQUE_WORDS.values()) + 2] = '<UNK>'
    int_to_vocab[max(UNIQUE_WORDS.values()) + 3] = '<GO>'
    int_to_vocab[max(UNIQUE_WORDS.values()) + 4] = '<EOS>'

    return int_to_vocab, vocab_to_int

def mapping_source_int(cor_ogm_train_data, UNIQUE_WORDS):
    source_int = []
    for plays in cor_ogm_train_data:
        temp = []
        for word in plays:
            temp.append(UNIQUE_WORDS[frozenset(word)])
        source_int.append(temp)
    return source_int

def mapping_target_int(ogm_train_data, UNIQUE_WORDS):
    target_int = []
    for plays in ogm_train_data:
        temp = []
        for word in plays:
            temp.append(UNIQUE_WORDS[frozenset(word)])
        temp.append(target_letter_to_int['<EOS>'])
        target_int.append(temp)
    return target_int
        

if __name__ == '__main__':
    print('autoencoder')
    path1=r'SoccerData/'
    #cor_ogm_train_data = ogm_train_data=pickle.load(open(path1+'drop_ogm_train_data', 'rb'), encoding='bytes') #for drop version
    cor_ogm_train_data = ogm_train_data=pickle.load(open(path1+'noise_ogm_train_data', 'rb'), encoding='bytes') #for noise version
    ogm_train_data=pickle.load(open(path1+'ogm_train_data', 'rb'), encoding='bytes')
    embed_mat=pickle.load(open(path1+'embed_mat', 'rb'), encoding='bytes')
    embed_mat = np.r_[embed_mat,np.random.rand(len(embed_mat[0])).reshape(1,-1)]
    embed_mat = np.r_[embed_mat,np.random.rand(len(embed_mat[0])).reshape(1,-1)]
    embed_mat = np.r_[embed_mat,np.random.rand(len(embed_mat[0])).reshape(1,-1)]
    embed_mat = np.r_[embed_mat,np.random.rand(len(embed_mat[0])).reshape(1,-1)]
    
    UNIQUE_WORDS=pickle.load(open(path1+'corpus', 'rb'), encoding='bytes')
    source_int_to_letter, source_letter_to_int = extract_character_vocab(cor_ogm_train_data, UNIQUE_WORDS)
    target_int_to_letter, target_letter_to_int = extract_character_vocab(ogm_train_data, UNIQUE_WORDS)   
    source_int = mapping_source_int(cor_ogm_train_data, UNIQUE_WORDS)
    target_int = mapping_target_int(ogm_train_data, UNIQUE_WORDS)
    #look transform
    print('source', source_int[:5])
    print('target', target_int[:5])
    
    # Number of Epochs
    epochs = 10
    # Batch Size
    batch_size = 10
    # RNN Size
    rnn_size = 50
    # Number of Layers
    num_layers = 2
    # Embedding Size
    encoding_embedding_size = 50
    decoding_embedding_size = 50
    # Learning Rate
    learning_rate = 0.01
    
    with tf.device(device_name):
        #building graph
        train_graph = tf.Graph()
        with train_graph.as_default():
            embed_seq, input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()
            training_decoder_output, predicting_decoder_output = seq2seq_model(embed_seq,
                                                                              input_data, 
                                                                              targets, 
                                                                              lr, 
                                                                              target_sequence_length, 
                                                                              max_target_sequence_length, 
                                                                              source_sequence_length,
                                                                              len(source_letter_to_int),
                                                                              len(target_letter_to_int),
                                                                              encoding_embedding_size, 
                                                                              decoding_embedding_size, 
                                                                              rnn_size, 
                                                                              num_layers)    
            training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
            predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')
            
            masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
        
            with tf.name_scope("optimization"):
                # Loss function
                cost = tf.contrib.seq2seq.sequence_loss(
                    training_logits,
                    targets,
                    masks)
                # Optimizer
                optimizer = tf.train.AdamOptimizer(lr)
        
                # Gradient Clipping
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)
        
        print('done building graph')
    
    # train and validation
    train_source = source_int[batch_size:]
    train_target = target_int[batch_size:]
    
    # leave one batch for validation
    valid_source = source_int[:batch_size]
    valid_target = target_int[:batch_size]
    
    (valid_embed_batch, valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(get_batches(valid_target, valid_source, batch_size,
                               source_letter_to_int['<PAD>'],
                               target_letter_to_int['<PAD>'],
                               embed_mat))
    
    display_step = 5
    
    checkpoint = path1 + "model_1/trained_model.ckpt"
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7) 
    with tf.Session(graph=train_graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
            
        for epoch_i in range(1, epochs+1):
            for batch_i, (embed_batch, targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    get_batches(train_target, train_source, batch_size,
                               source_letter_to_int['<PAD>'],
                               target_letter_to_int['<PAD>'],
                               embed_mat)):
                _ , loss = sess.run(
                       [train_op, cost],
                       {embed_seq: embed_batch,
                        input_data: sources_batch,
                        targets: targets_batch,
                        lr: learning_rate,
                        target_sequence_length: targets_lengths,
                        source_sequence_length: sources_lengths})
                if batch_i % display_step == 0:
                    # validation loss
                    validation_loss = sess.run(
                    [cost],
                    {embed_seq:valid_embed_batch,
                     input_data: valid_sources_batch,
                     targets: valid_targets_batch,
                     lr: learning_rate,
                     target_sequence_length: valid_targets_lengths,
                     source_sequence_length: valid_sources_lengths})
                    
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                          .format(epoch_i,
                                  epochs, 
                                  batch_i, 
                                  len(train_source) // batch_size, 
                                  loss, 
                                  validation_loss[0]))                        
        # save model
        saver = tf.train.Saver()
        saver.save(sess, checkpoint)
        print('Model Trained and Saved')