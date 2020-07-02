##############################################################################
####     Sentences Simplification with Deep Reinforcement Learning        ####
##############################################################################
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import re
import time

total_time_start = time.time()
##############################################################################
####                        Data Preprocessing                            ####
##############################################################################
complexes = open("/home/saman/Desktop/thesis_/ori.test.small/PWKP_108016.tag.80.aner.ori.test.src", encoding = 'utf-8', errors = 'ignore').read().split('\n')
simples = open("/home/saman/Desktop/thesis_/ori.test.small/PWKP_108016.tag.80.aner.ori.test.dst", encoding= 'utf-8', errors= 'ignore').read().split('\n')

# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    return text
# aval hazf bad word2cunt
clean_complexes = []
for _line in complexes:
    clean_complexes .append(clean_text(_line))

clean_simples = []
for _line in simples:
    clean_simples .append(clean_text(_line))

# Create word count :
word2count = {}
for _line in clean_complexes:
    for _i in _line.split():
        if _i not in word2count:
            word2count[_i] = 1
        else:
            word2count[_i] += 1

for _line in clean_simples:
    for _i in _line.split():
        if _i not in word2count:
            word2count[_i] = 1 
        else:
            word2count[_i] += 1

# Creating two dictionaries that map the complexes words and the simples words to a unique integer
threshold_complexes = 3
complexeswords2int = {}
word_number = 0
for _word, _count in word2count.items():
    if _count >= threshold_complexes :
        complexeswords2int[_word] = word_number
        word_number += 1

threshold_simples = 3
simpleswords2int = {}
word_number = 0
for _word, _count in word2count.items():
    if _count >= threshold_simples :
        simpleswords2int[_word] = word_number
        word_number += 1
        
# Adding the last tokens to these two dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    complexeswords2int[token] = len(complexeswords2int) + 1
for token in tokens:
    simpleswords2int[token] = len(simpleswords2int) + 1

# Creating the inverse dictionary of the simpleswords2int dictionary
### Chun dar Akhar mikhad output Reverse kone kari be input nadare
simplesint2word = {key: value for value, key in simpleswords2int.items()}

# Adding the End Of String token to the end of every simple
for i in range(len(clean_simples)):
    clean_simples[i] += ' <EOS>'

# Translating all the complexes and the simples into integers
# and Replacing all the words that were filtered out by <OUT>
complexes_into_int = []
for _line in clean_complexes:
    ints = []
    for _i in _line.split():
        if _i not in complexeswords2int:
            ints.append(complexeswords2int['<OUT>'])
        else:
            ints.append(complexeswords2int[_i])
    complexes_into_int.append(ints)

simples_into_int = []
for _line in clean_simples:
    ints = []
    for _i in _line.split():
        if _i not in simpleswords2int:
            ints.append(simpleswords2int['<OUT>'])
        else:
            ints.append(simpleswords2int[_i])
    simples_into_int.append(ints)

# Sorting complexes and simples by the length of complexes
#max_sequence_length = 50
sorted_clean_complexes = []
sorted_clean_simples = []
for length in range(1, 25 + 1):
    for i in enumerate(complexes_into_int):
        if len(i[1]) == length:
            sorted_clean_complexes.append(complexes_into_int[i[0]])
            sorted_clean_simples.append(simples_into_int[i[0]])
 
##############################################################################
####                PART 2 - BUILDING THE SEQ2SEQ MODEL                   ####
############################################################################## 
# Creating placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob
 
# Preprocessing the targets
def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets
 
# Creating the Encoder RNN
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state
 
# Decoding the training set
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)
 
# Decoding the test/validation set
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id,
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    return test_predictions
 
# Creating the Decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
    return training_predictions, test_predictions
 
# Building the seq2seq model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, simples_num_words, complexes_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, complexeswords2int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              simples_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, complexeswords2int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([complexes_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         complexes_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         complexeswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
 
##############################################################################
####                PART 3 - TRAINING THE SEQ2SEQ MODEL                   ####
############################################################################## 
# Setting the Hyperparameters
epochs = 1
batch_size = 32
rnn_size = 1024
num_layers = 3
encoding_embedding_size = 1024
decoding_embedding_size = 1024
learning_rate = 0.001
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.5
 
# Defining a session
tf.reset_default_graph()
session = tf.InteractiveSession()
 
# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()
 
# Setting the sequence length
sequence_length = tf.placeholder_with_default(25, None, name = 'sequence_length')
 
# Getting the shape of the inputs tensor
input_shape = tf.shape(inputs)
 
# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(simpleswords2int),
                                                       len(complexeswords2int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       complexeswords2int)
 
# Setting up the Loss Error, the Optimizer and Gradient Clipping
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0], sequence_length]))
    print("@@@@@@@@@@@@@@@@@")
    print(training_predictions)
    print("****************")
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5., 5.), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)
 
# Padding the sequences with the <PAD> token
def apply_padding(batch_of_sequences, word2int):
    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
 
# Splitting the data into batches of complexes and simples
def split_into_batches(complexes, simples, batch_size):
    for batch_index in range(0, len(complexes) // batch_size):
        start_index = batch_index * batch_size
        complexes_in_batch = complexes[start_index : start_index + batch_size]
        simples_in_batch = simples[start_index : start_index + batch_size]
        padded_complexes_in_batch = np.array(apply_padding(complexes_in_batch, complexeswords2int))
        padded_simples_in_batch = np.array(apply_padding(simples_in_batch, simpleswords2int))
        yield padded_complexes_in_batch, padded_simples_in_batch
 
# Splitting the complexes and simples into training and validation sets
training_validation_split = int(len(sorted_clean_complexes) * 0.15)
training_complexes = sorted_clean_complexes[training_validation_split:]
training_simples = sorted_clean_simples[training_validation_split:]
validation_complexes = sorted_clean_complexes[:training_validation_split]
validation_simples = sorted_clean_simples[:training_validation_split]

# Training
batch_index_check_training_loss = 100
batch_index_check_validation_loss = ((len(training_complexes)) // batch_size // 2) - 1
total_training_loss_error = 0
list_validation_loss_error = []
early_stopping_check = 0
early_stopping_stop = 100
checkpoint = "/home/saman/Desktop/thesis_/checkpoint/simplification_weights.ckpt"
session.run(tf.global_variables_initializer())
for epoch in range(1, epochs + 1):
    for batch_index, (padded_complexes_in_batch, padded_simples_in_batch) in enumerate(split_into_batches(training_complexes, training_simples, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_complexes_in_batch,
                                                                                               targets: padded_simples_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_simples_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_complexes) // batch_size,
                                                                                                                                       total_training_loss_error / batch_index_check_training_loss,
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
            total_training_loss_error = 0
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_complexes_in_batch, padded_simples_in_batch) in enumerate(split_into_batches(validation_complexes, validation_simples, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_complexes_in_batch,
                                                                       targets: padded_simples_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_simples_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
                
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_complexes) / batch_size)
            print('Validation Loss Error: {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('Transform better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
#                saver.save(session, checkpoint)
            else:
                print("Unfortunately, Model do not imporve, The Model need to train more.")
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
    if early_stopping_check == early_stopping_stop:
        print("Unfortunately, Model do not imporve, The Model need to train more.")
        break

checkpoint = "/home/saman/Desktop/thesis_/checkpoint/simplification_weights.ckpt"
#saver = tf.train.Saver()
#saver.save(session, checkpoint)
print("Game Over")

##############################################################################
####              PART 4 - My_TESTING THE SEQ2SEQ MODE                    ####
##############################################################################
test_complexes = open("/home/saman/Desktop/thesis_/ori.test.small/PWKP_108016.tag.80.aner.ori.test.src", encoding = 'utf-8', errors = 'ignore').read().split('\n')
test_complexes = test_complexes[:-1]
test_simples = open("/home/saman/Desktop/thesis_/ori.test.small/PWKP_108016.tag.80.aner.ori.test.dst", encoding= 'utf-8', errors= 'ignore').read().split('\n')
test_simples = test_simples[:-1]

checkpoint = "/home/saman/Desktop/thesis_/checkpoint/simplification_weights.ckpt"
session = tf.InteractiveSession()
session.run(tf.global_variables_initializer())
###
#init_op = tf.initialize_all_variables()
#_ = tf.Variable(initial_value='fake_variable')
###
saver = tf.train.Saver()
#saver.restore(session, checkpoint)

def s_convert_string2int(complexs, word2int):
    converted = []
    for line in complexs:
        ints = []
        for word in line.split():
            word = clean_text(word)
            ints.append(word2int.get(word, word2int['<OUT>']))
        converted.append(ints)
    return converted

test_complexes_convert = s_convert_string2int(test_complexes,complexeswords2int)

sorted_clean_test_complexes = []
for length in range(1, 25 + 1):
    for i in enumerate(test_complexes_convert):
        if len(i[1]) == length:
            sorted_clean_test_complexes.append(test_complexes_convert[i[0]])

test_complexes_pad = []
for test in sorted_clean_test_complexes:
        test_complexes_pad.append(test + [complexeswords2int['<PAD>']] * (25 - len(test)))
#while(True):
fake_batch = np.zeros((batch_size, 25))
total_predicted_simples = []
strat_test_time = time.time()
for sent in test_complexes_pad:
    fake_batch = np.zeros((batch_size, 25))
    fake_batch[0] = np.array(sent)
    pred_int = session.run(test_predictions, {inputs: fake_batch, keep_prob: 0.5})[0]
    pred_arg = np.argmax(pred_int, 1)
    init = []
    for word in pred_arg:
        init.append(simplesint2word[word])
    total_predicted_simples.append(init)
end_test_time = time.time()
print("Test Time: {}".format(int(end_test_time - strat_test_time) // 60))

total_time_end = time.time()
print("Total Time: {}".format(int(total_time_end - total_time_start) // 60))

total_predicted_simples_string = []
for line in total_predicted_simples:
    l = ' '.join(line)
    total_predicted_simples_string.append(l)
##
##############################################################################
####                    PART 7 - SARI, BLEU, FKGL                         ####
##############################################################################

############    SARI Score  ################
# Requirement: SARI file
# Note: Each file must have same Length
from sari.SARI import SARIsent

def sari_score(inputs, predicted, reference):
    SARI_score = []
    if len(test_complexes) == len(test_simples) == len(test_simples):
        for line in range(len(test_complexes)):
            SARI = SARIsent(inputs[line], predicted[line], reference[line])
            SARI_score.append(SARI)
        SARI_score_total = np.average(SARI_score)
    else:
        print("SARI: The number of Predicted and their references should be the same...")
    return SARI_score_total

total_sari_score = sari_score(test_complexes, test_simples, test_simples)
print("SARI Score is : {}".format(total_sari_score))

############   /SARI Score  ################

############    BELU Score  ################
# Note: Each file must have same Length
# Note: The reference sentences must be provided as a list of sentences where each reference is a list of tokens.
# Note: The candidate sentence is provided as a list of tokens.
from nltk.translate.bleu_score import corpus_bleu

def BLEU_same_length(predicted, refrence):
    BLEU_same_length = []
    if len(predicted) == len(refrence):
        for pre, ref in zip(predicted, refrence):
            if len(pre.split()) != len(ref.split()):
                BLEU_same_length.append([ref + ' <PAD> ' * (len(pre.split()) - len(ref.split())) ])
            else:
                BLEU_same_length.append([ref])
    else:
        print("BLEU: The number of Predicted and their reference should be the same... ")
    return BLEU_same_length

BLEU_output_padded = BLEU_same_length(test_complexes, test_complexes)
blue_score = corpus_bleu(BLEU_output_padded, test_complexes)
print("Corpus Bleu Score is : {}".format(blue_score))

############   /BELU Score  ################

############    FKGL Score  ################
# Requirement: syllables_en file
# Note: Check with both Functions
# Note: Text must be List of the strings
import nltk
from nltk.tokenize import RegexpTokenizer
# import syllables_en

TOKENIZER = RegexpTokenizer('(?u)\W+|\$[\d\.]+|\S+')
SPECIAL_CHARS = ['.', ',', '!', '?']
def FleschKincaidGradeLevel(text):
    sentences = []
    filtered_words = []
    syllableCount = 0
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for line in text:
        line.lower()
        words = []
        words = TOKENIZER.tokenize(line)
        
        for word in words:
            if word in SPECIAL_CHARS :
                pass
            else:
                strip_word = word.strip()
                new_word = strip_word.replace(",","").replace(".","")
                new_word = new_word.replace("!","").replace("?","")
                if new_word != "" and new_word != " " and new_word != "  " :
                    filtered_words.append(new_word)
            for char in word:
                syllableCount += syllables_en.count(char)
        sentences.append(tokenizer.tokenize(line)) # .decode('utf-8')
    word_count = len(filtered_words)
    sentence_count = len(sentences)
    syllable_count = syllableCount
    avg_words_p_sentence = word_count / sentence_count
    score = 0.0 
    if float(word_count) > 0.0:
        score = 0.39 * (float(avg_words_p_sentence)) + 11.8 * (float(syllable_count) / float(word_count)) - 15.59
    return round(score, 4)

fkgl_score = FleschKincaidGradeLevel(total_predicted_simples_string)
print("FKGL Score is : {}".format(fkgl_score))
#
#rd = Readability(test_complexes)
#print('FleschKincaidGradeLevel: ', rd.FleschKincaidGradeLevel())

############   /FKGL Score  ################


############   EASSE Score  ################
# Will be add
############  /EASSE Score  ################

############   SAMSA Score  ################
# Will be add
############  /SAMSA Score  ################

##############################################################################
####                PART 6 - Reinforcement Learning                       ####
##############################################################################

############    RL: PART 1 - Simplicity(1 - SARI)    ################
#r S = β S ARI (X, Ŷ , Y ) + (1 − β) S ARI (X, Y, Ŷ )
beta = 1.0
def simplicity(inputs, predicted, reference):
    sari = sari_score(inputs, predicted, reference)
    sari_reverse = sari_score(inputs, reference, predicted)
    return (beta * sari) + ((1.0 -beta) * sari_reverse)
r_s = simplicity(test_complexes, test_simples, test_simples)

############   /RL: PART 1 - Simplicity(1 - SARI)    ################

############   RL: PART 2 - Relevance(Auto-Encoder)  ################
#r = cos(q X , q Ŷ ) = q X · q Ŷ / ||q X || ||q Ŷ ||
####################
#model = Sequential()
#model.add(Embedding(1000, 64))
#model.compile('rmsprop', 'mse')
#input_i = Input(shape=(200,100))
#encoded_h1 = Dense(64, activation='tanh')(input_i)
#encoded_h2 = Dense(32, activation='tanh')(encoded_h1)
#encoded_h3 = Dense(16, activation='tanh')(encoded_h2)
#encoded_h4 = Dense(8, activation='tanh')(encoded_h3)
#encoded_h5 = Dense(4, activation='tanh')(encoded_h4)
#latent = Dense(2, activation='tanh')(encoded_h5)
#decoder_h1 = Dense(4, activation='tanh')(latent)
#decoder_h2 = Dense(8, activation='tanh')(decoder_h1)
#decoder_h3 = Dense(16, activation='tanh')(decoder_h2)
#decoder_h4 = Dense(32, activation='tanh')(decoder_h3)
#decoder_h5 = Dense(64, activation='tanh')(decoder_h4)
#
#output = Dense(100, activation='tanh')(decoder_h5)
#
#autoencoder = Model(input_i,output)
#
#autoencoder.compile('adadelta','mse')
### After adapting the above models parameters to your case, this should work fine:
#
#X_embedded = model.predict(X_train)
#autoencoder.fit(X_embedded,X_embedded,epochs=10,
#            batch_size=256, validation_split=.1)

#############
#x_train=np.reshape(x_train, (len(x_train), 1, 1))
#from keras.models import Sequential
#from keras.layers import LSTM, Dense, Masking
#
#model = Sequential()
#model.add(Masking(mask_value=0.0, input_shape=(timesteps, features)))
#model.add(LSTM(20, activation='tanh',return_sequences=True))
#model.add(LSTM(15, activation='tanh', return_sequences=True))
#model.add(LSTM(5, activation='tanh', return_sequences=True))
#model.add(LSTM(15, activation='tanh', return_sequences=True))
#model.add(LSTM(20, activation='tanh', return_sequences=True))
#model.add((Dense(1,activation='tanh')))

#nb_epoch = 10
#model.compile(optimizer='rmsprop', loss='mse')
#checkpointer = ModelCheckpoint(filepath="text_model.h5",
#                               verbose=0,
#                               save_best_only=True)
#
#es_callback = keras.callbacks.EarlyStopping(monitor='val_loss')
#
#history = model.fit(x_train, x_train,
#                    epochs=nb_epoch,
#                    shuffle=True,
#                    validation_data=(x_test, x_test),
#                    verbose=0,
#                    callbacks=[sarer,es_callback])
################

############  /RL: PART 2 - Relevance(Auto-Encoder)  ################

############    RL: PART 3 - Fluency(Exponential)    ################
#rf = np.exp(1/|Ŷ| * np.sum(np.log( Plm(Ŷ_i| Ŷ 0:i-1 ))))

############   /RL: PART 3 - Fluency(Exponential)    ################

############   /RL: PART 4 - Reward    ################
# It is the normalized sentence probability assigned by an LSTM language model
# trained on simple sentences:
# Exponential of Ŷ ’s perplexity
#r( Ŷ ) = λ S r S + λ R r R + λ F r F
############   /RL: PART 3 - Reward    ################
##############################################################################
####                           PART 8 - NER                               ####
##############################################################################

##############################################################################
####                        PART 9 - Others                               ####
##############################################################################

##############################################################################
####                         -- COMMENT --                                ####
##############################################################################

############    Test Commented  ################
#batch_size = 32
#def s_apply_padding(batch_of_sequences, word2int):
#    max_sequence_length = max([len(sequence) for sequence in batch_of_sequences])
#    return [sequence + [word2int['<PAD>']] * (max_sequence_length - len(sequence)) for sequence in batch_of_sequences]
#
#def s_split_into_batches(sorted_clean_test_complexes, batch_size):
#    for batch_index in range(0, len(sorted_clean_test_complexes) // batch_size):
#        start_index = batch_index * batch_size
#        input_test_in_batch = sorted_clean_test_complexes[start_index : start_index + batch_size]
#        padded_complexes_in_batch = np.array(s_apply_padding(input_test_in_batch, inputswords2int))
#        yield padded_complexes_in_batch

#total = []
#for batch_index_test, padded_inputs_in_batch in enumerate(s_split_into_batches(sorted_clean_test_complexes1, batch_size)):
#    print(batch_index_test)
#    print(len(padded_complexes_in_batch))
#    predicted_simples = session.run(test_predictions, {inputs: padded_inputs_in_batch, keep_prob: 0.5})
#    total.append(predicted_simples)
#
#count = 0
#count1 = 0
#prediction_final = []
#for i in range(0,(len(sorted_clean_test_complexes1) // batch_size)): # 7
#    count1 += 1
#    for k in range(0, batch_size): # 32
#        ints_sent = []
#        count += 1
#        for j in range(0, 25 ): # 25
#            ints_sent.append(str(np.argmax(total[i][k][j])))
#            
#        prediction_final.append(ints_sent)    
# .shape[1]
############   /Test Commented  ################

############    SARI Commented  ################
### for multi reference testing use list for ref arg ###
#out_csent3 = "About 95 species are currently agreed ."
#ref_rsents = ["About 95 species are currently known .", "About 95 species are now accepted .", "95 species are now accepted ."]
#print (SARIsent(ssent_input, out_csent1, ref_rsents))
############   /SARI Commented  ################

############    BLEU Commented  ################
#from nltk.translate.bleu_score import sentence_bleu
#reference = [['the', 'quick', 'brown', 'fox']]
#candidate = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
#score = sentence_bleu(test_simples, test_simples)
#print(score)
###
#from nltk.translate.bleu_score import SmoothingFunction
#smoother = SmoothingFunction()
#score = sentence_bleu(reference, candidate, smoother.method4)
##reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
####
##Corpus BLEU Score
## two references for one document
##references = [[['this', 'is', 'a', 'test'], ['this', 'is' 'test']]]
##candidates = [['this', 'is', 'a', 'test']]
####
## 1-gram individual BLEU
#from nltk.translate.bleu_score import sentence_bleu
#reference = [['this', 'is', 'small', 'test']]
#candidate = ['this', 'is', 'a', 'test']
#score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
#print(score)
###
#hypothesis1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always','obeys', 'the', 'commands', 'of', 'the', 'party']
#hypothesis2 = ['It', 'is', 'to', 'insure', 'the', 'troops','forever', 'hearing', 'the', 'activity', 'guidebook', 'that', 'party', 'direct']
#reference1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'that','ensures', 'that', 'the', 'military', 'will', 'forever','heed', 'Party', 'commands']
#reference2 = ['It', 'is', 'the', 'guiding', 'principle', 'which','guarantees', 'the', 'military', 'forces', 'always','being', 'under', 'the', 'command', 'of', 'the','Party']
#reference3 = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', 'army', 'always', 'to', 'heed', 'the', 'directions','of', 'the', 'party']
#sentence_bleu([reference1, reference2, reference3], hypothesis1) # doctest: +ELLIPSIS
##0.5045...
#sentence_bleu([reference1, reference2, reference3], hypothesis2) # doctest: +ELLIPSIS
##0.3969...
############   /BLEU Commented  ################

############    FKGL Commented  ################
####
#def get_words(text):
#    s_get_words = []
#    words = []
#    for line in text:
#        filtered_words = []
#        words = TOKENIZER.tokenize(line)
#        for word in words:
#            if word in SPECIAL_CHARS :
#                pass
#            else:
#                word = word.strip()
#                new_word = word.replace(",","").replace(".","")
#                new_word = new_word.replace("!","").replace("?","")
#                if new_word != "" and new_word != " " and new_word != "  " :
#                    filtered_words.append(new_word)
#        s_get_words.append(filtered_words)
#    return s_get_words
####
#all_get_words = get_words(test_complexes)
####
#def get_sentences(text=''):
#    sentences = []
#    for line in text:
#        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
##        sentences = tokenizer.tokenize(line.decode('utf-8'))
#        sentences.append(tokenizer.tokenize(line))
#    return sentences
#
####
#all_get_sentences = get_sentences(test_complexes)
####
####
#def count_syllables(words):
#    syllableCount = 0
#    for word in words:
#        syllableCount += syllables_en.count(word)
#    return syllableCount
####
#all_count_syllables = []
#for line in test_complexes:
#    all_count_syllables.append(count_syllables(line))
####
####
#def count_syllables(words):
#    syllableCount = 0
#    for word in words:
#        syllableCount += syllables_en.count(word)
#    return syllableCount
####
#all_count_syllables = []
#for line in test_complexes:
#    all_count_syllables.append(count_syllables(line))
####
## readability and utils and syllables_en Files need ...
#import readability
#rd = Readability(test_complexes)
#print('FleschKincaidGradeLevel: ', rd.FleschKincaidGradeLevel())
####
#b3 = 'The Australian platypus is seemingly a hybrid of a mammal and reptilian creature'
############   /FKGL Commented  ################


############     RL: PART 1 - Simplicity Commented     ################
############    /RL: PART 1 - Simplicity Commented     ################

############      RL: PART 2 - Relevance Commented     ################
############     /RL: PART 2 - Relevance Commented     ################

############       RL: PART 3 - Fluency Commented      ################
############      /RL: PART 3 - Fluency Commented      ################


##############################################################################
####                           -- END --                                  ####
##############################################################################







