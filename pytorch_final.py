"""
tutorial chatbot site pytorcho b ye taghirati tu
pre va test va afzudan metric baraye dress amade kardam.
class ro tu faze pre hazf kardam , 6om tir ham evaluate ha ro
tush copy kardam.
fk konam final derss ba torch in bashe.

Created on Sun Mar 15 12:55:12 2020

@author: saman
"""
##############################################################################
####     Sentences Simplification with Deep Reinforcement Learning        ####
#########################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

import numpy as np
import time

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

total_time_start = time.time()

##############################################################################
####                    PART 1 - Data Preprocessing                       ####
##############################################################################
inputs = open("/home/saman/Desktop/thesis_/ori.test.small/PWKP_108016.tag.80.aner.ori.test.src", encoding = 'utf-8', errors = 'ignore').read().split('\n')
inputs = inputs[:-1]
outputs = open("/home/saman/Desktop/thesis_/ori.test.small/PWKP_108016.tag.80.aner.ori.test.dst", encoding= 'utf-8', errors= 'ignore').read().split('\n')
outputs = outputs[:-1]
save_dir = "/home/saman/Desktop/thesis_/ori.test.small/save/"
corpus_name = "/home/saman/Desktop/thesis_/ori.test.small/"

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
    text = re.sub(r"its", "it is", text)
    text = re.sub(r"it's", "it is", text)
    return text

clean_inputs = []
for _line in inputs:
    clean_inputs.append(clean_text(_line))

clean_outputs = []
for _line in outputs:
    clean_outputs.append(clean_text(_line))

# Create word count :
word2count = {}
for _line in clean_inputs:
    for _i in _line.split():
        if _i not in word2count:
            word2count[_i] = 1
        else:
            word2count[_i] += 1

for _line in clean_outputs:
    for _i in _line.split():
        if _i not in word2count:
            word2count[_i] = 1 
        else:
            word2count[_i] += 1

# Creating two dictionaries that map the questions words and the answers words to a unique integer
MIN_COUNT = 2
threshold_inputs = MIN_COUNT
inputswords2int = {}
word_number = 0
for _word, _count in word2count.items():
    if _count >= threshold_inputs :
        inputswords2int[_word] = word_number
        
        word_number += 1

threshold_outputs = MIN_COUNT
outputswords2int = {}
word_number = 0
for _word, _count in word2count.items():
    if _count >= threshold_outputs :
        outputswords2int[_word] = word_number
        word_number += 1
#$
clean_inp_unk_str = []
clean_inp_unk_list = []
for line in clean_inputs:
    temp = []
    for word in line.split():
        if word in word2count:
            if word2count[word] >= threshold_inputs:
                temp.append(word)
            else:
                temp.append('<OUT>')
        else:
            temp.append('<OUT>')
    clean_inp_unk_str.append(' '.join(temp))
    clean_inp_unk_list.append(temp)

clean_out_unk_str = []
clean_out_unk_list = []
for line in clean_outputs:
    temp = []
    for word in line.split():
        if word in word2count and word2count[word] >= threshold_inputs :
#            if word2count[word] >= threshold_inputs:
            temp.append(word)
        else:
            temp.append('<OUT>')
    clean_out_unk_str.append(' '.join(temp))
    clean_out_unk_list.append(temp)
#$ Check Both Max Lenght be the Same
MAX_LENGTH = 28
clean_inp_unk_list_len = []
clean_out_unk_list_len = []
for i, j in zip(clean_inp_unk_list, clean_out_unk_list):
    if MAX_LENGTH >= len(i) and MAX_LENGTH >= len(j) :
        clean_inp_unk_list_len.append(i)
        clean_out_unk_list_len.append(j)
#$
# Adding the last tokens to these two dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    inputswords2int[token] = len(inputswords2int) + 1
for token in tokens:
    outputswords2int[token] = len(outputswords2int) + 1

# Creating the inverse dictionary of the answerswords2int dictionary
### Ehtemalan chun outputswords2int va inputswords2int yeki hastan yebar neveshte ke baiide
outputsint2word = {key: value for value, key in outputswords2int.items()}

# Adding the End Of String token to the end of every answer
for i in range(len(clean_outputs)):
    clean_outputs[i] += ' <EOS>'

# Translating all the questions and the answers into integers
# and Replacing all the words that were filtered out by <OUT>
inputs_into_int = []
for _line in clean_inputs:
    ints = []
    for _i in _line.split():
        if _i not in inputswords2int:
            ints.append(inputswords2int['<OUT>'])
        else:
            ints.append(inputswords2int[_i])
    inputs_into_int.append(ints)

outputs_into_int = []
for _line in clean_outputs:
    ints = []
    for _i in _line.split():
        if _i not in outputswords2int:
            ints.append(outputswords2int['<OUT>'])
        else:
            ints.append(outputswords2int[_i])
    outputs_into_int.append(ints)

# Sorting questions and answers by the length of questions
#max_sequence_length = 10
sorted_clean_inputs = []
sorted_clean_outputs = []
for length in range(1, MAX_LENGTH):
    for i in enumerate(inputs_into_int):
        if len(i[1]) == length:
            sorted_clean_inputs.append(inputs_into_int[i[0]])
            sorted_clean_outputs.append(outputs_into_int[i[0]])

#######################################################################
# SOS
# def preprocess_targets(targets, word2int, batch_size):
#     left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
#     right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
#     preprocessed_targets = tf.concat([left_side, right_side], 1)
#     return preprocessed_targets
#batch_size = 32
# preprocess_targets(targets, questionswords2int, batch_size) # for embedding # targets : torch tensor

#######################################################################
# Splitting the questions and answers into training and validation sets
#training_validation_split = int(len(sorted_clean_inputs) * 0.15)
#training_questions = sorted_clean_inputs[training_validation_split:]
#training_answers = sorted_clean_outputs[training_validation_split:]
#validation_questions = sorted_clean_inputs[:training_validation_split]
#validation_answers = sorted_clean_outputs[:training_validation_split]

#######################################################################
#def apply_padding(batch_of_sequences, word2int):
#    MAX_LENGTH = max([len(sequence) for sequence in batch_of_sequences])
#    return [sequence + [word2int['<PAD>']] * (MAX_LENGTH - len(sequence)) for sequence in batch_of_sequences]

# Splitting the data into batches of questions and answers
# def split_into_batches(questions, answers, batch_size):
#     for batch_index in range(0, len(questions) // batch_size):
#         start_index = batch_index * batch_size
#         questions_in_batch = questions[start_index : start_index + batch_size]
#         answers_in_batch = answers[start_index : start_index + batch_size]
#         padded_questions_in_batch = np.array(apply_padding(questions_in_batch, inputswords2int))
#         padded_answers_in_batch = np.array(apply_padding(answers_in_batch, outputswords2int))
#         yield padded_questions_in_batch, padded_answers_in_batch

# enumerate(split_into_batches(training_questions, training_answers, batch_size) # when model run

#######################################################################
# Prepare Data for Torch Models
#for pair in pairs:
##    for sentence in pair[0]:
##        print(pair[0])
##        print(sentence)
#    print([inputswords2int[word] for word in pair.split(' ')] + [inputswords2int['<EOS>']])

def indexesFromSentence(inputswords2int, sentence):
    return [inputswords2int[word] for word in sentence.split()] + [inputswords2int['<EOS>']]
###$
#    returnn = []
#    aa = []
#    for i in sentence:
#        aa.append(i)
#    sentence = ''.join(aa)
##    try:
##        for word in sentence.split(' '):
##            returnn.append(inputswords2int[word])
###$
    
#indexesFromSentence = sorted_clean_inputs
# sorted_clean_outputs
def zeroPadding(l, fillvalue=inputswords2int['<PAD>']):
#    aa = list(itertools.zip_longest(*l, fillvalue=fillvalue))
#    print("AA")
#    print(aa)
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=inputswords2int["<PAD>"]):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == inputswords2int["<PAD>"]:
                m[i].append(0)
            else:
                m[i].append(1)
    return m
#sort_revese = sorted_clean_inputs[::-1]
#l = pair[0] and pair[1]
# clean_inputs
#$
s_pairs = []
for i, j in zip(clean_inp_unk_str, clean_out_unk_str):
    s_pairs.append([''.join(i),''.join(j)])
####$
#indexes_batch_inp = []
#for line in clean_inp_unk_str:
#    temp = []
#    for word in line.split(' '):
#        temp.append(inputswords2int[word])
#    indexes_batch_inp.append(temp + [inputswords2int['<EOS>']])
####
#indexes_batch_out = []
#for line in clean_out_unk_str:
#    temp = []
#    for word in line.split(' '):
#        temp.append(inputswords2int[word])
#    indexes_batch_out.append(temp + [inputswords2int['<EOS>']])
####$
#$
def inputVar(indexes_batch):
#    indexes_batch = [indexesFromSentence(inputswords2int, sentence) for sentence in input_batch]
#    indexes_batch = indexes_batch_inp
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(indexes_batch):
#    indexes_batch = [indexesFromSentence(inputswords2int, sentence) for sentence in output_batch]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len
###$
#MAX_LENGTH = 25 # 10
#sorted_clean_inputs = []
#sorted_clean_outputs = []
#for length in range(1, MAX_LENGTH + 1):
#    for i in enumerate(inputs_into_int):
#        if len(i[1]) == length:
#            sorted_clean_inputs.append(inputs_into_int[i[0]])
#            sorted_clean_outputs.append(outputs_into_int[i[0]])
#
#pair_in = []
#pair_out = []
#for length in range(1, MAX_LENGTH + 1):
#    for i in enumerate(clean_inp_unk_str):
#        if len(i[1].split()) == length:
#            pair_in.append(clean_inp_unk_str[i[0]])
#            pair_out.append(clean_out_unk_str[i[0]])

###$
# Returns all items for a given batch of pairs
def batch2TrainData(clean_inp, clean_out):
    pair_in = []
    pair_out = []
    for length in range(MAX_LENGTH , 0, -1):
        for i in enumerate(clean_inp):
            if len(i[1]) == length:
                pair_in.append(clean_inp[i[0]])
                pair_out.append(clean_out[i[0]])

    pair_inp_int = []
    for line in pair_in:
        temp = []
        for word in line:
            temp.append(inputswords2int[word])
        pair_inp_int.append(temp + [inputswords2int['<EOS>']])
    inp, lengths = inputVar(pair_inp_int)
    
    pair_out_int = []
    for line in pair_out:
        temp = []
        for word in line:
            temp.append(inputswords2int[word])
        pair_out_int.append(temp + [inputswords2int['<EOS>']])
    output, mask, max_target_len = outputVar(pair_out_int)
    
    return inp, lengths, output, mask, max_target_len
#
## Returns all items for a given batch of pairs
small_batch_size = 5
# sorted_clean_inputs : hame jur token okey shode, amade baraye emmbeding hast
batches = batch2TrainData(clean_inp_unk_list_len[:small_batch_size], clean_out_unk_list_len[:small_batch_size])
input_variable, lengths, target_variable, mask, max_target_len = batches

# Transpose
#input_variable = input_variable.t()
#target_variable = target_variable.t()
#mask = mask.t()
#max_target_len = max_target_len.t()

print("input_variable:", input_variable)
print("lengths:", lengths)
print("target_variable:", target_variable)
print("mask:", mask)
print("max_target_len:", max_target_len)


#######################################################################
def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

##############################################################################
####                PART 2 - BUILDING THE SEQ2SEQ MODEL                   ####
##############################################################################
## Encoder
#MAX_LENGTH = 10 #$ 25  # Maximum sentence length to consider

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden

##############################################################################
## Decoder
class Attn(torch.nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
##############################################################################

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden

##############################################################################
## Single training iteration
def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[inputswords2int['<SOS>'] for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

## Training iterations

def trainIters(model_name, sorted_clean_inputs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, loadFilename): # corpus_name

    # Load batches for each iteration
    training_batches = [batch2TrainData(clean_inp_unk_list_len[:batch_size], clean_out_unk_list_len[:batch_size]) for _ in range(n_iteration)]
#([random.choice(sorted_clean_inputs) for _ in range(batch_size)], [random.choice(sorted_clean_outputs) for _ in range(batch_size)])
    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Save checkpoint
        # if (iteration % save_every == 0):
        #     directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
        #     if not os.path.exists(directory):
        #         os.makedirs(directory)
        #     torch.save({
        #         'iteration': iteration,
        #         'en': encoder.state_dict(),
        #         'de': decoder.state_dict(),
        #         'en_opt': encoder_optimizer.state_dict(),
        #         'de_opt': decoder_optimizer.state_dict(),
        #         'loss': loss,
        #         # 'voc_dict': voc.__dict__,
        #         'embedding': embedding.state_dict()
        #     }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


##############################################################################
# Define Evaluation
# Greedy decoding
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * inputswords2int['<SOS>']
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores
##############################################################################
## Evaluate my text
def evaluate(encoder, decoder, searcher, sorted_clean_inputs, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = sorted_clean_inputs
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [outputsint2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, sorted_clean_inputs):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = clean_text(input_sentence)
            # Evaluate sentence
            output_words = evaluate(encoder, decoder, searcher, sorted_clean_inputs, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))

        except KeyError:
            print("Error: Encountered unknown word.")

##############################################################################z
# Configure models
model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 4000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    # voc.__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(word_number + 5 , hidden_size) # word2count #voc.num_words
if loadFilename:
    embedding.load_state_dict(embedding_sd)
# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, word_number, decoder_n_layers, dropout) #  word2count
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')
##############################################################################
## Run Training

# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 1 # 4000
print_every = 1
save_every = 500

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# Run training iterations
print("Starting Training!")
trainIters(model_name, sorted_clean_inputs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, loadFilename) # corpus_name

##############################################################################
"""Run Evaluation
~~~~~~~~~~~~~~

To chat with your model, run the following block.
"""

# Set dropout layers to eval mode
encoder.eval()
decoder.eval()

# Initialize search module
#searcher = GreedySearchDecoder(encoder, decoder)
print("Well Done SaMaNian")
##############################################################################
# 99-4-6
######################################################################
# $

my_testinput = open("/home/saman/Desktop/thesis_/ori.test.small/PWKP_108016.tag.80.aner.ori.test.src", encoding='utf-8',
                    errors='ignore').read().split('\n')
my_testinput = my_testinput[:-1]
test_output = open("/home/saman/Desktop/thesis_/ori.test.small/PWKP_108016.tag.80.aner.ori.test.dst", encoding='utf-8',
                   errors='ignore').read().split('\n')
test_output = test_output[:-1]
my_output = []
indexes = []
indexes_batch = []


########################################################################
def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


##############################################################################
####                    PART 7 - SARI, BLEU, FKGL                         ####
##############################################################################
test_input = open("/home/saman/Desktop/thesis_/ori.test.small/PWKP_108016.tag.80.aner.ori.test.src", encoding='utf-8',
                  errors='ignore').read().split('\n')
test_input = test_input[:-1]
test_output = open("/home/saman/Desktop/thesis_/ori.test.small/PWKP_108016.tag.80.aner.ori.test.dst", encoding='utf-8',
                   errors='ignore').read().split('\n')
test_output = test_output[:-1]
# $
line = my_testinput[1]
aa = []
bb = []
a = "malthusianism malthusianism malthusianism malthusianism has has*malthusianism*awarded called called higher higher higher higher*collections damage higher specific scenic age*rainfall organised celestial laws scenic SCENIC*moved she she she she palladium*mine mine eat old work work three on on age into into other area into area*into area scientific other other area other area scientific other*other area scientific other*german groups president make singapore gives*symphony fields symphony mine celestial reelection"
for i in a.split('*'):
    aa.append(i)

bbb = []
for line in aa:
    input_sentence = normalizeString(line)
    bb = evaluate(encoder, decoder, searcher, voc, input_sentence)
    bb[:] = [x for x in bb if not (x == 'EOS' or x == 'PAD')]
    bbb.append(' '.join(bb))
# $
############    SARI Score  ################
# Requirement: SARI file
# Note: Each file must have same Length
from sari.SARI import SARIsent


def sari_score(inputs, predicted, reference):
    SARI_score = []
    if len(test_input) == len(test_output) == len(test_output):
        for line in range(len(test_input)):
            SARI = SARIsent(inputs[line], predicted[line], reference[line])
            SARI_score.append(SARI)
        SARI_score_total = np.average(SARI_score)
    else:
        print("SARI: The number of Predicted and their references should be the same...")
    return SARI_score_total


total_sari_score = sarii(test_input, test_output, test_output)
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
                BLEU_same_length.append([ref + ' <PAD> ' * (len(pre.split()) - len(ref.split()))])
            else:
                BLEU_same_length.append([ref])
    else:
        print("BLEU: The number of Predicted and their reference should be the same... ")
    return BLEU_same_length


BLEU_output_padded = BLEU_same_length(test_input, test_input)
blue_score = corpus_bleu(BLEU_output_padded, test_input)
print("Corpus Bleu Score is : {}".format(blue_score))

############   /BELU Score  ################

############    FKGL Score  ################
# Requirement: syllables_en file
# Note: Check with both Functions
# Note: Text must be List of the strings
import nltk
from nltk.tokenize import RegexpTokenizer
import syllables_en

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
            if word in SPECIAL_CHARS:
                pass
            else:
                strip_word = word.strip()
                new_word = strip_word.replace(",", "").replace(".", "")
                new_word = new_word.replace("!", "").replace("?", "")
                if new_word != "" and new_word != " " and new_word != "  ":
                    filtered_words.append(new_word)
            for char in word:
                syllableCount += syllables_en.count(char)
        sentences.append(tokenizer.tokenize(line))  # .decode('utf-8')
    word_count = len(filtered_words)
    sentence_count = len(sentences)
    syllable_count = syllableCount
    avg_words_p_sentence = word_count / sentence_count
    score = 0.0
    if float(word_count) > 0.0:
        score = 0.39 * (float(avg_words_p_sentence)) + 11.8 * (float(syllable_count) / float(word_count)) - 15.59
    return round(score, 4)


fkgl_score = FleschKincaidGradeLevel(total_predicted_answer_string)
print("FKGL Score is : {}".format(fkgl_score))
#
# rd = Readability(test_input)
# print('FleschKincaidGradeLevel: ', rd.FleschKincaidGradeLevel())

############   /FKGL Score  ################

##############################################################################
####                     ///  SARI, BLEU, FKGL                            ####
##############################################################################

########################################################################
# # def evaluateInput(encoder, decoder, searcher, voc):
# for line in my_testinput:
#     input_sentence = normalizeString(line)
#     # output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
#     indexes_batch = [voc.word2index[word] for word in input_sentence.split(' ')] + [EOS_token]
#     #

#     lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
#     input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
#     input_batch = input_batch.to(device)
#     lengths = lengths.to(device)
#     tokens, scores = searcher(input_batch, lengths, max_length)
#     output_words = [voc.index2word[token.item()] for token in tokens]

#     # output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
#     # print('Bot:', ' '.join(output_words))
#     # my_output.append(' '.join(output_words))
# ##
# f = []
# inp = "the the the jadgal"
# for word in inp.split(' '):
#     print(word)
#     f.append(voc.word2index[word])
# ##
# #$
# for line in my_testinput:
#     input_sentence = normalizeString(line)
#     #output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
#     #output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
#     #my_output.append(' '.join(output_words))

#     #indexes_batch = [indexesFromSentence(voc, line)]
#     for word in input_sentence.split(' '):
#         #print(word)
#         flag = word in voc.word2index
#         if flag:
#             indexes.append(voc.word2index[word])
#     indexes_batch.append(indexes + [EOS_token])
# lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
# input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
# input_batch = input_batch.to(device)
# lengths = lengths.to(device)
# tokens, scores = searcher(input_batch, lengths, max_length)
# decoded_words = [voc.index2word[token.item()] for token in tokens]


##$
######################################################################


######################################################################
# $
## My Addition
# Code to read files into Colaboratory:
# !pip install -U -q PyDrive
# from pydrive.auth import GoogleAuth
# from pydrive.drive import GoogleDrive
# from google.colab import auth
# from oauth2client.client import GoogleCredentials
## Authenticate and create the PyDrive client.
# auth.authenticate_user()
# gauth = GoogleAuth()
# gauth.credentials = GoogleCredentials.get_application_default()
# drive = GoogleDrive(gauth)
# link = "https://drive.google.com/open?id=1AFjKBFSJpK2xKHlrSJ3Xd9LRgtMj1RTo"
# fluff, id = link.split('=')
# corpus_name = drive.CreateFile({'id':id})
# corpus_name.GetContentFile('movie_lines.txt')
##$
######################################################################
