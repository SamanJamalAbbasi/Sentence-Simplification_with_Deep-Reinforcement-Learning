"""
Author: SaMaN
"""
import os
import re
import pickle
import collections
import numpy as np
from nltk.tokenize import word_tokenize

file_dir = "/home/saman/Desktop/thesis_/thesis_DERSS/data/wikismall/wikismall_final/both_complex_simple_train.txt"
input_text = open(file_dir, encoding='utf-8', errors='ignore').read().split('\n')
TRAIN_PATH = file_dir
TEST_PATH = file_dir


def clean_str(text):
    text = re.sub(r"[^A-Za-z0-9(),!?\'`\"]", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text


def build_word_dict():
    if not os.path.exists("/home/saman/Desktop/thesis_/thesis_DERSS/rewards/auto_encoder/word_dict.pickle"):
        contents = input_text
        words = list()
        for content in contents:
            for word in word_tokenize(clean_str(content)):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<pad>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        for word, count in word_counter:
            if count > 1:
                word_dict[word] = len(word_dict)

        with open("/home/saman/Desktop/thesis_/thesis_DERSS/rewards/auto_encoder/word_dict.pickle",
                  "wb") as f:
            pickle.dump(word_dict, f)

    else:
        with open("/home/saman/Desktop/thesis_/thesis_DERSS/rewards/auto_encoder/word_dict.pickle",
                  "rb") as f:
            word_dict = pickle.load(f)

    return word_dict


def build_word_dataset(word_dict, document_max_len):
    df = input_text
    x = list(map(lambda d: word_tokenize(clean_str(d)), df))
    x = list(map(lambda d: list(map(lambda w: word_dict.get(w, word_dict["<unk>"]), d)), x))
    x = list(map(lambda d: d[:document_max_len], x))
    x = list(map(lambda d: d + (document_max_len - len(d)) * [word_dict["<pad>"]], x))
    y = list(map(lambda d: d, list(df)))

    return x[:70], y[:70]


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)
    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]
