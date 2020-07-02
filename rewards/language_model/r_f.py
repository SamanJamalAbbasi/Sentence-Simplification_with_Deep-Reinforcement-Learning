"""
Author: SaMaN
Train and test Language Model

TO DO:
Check Max-Length, Pre-Processing(Simple va Query) and min Count .
Documentation for class's and def's.
"""
import pickle
import tensorflow as tf
from rewards.language_model.pre_train import train
from rewards.language_model.query import QueryEmbedding
from rewards.language_model.data_utils import build_word_dict, build_word_dataset


class Fluency:
    def __init__(self, predicted, max_sequence_length):
        self.max_length = max_sequence_length
        self.query = predicted

    def train_lm(self):
        # Training all data and Save it
        word_dict = build_word_dict()
        # Preprocessing dataset..
        train_x, train_y = build_word_dataset(word_dict, self.max_length)
        train(train_x, train_y, word_dict)

    def query_perplexity(self):
        tf.reset_default_graph()
        with open("/home/saman/Desktop/thesis_/thesis_DERSS/rewards/language_model/lm_word_dict.pickle",
                  "rb") as wd:
            word_dict = pickle.load(wd)
        lm_output = QueryEmbedding(self.query, word_dict)
        r_f = lm_output.perplexity()
        return r_f


if __name__ == '__main__':
    max_sequence_length = 50
    predicted = "Pre-Columbian Art Research Institute ."
    fluency = Fluency(predicted, max_sequence_length)
    r_f = fluency.query_perplexity()
    print(r_f)
