"""
Author: SaMaN
Note: must evaluate cosine similarity between y_hat and
Corresponding X (simple)
"""
import pickle
import numpy as np
import tensorflow as tf
from rewards.auto_encoder.pre_train import train
from rewards.auto_encoder.query import QueryEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from rewards.auto_encoder.data_utils import build_word_dict, build_word_dataset


class Relevance:
    def __init__(self, complex_input, predicted, max_sequence_length):
        self.max_length = max_sequence_length
        self.query = predicted
        self.complex_input = complex_input

    def encoder_output(self):
        # Building dictionary..
        word_dict = build_word_dict()
        # Preprocessing dataset..
        train_x, train_y = build_word_dataset(word_dict, self.max_length)
        embedded_data = train(train_x, train_y, word_dict)
        return embedded_data

    def complex_encoder_output(self):
        # Convert Sentence to embedded representation base on previous Trained model weights.
        tf.reset_default_graph()
        with open("/home/saman/Desktop/thesis_/thesis_DERSS/rewards/auto_encoder/word_dict.pickle",
                  "rb") as f:
            word_dict = pickle.load(f)
        complex_embedding = QueryEmbedding(self.complex_input, word_dict)
        embed_complex = complex_embedding.query()
        return embed_complex

    def predicted_encoder_output(self):
        # Convert Sentence to embedded representation base on previous Trained model weights.
        tf.reset_default_graph()
        with open("/home/saman/Desktop/thesis_/thesis_DERSS/rewards/auto_encoder/word_dict.pickle",
                  "rb") as f:
            word_dict = pickle.load(f)
        predicted_embedding = QueryEmbedding(self.query, word_dict)
        embed_pred = predicted_embedding.query()
        return embed_pred

    def sum_embedded_words(self, embed_sent):
        sent_embedded = []
        for word in embed_sent:
            sent_embedded.append(sum(word).tolist())
        return sent_embedded

    def similarity(self, x, p):
        # sentence = np.reshape(sentence, [1, 512])
        return cosine_similarity(x, p)

    def run(self):
        predicted_encoder = self.predicted_encoder_output()
        embedded_predicted = self.sum_embedded_words(predicted_encoder[:1])
        embedded_predicted = np.reshape(embedded_predicted, [1, 512])
        complex_encoder = self.complex_encoder_output()
        embedded_complex = self.sum_embedded_words(complex_encoder[:1])
        embedded_complex = np.reshape(embedded_complex, [1, 512])
        score = self.similarity(embedded_complex, embedded_predicted)
        return score


# if __name__ == '__main__':
#     max_sequence_length = 50
#     predicted = "Pre-Columbian Art Research Institute ."
#     complex_input = "Pre-Columbian Art Research Institute ."
#     relevance = Relevance(complex_input, predicted, max_sequence_length)
#     score = relevance.run()
#     print(score[0][0])
