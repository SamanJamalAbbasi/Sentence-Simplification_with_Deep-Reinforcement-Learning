"""
Author: SaMaN
"""
import numpy as np
import tensorflow as tf
from rewards.language_model.model.language_model import LanguageModel


class QueryEmbedding:
    def __init__(self, input_query, word_dict):
        self.query = input_query
        self.max_length = 50
        self.wordDict = word_dict
        self.batch_size = 1

    def build_dataset(self):
        y_hat = []
        for word in self.query.split():
            y_hat.append(word)
        y_hat = list(map(lambda w: self.wordDict.get(w, self.wordDict["<unk>"]), y_hat))
        # In the test phase ,Is it necessary to impose restrictions on the length of the sequence??
        y_hat = list(y_hat[:self.max_length])
        y_hat = y_hat + (self.max_length - len(y_hat)) * [self.wordDict["<pad>"]]
        return y_hat

    def lm_prob_query(self):
        tf.reset_default_graph()
        model = LanguageModel(self.wordDict, self.max_length)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pre_trained_variables = [v for v in tf.global_variables()
                                     if (v.name.startswith("embedding") or v.name.startswith("birnn"))
                                     and "Adam" not in v.name]
            saver = tf.train.Saver(pre_trained_variables)
            ckpt = tf.train.get_checkpoint_state("/home/saman/Desktop/thesis_/thesis_DERSS/rewards/language_model"
                                                 "/checkpoint/")
            saver.restore(sess, ckpt.model_checkpoint_path)
            word2int = self.build_dataset()
            fake_batch = np.zeros((self.batch_size, self.max_length))
            fake_batch[0] = word2int
            feed_dict = {model.x: fake_batch}
            lm_logits = sess.run([model.logits], feed_dict=feed_dict)
        return lm_logits

    def perplexity(self):
        lm_logits = self.lm_prob_query()
        sum_score = 0
        for index in range(0, len(self.query.split())):
            sum_score += np.log(max(lm_logits[0][0][index]))
        return np.exp(sum_score / len(self.query.split()))
