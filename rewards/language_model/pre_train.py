"""
Author: SaMaN
"""
import tensorflow as tf
from rewards.language_model.data_utils import batch_iter
from rewards.language_model.model.language_model import LanguageModel

BATCH_SIZE = 64
NUM_EPOCHS = 2 #10
MAX_DOCUMENT_LEN = 50


def train(input_file, target, word_dict):
    with tf.Session() as sess:
        model = LanguageModel(word_dict, MAX_DOCUMENT_LEN)
        # Define training procedure
        global_step = tf.Variable(0, trainable=False)
        params = tf.trainable_variables()
        gradients = tf.gradients(model.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        optimizer = tf.train.AdamOptimizer(0.001)
        train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)
        # Summary
        tf.summary.scalar("loss", model.loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter("language_model_weights", sess.graph)
        # Checkpoint
        saver = tf.train.Saver(tf.global_variables())
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        def train_step(batch_inputs):
            feed_dict = {model.x: batch_inputs}
            _, step, summaries, loss = sess.run([train_op, global_step, summary_op, model.loss],
                                                        feed_dict=feed_dict)
            summary_writer.add_summary(summaries, step)
            if step % 100 == 0:
                print("step {0} : loss = {1}".format(step, loss))
            # return loss, logits
        # Training loop
        batches = batch_iter(input_file, target, BATCH_SIZE, NUM_EPOCHS)
        for batch_inputs, _ in batches:
            train_step(batch_inputs)
            step = tf.train.global_step(sess, global_step)
        saver.save(sess, "/home/saman/Desktop/thesis_/thesis_DERSS/rewards/language_model/"
                         "checkpoint/model-100epc.ckpt", global_step=step)
