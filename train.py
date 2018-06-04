#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 5/21/2018 11:05 AM
# @Author  : Siqi
#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from sklearn.metrics import precision_score, recall_score, f1_score
from Bi_LSTM_Model import BiLSTM

from tensorflow.contrib import learn
from gensim.models.keyedvectors import KeyedVectors
import sklearn.metrics
import time



np.set_printoptions(threshold=np.inf)
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("train_label_0", "./train/label_0.txt", "Data source for label 0")
tf.flags.DEFINE_string("train_label_1", "./train/label_1.txt", "Data source for label 1")
tf.flags.DEFINE_string("train_label_2", "./train/label_2.txt", "Data source for label 2")
tf.flags.DEFINE_string("train_label_3", "./train/label_3.txt", "Data source for label 3")
tf.flags.DEFINE_string("train_label_4", "./train/label_4.txt", "Data source for label 4")
tf.flags.DEFINE_string("train_label_5", "./train/label_5.txt", "Data source for label 5")

#Model


# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 1, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning_rate")
#LSTM
tf.flags.DEFINE_integer("hidden_sizes", 128, "Number of hidden sizes (default: 128)")
#tf.flags.DEFINE_integer("attention_size", 300, "ATTENTION_SIZE")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
#FLAGS._parse_flags()
FLAGS.flag_values_dict()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")



#load
x_text, y = data_helpers.load_data_and_labels(FLAGS.train_label_0, FLAGS.train_label_1,
                                              FLAGS.train_label_2, FLAGS.train_label_3,
                                              FLAGS.train_label_4, FLAGS.train_label_5
                                              )

max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/dev set
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))





# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        np.random.seed(1)
        tf.set_random_seed(2)

        cnn = BiLSTM(
             input_embedding_size = FLAGS.embedding_dim,
             sequence_length = x_train.shape[1],
             #hidden_size = FLAGS.num_filters * len(list(map(int, FLAGS.filter_sizes.split(",")))),
             hidden_size=FLAGS.hidden_sizes,
             output_size = y_train.shape[1],
             vocab_size = len(vocab_processor.vocabulary_),
             learning_rate = FLAGS.learning_rate)


        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

        # Write vocabulary
        vocab_processor.save(os.path.join(out_dir, "vocab"))

        # Initialize all variables
        #sess.run(tf.global_variables_initializer())

        #load word2vec
        # print("Start Loading Embedding!")
        # word2vec = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        # print("Finish Loading Embedding!")
        # my_embedding_matrix = np.zeros(shape=(len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
        # for word in vocab_processor.vocabulary_._mapping:
        #     id = vocab_processor.vocabulary_._mapping[word]
        #     if word in word2vec.vocab:
        #         my_embedding_matrix[id] = word2vec[word]
        #     else:
        #         my_embedding_matrix[id] = np.random.uniform(low=-0.0001, high=0.0001, size=FLAGS.embedding_dim)
        # W = tf.placeholder(tf.float32, [None, None], name="pretrained_embeddings")
        # set_x = cnn.W.assign(my_embedding_matrix)
        # sess.run(set_x, feed_dict={W: my_embedding_matrix})

        print("Finish transfer")
        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }

            _, step, summaries, loss, accuracy, predictions,y_actual = sess.run(
                     [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy, cnn.predictions,cnn.y],
                     feed_dict)

            time_str = datetime.datetime.now().isoformat()
            # print("train_f1_score:", f1_score(y_actual, predictions, average=None))
            # print (predictions)
            # print(y_actual)
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
            return accuracy

            train_summary_writer.add_summary(summaries, step)

        def dev_step(x_batch, y_batch, writer=None):
            """
            Evaluates model on a dev set
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: 1.0
            }

            step, summaries, loss, accuracy ,predictions,y_actual= sess.run(
                [global_step, dev_summary_op, cnn.loss, cnn.accuracy, cnn.predictions,cnn.y],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            if writer:
                writer.add_summary(summaries, step)
            return accuracy






        if __name__ == "__main__":
            # Save the maximum accuracy value for validation data
            sess.run(tf.global_variables_initializer())
            max_acc_dev = 0.
            max_epoch = 0
            for epoch in range(FLAGS.num_epochs):
                time_start = time.time()
                epochs = data_helpers.epochs_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                for batch in epochs:
                    x_batch , y_batch = zip(*batch)
                    train_accuracy = train_step(x_batch , y_batch)

                    current_step = tf.train.global_step(sess, global_step)
                    # path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    # print("Saved model checkpoint to {}\n".format(path))
                print("\nEvaluation:")
                print("Epoch: %03d" % (epoch))
                dev_accuracy= dev_step(x_dev, y_dev, writer=dev_summary_writer)
                print ("Dev_accuracy:", dev_accuracy)

                if dev_accuracy > max_acc_dev:
                    max_acc_dev = dev_accuracy
                    max_epoch = epoch
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("max_acc_dev %f" %max_acc_dev)
                if (epoch - max_epoch) > 5:
                    break
