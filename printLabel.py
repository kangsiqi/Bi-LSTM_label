import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn
import time
import json
from pyltp import Segmentor
import re

# Parameters
# ==================================================

def emotion_label(test_data):
    # change this to a directory with the desired checkpoint


    #tf.flags.DEFINE_string("checkpoint_dir", ".\\runs\\1526896170\\checkpoints", "Checkpoint directory from training run")
    tf.flags.DEFINE_string("checkpoint_dir", "./runs/1526896170/checkpoints", "Checkpoint directory from training run")
    tf.flags.DEFINE_string("test_file", "twitter-datasets/test_data.txt", "Path and name of test file")
    tf.flags.DEFINE_string("submission_filename", "submission_predictions" + str(int(time.time())), "Path and name of submission file")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    FLAGS = tf.flags.FLAGS




    test_data = [data_helpers.clean_str(test) for test in test_data]
    print (test_data)
    # Map data into vocabulary
    # vocab_path = ".\\runs\\1526896170\\vocab"
    vocab_path = "./runs/1526896170/vocab"
    #vocab_path = FLAGS.checkpoint_dir + "\\..\\vocab"
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
    x_test = np.array(list(vocab_processor.transform(test_data)))


    # Evaluation
    # ==================================================
    checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("train_input").outputs[0]
            #input_x = graph.get_operation_by_name("train_input").outputs[0]
            #train_input
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors to evaluate
            predictions = graph.get_operation_by_name("Model/predictions").outputs[0]


            #Prediction
            predictions = sess.run(predictions, {input_x: x_test, dropout_keep_prob: 1.0})
            #print (predictions)
            labellist = predictions.tolist()
            fileobj = open('label.json','w')
            fileobj.write(json.dumps(labellist))
            fileobj.close()

            return labellist

def load_par_data(filepath):
    raw_data=[]
    par_data = open(filepath)
    jsonString=json.load(par_data)
    for str in jsonString:
        raw_data.append(str+'\n')
    return raw_data

def load_pre_data(filepath):
    raw_data=[]
    par_data = open(filepath)
    jsonString=json.load(par_data)
    for str in jsonString:
        raw_data.append(str)
    return raw_data

def output(l1,l2):
    list2=[]
    for x,y in zip(l1,l2):
        list=[]
        list.append(x)
        list.append(y)
        list2.append(list)
    list3=[]
#   print (list2)
    for index,item in enumerate(list2):
        if index%2 == 0:
            list4=[]
            list4.append(item)
            list4.append(list2[index+1])
            list3.append(list4)
#       print (list3)
    fileobj = open('zhihu_Bi-LSTM_labeled.json','w')
    fileobj.write(json.dumps(list3,ensure_ascii=False))
        #for pair in list3:
        #fileobj.write(json.dumps(str(pair)))
    fileobj.close()


if __name__ == "__main__":

    par = load_par_data("/Users/siqikang/Documents/master_grade1/semester2/EmotionRecog/model/zhihu_partition.json")

    pre = load_pre_data("/Users/siqikang/Documents/master_grade1/semester2/EmotionRecog/model/label.json")

    output(par,pre)


