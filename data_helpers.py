import numpy as np
import re
import itertools
from collections import Counter
import os


def clean_str(string):

    string = re.sub(r"\s{2,}", " ", string)

    return string.strip()



def load_data_and_labels(train_label_0, train_label_1,train_label_2,train_label_3,train_label_4,train_label_5):

    # Load data from files
    example_0 = list(open(train_label_0, "r",encoding='utf-8').readlines())
    example_0 = [s.strip() for s in example_0]
    print ("len of 0:" + str(len(example_0)))
    example_1 = list(open(train_label_1, "r",encoding='utf-8').readlines())
    example_1 = [s.strip() for s in example_1]
    print ("len of 1:" + str(len(example_1)))
    example_2 = list(open(train_label_2, "r",encoding='utf-8').readlines())
    example_2 = [s.strip() for s in example_2]
    print ("len of 2:"+ str(len(example_2)))
    example_3 = list(open(train_label_3, "r",encoding='utf-8').readlines())
    example_3 = [s.strip() for s in example_3]
    print ("len of 3:"+ str(len(example_3)))
    example_4 = list(open(train_label_4, "r",encoding='utf-8').readlines())
    example_4 = [s.strip() for s in example_4]
    print ("len of 4:"+str(len(example_4)))
    example_5 = list(open(train_label_5, "r",encoding='utf-8').readlines())
    example_5 = [s.strip() for s in example_5]
    print ("len of 5:"+str(len(example_5)))
    # Split by words
    x_text = example_0 + example_1 + example_2 + example_3 + example_4 + example_5
    x_text = [clean_str(sent) for sent in x_text]
    # with open ("test.txt","w",encoding='utf-8') as f:
    #     for i in x_text:
    #         f.write(i + "\n")
    # Generate labels
    label_0 = [[1, 0, 0, 0, 0, 0] for _ in example_0]
    label_1 = [[0, 1, 0, 0, 0, 0] for _ in example_1]
    label_2 = [[0, 0, 1, 0, 0, 0] for _ in example_2]
    label_3 = [[0, 0, 0, 1, 0, 0] for _ in example_3]
    label_4 = [[0, 0, 0, 0, 1, 0] for _ in example_4]
    label_5 = [[0, 0, 0, 0, 0, 1] for _ in example_5]


    y = np.concatenate([label_0, label_1, label_2, label_3, label_4, label_5], 0)
    return [x_text, y]



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def epochs_iter(data, batch_size, num_epochs,shuffle=True):

    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    # Shuffle the data at each epoch
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]