"""Convert input data to Standard Tensorflow Format"""

import sys
from collections import defaultdict
from tqdm import tqdm

import tensorflow as tf

MAX_LEN = int(sys.argv[1])

def read_vocabulary_file():
    """Read the vocabulary dictionary"""
    vocab_list = open('./vocab_default.txt', 'r').read().splitlines()
    dct = defaultdict(lambda: 3, [[word, i] for i, word in enumerate(vocab_list)])
    return dct

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _list_feature(lst):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=lst))

def convert_to_tf_format(mode):
    """ the main function to convert data """

    seqs = open('./train_ids.txt', 'r').read().splitlines()
    seqs = [seq.split(' ') for seq in seqs]

    for i, seq in enumerate(seqs):
        for j, word in enumerate(seq):
            if word == '3':
                raise ValueError('UNKNOWN SONG: ({}, {})'.format(i + 1, j + 1))

    input_list = [seq[0:-1] for seq in seqs]
    input_list_len = [len(seq) - 1  for seq in seqs]
    output_list = [seq[-1] for seq in seqs]

    input_list = [[int(word) for word in seq] for seq in input_list]
    output_list = [int(word) for word in output_list]

    input_list = [seq + [0] * (MAX_LEN - len(seq)) for seq in input_list]

    writer = tf.python_io.TFRecordWriter('{}.tfrecords'.format(mode))
    for i in tqdm(range(len(input_list))):
        example = tf.train.Example(features=tf.train.Features(feature={
            'rnn_input': _list_feature(input_list[i]),
            'rnn_input_len': _int64_feature(input_list_len[i]),
            'target_output': _int64_feature(output_list[i]),
        }))
        writer.write(example.SerializeToString())
    writer.close()

if __name__ == '__main__':
    print('convert training data to Standard TensorFlow Format')
    convert_to_tf_format('train')
