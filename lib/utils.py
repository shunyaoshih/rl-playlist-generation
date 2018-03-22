"""Utility functions"""

from collections import defaultdict
import logging

import numpy as np
from numpy import linalg as LA

EPSILON = 1e-8

def num_of_lines(file_name):
    """Count the number of lines in a file"""
    with open(file_name) as input_file:
        for i, _ in enumerate(input_file):
            pass
        return i + 1 # pylint: disable=undefined-loop-variable

def read_vocabulary_file():
    """Read vocabulary file"""
    vocab_list = open('./data/vocab_default.txt', 'r').read().splitlines()
    dct = defaultdict(lambda: 3, [[word, i] for i, word in enumerate(vocab_list)])
    return dct

def read_rev_vocab():
    """Read the mapping from id to word"""
    return open('./data/vocab_default.txt', 'r').read().splitlines()

def read_all_song_ids():
    """Read all song ids"""
    vocab_list = open('./data/vocab_default.txt', 'r').readlines()
    vocab_list = [word.strip() for word in vocab_list]
    return vocab_list

def print_parameters(para):
    """Print all important parameters"""
    important_args = ['mode', 'batch_size', 'max_len', 'num_units', 'attention',
                      'num_layers', 'vocab_size', 'embedding_size', 'optimizer',
                      'concatenate']

    if para.mode == 'pretrain' or para.mode == 'rl':
        output_dir = para.model_dir
        if para.mode == 'rl':
            output_dir = para.rl_model_dir
        with open(output_dir + '/para.txt', 'w') as para_file:
            for arg in vars(para):
                para_file.write(str(arg) + ": " + str(getattr(para, arg)) + '\n')

    if para.mode == 'rl':
        important_args.append('distance')
        important_args.append('dist')
        important_args.append('pop')
        important_args.append('date')
        important_args.append('probs')
    value = [0] * len(important_args)
    for arg in vars(para):
        for i, prop in enumerate(important_args):
            if arg == prop:
                value[i] = getattr(para, arg)
                break
    for i, prop in enumerate(important_args):
        logging.info("%s: %s", prop, value[i])

def read_pretrained_embedding(para, embed_dct, raw_input_ids):
    """Read pretrained embeddings, and return as an array (fake id -> embedding)"""
    rev_vocab = open('./data/vocab_default.txt', 'r').read().splitlines()
    rnn_inputs_embedded = []
    for i in range(para.batch_size):
        now_batch = []
        for j in range(para.max_len):
            now_batch.append(np.array(embed_dct[rev_vocab[raw_input_ids[i][j]]]))
        rnn_inputs_embedded.append(np.array(now_batch))
    return np.asarray(rnn_inputs_embedded)

def read_all_embedding():
    """Read pretrained embeddings, and return as an dictionary (real id -> embeding)"""
    embed_file = open('./data/embedding.txt', 'r').read().splitlines()
    embed_file = [seq.split(' ') for seq in embed_file]
    seed_ids = [seq[0] for seq in embed_file]
    embed_seqs = [[float(seq[i]) for i in range(len(seq)) if i != 0] for seq in embed_file]
    embed_dct = {}
    for i, seq in enumerate(embed_seqs):
        embed_dct[seed_ids[i]] = seq
    return embed_dct

def read_all_seed_ids():
    """Read all seed song ids"""
    raw_file = open('./data/raw/raw_data.txt', 'r').read().splitlines()
    raw_file = [seq.split(' ') for seq in raw_file]
    seed_id_list = [seq[1] for seq in raw_file]
    return seed_id_list

def read_metrics(file_path):
    """Read all metrics

    file format: song_id artist_id popularity_score release_date
    """
    raw_file = open(file_path, 'r').readlines()
    raw_file = [seq.strip().split(' ') for seq in raw_file]
    for i, seq in enumerate(raw_file):
        if seq[2] == '':
            raw_file[i][2] = '0.0'
    return raw_file

def read_popularity(file_path='./data/metrics.txt'):
    """Read the metrics of novelty"""
    raw_file = read_metrics(file_path)

    mapping = {}
    vocab_list = read_all_song_ids()
    for word in vocab_list:
        mapping[word] = 1.0

    for seq in raw_file:
        song_id = seq[0]
        popularity_score = float(seq[2])
        mapping[song_id] = popularity_score
    return mapping

def read_artist(file_path='./data/metrics.txt'):
    """Read the metrics of diversity"""
    raw_file = read_metrics(file_path)

    mapping = {}
    vocab_list = read_all_song_ids()
    for word in vocab_list:
        mapping[word] = '-1'

    for seq in raw_file:
        song_id = seq[0]
        artist_id = seq[1]
        mapping[song_id] = artist_id
    return mapping

def read_date(file_path='./data/metrics.txt'):
    """Read the metrics of freshness"""
    constant_date = int((1900 + 2017) / 2)

    raw_file = read_metrics(file_path)

    mapping = {}
    vocab_list = read_all_song_ids()
    for word in vocab_list:
        mapping[word] = 2017

    for seq in raw_file:
        song_id = seq[0]
        date = int(seq[3])
        if date == 0:
            mapping[song_id] = constant_date
        else:
            mapping[song_id] = date
    return mapping

def log(data):
    """Safe log"""
    return np.log(data + EPSILON)

def cal_diff_distance(msg, para, rnn_inputs, rnn_inputs_len, sampled_song_ids):
    """
    Given a playlist {s_1, s_2, ..., s_n} and the next song a,
    calculate the euclidean distance of the embeddings between s_n and a
    """
    rev_vocab = cal_diff_distance.rev_vocab
    embed_dct = cal_diff_distance.embed_dct

    diff_distance = np.zeros(para.batch_size)
    if para.dist > EPSILON:
        for i in range(para.batch_size):
            target_idx = rnn_inputs_len[i] - 1
            target_song_id = rev_vocab[rnn_inputs[i][target_idx]]
            target_vector = np.asarray(embed_dct[target_song_id])

            sampled_song_id = sampled_song_ids[i]
            sampled_vector = np.asarray(embed_dct[sampled_song_id])

            diff_distance[i] = LA.norm(np.subtract(sampled_vector, target_vector))
        msg['diff_distance'] = np.mean(diff_distance, axis=0)
        diff_distance = np.absolute(diff_distance - para.distance)
        diff_distance = -log(diff_distance)
    return diff_distance
cal_diff_distance.rev_vocab = read_rev_vocab()
cal_diff_distance.embed_dct = read_all_embedding()

def cal_popularity_score(msg, para, sampled_song_ids):
    """Calculate popularity scores"""
    popularity_mapping = cal_popularity_score.popularity_mapping

    popularity_score = np.zeros(para.batch_size)
    if para.pop > EPSILON:
        for i in range(para.batch_size):
            sampled_song_id = sampled_song_ids[i]
            popularity_score[i] = popularity_mapping[sampled_song_id]
        msg['popularity_score'] = np.mean(popularity_score, axis=0)
        popularity_score = -log(popularity_score)
    return popularity_score
cal_popularity_score.popularity_mapping = read_popularity()

def cal_date_score(msg, para, sampled_song_ids):
    """Calculate release years"""
    date_mapping = cal_date_score.date_mapping

    date_score = np.zeros(para.batch_size)
    if para.date > EPSILON:
        for i in range(para.batch_size):
            sampled_song_id = sampled_song_ids[i]
            date = date_mapping[sampled_song_id]
            date_score[i] = (date - 1900) / (2017 - 1900)
        msg['date'] = np.mean(date_score, axis=0) * (2017 - 1900) + 1900
        date_score = -log(date_score)
    return date_score
cal_date_score.date_mapping = read_date()

def cal_original_probs(msg, para, sampled_ids, probs):
    """Calculate the corresponding probabilities of the sampled ids"""
    original_probs = np.zeros(para.batch_size)
    if para.probs > EPSILON:
        for i in range(para.batch_size):
            original_probs[i] = probs[i][sampled_ids[i]]
        msg['original_probs'] = np.mean(original_probs, axis=0)
        original_probs = log(original_probs)
    return original_probs

def reward_functions(para, rnn_inputs, rnn_inputs_len, sampled_ids, probs):
    """
        rnn_inputs: [batch_size, max_len]
        rnn_inputs_len: [batch_size]
        sampled_ids: [batch_size]
        rewards: [batch_size]
    """
    rewards = np.zeros(para.batch_size)
    msg = {}

    rev_vocab = reward_functions.rev_vocab
    sampled_song_ids = [rev_vocab[sampled_ids[i]] for i in range(para.batch_size)]

    diff_distance = cal_diff_distance(msg, para, rnn_inputs, rnn_inputs_len, sampled_song_ids)
    popularity_score = cal_popularity_score(msg, para, sampled_song_ids)
    date_score = cal_date_score(msg, para, sampled_song_ids)
    original_probs = cal_original_probs(msg, para, sampled_ids, probs)

    # sum up
    rewards = para.dist * diff_distance + \
              para.pop * popularity_score + \
              para.date * date_score + \
              para.probs * original_probs

    msg['mean_reward'] = np.mean(rewards, axis=0)
    rewards -= np.mean(rewards, axis=0)

    return rewards, msg
reward_functions.rev_vocab = read_rev_vocab()
