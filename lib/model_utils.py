"""Utility functions related to the model"""
import os
import logging
import tensorflow as tf
import numpy as np

from lib.model import RNN
from lib.utils import read_vocabulary_file

def create_model_dir(para):
    """Create model directory if not exists"""
    try:
        if para.mode == 'pretrain':
            os.makedirs(para.model_dir)
        else:
            os.makedirs(para.rl_model_dir)
    except os.error:
        pass

def save_model_dir(para):
    """Decide where the model should be saved based on the parameters"""
    if para.mode == 'pretrain':
        return para.model_dir
    return para.rl_model_dir

def create_graph(para):
    """Function to create graph given parameters"""
    with tf.Graph().as_default() as graph: # pylint: disable=not-context-manager
        initializer = tf.random_uniform_initializer(
            -para.init_weight, para.init_weight)
        with tf.variable_scope('model', initializer=initializer):
            model = RNN(para)
        return graph, model

def load_weights(para, sess, model):
    """Load weights if it exists; otherwise, create a fresh one"""
    # pylint: disable=no-member
    if para.mode == 'pretrain' or para.mode == 'test':
        ckpt = tf.train.get_checkpoint_state(para.model_dir)
        if ckpt:
            logging.info('Loading model from %s', ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            logging.info('Loading model with fresh parameters')
            sess.run(tf.global_variables_initializer())
    else:
        ckpt = tf.train.get_checkpoint_state(para.rl_model_dir)
        if ckpt:
            logging.info('Loading model from %s', ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            ckpt = tf.train.get_checkpoint_state(para.model_dir)
            if ckpt:
                logging.info('Loading model from %s', ckpt.model_checkpoint_path)
                model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                logging.info('Loading model with fresh parameters')
                sess.run(tf.global_variables_initializer())
    # pylint: enable=no-member

def save_model(para, sess, model):
    """Save the model based on the parameters"""
    [global_step] = sess.run([model.global_step])
    checkpoint_path = os.path.join(save_model_dir(para), "model.ckpt")
    model.saver.save(sess, checkpoint_path, global_step=global_step)

# pylint: disable=too-many-locals, too-many-arguments
def predict(rev_vocab, embed_dct, para, sess, model, seed_ids,
            output_lengths=None, sample=False):
    """Given a list of seed songs, output playlists by sampling or argmax"""
    if output_lengths is None:
        output_lengths = para.max_len
    vocab_dct = read_vocabulary_file()
    ids = []
    song_ids = []
    song_ids_embed = []
    for i, seed in enumerate(seed_ids):
        now = [0] * para.max_len
        now[0] = vocab_dct[seed]
        ids.append(now)

        raw_now = [0] * para.max_len
        raw_now[0] = seed
        song_ids.append(raw_now)

        now_embed = [np.array([0] * para.embedding_size)] * para.max_len
        now_embed[0] = np.array(embed_dct[seed])
        song_ids_embed.append(now_embed)

    for song_num in range(1, para.max_len):
        if song_num > output_lengths - 1:
            predicted_ids = [0] * para.batch_size
        elif sample and song_num == output_lengths - 1:
            [predicted_ids] = sess.run(
                fetches=[model.sampled_ids],
                feed_dict={
                    model.rnn_inputs_embedded: np.array([np.array(s) for s in song_ids_embed]),
                    model.rnn_inputs_len: np.array([song_num] * para.batch_size),
                }
            )
            predicted_ids = np.reshape(predicted_ids, (para.batch_size))
        else:
            [predicted_ids] = sess.run(
                fetches=[model.predicted_ids],
                feed_dict={
                    model.rnn_inputs_embedded: np.array([np.array(s) for s in song_ids_embed]),
                    model.rnn_inputs_len: np.array([song_num] * para.batch_size),
                }
            )
        for i in range(para.batch_size):
            now_song_id = rev_vocab[predicted_ids[i]]
            ids[i][song_num] = predicted_ids[i]
            song_ids[i][song_num] = now_song_id
            song_ids_embed[i][song_num] = np.array(embed_dct[now_song_id])

    return np.array([np.array(seq) for seq in ids]), np.array([np.array(seq) for seq in song_ids])
