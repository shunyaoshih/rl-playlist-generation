"""The procedure of policy gradient reinforcement learning"""

from copy import deepcopy
import logging
import random
import time

import numpy as np
import tensorflow as tf

from lib.setup import config_setup
from lib.utils import read_all_embedding, read_all_seed_ids, read_rev_vocab, \
                      read_pretrained_embedding, reward_functions
from lib.model_utils import load_weights, save_model_dir, save_model, predict
from lib.model import RNN

def create_original_graph(para):
    """Create the pretrained model"""
    original_para = deepcopy(para)
    original_para.mode = 'pretrain'
    original_para.model_dir = para.original_model_dir

    with tf.Graph().as_default() as original_graph: # pylint: disable=not-context-manager
        initializer = tf.random_uniform_initializer(
            -original_para.init_weight, original_para.init_weight
        )
        with tf.variable_scope('model', initializer=initializer):
            original_model = RNN(original_para)
        return original_para, original_graph, original_model

def load_original_weights(original_para, original_sess, original_model):
    "Load weights of the pretrained model"
    original_sess.run(tf.global_variables_initializer())
    logging.info('Loading original model')
    load_weights(original_para, original_sess, original_model)

def policy_gradient(para, sess, model): # pylint: disable=too-many-locals
    """The procedure of policy gradient reinforcement learning"""
    embed_dct = read_all_embedding()

    seed_id_list = read_all_seed_ids()

    rev_vocab = read_rev_vocab()

    original_para, original_graph, original_model = create_original_graph(para)

    with tf.Session(config=config_setup(), graph=original_graph) as original_sess:
        load_original_weights(original_para, original_sess, original_model)

        step_time = 0.0
        for step in range(1, para.steps + 1):
            start_time = time.time()

            chosen_ids = random.sample(range(0, len(seed_id_list)), para.batch_size)
            seed_ids = [seed_id_list[idx] for idx in chosen_ids]

            output_lengths = random.randint(1, para.max_len - 1)

            # raw_rnn_inputs: [batch_size, output_lengths]
            raw_rnn_inputs, _ = predict(rev_vocab, embed_dct, para, sess, model,
                                        seed_ids, output_lengths, True)
            # raw_rnn_inputs_len: [batch_size]
            raw_rnn_inputs_len = np.array([output_lengths] * para.batch_size)

            # raw_inputs_embedded: [batch_size, output_lengths, embedding_size]
            rnn_inputs_embedded = read_pretrained_embedding(para, embed_dct, raw_rnn_inputs)

            # get original probs
            [probs] = original_sess.run(
                fetches=[original_model.probs],
                feed_dict={
                    original_model.rnn_inputs_embedded: rnn_inputs_embedded,
                    original_model.rnn_inputs_len: raw_rnn_inputs_len,
                })

            # get sampled ids
            [sampled_ids] = sess.run(
                fetches=[model.sampled_ids],
                feed_dict={
                    model.rnn_inputs_embedded: rnn_inputs_embedded,
                    model.rnn_inputs_len: raw_rnn_inputs_len,
                })
            sampled_ids = np.reshape(sampled_ids, (para.batch_size))

            # get reward
            rewards, msg = reward_functions(para, raw_rnn_inputs, raw_rnn_inputs_len,
                                            sampled_ids, probs)
            [_] = sess.run(
                fetches=[model.rl_update],
                feed_dict={
                    model.rnn_inputs_embedded: rnn_inputs_embedded,
                    model.rnn_inputs_len: raw_rnn_inputs_len,
                    model.sampled_ids_inputs: sampled_ids,
                    model.rewards: rewards
                })

            step_time += (time.time() - start_time)
            if step % para.steps_per_stats == 0 or step == 1:
                logging.info('step: %d, reward: %.5f, step_time: %.2f => save model to %s',
                             step, msg['mean_reward'], step_time / para.steps_per_stats,
                             save_model_dir(para))
                for key, value in msg.items():
                    if key == 'mean_reward':
                        continue
                    logging.info('%s: %.2f', key, value)
                save_model(para, sess, model)
                step_time = 0

                if para.debug == 1:
                    exit()
