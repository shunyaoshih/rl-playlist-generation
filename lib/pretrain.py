"""The procedure to pretrain the model"""

import logging
import time
import numpy as np

from lib.utils import read_all_embedding, read_pretrained_embedding
from lib.model_utils import save_model_dir, save_model

def pretrain(para, sess, model):
    """The procedure to pretrain the model"""
    embed_dct = read_all_embedding()

    step_time = 0.0
    for step in range(1, para.steps + 1):
        start_time = time.time()

        [raw_rnn_inputs, raw_rnn_inputs_len, raw_target_outputs] = \
            sess.run(
                fetches=[
                    model.raw_rnn_inputs,
                    model.raw_rnn_inputs_len,
                    model.raw_target_outputs,
                ]
            )

        rnn_inputs_embedded = read_pretrained_embedding(para, embed_dct, raw_rnn_inputs)

        [loss, _] = sess.run(
            fetches=[
                model.loss,
                model.update,
            ],
            feed_dict={
                model.rnn_inputs_embedded: rnn_inputs_embedded,
                model.rnn_inputs_len: raw_rnn_inputs_len,
                model.target_outputs: raw_target_outputs,
            }
        )

        step_time += (time.time() - start_time)
        if step % para.steps_per_stats == 0 or step == 1:
            logging.info('step: %d, perplexity: %.2f, step_time: %.2f => save model to %s', \
                         step, np.exp(loss), step_time / para.steps_per_stats,
                         save_model_dir(para))
            save_model(para, sess, model)
            step_time = 0.0
            if para.debug == 1:
                exit()
