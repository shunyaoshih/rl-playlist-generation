"""Main function to decide the execution mode"""

import os
import tensorflow as tf

from lib.config import params_setup
from lib.utils import print_parameters
from lib.model_utils import create_model_dir, load_weights, create_graph
from lib.setup import config_setup, logging_config_setup

from lib.pretrain import pretrain
from lib.rl import policy_gradient
from lib.test import test

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    PARA = params_setup()
    create_model_dir(PARA)
    logging_config_setup(PARA)
    print_parameters(PARA)

    GRAPH, MODEL = create_graph(PARA)

    with tf.Session(config=config_setup(), graph=GRAPH) as sess:
        sess.run(tf.global_variables_initializer())
        load_weights(PARA, sess, MODEL)

        COORD = tf.train.Coordinator()
        THREADS = tf.train.start_queue_runners(sess=sess, coord=COORD)
        try:
            if PARA.mode == 'pretrain':
                pretrain(PARA, sess, MODEL)
            elif PARA.mode == 'rl':
                policy_gradient(PARA, sess, MODEL)
            elif PARA.mode == 'test':
                test(PARA, sess, MODEL)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
        finally:
            print('Stop')
            COORD.request_stop()
            COORD.join(THREADS)
