"""Funcstions to define configurations"""

import logging
import tensorflow as tf

from lib.model_utils import save_model_dir

def config_setup():
    """GPU configuration"""
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=no-member
    config.allow_soft_placement = True
    return config

def logging_config_setup(para):
    """logging configuration"""
    if para.file_output == 0:
        logging.basicConfig(level=para.logging_level,
                            format='%(levelname)-8s - %(message)s')
    else:
        logging.basicConfig(level=para.logging_level,
                            format='%(levelname)-8s - %(message)s',
                            filename=save_model_dir(para) + '/progress.txt')
    tf.logging.set_verbosity(tf.logging.ERROR)
